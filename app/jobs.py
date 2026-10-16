# ============================================================================
# app/jobs.py — run folders, the shared queue, progress and log plumbing
# ----------------------------------------------------------------------------
# Front-end agnostic ON PURPOSE. It sits beside app/web/ rather than inside it
# because a task's cli.py will want exactly the same run-folder layout and the
# same progress reporting as its web page -- that is what makes a CLI run and a
# web run produce byte-identical output folders.
#
# Nothing here imports gradio, and nothing here knows what a task computes.
# ============================================================================
import csv
import logging
import shutil
import threading
import time
import uuid
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, List, Optional

import settings

logger = logging.getLogger(__name__)

# A task hands this to service code so it can say what it is doing. `None` is
# always acceptable and means "nobody is listening".
Reporter = Optional[Callable[[str], None]]


# ============================================================================
# Run folder
# ----------------------------------------------------------------------------
# data/runs/<task>/<YYYYmmdd-HHMMSS>_<6 hex>/
#     inputs/    exactly what the user supplied, staged server-side
#     outputs/   everything the job produced -- this is what gets zipped
#     run.log    the full transcript, including what the progress box elided
# ============================================================================
@dataclass
class RunDir:
    task_slug: str
    job_id: str
    root: Path
    inputs: Path
    outputs: Path
    log_file: Path

    def output(self, *parts) -> Path:
        """Path inside outputs/, with parent folders created."""
        p = self.outputs.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def zip_outputs(self, archive_name: Optional[str] = None,
                    nest: bool = False,
                    nest_folder: Optional[str] = None,
                    subject_id: Optional[str] = None,
                    model: Optional[str] = None) -> Path:
        """Zip outputs/ into the run root and return the archive path.

        Written next to outputs/ rather than inside it, so re-zipping a run can
        never pick up the previous archive.

        The archive is named `<subject>_<task slug>[_<model>]___output.zip`
        (output_zip_name) -- the convention every task follows -- from
        `subject_id` and, for a task tied to one conductivity model, `model`.
        `archive_name` overrides it; only the older tasks still pass one.

        The files sit at the archive's root: unpacking into a folder named
        after the archive (what every unzip tool offers) gives
        `<subject>_<task>_<model>___output/vtk/...`, and a second folder
        inside it would only repeat the name. `nest` / `nest_folder` keep the
        old layout available for an archive some reader still expects nested.
        """
        name = archive_name or output_zip_name(self.task_slug, subject_id,
                                               model)
        archive = self.root / f"{name}.zip"
        nest = nest or nest_folder is not None
        root = f"{nest_folder or name}/" if nest else ""
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(self.outputs.rglob("*")):
                if path.is_file():
                    zf.write(path, f"{root}{path.relative_to(self.outputs)}")
        return archive


# What a task's "download all" archive is called: the subject, the task and
# -- when the task is tied to one conductivity model -- that model's tag,
# then `___output`: `IXI258_mesh_sim_tbe___output.zip`. One archive per run,
# flat inside, so a downloads folder holding runs of several tasks and
# subjects reads by itself and each unpacks into a folder of that name.
OUTPUT_ZIP_SUFFIX = "___output"


def output_zip_name(task_slug: str, subject_id: Optional[str] = None,
                    model: Optional[str] = None) -> str:
    """`<subject>_<task slug>[_<model>]___output` -- the archive's stem."""
    parts = [p for p in (subject_id, task_slug, model) if p]
    return "_".join(parts) + OUTPUT_ZIP_SUFFIX

def new_run(task_slug: str) -> RunDir:
    """Allocate a fresh run folder for one job."""
    job_id = f"{datetime.now():%Y%m%d-%H%M%S}_{uuid.uuid4().hex[:6]}"
    root = Path(settings.RUNS_DIR) / task_slug / job_id
    inputs, outputs = root / "inputs", root / "outputs"
    inputs.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    return RunDir(task_slug, job_id, root, inputs, outputs, root / "run.log")


def prune_old_runs(task_slug: Optional[str] = None) -> int:
    """Delete run folders older than settings.KEEP_RUNS_DAYS. Returns the count.

    Best-effort: a folder that cannot be removed (open file, permissions) is
    logged and skipped rather than failing the job that triggered the sweep.
    """
    days = int(getattr(settings, "KEEP_RUNS_DAYS", 0) or 0)
    if days <= 0:
        return 0
    cutoff = time.time() - timedelta(days=days).total_seconds()
    roots = ([Path(settings.RUNS_DIR) / task_slug] if task_slug
             else list(Path(settings.RUNS_DIR).glob("*")))
    n = 0
    for root in roots:
        if not root.is_dir():
            continue
        for run in root.iterdir():
            if run.is_dir() and run.stat().st_mtime < cutoff:
                try:
                    shutil.rmtree(run)
                    n += 1
                except OSError as exc:
                    logger.warning("Could not prune %s: %s", run, exc)
    return n


def stage_upload(src, run: RunDir, name: Optional[str] = None) -> Path:
    """Copy an uploaded file into the run's inputs/ and return the new path.

    The browser hands Gradio a temp path that is not ours to keep; copying it in
    makes the run folder a complete record of what was computed from.
    """
    src = Path(str(src))
    dst = run.inputs / (name or src.name)
    if src.resolve() != dst.resolve():
        shutil.copyfile(src, dst)
    return dst


# ============================================================================
# The shared queue
# ----------------------------------------------------------------------------
# Whole-brain tractography and charm each want the whole machine, so jobs are
# serialized across every task rather than per task. A caller that has to wait
# is told how many jobs are ahead of it instead of staring at a frozen page.
# ============================================================================
class RunQueue:
    def __init__(self, max_concurrent: int = 1):
        self._sem = threading.Semaphore(max_concurrent)
        self._lock = threading.Lock()
        self._waiting = 0
        self._running = 0

    @property
    def busy(self) -> int:
        with self._lock:
            return self._running + self._waiting

    @contextmanager
    def slot(self, report: Reporter = None, run: Optional[RunDir] = None):
        """Block until this job may run. Use as a `with` around the heavy work.
        `run` stamps the usage log's started_at the moment the slot is
        taken -- the queue wait is the difference to requested_at."""
        with self._lock:
            ahead = self._running + self._waiting
            self._waiting += 1
        if ahead and report:
            report(f"Queued behind {ahead} running/waiting job(s) — "
                   f"this page will start on its own, no need to resubmit.")
        self._sem.acquire()
        with self._lock:
            self._waiting -= 1
            self._running += 1
        if run is not None:
            usage_update(run.job_id, started_at=_usage_now())
        try:
            yield
        finally:
            with self._lock:
                self._running -= 1
            self._sem.release()


QUEUE = RunQueue(getattr(settings, "MAX_CONCURRENT_JOBS", 1))


# ============================================================================
# Usage log
# ----------------------------------------------------------------------------
# One CSV for the whole app, data/usage_log.csv (settings.USAGE_LOG): a row
# per calculation request -- which task, from which browser session and IP,
# when it was requested, when it actually started (after the queue) and when
# it finished. The row is appended when the request is handed to its thread
# (usage_start, from app/web/common.stream_job) and completed IN PLACE when
# the job ends (usage_finish), so the columns themselves tell the state: no
# started_at = still queued, no completed_at = still running (or the server
# died under it). completed_at is a time of day only; its date is
# requested_at's (a job does not run for a day), and duration_s is exact.
# The file is small (one line per job) and rewritten whole under a lock;
# nothing in a run folder depends on it, a lost log costs only statistics,
# so every failure here is logged and swallowed.
#
# Front-end agnostic like the rest of this file: the session id and the IP
# arrive as strings (the web layer reads them off gradio's Request); a cli
# passes nothing and the columns stay empty.
# ============================================================================
USAGE_COLUMNS = ("job_id", "task", "session_id", "ip", "requested_at",
                 "started_at", "completed_at", "duration_s")
_USAGE_LOCK = threading.Lock()
_USAGE_STAMP = "%Y-%m-%d %H:%M:%S"
_USAGE_TIME = "%H:%M:%S"


def _usage_now() -> str:
    return datetime.now().strftime(_USAGE_STAMP)


def usage_start(run: RunDir, session_id: str = "", ip: str = "") -> None:
    """Append the row for `run`: requested now, not yet started."""
    row = {column: "" for column in USAGE_COLUMNS}
    row.update(job_id=run.job_id, task=run.task_slug,
               session_id=session_id or "", ip=ip or "",
               requested_at=_usage_now())
    path = Path(settings.USAGE_LOG)
    try:
        with _USAGE_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            fresh = not path.exists() or path.stat().st_size == 0
            if not fresh:
                # A log written with an older column set is set aside, not
                # appended to with a different shape.
                with open(path, newline="", encoding="utf-8") as fh:
                    header = next(csv.reader(fh), [])
                if tuple(header) != USAGE_COLUMNS:
                    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    path.rename(path.with_name(f"{path.stem}.{stamp}.old.csv"))
                    fresh = True
            with open(path, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=USAGE_COLUMNS)
                if fresh:
                    writer.writeheader()
                writer.writerow(row)
    except OSError as exc:
        logger.warning("usage log: could not append %s: %s", run.job_id, exc)


def usage_update(job_id: str, **fields) -> None:
    """Fill columns of one row in place. Unknown columns are ignored; a job
    the log does not know (an older log, a log that failed to append) is
    left alone rather than invented."""
    path = Path(settings.USAGE_LOG)
    try:
        with _USAGE_LOCK:
            if not path.exists():
                return
            with open(path, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            hit = False
            for row in rows:
                if row.get("job_id") == job_id:
                    row.update({k: v for k, v in fields.items()
                                if k in USAGE_COLUMNS})
                    hit = True
            if not hit:
                return
            tmp = path.with_name(path.name + ".tmp")
            with open(tmp, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=USAGE_COLUMNS,
                                        extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            tmp.replace(path)
    except OSError as exc:
        logger.warning("usage log: could not update %s: %s", job_id, exc)


def usage_finish(run: RunDir) -> None:
    """Stamp completed_at (time of day) and the duration in seconds from
    started_at (from requested_at when the job never reached the queue)."""
    path = Path(settings.USAGE_LOG)
    started = ""
    try:
        with _USAGE_LOCK:
            if path.exists():
                with open(path, newline="", encoding="utf-8") as fh:
                    for row in csv.DictReader(fh):
                        if row.get("job_id") == run.job_id:
                            started = (row.get("started_at")
                                       or row.get("requested_at") or "")
    except OSError:
        pass
    now = datetime.now()
    duration = ""
    if started:
        try:
            duration = f"{(now - datetime.strptime(started, _USAGE_STAMP)).total_seconds():.1f}"
        except ValueError:
            duration = ""
    usage_update(run.job_id, completed_at=now.strftime(_USAGE_TIME),
                 duration_s=duration)


# ============================================================================
# Progress + logging
# ----------------------------------------------------------------------------
# One transcript, two destinations: the run's own run.log (complete, kept) and
# whatever the front-end passed in (a textbox, a terminal, nothing).
# ============================================================================
@dataclass
class Transcript:
    """Collects progress lines and log records for one job."""
    log_file: Optional[Path] = None
    sink: Reporter = None                       # extra per-line destination
    lines: List[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def report(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        with self._lock:
            self.lines.append(line)
            if self.log_file is not None:
                with open(self.log_file, "a") as fh:
                    fh.write(line + "\n")
        if self.sink is not None:
            self.sink(line)

    def text(self, last: Optional[int] = None) -> str:
        with self._lock:
            rows = self.lines[-last:] if last else list(self.lines)
        return "\n".join(rows)


class _TranscriptHandler(logging.Handler):
    """Mirrors log records of the captured loggers into a Transcript."""

    def __init__(self, transcript: Transcript):
        super().__init__()
        self.transcript = transcript

    def emit(self, record):
        try:
            self.transcript.report(self.format(record))
        except Exception:       # a broken log line must never kill a job
            pass


class _ExcludeLoggers(logging.Filter):
    """Drop records from loggers that already report through a progress callback.

    Without this, a service layer that both calls `report(...)` and logs the
    same message puts every step in the box twice.
    """

    def __init__(self, prefixes):
        super().__init__()
        self.prefixes = tuple(prefixes)

    def filter(self, record) -> bool:
        return not any(record.name == p or record.name.startswith(p + ".")
                       for p in self.prefixes)


@contextmanager
def capture_logs(transcript: Transcript,
                 logger_names=None,
                 level: Optional[str] = None):
    """Route the named loggers into `transcript` for the duration of the block."""
    names = tuple(logger_names if logger_names is not None
                  else settings.CAPTURED_LOGGERS)
    handler = _TranscriptHandler(transcript)
    handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    handler.setLevel(level or settings.LOG_LEVEL)
    excluded = getattr(settings, "UNCAPTURED_LOGGERS", ())
    if excluded:
        handler.addFilter(_ExcludeLoggers(excluded))
    attached = []
    for name in names:
        lg = logging.getLogger(name)
        lg.addHandler(handler)
        attached.append(lg)
    try:
        yield transcript
    finally:
        for lg in attached:
            lg.removeHandler(handler)

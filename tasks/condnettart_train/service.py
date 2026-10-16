# ============================================================================
# tasks/condnettart_train/service.py — train CondNet-TART at a new frequency
# ----------------------------------------------------------------------------
# What the page does, minus the widgets (web.py). A conductivity table (the
# tissue-conductivity utility's <freq>.yaml) goes to the condnettart_training
# project's own scripts, run one after another in that project's environment:
#
#   03_train_conductivity.py --exp <EXPERIMENT> --table <freq>.yaml --dated
#       default data, model/<freq>/ -- or model/<freq>_<YYYYMMDD>/ if
#       model/<freq>/ already holds a model (replacing one there)
#   04_evaluate.py --run <that folder>        test split, for the model card
#   05_export_for_app.py --run <that folder>  the model card
#
# and only its result is collected: condnettart_<freq>_stage0.pth,
# _stage1.pth and condnettart_<freq>.json, copied into this job's outputs.
#
# NOT a background job: train() is a generator the page iterates, and the
# scripts run as the app's child processes. Whenever it stops -- finished,
# the page closed, Stop pressed -- the running script is killed (finally);
# and if the app process itself ends, however it ends (Ctrl+C, kill, even
# kill -9), the kernel ends the script too (PR_SET_PDEATHSIG, see _spawn).
# Nothing is left running out of reach. One training at a time (one GPU).
# ============================================================================
import atexit
import ctypes
import os
import queue
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

import yaml

from app import jobs
from tasks.condnettart_train import settings as task_settings

TASK_SLUG = "condnettart_train"
NUM_CLASSES = 15

_busy = threading.Lock()                  # one training at a time
_children: set = set()                    # running script processes


def _kill(proc: subprocess.Popen) -> None:
    """End a script and anything it started (its own process group)."""
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    _children.discard(proc)


@atexit.register
def _kill_all() -> None:                  # the app is shutting down
    for proc in list(_children):
        _kill(proc)


# ----------------------------------------------------------------------------
# Starting a script so that it cannot outlive the app. PR_SET_PDEATHSIG asks
# the kernel to send the child SIGTERM when its parent dies -- where "parent"
# is the THREAD that started it. Gradio runs handlers on pool threads that
# come and go, so every script is started from this one thread instead, which
# lives exactly as long as the app.
# ----------------------------------------------------------------------------
_PR_SET_PDEATHSIG = 1
_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_spawn_requests: "queue.Queue" = queue.Queue()


def _die_with_app() -> None:              # runs in the child, before exec
    os.setsid()                           # its own group: Stop kills it whole
    _libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM)


def _spawner() -> None:
    while True:
        argv, kwargs, reply = _spawn_requests.get()
        try:
            reply.put(subprocess.Popen(argv, preexec_fn=_die_with_app, **kwargs))
        except Exception as exc:          # noqa: BLE001 - handed back
            reply.put(exc)


threading.Thread(target=_spawner, name="condnettart-train-spawner", daemon=True).start()


def _spawn(argv, **kwargs) -> subprocess.Popen:
    reply: "queue.Queue" = queue.Queue()
    _spawn_requests.put((argv, kwargs, reply))
    result = reply.get()
    if isinstance(result, Exception):
        raise result
    return result


def read_table(path) -> dict:
    """The uploaded conductivity table, checked: frequency, normalization,
    sigma for classes 0..14."""
    try:
        t = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Not a YAML file: {exc}") from None
    missing = [k for k in ("frequency", "normalization", "sigma") if k not in t]
    if missing:
        raise ValueError(f"The table has no {', '.join(missing)} (use the Tissue "
                         f"Conductivity utility's <freq>.yaml)")
    classes = {int(k) for k in t["sigma"]}
    if not set(range(NUM_CLASSES)) <= classes:
        raise ValueError(f"sigma must cover classes 0-{NUM_CLASSES - 1}; missing "
                         f"{sorted(set(range(NUM_CLASSES)) - classes)}")
    return t


@dataclass
class Progress:
    frequency: str
    state: str                    # "running" | "done" | "failed" | "stopped"
    started: float
    log: List[str] = field(default_factory=list)
    run_dir: Optional[str] = None
    files: List[str] = field(default_factory=list)

    def summary(self) -> str:
        minutes = (time.time() - self.started) / 60
        what = {"running": f"running, {minutes:.0f} min so far",
                "done": f"done in {minutes:.0f} min",
                "failed": "FAILED (see the log)",
                "stopped": "stopped"}[self.state]
        return (f"**{self.frequency}** · {what}"
                + (f" · `{self.run_dir}`" if self.run_dir else ""))

    def tail(self) -> str:
        return "\n".join(self.log[-task_settings.CONSOLE_TAIL_LINES:])


def _steps(table_copy: Path, smoke: bool):
    """The commands, as (label, argv) -- the run folder filled in later."""
    py = str(task_settings.TRAINING_PYTHON)
    yield ("train", [py, "-u", "script/03_train_conductivity.py",
                     "--exp", task_settings.EXPERIMENT, "--table", str(table_copy),
                     "--dated"] + (["--smoke"] if smoke else []))
    if not smoke:                         # export refuses a quick-test model
        yield ("evaluate", [py, "-u", "script/04_evaluate.py", "--run", "{run}",
                            "--examples", "0"])
        yield ("export", [py, "-u", "script/05_export_for_app.py", "--run", "{run}"])


def train(table_path, smoke: bool = False) -> Iterator[Progress]:
    """Run the training, yielding its progress every POLL_SECONDS until done."""
    table = read_table(table_path)
    for p in (task_settings.TRAINING_PROJECT, task_settings.TRAINING_PYTHON):
        if not p.exists():
            raise FileNotFoundError(f"Not found: {p}")
    if not _busy.acquire(blocking=False):
        raise ValueError("A training is already running; one at a time")
    proc = None
    try:
        freq = str(table["frequency"])
        run = jobs.new_run(TASK_SLUG)
        table_copy = run.root / f"{freq}.yaml"
        shutil.copyfile(table_path, table_copy)
        prog = Progress(freq, "running", time.time())
        log_file = open(run.root / "train.log", "a", encoding="utf-8")
        try:
            for label, argv in _steps(table_copy, smoke):
                argv = [a.replace("{run}", prog.run_dir or "") for a in argv]
                prog.log.append("$ " + " ".join(Path(a).name if a.endswith(".py") else a
                                                for a in argv[1:]))
                proc = _spawn(argv, cwd=task_settings.TRAINING_PROJECT, text=True,
                              bufsize=1, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)
                _children.add(proc)
                lines: List[str] = []
                reader = threading.Thread(
                    target=lambda: lines.extend(iter(proc.stdout.readline, "")), daemon=True)
                reader.start()
                last = 0
                while reader.is_alive() or last < len(lines):
                    new, last = lines[last:], len(lines)
                    for line in new:
                        line = line.rstrip("\n")
                        prog.log.append(line)
                        log_file.write(line + "\n")
                        if label == "train" and prog.run_dir is None and line.startswith("Run "):
                            prog.run_dir = line.split()[1]
                    log_file.flush()
                    yield prog
                    reader.join(timeout=task_settings.POLL_SECONDS)
                code = proc.wait()
                _children.discard(proc)
                if code != 0:
                    prog.state = "failed"
                    prog.log.append(f"[{label} exited with code {code}]")
                    yield prog
                    return
            prog.files = _collect(run.outputs, prog)
            prog.state = "done"
            prog.log.append("Finished.")
            yield prog
        finally:
            log_file.close()
    finally:
        if proc is not None and proc.poll() is None:
            _kill(proc)                   # page closed / Stop / shutdown
        _busy.release()


def _collect(out: Path, prog: Progress) -> List[str]:
    """Copy the model files out of the training project."""
    out.mkdir(parents=True, exist_ok=True)
    names = [f"condnettart_{prog.frequency}_stage0.pth",
             f"condnettart_{prog.frequency}_stage1.pth",
             f"condnettart_{prog.frequency}.json"]
    src = task_settings.TRAINING_PROJECT / (prog.run_dir or "")
    for n in names:
        if (src / n).is_file():
            shutil.copyfile(src / n, out / n)
    return [str(out / n) for n in names if (out / n).is_file()]

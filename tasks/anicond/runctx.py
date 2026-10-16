# ============================================================================
# tasks/anicond/runctx.py — per-job configuration
# ----------------------------------------------------------------------------
# This is what replaces the study's `core/general as gnr` at the JOB level.
#
# vendor/core/general.py holds the static defaults the vendored libraries read
# (window size, tract thresholds, the tissue list). What a job varies -- the
# target conductivity table the user uploaded and which label is the
# anisotropic tissue -- is NOT written back into that module. It is parsed here
# and handed to each model through its constructor (`dict_cond=`, `wm_id=`,
# `WM_segs=`, `GM_segs=`), which the vendored models already support.
#
# That is deliberate: a mutated global would make two queued jobs able to see
# each other's conductivity table. A JobContext is frozen and passed down.
# ============================================================================
import csv
import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from core import general as gnr
from core.cond import lib_cond as lcond

from tasks.anicond import settings as task_settings

logger = logging.getLogger(__name__)

# Accepted spellings of the tissue id column, lower-cased.
_ID_COLUMNS = ("tissue_id", "seg", "seg_id", "label", "id")
# A conductivity column names its frequency: conductivity_iso_<freq>, e.g.
# conductivity_iso_10khz. A table may carry several; the job picks one.
_COND_PREFIX = "conductivity_iso_"
# Optional: the SimNIBS tissue this row stands for (its name in
# simnibs.utils.mesh_element_properties.tissue_names: WM, GM, CSF, Scalp,
# Compact_bone, Spongy_bone, Blood, Eye_balls, ...). Blank when the
# segmentation's tissue has no counterpart in a charm head model. anicond
# never reads it; the TMS-simulation task does, to give SimNIBS THIS table's
# isotropic values instead of its own built-in ones.
_SIMNIBS_COLUMN = "simnibs_tissue"


@dataclass(frozen=True)
class JobContext:
    """Everything one job needs that the user chose, validated."""
    case_name: str
    dict_cond: Dict[int, float]       # tissue id -> isotropic conductivity (S/m)
    aniso_ids: Tuple[int, ...]        # tissue(s) given an anisotropic tensor
    iso_ref_ids: Tuple[int, ...]      # reference tissue for directmapping's scale
    model_names: Tuple[str, ...]
    freq: str                         # the conductivity_iso_<freq> column used
    conductivity_source: str          # human-readable provenance of dict_cond

    @property
    def needs_tracts(self) -> bool:
        """True if any selected model reads whole-brain tracts."""
        return any(n in task_settings.TRACT_BASED_MODELS for n in self.model_names)

    def describe(self) -> str:
        rows = [f"{tid} ({gnr.dict_tissue.get(tid, '?')}) = {c:g} S/m"
                for tid, c in sorted(self.dict_cond.items())]
        return (
            f"case            : {self.case_name}\n"
            f"models          : {', '.join(self.model_names)}\n"
            f"anisotropic ids : {', '.join(str(i) for i in self.aniso_ids)}\n"
            f"reference ids   : {', '.join(str(i) for i in self.iso_ref_ids)}\n"
            f"frequency       : {self.freq}\n"
            f"conductivity    : {self.conductivity_source}\n"
            f"                  " + "\n                  ".join(rows)
        )


# ============================================================================
# Parsing the user's inputs
# ============================================================================
def parse_tissue_ids(text, field_name: str = "tissue id") -> Tuple[int, ...]:
    """'13, 14,15' -> (13, 14, 15), validated against the tissue list."""
    if text is None or not str(text).strip():
        raise ValueError(f"No {field_name} given")
    ids = []
    for chunk in str(text).replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            ids.append(int(chunk))
        except ValueError:
            raise ValueError(f"{field_name}: {chunk!r} is not an integer") from None
    if not ids:
        raise ValueError(f"No {field_name} given")

    unknown = [i for i in ids if i not in gnr.dict_tissue]
    if unknown:
        known = ", ".join(f"{k}={v}" for k, v in sorted(gnr.dict_tissue.items()))
        raise ValueError(f"{field_name}: unknown label(s) {unknown}. Known: {known}")
    return tuple(dict.fromkeys(ids))            # de-duplicated, order kept


def _read_rows(path: Path):
    """The CSV's data lines (comments and blanks dropped) as a DictReader,
    plus {lower-cased header: header as written}."""
    with open(path, newline="") as fh:
        rows = [ln for ln in fh
                if ln.strip() and not ln.lstrip().startswith("#")]
    if not rows:
        raise ValueError(f"Conductivity CSV is empty: {path.name}")
    reader = csv.DictReader(rows)
    if not reader.fieldnames:
        raise ValueError(f"Conductivity CSV has no header row: {path.name}")
    header = {(name or "").strip().lower(): name for name in reader.fieldnames}
    return reader, header


def csv_frequencies(path) -> List[str]:
    """The <freq> of every conductivity_iso_<freq> column, in file order."""
    _reader, header = _read_rows(Path(str(path)))
    return [key[len(_COND_PREFIX):] for key in header
            if key.startswith(_COND_PREFIX) and key[len(_COND_PREFIX):]]


def default_freq(conductivity_csv=None) -> str:
    """The frequency a job starts from: the config's `conductivity_freq_in_use`
    when the table carries it, else the table's first frequency column."""
    try:
        freqs = csv_frequencies(conductivity_csv
                                or task_settings.DEFAULT_CONDUCTIVITY_CSV)
    except (OSError, ValueError) as exc:
        logger.warning("Cannot read the conductivity table for its "
                       "frequencies: %s", exc)
        return gnr.cond_freq_in_use
    if not freqs:
        return gnr.cond_freq_in_use
    return gnr.cond_freq_in_use if gnr.cond_freq_in_use in freqs else freqs[0]


def parse_conductivity_csv(path, freq: Optional[str] = None) -> Dict[int, float]:
    """Read one frequency's column of the target conductivity table into
    {tissue id: S/m}.

    Required columns (case-insensitive, any order): a tissue id column
    (tissue_id / seg / label / id) and at least one conductivity column named
    conductivity_iso_<freq>. `freq` picks the column; None -> default_freq().
    A `tissue` name column is optional and, when present, is checked against
    the tissue list -- a mismatch is a warning, not an error, because the id is
    what keys the table. A `simnibs_tissue` column is optional too and ignored
    here (see parse_simnibs_tissues). Blank lines and lines starting with #
    are ignored.
    """
    path = Path(str(path))
    reader, header = _read_rows(path)

    id_col = next((header[c] for c in _ID_COLUMNS if c in header), None)
    freqs = [key[len(_COND_PREFIX):] for key in header
             if key.startswith(_COND_PREFIX) and key[len(_COND_PREFIX):]]
    if id_col is None or not freqs:
        raise ValueError(
            f"{path.name}: need a tissue id column (one of {', '.join(_ID_COLUMNS)}) "
            f"and at least one conductivity column named "
            f"{_COND_PREFIX}<freq> (e.g. {_COND_PREFIX}10khz); "
            f"found {reader.fieldnames}")
    freq = (freq or "").strip().lower() or default_freq(path)
    if freq not in freqs:
        raise ValueError(f"{path.name}: no column {_COND_PREFIX}{freq}; "
                         f"frequencies available: {', '.join(freqs)}")
    cond_col = header[_COND_PREFIX + freq]
    name_col = header.get("tissue") or header.get("tissue_name")

    table: Dict[int, float] = {}
    for n, row in enumerate(reader, start=2):
        raw_id = (row.get(id_col) or "").strip()
        raw_cond = (row.get(cond_col) or "").strip()
        if not raw_id and not raw_cond:
            continue
        try:
            tid = int(float(raw_id))
            cond = float(raw_cond)
        except ValueError:
            raise ValueError(f"{path.name} line {n}: cannot read "
                             f"({raw_id!r}, {raw_cond!r}) as (id, conductivity)"
                             ) from None
        if cond < 0:
            raise ValueError(f"{path.name} line {n}: negative conductivity {cond}")
        if tid in table:
            raise ValueError(f"{path.name} line {n}: tissue id {tid} listed twice")

        known = gnr.dict_tissue.get(tid)
        if known is None:
            logger.warning("Conductivity CSV: tissue id %d is not in the tissue "
                           "list; kept, but no voxel will carry it", tid)
        elif name_col:
            given = (row.get(name_col) or "").strip()
            if given and given.lower() != known.lower():
                logger.warning("Conductivity CSV: id %d is %r in the tissue list "
                               "but %r in the file; the id wins", tid, known, given)
        table[tid] = cond

    if not table:
        raise ValueError(f"{path.name}: no usable rows")
    return table


def parse_simnibs_tissues(path) -> Dict[int, str]:
    """{tissue id: SimNIBS tissue name} from the table's `simnibs_tissue`
    column -- the rows that have one. Empty when the table has no such column.

    The same id column and comment/blank rules as parse_conductivity_csv, so
    the two read one file the same way; the values are returned as written
    (case kept), and resolving them to SimNIBS element tags is the caller's
    job, because this module does not know SimNIBS.
    """
    path = Path(str(path))
    reader, header = _read_rows(path)
    id_col = next((header[c] for c in _ID_COLUMNS if c in header), None)
    simnibs_col = header.get(_SIMNIBS_COLUMN)
    if id_col is None or simnibs_col is None:
        return {}
    out: Dict[int, str] = {}
    for n, row in enumerate(reader, start=2):
        raw_id = (row.get(id_col) or "").strip()
        name = (row.get(simnibs_col) or "").strip()
        if not raw_id or not name:
            continue
        try:
            tid = int(float(raw_id))
        except ValueError:
            raise ValueError(f"{path.name} line {n}: cannot read {raw_id!r} "
                             f"as a tissue id") from None
        if tid in out:
            raise ValueError(f"{path.name} line {n}: tissue id {tid} listed twice")
        out[tid] = name
    return out


# ============================================================================
# Defaults read off the table in use
# ----------------------------------------------------------------------------
# Which labels are "white matter related" is a property of the LABELLING, not
# of this app, so it is answered from the conductivity table the job will run
# with rather than from a constant here: upload a table for a different
# segmentation and the page's defaults move with it.
#
# gnr.ls_wm_seg / ls_gm_seg (from the vendored config.yaml) say which ids those
# are; the table says which of them this job actually has a conductivity for.
# ============================================================================
def _default_ids(conductivity_csv, wanted: Sequence[int]) -> Tuple[int, ...]:
    try:
        table = parse_conductivity_csv(
            conductivity_csv or task_settings.DEFAULT_CONDUCTIVITY_CSV)
    except (OSError, ValueError) as exc:
        # A table the page cannot read is the run's problem, not the default
        # box's: fall back to the full list so the widget still shows something
        # sensible, and let build() raise when Run is pressed.
        logger.warning("Cannot read the conductivity table for the defaults: %s",
                       exc)
        return tuple(wanted)
    present = tuple(i for i in wanted if i in table)
    return present or tuple(wanted)


def default_aniso_ids(conductivity_csv=None) -> Tuple[int, ...]:
    """The WM-related tissue ids the given table carries."""
    return _default_ids(conductivity_csv, gnr.ls_wm_seg)


def default_ref_ids(conductivity_csv=None) -> Tuple[int, ...]:
    """The GM-related tissue ids Direct Mapping calibrates its scale against.

    Not offered on the page: it is a property of the model, not of the case, and
    only directmapping reads it (see build_model).
    """
    return _default_ids(conductivity_csv, gnr.ls_gm_seg)


def ids_text(ids: Sequence[int]) -> str:
    """(13, 14, 15) -> '13, 14, 15', the form the page's box shows."""
    return ", ".join(str(i) for i in ids)


def describe_ids(ids: Sequence[int]) -> str:
    """'13 WM, 14 Brainstem' — for an info line under the box."""
    return ", ".join(f"{i} {gnr.dict_tissue.get(i, '?')}" for i in ids)


# ============================================================================
# Building the context and the models
# ============================================================================
def build(case_name: str,
          model_names: Sequence[str],
          aniso_ids_text: Optional[str] = None,
          iso_ref_ids_text: Optional[str] = None,
          conductivity_csv: Optional[Path] = None,
          freq: Optional[str] = None) -> JobContext:
    """Validate one job's choices into a frozen JobContext.

    `aniso_ids_text` / `iso_ref_ids_text` left as None are taken from the
    conductivity table in use (default_aniso_ids / default_ref_ids); `freq`
    left as None is default_freq() of that table.
    """
    models = tuple(dict.fromkeys(str(m).strip() for m in model_names if str(m).strip()))
    if not models:
        raise ValueError("Select at least one conductivity model")

    available = set(lcond.get_available_cond_models(gnr.cond_model_fdpath))
    unknown = [m for m in models if m not in available]
    if unknown:
        raise ValueError(f"Unknown model(s) {unknown}; available: "
                         f"{', '.join(sorted(available))}")

    table_path = Path(str(conductivity_csv or task_settings.DEFAULT_CONDUCTIVITY_CSV))
    freq = (freq or "").strip().lower() or default_freq(table_path)
    dict_cond = parse_conductivity_csv(table_path, freq)
    source = (f"{'uploaded' if conductivity_csv else 'default'} {table_path.name}, "
              f"column {_COND_PREFIX}{freq} ({len(dict_cond)} tissues)")

    aniso_ids = parse_tissue_ids(
        aniso_ids_text if (aniso_ids_text or "").strip()
        else ids_text(default_aniso_ids(conductivity_csv)),
        "anisotropic tissue id")
    iso_ref_ids = parse_tissue_ids(
        iso_ref_ids_text if (iso_ref_ids_text or "").strip()
        else ids_text(default_ref_ids(conductivity_csv)),
        "reference tissue id")

    missing = [i for i in aniso_ids + iso_ref_ids if i not in dict_cond]
    if missing:
        raise ValueError(f"The conductivity table has no row for tissue id(s) "
                         f"{missing}, but they were selected")

    return JobContext(
        case_name=case_name,
        dict_cond=dict(dict_cond),
        aniso_ids=aniso_ids,
        iso_ref_ids=iso_ref_ids,
        model_names=models,
        freq=freq,
        conductivity_source=source,
    )


def build_model(model_name: str, ctx: JobContext):
    """Instantiate one conductivity model, configured for this job.

    Resolved by NAME through the vendored dispatcher, never by importing a model
    module -- the study's rule, kept. `load_cond_model` returns a default-built
    instance, so its class is taken and rebuilt with the job's arguments.

    Which arguments a model accepts differs (directmapping wants WM_segs and
    GM_segs, the tract-based models want wm_id, volumenormalized wants neither),
    so the candidate set is filtered against the constructor signature. A new
    model that names its arguments the same way is configured automatically.
    """
    model_cls = type(lcond.load_cond_model(model_name))
    candidates = {
        "dict_cond": ctx.dict_cond,
        "wm_id": list(ctx.aniso_ids),           # tract-based models
        "WM_segs": list(ctx.aniso_ids),         # directmapping
        "GM_segs": list(ctx.iso_ref_ids),       # directmapping
    }
    accepted = inspect.signature(model_cls.__init__).parameters
    kwargs = {k: v for k, v in candidates.items() if k in accepted}
    dropped = sorted(set(candidates) - set(kwargs))
    if dropped:
        logger.debug("%s takes no %s; using its own defaults for those",
                     model_name, ", ".join(dropped))
    return model_cls(**kwargs)


def model_labels() -> List[Tuple[str, str]]:
    """[(label, name)] for the page's model picker, in ENABLED_MODELS order."""
    out = []
    for name in task_settings.ENABLED_MODELS:
        short = lcond.model_short_name(name)
        out.append((f"{short} — {name}", name))
    return out

# ============================================================================
# tasks/tissue_conductivity/service.py — conductivity per tissue at one frequency
# ----------------------------------------------------------------------------
# What the page does, minus the widgets (web.py):
#
#   gabriel_sigma(tissue, f)   Gabriel, Lau & Gabriel, Phys. Med. Biol.
#                              41:2271-2293 (1996), eq. (3): four Cole-Cole
#                              dispersions plus ionic conductivity,
#                                eps(w) = eps_inf + sum_n deps_n / (1 + (j w tau_n)^(1-alpha_n))
#                                         + sigma_i / (j w eps0)
#                                sigma(w) = -w eps0 Im eps(w)        [S/m]
#   class_table(f)             the 15 classes at f, ANCHORED at 10 kHz: each
#                              class keeps the anicond default table's 10 kHz
#                              value and follows its Gabriel tissue's change
#                              from 10 kHz to f,
#                                sigma(f) = anchor + gabriel(f) - gabriel(10 kHz)
#                              (an offset, not a ratio: a ratio would multiply
#                              wet skin's 0.003 S/m by ~24 to reach the
#                              table's 0.07 and give >5 S/m at 1 MHz).
#   write_outputs(f, run)      <freq>.yaml (frequency, frequency_hz,
#                              normalization, sigma)
#                              and tissues.csv (class_id, name).
# ============================================================================
import csv
import io
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from app import jobs
from tasks.tissue_conductivity import settings as task_settings

TASK_SLUG = "tissue_conductivity"
EPS0 = 8.8541878128e-12                      # F/m
TAU_SCALE = (1e-12, 1e-9, 1e-6, 1e-3)        # tau1 ps, tau2 ns, tau3 us, tau4 ms


def _read_csv(path) -> List[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        lines = [l for l in fh if l.strip() and not l.lstrip().startswith("#")]
    return list(csv.DictReader(io.StringIO("".join(lines))))


def load_parameters() -> Dict[str, dict]:
    """{tissue: {column: float}} from the Gabriel 1996 table."""
    return {r["tissue"]: {k: float(v) for k, v in r.items() if k != "tissue"}
            for r in _read_csv(task_settings.PARAMETERS_CSV)}


def gabriel_sigma(tissue: str, f_hz, params: Optional[dict] = None):
    """Conductivity (S/m) of a Gabriel 1996 tissue at f_hz (scalar or array)."""
    p = (params or load_parameters())[tissue]
    w = 2 * np.pi * np.asarray(f_hz, dtype=float)
    eps = p["eps_inf"] + p["sigma_i"] / (1j * w * EPS0)
    for n, (tau_key, scale) in enumerate(zip(("tau1_ps", "tau2_ns", "tau3_us", "tau4_ms"),
                                             TAU_SCALE), start=1):
        d = p[f"deps{n}"]
        if d:
            eps = eps + d / (1 + (1j * w * p[tau_key] * scale) ** (1 - p[f"alpha{n}"]))
    return -w * EPS0 * np.imag(eps)


# ============================================================================
# Frequencies: "128 MHz" <-> 1.28e8 <-> the tag "128MHz"
# ============================================================================
def to_hz(value: float, unit: str) -> float:
    if unit not in task_settings.UNITS:
        raise ValueError(f"Unknown unit {unit!r}")
    return float(value) * task_settings.UNITS[unit]


def frequency_tag(f_hz: float) -> str:
    """1.28e8 -> '128MHz', 1500 -> '1.5kHz': the largest unit with a value
    >= 1, up to 6 significant digits, no trailing zeros -- the form the
    CondNet-TART file names use."""
    for unit in ("GHz", "MHz", "kHz", "Hz"):
        v = f_hz / task_settings.UNITS[unit]
        if v >= 1 or unit == "Hz":
            return f"{float(f'{v:.6g}'):g}{unit}".replace("e+", "e")
    raise AssertionError


def check_frequency(f_hz: float) -> None:
    if not task_settings.MIN_HZ <= f_hz <= task_settings.MAX_HZ:
        raise ValueError(f"{frequency_tag(f_hz)} is outside the model's range, "
                         f"{frequency_tag(task_settings.MIN_HZ)} to "
                         f"{frequency_tag(task_settings.MAX_HZ)}")


def low_frequency_warning(f_hz: float) -> Optional[str]:
    if f_hz >= task_settings.LOW_FREQUENCY_WARN_HZ:
        return None
    return (f"**Low frequency ({frequency_tag(f_hz)}).** Gabriel et al. (1996) "
            f"note that their parametric model is least certain below about 1 MHz "
            f"(and most of all below 100 Hz): the tissue data there are sparse and "
            f"scattered, so the values below carry a larger uncertainty."
            + (f" Below {frequency_tag(task_settings.FIT_MIN_HZ)}, the lower end of the "
               f"range their parameters were fitted over, the values are an "
               f"extrapolation of the model." if f_hz < task_settings.FIT_MIN_HZ else ""))


# ============================================================================
# The 15 classes at one frequency
# ============================================================================
@dataclass
class ClassRow:
    class_id: int
    name: str
    gabriel_tissue: str            # "" for background
    anchor_10khz: float            # from the anicond default table
    gabriel_10khz: Optional[float]
    gabriel_f: Optional[float]
    sigma: float                   # the result at f, S/m


def _anchors() -> Dict[int, float]:
    return {int(r["tissue_id"]): float(r[task_settings.ANCHOR_COLUMN])
            for r in _read_csv(task_settings.ANCHOR_CSV)}


def class_table(f_hz: float) -> List[ClassRow]:
    check_frequency(f_hz)
    params, anchors = load_parameters(), _anchors()
    rows = []
    for r in _read_csv(task_settings.CLASSES_CSV):
        k, tissue = int(r["class_id"]), r["gabriel_tissue"].strip()
        if not tissue:                                   # background
            rows.append(ClassRow(k, r["name"], "", 0.0, None, None, 0.0))
            continue
        if tissue not in params:
            raise ValueError(f"classes.csv: class {k} names {tissue!r}, which is not "
                             f"in {task_settings.PARAMETERS_CSV.name}")
        if k not in anchors:
            raise ValueError(f"{task_settings.ANCHOR_CSV.name} has no tissue_id {k}")
        g10 = float(gabriel_sigma(tissue, task_settings.ANCHOR_HZ, params))
        gf = float(gabriel_sigma(tissue, f_hz, params))
        rows.append(ClassRow(k, r["name"], tissue, anchors[k], g10, gf,
                             max(0.0, anchors[k] + gf - g10)))
    return rows


# ============================================================================
# The two files
# ============================================================================
def yaml_text(f_hz: float, rows: List[ClassRow]) -> str:
    tag = frequency_tag(f_hz)
    peak = max(rows, key=lambda r: r.sigma)
    sigma_max, top = float(f"{peak.sigma:.6g}"), peak.name
    lines = [
        "# " + "=" * 76,
        f"# {tag}: tissue conductivity",
        "# " + "-" * 76,
        "# Gabriel, Lau & Gabriel, Phys. Med. Biol. 41:2271-2293 (1996), 4-Cole-Cole",
        "# model, anchored at 10 kHz to the anicond default table:",
        "#   sigma(f) = sigma_table(10 kHz) + sigma_Gabriel(f) - sigma_Gabriel(10 kHz)",
        "# Comment after each value: class name, then the Gabriel tissue it follows.",
        "# " + "=" * 76,
        f"frequency: {tag}",
        f"frequency_hz: {f_hz:.10g}",
        "",
        "normalization:",
        f"  sigma_max: {sigma_max:.6g}   # S/m, largest tissue conductivity at {tag} ({top})",
        f"  tau: {task_settings.TAU:g}         # headroom: targets span [0, 1 - tau]",
        "",
        "sigma:             # S/m",
    ]
    for r in rows:
        value = f"{r.sigma:.6g}" if r.sigma else "0.0"
        note = r.name + (f"  ({r.gabriel_tissue})" if r.gabriel_tissue else "")
        lines.append(f"  {str(r.class_id) + ':':<4}{value:<12} # {note}")
    return "\n".join(lines) + "\n"


def tissues_csv_text(rows: List[ClassRow]) -> str:
    return "class_id,name\n" + "".join(f"{r.class_id},{r.name}\n" for r in rows)


@dataclass
class Result:
    f_hz: float
    tag: str
    rows: List[ClassRow]
    yaml_path: str
    csv_path: str
    warning: Optional[str]


def write_outputs(f_hz: float) -> Result:
    rows = class_table(f_hz)
    tag = frequency_tag(f_hz)
    jobs.prune_old_runs(TASK_SLUG)
    run = jobs.new_run(TASK_SLUG)
    yaml_path = run.output(f"{tag}.yaml")
    yaml_path.write_text(yaml_text(f_hz, rows), encoding="utf-8")
    csv_path = run.output("tissues.csv")
    csv_path.write_text(tissues_csv_text(rows), encoding="utf-8")
    return Result(f_hz, tag, rows, str(yaml_path), str(csv_path),
                  low_frequency_warning(f_hz))

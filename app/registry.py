# ============================================================================
# app/registry.py — the task list
# ============================================================================
import importlib
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TaskSpec:
    slug: str          # URL path suffix
    title: str         # task title
    blurb: str         # task subtitle
    module: str        # module exposing build_page()
    enabled: bool = True
    # `enabled` decides whether the page EXISTS (its route is mounted);
    # `visible` decides whether it is ADVERTISED -- a card on the home menu
    # and an entry in the navbar. A task that is enabled but not visible is
    # reachable by anyone who knows its URL (/<slug>), and linked from the
    # footer of the OTHER hidden pages (app/web/main._hidden_links) -- the
    # hidden tasks see each other, the visible ones never mention them. The
    # way to keep an analysis step available without putting it in front of
    # every visitor.
    visible: bool = True
    # A UTILITY is a small, self-contained tool (a calculator, a converter)
    # rather than a full task: when visible, it gets a compact tile in the
    # home page's "Utilities" section instead of a card under "Tasks", and no
    # navbar entry. Its page is reached from the home page only.
    utility: bool = False
    # Optional demo thumbnail for the home page card. Either a bare filename
    # resolved against tasks/<slug>/assets/, or a full URL. None -> no image,
    # the card's text simply fills the row.
    image: Optional[str] = None

    def build_page(self) -> None:
        """Import the task's web module and let it lay out the current page."""
        importlib.import_module(self.module).build_page()


TASKS = (
    TaskSpec(
        slug="isocond",
        title="T1/T2 → Conduc. Map",
        blurb="DL-based estimation of conductivity map given T1/T2 volumes.<br/>Ref.: CondNet-TART (Kubota Y., et al. 2025), CondNet. (Rashed E., et al. 2019)",
        module="tasks.isocond.web",
        image="isocond.png",
    ),
    TaskSpec(
        slug="anicond",
        title="DTI → Conduc. Tensors",
        blurb="Estimation of conductivity tensors given DWI volumes & tissue segments. "
              "<br/>Ref.: DCond (My NN., et al. 2026)",
        module="tasks.anicond.web",
        image="anicond.png",
    ),
    TaskSpec(
        slug="mesh_head",
        title="T1/T2 → Head Mesh",
        blurb="Estimation of cortical surfaces given T1/T2 volumes via SimNIBS software.",
        module="tasks.mesh_head.web",
        image="head_mesh.png",
    ),
    TaskSpec(
        slug="mesh_sim",
        title="Head Mesh → TMS Simulation",
        blurb="TMS simulation on the head mesh from the previous task using SimNIBS software.",
        module="tasks.mesh_sim.web",
        image="sim.png",
    ),
    TaskSpec(
        slug="tract_efield",
        title="TMS Field → Tracts",
        blurb="Tract streamlines reaching the white-matter hot spot of a TMS "
              "simulation (the previous task's archive + the tractogram from "
              "DTI → Conduc. Tensors).",
        module="tasks.tract_efield.web",
        visible=False,      # reachable at /tract_efield, not listed
    ),
    TaskSpec(
        slug="tract_simplify",
        title="Tractogram → Visual Tracts",
        blurb="A tractogram healed, thinned and smoothed for pictures only: "
              "broken tracts joined end to end, short fragments cut, each "
              "bundle kept at a chosen fraction.",
        module="tasks.tract_simplify.web",
        visible=False,      # reachable at /tract_simplify, not listed
    ),
    TaskSpec(
        slug="isocond_debug",
        title="Isocond Result Check",
        blurb="Slice-by-slice check of an existing conductivity map (and "
              "segmentation) against its T1/T2: values under the cursor, a "
              "per-tissue table per slice, and the checkpoints' architecture.",
        module="tasks.isocond_debug.web",
        visible=False,      # reachable at /isocond_debug, not listed
    ),
    TaskSpec(
        slug="condnettart_train",
        title="CondNet-TART Training",
        blurb="Train CondNet-TART on the default dataset with a conductivity "
              "table of another frequency; returns the best model files.",
        module="tasks.condnettart_train.web",
        visible=False,      # reachable at /condnettart_train, not listed
    ),
    # --- Utilities: small tools, tiles on the home page, not in the navbar ---
    TaskSpec(
        slug="tissue_conductivity",
        title="Conductivity σ",
        blurb="Generate σ at any frequency<br/>"
              "(Cole-Cole model)",
        module="tasks.tissue_conductivity.web",
        utility=True,
    ),
    TaskSpec(
        slug="registration",
        title="Image Alignment",
        blurb="Register a moving NIfTI onto a fixed one.",
        module="tasks.registration.web",
        utility=True,
    ),
    TaskSpec(
        slug="resample",
        title="Resampling",
        blurb="Resample a NIfTI to a desired voxel size.",
        module="tasks.resample.web",
        utility=True,
    ),
)


def enabled_tasks():
    """Every task that gets a page, listed or not."""
    return tuple(t for t in TASKS if t.enabled)


def visible_tasks():
    """The tasks the home menu's "Tasks" cards and the navbar show (no utilities)."""
    return tuple(t for t in TASKS if t.enabled and t.visible and not t.utility)


def utilities():
    """The utilities the home menu shows as tiles under "Utilities"."""
    return tuple(t for t in TASKS if t.enabled and t.visible and t.utility)


def hidden_tasks():
    """Enabled tasks reachable only by URL."""
    return tuple(t for t in TASKS if t.enabled and not t.visible)


def by_slug(slug: str) -> TaskSpec:
    for task in TASKS:
        if task.slug == slug:
            return task
    raise KeyError(f"No task registered under slug {slug!r}")

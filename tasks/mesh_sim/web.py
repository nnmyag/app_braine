# ============================================================================
# tasks/mesh_sim/web.py — the Gradio page for head mesh -> TMS simulation
# ----------------------------------------------------------------------------
# Widgets and callbacks ONLY. Every line of pipeline logic lives in
# service.py, every figure in preview.py and the SimNIBS half in worker.py,
# so a cli.py added later reproduces this page's behaviour by calling the
# same run_job().
#
# The layout mirrors tasks 1-3, with one difference: there is no T1/T2
# upload. The input is the *finished head model* -- the .zip that
# tasks/mesh_head's "Download all" link produces -- so the page asks for
# that instead, the same hand-off anicond already makes to isocond for a
# segmentation.
#
#     left  | head model upload + detected subject id + anisotropic
#           |   conductivity checkbox, one row -- upload, identify, opt in
#           |   to a fancier conductivity model are all one decision each
#           | coil placement (center + handle direction electrodes) --
#           |   the one choice that actually depends on the target, so it
#           |   comes first among the coil rows even though it too starts
#           |   from a plausible default (C3, ~45 deg postero-lateral)
#           | coil & stimulation defaults (coil, cap, dI/dt, distance) --
#           |   grouped below placement because every one of them has a
#           |   sensible default most users will keep as is
#           | Run
#           | -> downloads + duration (hidden until a run finishes)
#           | -> console
#     right | small "Show" combobox
#           | -> large canvas
#
# The dropdowns fill themselves from what is actually there: the coil list
# from the SimNIBS install, the cap list and both electrode lists from the
# uploaded head model, with the handle-direction angles computed against
# THIS subject's own scalp (see service.handle_choices). settings.py holds
# only the fallbacks used before anything has been uploaded.
# ============================================================================
import logging
from pathlib import Path

import gradio as gr

from app import jobs
from app.web import common
from tasks.mesh_sim import service
from tasks.mesh_sim import settings as task_settings

logger = logging.getLogger(__name__)

INTRO = ("Upload the head model built by the previous task, pick a coil and "
         "where to place it, and Run. SimNIBS solves the induced electric "
         "field on the head mesh and maps it onto the cortical surface.  \n"
         "Read from the uploaded head-model archive: "
         "`m2m_<id>/<id>.msh` — the head mesh the field is solved on; "
         "`m2m_<id>/eeg_positions/*.csv` + `Fiducials.csv` — the cap: coil "
         "centre and handle electrodes, handle angles; "
         "`m2m_<id>/T1.nii.gz` — the grid the field is written to as NIfTI, "
         "the pictures' background; "
         "`m2m_<id>/surfaces/{lh,rh}.{central,pial,white}.gii` — the "
         "cortical surfaces the field is read on; "
         "`m2m_<id>/final_tissues.nii.gz` — anisotropic runs only: the "
         "white/gray label map. The rest of the archive (charm's report, "
         "the MNI warps, the T1/T2 copies, the vtk/ folder) is unpacked and "
         "carried along unused.")

ZIP_TYPES = [".zip"]
NIFTI_TYPES = [".nii", ".gz"]

UPLOAD_BOX_HEIGHT = 70
# The head-model upload shares its row with only Subject id now, so a
# shorter box (rather than the full drop-zone height used elsewhere) keeps
# that row compact -- the upload was asked to be the shorter of the two.
TOP_ROW_UPLOAD_HEIGHT = 56
CANVAS_HEIGHT = 620
CONSOLE_LINES = 15

# The generic 10-20 reference diagram shown beside the coil-placement and
# coil-model rows, so the electrode names in those dropdowns mean something
# without leaving the page. Static and the same for every run, unlike the
# per-subject "Coil position" canvas view below, which will show where THIS
# run's chosen electrode actually maps onto THIS subject's scalp.
ELECTRODE_FIGURE = task_settings.ASSETS_DIR / "electrode_positions.png"
# Sized explicitly (not just a max height) so the box fits the picture's own
# 568x671 aspect ratio with no dead space -- a height-only cap left the
# actual content width-bound by the column, well short of the requested box.
# 156x185 was this column's previous rendered size; both are that x 1.25.
ELECTRODE_FIGURE_WIDTH = 195
ELECTRODE_FIGURE_HEIGHT = 230

# The canvas views belong to service.py, which knows what a finished run
# actually holds; the page only lists them and asks for one to be drawn.
#
# Six views: the T1, the slice the coil sits on, the field on the cortex
# with the coil drawn over it, the field on the white surface, the field
# volume across gray + white matter with the boundary drawn, and the
# tensor's orientation for an anisotropic run. SimNIBS's other surface
# quantities (normal, tangential, angle) ship in the run's overlays and the
# normal component as a VTK, but are not drawn on the page.
VIEW_LABELS = service.view_labels()
DEFAULT_VIEW = service.DEFAULT_VIEW

# Gradio serves any file under `allowed_paths` from this route; run_web.py
# allows data/runs, which is where a job writes.
FILE_ROUTE = "/gradio_api/file="

PAGE_CSS = """
<style>
.mesh-sim-upload .wrap:has(> .icon-wrap) {
    font-size: 0 !important;
    line-height: 0 !important;
    min-height: 0 !important;
    padding-top: 0 !important;
}
.mesh-sim-upload .wrap:has(> .icon-wrap) .or { display: none !important; }
.mesh-sim-upload .wrap:has(> .icon-wrap) .icon-wrap {
    margin-bottom: 0 !important;
    width: 26px !important;
}
.mesh-sim-note { font-size: 0.3em; }
.mesh-sim-note p { font-size: inherit !important; color: #6b7280 !important; line-height: 0.5 }
.mesh-sim-unsupported {
    border: 1px solid #e0a0a0; background: #fff4f4; color: #7a1f1f;
    border-radius: 8px; padding: 0.6em 0.9em; margin: 0.2em 0 0.6em;
}
@media (prefers-color-scheme: dark) {
    .mesh-sim-unsupported { background: #3a1e1e; color: #ffb3b3; border-color: #7a3b3b; }
}
.mesh-sim-console textarea {
    background: #101214 !important;
    color: #c9d1d9 !important;
    font-family: ui-monospace, "DejaVu Sans Mono", Menlo, Consolas, monospace !important;
    font-size: 10px !important;
    line-height: 1.35 !important;
    border: 1px solid #2b3137 !important;
    resize: vertical !important;
    min-height: 90px;
    overflow-y: auto !important;
}
.mesh-sim-guide table { font-size: 0.85em; }
.mesh-sim-guide td, .mesh-sim-guide th { padding: 2px 8px !important; }
/* Any dropdown whose selected value can run long (coil/cap catalog names,
   the annotated coil-placement options): shown smaller so the text clears
   the dropdown's own arrow icon instead of running behind it. Verified
   against the real Gradio markup -- the options <ul> nests inside the
   component's own .block, which is where elem_classes lands, so this stays
   scoped to the dropdowns it's put on and does not shrink the others (the
   view selector, the anisotropy-mapping dropdown). */
.mesh-sim-compact-dropdown input {
    font-size: 0.8em !important;
    text-overflow: ellipsis;
    padding-right: 22px !important;    /* clears the caret icon */
}
.mesh-sim-compact-dropdown .options, .mesh-sim-compact-dropdown .options .item {
    font-size: 0.8em !important;
}
/* dI/dt and the coil-scalp distance: narrow by design, but the value must
   still clear the up/down spinner. */
.mesh-sim-compact-number input { font-size: 0.85em !important; padding-right: 4px !important; }
/* Label and box on one line, for the dropdowns in the coil rows: verified
   against the real markup -- a Dropdown's label and its box are plain block
   siblings under one .container, so turning that container into a row flex
   puts them side by side. Scoped to instances carrying this class, so the
   view selector and other pages are untouched. Only safe once an instance
   has no `info=` text (a 3rd sibling), which is why coil model / handle
   direction lost theirs. */
.mesh-sim-inline-label .container {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    gap: 8px;
}
.mesh-sim-inline-label .container > span:first-child {
    flex: 0 0 auto;
    white-space: nowrap;
    margin-bottom: 0 !important;
}
.mesh-sim-inline-label .container > .wrap {
    flex: 1 1 auto;
    min-width: 0;
}
/* The electrode-position reference picture: no border/shadow of its own,
   so it reads as a diagram next to the form rather than a photo in a frame. */
.mesh-sim-electrode-fig img { object-fit: contain; }
/* The frequency line under the 10-20 picture: small, centred, no margin. */
.mesh-sim-frequency { text-align: center; font-size: 0.78em; }
.mesh-sim-frequency p { margin: 2px 0 0 !important; color: #4b5563 !important; line-height: 1.3; }
</style>
"""

ANISO_NOTE = (
    "No tensor? Generate one on the "
    '<a href="/anicond">DTI → Cond. Tensor</a> page and bring it back here. '
    "Applies to white matter and gray matter only; every other tissue keeps "
    "its standard isotropic value."
)


def _frequency_note() -> str:
    """'Conductivities at 10 kHz' -- or a plain fallback if the table
    cannot be read at page build (the run itself will fail loudly)."""
    try:
        freq = service.conductivity_frequency()
    except Exception:                     # noqa: BLE001 - a caption, not a job
        logger.exception("mesh_sim: could not read the conductivity frequency")
        return "Tissue conductivities: frequency unknown (table unreadable)"
    note = (f"**TMS Freq.: {freq}**")
    try:
        hz = service.conductivity_frequency_hz()
        limits = service.icnirp_basic_restriction(hz) if hz else {}
    except Exception:                     # noqa: BLE001 - a caption, not a job
        logger.exception("mesh_sim: could not evaluate the ICNIRP restriction")
        limits = {}
    if limits:
        peak = task_settings.ICNIRP_PEAK_FACTOR
        note += (f"<br/>ICNIRP 2010 basic restriction for E-field: "
                 + " · ".join(f"{v:.3g} V/m rms ({v * peak:.3g} peak) "
                              f"{k.replace('_', ' ')}"
                              for k, v in limits.items()))
    return note


# ============================================================================
# Download links + file guide
# ============================================================================
def _link(path, text=None) -> str:
    return f"[{text or Path(path).name}]({FILE_ROUTE}{Path(path).resolve()})"


def _download_links(result) -> str:
    """The archive first, then every guided file with what it is."""
    if result is None:
        return ""
    blocks = []
    if result.zip_path and Path(result.zip_path).is_file():
        size = Path(result.zip_path).stat().st_size / 1e6
        blocks.append(f"**[⬇ Download all — every file below (.zip, "
                      f"{size:.0f} MB)]"
                      f"({FILE_ROUTE}{Path(result.zip_path).resolve()})**")
    rows = ["| file | what it is |", "|---|---|"]
    for _role, path, meaning in result.guide():
        rel = Path(path).relative_to(result.run.outputs)
        rows.append(f"| {_link(path, str(rel))} | {meaning} |")
    for folder, meaning in result.folders():
        rel = folder.relative_to(result.run.outputs)
        rows.append(f"| `{rel}/` (in the archive) | {meaning} |")
    blocks.append("\n".join(rows))
    return "\n\n".join(blocks)


# ============================================================================
# Filling the dropdowns from what was uploaded
# ============================================================================
def _on_head_model(zip_path):
    """A head model was chosen: read its subject id and its own cap files.

    Only the archive's index and a couple of small CSVs are read here, never
    the whole thing -- a head model is several hundred megabytes and the page
    has to stay responsive. The unpacking happens when a job runs.
    """
    if not zip_path:
        return (gr.update(value=""), gr.update(), gr.update(), gr.update())
    subject = service.subject_id_from_zip(zip_path)
    caps = service.cap_choices(zip_path)
    cap = service.default_cap(caps)
    centres, handles, centre, handle = _placement_choices(zip_path, cap)
    return (gr.update(value=subject or ""),
            gr.update(choices=caps, value=cap),
            gr.update(choices=centres, value=centre),
            gr.update(choices=handles, value=handle))


def _placement_choices(zip_path, cap_name, centre=None):
    """(centre choices, handle choices, centre value, handle value).

    The handle angles are computed against this subject's own electrode
    coordinates, so they follow both the cap and the centre that is chosen.
    """
    try:
        caps, fiducials = service.read_cap_from_zip(zip_path, cap_name)
    except Exception:                     # noqa: BLE001 - fall back, don't fail
        logger.exception("mesh_sim: could not read %s from %s", cap_name,
                         zip_path)
        caps, fiducials = {}, {}
    centres = service.electrode_choices(caps)
    names = [value for _label, value in centres]
    if centre not in names:
        centre = (task_settings.DEFAULT_COIL_CENTER
                  if task_settings.DEFAULT_COIL_CENTER in names
                  else (names[0] if names else task_settings.DEFAULT_COIL_CENTER))
    handles = service.handle_choices(caps, fiducials, centre)
    return centres, handles, centre, service.default_handle(handles, centre)


def _on_cap_change(zip_path, cap_name, centre):
    if not zip_path:
        return gr.update(), gr.update()
    centres, handles, centre, handle = _placement_choices(zip_path, cap_name,
                                                          centre)
    return (gr.update(choices=centres, value=centre),
            gr.update(choices=handles, value=handle))


def _on_centre_change(zip_path, cap_name, centre):
    """The handle angles are relative to the centre, so they move with it."""
    if not zip_path or not centre:
        return gr.update()
    _centres, handles, _centre, handle = _placement_choices(zip_path, cap_name,
                                                            centre)
    return gr.update(choices=handles, value=handle)


# ============================================================================
# dI/dt, capped at the coil's own ceiling
# ----------------------------------------------------------------------------
# The box stays a plain number in A/s (the unit SimNIBS takes); what the coil
# adds is the ceiling -- its measured 100% MSO dI/dt -- shown in the label
# and enforced as the box's maximum. Both follow the coil dropdown.
# ============================================================================
def _didt_label(coil: str) -> str:
    ceiling = service.coil_didtmax(coil)
    if ceiling is None:
        return "dI/dt (A/s)"
    return (f"dI/dt (A/s), max: {ceiling:.4g} ")


def _on_coil_change(coil: str, didt):
    """The coil changed: relabel the dI/dt box with the new ceiling and pull
    the value down to it if it is now above."""
    ceiling = service.coil_didtmax(coil)
    value = didt
    if ceiling is not None and didt is not None and float(didt) > ceiling:
        value = ceiling
    return gr.update(label=_didt_label(coil), maximum=ceiling, value=value)


# ============================================================================
# Canvas
# ============================================================================
def _render(view_label, result):
    """Every view comes from a file the run wrote, so before a run there is
    nothing to draw."""
    try:
        return service.render_view(view_label, result)
    except Exception:
        logger.exception("Could not render %s", view_label)
        return None


# ============================================================================
# The run
# ============================================================================
def _run(head_model, subject_id, coil_model, eeg_cap, didt, distance,
         coil_center, coil_ydir, anisotropic, tensor_file,
         conductivity_type, request: gr.Request = None):
    """Yields (console, downloads box, duration, links, canvas, view, state)."""
    hidden = gr.update(visible=False)
    idle = (hidden, "", "", gr.update(), gr.update(), gr.update())

    status = service.simnibs_status()
    if not status.available:
        yield (f"This function is not supported on this server: "
               f"{status.message}", *idle)
        return
    missing = []
    if not head_model:
        missing.append("head model (.zip)")
    if anisotropic and not tensor_file:
        missing.append("conductivity tensor (anisotropic conductivity is on)")
    if missing:
        yield (f"Missing input: {', '.join(missing)}.", *idle)
        return

    jobs.prune_old_runs(service.TASK_SLUG)
    run = jobs.new_run(service.TASK_SLUG)
    transcript = jobs.Transcript(log_file=run.log_file)

    def target():
        with jobs.capture_logs(transcript):
            with jobs.QUEUE.slot(report=transcript.report, run=run):
                return service.run_job(
                    head_model_zip=head_model,
                    coil=coil_model,
                    eeg_cap=eeg_cap,
                    coil_center=coil_center,
                    coil_ydir=coil_ydir,
                    didt=float(didt),
                    distance=float(distance),
                    anisotropic=bool(anisotropic),
                    tensor_path=tensor_file,
                    conductivity_type=conductivity_type,
                    run=run,
                    report=transcript.report,
                )

    for _text, result, error in common.stream_job(target, transcript,
                                                  run=run, request=request):
        # The console gets the WHOLE transcript, not a tail: it scrolls, and a
        # user chasing a SimNIBS error wants to scroll back to where it began.
        console = transcript.text()

        if error is not None:
            logger.exception("mesh_sim job %s failed", run.job_id,
                             exc_info=error)
            transcript.report(f"FAILED: {type(error).__name__}: {error}")
            transcript.report(f"Full transcript: {run.log_file}")
            yield (transcript.text(), *idle)
            return

        if result is None:
            yield (console, *idle)
            continue

        for line in result.summary.splitlines():
            transcript.report(line)

        peak = (f" &nbsp;·&nbsp; peak |E| {result.peak_field:.3g} V/m"
                if result.peak_field is not None else "")
        yield (transcript.text(),
               gr.update(visible=True),
               f"**Completed in {result.elapsed_s / 60:.1f} min** "
               f"&nbsp;·&nbsp; subject `{result.subject_id}` "
               f"&nbsp;·&nbsp; {result.coil_center} → {result.coil_ydir} "
               f"&nbsp;·&nbsp; `{result.conductivity_type}`"
               f"{peak} &nbsp;·&nbsp; job `{run.job_id}`",
               _download_links(result),
               _render(service.RESULT_VIEW, result),
               gr.update(value=service.RESULT_VIEW),  # jump to what they came for
               result)


# ============================================================================
# Layout
# ============================================================================
def build_page() -> None:
    """Lay out this task's page. Called by app/web/main.py inside a route."""
    common.page_header("head mesh → TMS simulation", INTRO)
    gr.HTML(PAGE_CSS)

    # service.simnibs_status() is mesh_head's own probe, reused rather than
    # copied: both tasks ask the same question of the same install.
    status = service.simnibs_status()
    result_state = gr.State(None)

    # Scanned once per page build; the cap and electrode lists below start
    # from settings.py's fallbacks and are replaced the moment a head model
    # is chosen (see _on_head_model).
    coils = service.coil_choices()

    with gr.Row():
        # ----------------------------------------------------- controls --
        with gr.Column(scale=2):
            common.section_label("Input")
            if not status.available:
                gr.Markdown(
                    f"⚠ **This function is not supported on this server.** "
                    f"{status.message} — set `SIMNIBS_ROOT` in `settings.py` "
                    f"to a SimNIBS 4 install and restart the app.",
                    elem_classes="mesh-sim-unsupported")

            # Upload and identification: each is a single small decision,
            # and neither needs a heading of its own to be legible next to
            # its label.
            with gr.Row(equal_height=True):
                head_model = gr.File(label="Head model (.zip)",
                                     file_types=ZIP_TYPES, type="filepath",
                                     height=TOP_ROW_UPLOAD_HEIGHT, scale=2,
                                     elem_classes="mesh-sim-upload")
                subject_id = gr.Textbox(label="Subject id", interactive=False,
                                        placeholder="from the .zip", scale=1)
            # The conductivity-model choice gets its own row: it is a
            # decision about HOW to model the subject, not part of
            # identifying which one it is.
            with gr.Row(equal_height=True):
                anisotropic = gr.Checkbox(
                    label="Anisotropic WM/GM conductivity", value=False,
                    scale=2)
                # The label this run is FILED under in the results CSV, and
                # what a cohort's box plots get grouped by. Deliberately not
                # derived from the checkbox or the tensor's filename: those
                # cannot tell dm from vn from tbe, and a mislabelled row is
                # invisible until the cohort plot comes out wrong.
                conductivity_type = gr.Dropdown(
                    label="Conductivity type (for the results CSV)",
                    choices=task_settings.CONDUCTIVITY_TYPES,
                    value=task_settings.DEFAULT_CONDUCTIVITY_TYPE, scale=3,
                    elem_classes="mesh-sim-compact-dropdown")
            # No mapping choice here: every tensor this app ever receives is
            # a ready-made conductivity tensor (task 2's own output), never a
            # bare diffusion tensor SimNIBS would need to rescale, so the
            # mapping is fixed to 'dir' (task_settings.DEFAULT_ANISOTROPY_TYPE)
            # rather than offered as a control -- see settings.py for why.
            with gr.Row(equal_height=True, visible=False) as aniso_row:
                tensor_file = gr.File(
                    label="Conductivity tensor (.nii.gz)",
                    file_types=NIFTI_TYPES, type="filepath",
                    height=UPLOAD_BOX_HEIGHT,
                    elem_classes="mesh-sim-upload")
            gr.Markdown(ANISO_NOTE, elem_classes="mesh-sim-note")

            # Every remaining control on the left; a generic 10-20 diagram on
            # the right, so "C3", "CP5", "Cz" mean something while picking
            # them, without a second lookup away from the page.
            with gr.Row(equal_height=False):
                with gr.Column(scale=3, min_width=260):
                    # Coil placement first: the one choice that actually
                    # depends on the target, even though it too starts from
                    # a plausible default. One dropdown per row here -- with
                    # the reference figure eating into this column's width,
                    # two inline-label dropdowns side by side would leave
                    # neither enough room to show its value.
                    with gr.Group():
                        coil_center = gr.Dropdown(
                            label="Coil center",
                            choices=task_settings.COIL_CENTER_CHOICES,
                            value=task_settings.DEFAULT_COIL_CENTER,
                            elem_classes="mesh-sim-compact-dropdown "
                                         "mesh-sim-inline-label")
                        coil_ydir = gr.Dropdown(
                            label="Handle direction",
                            choices=task_settings.COIL_YDIR_CHOICES,
                            value=task_settings.DEFAULT_COIL_YDIR,
                            elem_classes="mesh-sim-compact-dropdown "
                                         "mesh-sim-inline-label")

                    # Coil & stimulation defaults: everyone keeps these as
                    # is, so they come after placement, not before it.
                    with gr.Group():
                        coil_model = gr.Dropdown(
                            label="Coil model", choices=coils,
                            value=service.default_coil(coils),
                            elem_classes="mesh-sim-compact-dropdown "
                                         "mesh-sim-inline-label")
                        eeg_cap = gr.Dropdown(
                            label="EEG cap",
                            choices=task_settings.EEG_CAP_CHOICES,
                            value=task_settings.DEFAULT_EEG_CAP,
                            elem_classes="mesh-sim-compact-dropdown "
                                         "mesh-sim-inline-label")
                        with gr.Row(equal_height=True):
                            initial_coil = service.default_coil(coils)
                            didt = gr.Number(
                                label=_didt_label(initial_coil),
                                value=task_settings.DEFAULT_DIDT,
                                minimum=0.0,
                                maximum=service.coil_didtmax(initial_coil),
                                scale=1, min_width=110,
                                elem_classes="mesh-sim-compact-number")
                            distance = gr.Number(
                                label="Coil–scalp distance (mm)",
                                value=task_settings.DEFAULT_DISTANCE_MM,
                                scale=1, min_width=110,
                                elem_classes="mesh-sim-compact-number")

                    run_btn = gr.Button("Run", variant="primary",
                                        interactive=status.available)

                with gr.Column(scale=1, min_width=ELECTRODE_FIGURE_WIDTH):
                    gr.Image(value=str(ELECTRODE_FIGURE)
                            if ELECTRODE_FIGURE.is_file() else None,
                            label="10-20 reference", show_label=False,
                            interactive=False, container=False,
                            width=ELECTRODE_FIGURE_WIDTH,
                            height=ELECTRODE_FIGURE_HEIGHT,
                            elem_classes="mesh-sim-electrode-fig")
                    # The frequency the tissue conductivities are taken at
                    # -- the only frequency this quasi-static simulation
                    # has, and the one an exposure limit is read at.
                    gr.Markdown(_frequency_note(),
                                elem_classes="mesh-sim-frequency")

            with gr.Group(visible=False) as downloads_box:
                common.section_label("Output")
                duration = gr.Markdown("")
                links = gr.Markdown("", elem_classes="mesh-sim-guide")

            console = gr.Textbox(
                label="Console", lines=CONSOLE_LINES, max_lines=CONSOLE_LINES,
                interactive=False, autoscroll=True,
                elem_classes="mesh-sim-console",
                value=("Idle. Upload a head model, pick the coil and its "
                       "position, then press Run."
                       if status.available else
                       f"Not supported on this server: {status.message}"))

        # ------------------------------------------------------ display --
        with gr.Column(scale=3):
            view = gr.Dropdown(choices=VIEW_LABELS, value=DEFAULT_VIEW,
                               label="Show")
            canvas = gr.Image(label="Mosaic", type="pil", height=CANVAS_HEIGHT)

    common.page_footer()

    # -------------------------------------------------------- wiring --
    # Which conductivity fields are visible.
    anisotropic.change(lambda on: gr.update(visible=on),
                       inputs=anisotropic, outputs=aniso_row)

    # A head model was chosen: its subject id, its own cap files and its own
    # electrode coordinates replace the fallbacks, and the handle angles are
    # computed against this subject's scalp rather than a reference one.
    head_model.change(_on_head_model, inputs=head_model,
                      outputs=[subject_id, eeg_cap, coil_center, coil_ydir])
    eeg_cap.change(_on_cap_change,
                   inputs=[head_model, eeg_cap, coil_center],
                   outputs=[coil_center, coil_ydir])
    coil_center.change(_on_centre_change,
                       inputs=[head_model, eeg_cap, coil_center],
                       outputs=coil_ydir)
    # A different coil has a different ceiling: relabel and re-cap dI/dt.
    coil_model.change(_on_coil_change, inputs=[coil_model, didt],
                      outputs=didt)

    view.change(_render, inputs=[view, result_state], outputs=canvas)

    run_btn.click(
        _run,
        inputs=[head_model, subject_id, coil_model, eeg_cap, didt, distance,
                coil_center, coil_ydir, anisotropic, tensor_file,
                conductivity_type],
        outputs=[console, downloads_box, duration, links, canvas, view,
                 result_state],
    )

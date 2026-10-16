# ============================================================================
# tasks/anicond/web.py — the Gradio page for DTI -> conductivity tensor
# ----------------------------------------------------------------------------
# Widgets and callbacks ONLY. Every line of pipeline logic lives in service.py,
# every validation rule in runctx.py and every figure in preview.py, so a
# cli.py added later reproduces this page's behaviour by calling the same
# run_job().
#
# The layout is task 1's, because the two pages are read the same way:
#
#     left  | inputs: DWI + gradients / segmentation + table / tissues /
#           |         model + Run
#           | -> downloads + duration (hidden until a run finishes)
#           | -> console
#     right | small "Show" combobox
#           | -> large canvas
#
# The one thing that IS a front-end concern and lives here: turning a job that
# blocks for minutes into a page that keeps updating (see common.stream_job).
# ============================================================================
import logging
from pathlib import Path

import gradio as gr

from app import jobs
from app.web import common
from tasks.anicond import runctx, service
from tasks.anicond import settings as task_settings

logger = logging.getLogger(__name__)

INTRO = (
    "Upload a DWI series with its gradient table and a tissue segmentation. "
    "The app fits the diffusion tensor, runs whole-brain tractography and "
    "returns the anisotropic conductivity tensor as a 6-component, "
    "FSL-compatible NIfTI (Cxx, Cxy, Cxz, Cyy, Cyz, Czz)."
)

NIFTI_TYPES = [".nii", ".gz"]
# What load_streamlines_from_vtk reads. The page says VTK because that is what
# a run writes; a tractogram of the same DWI in another container works too.
TRACT_TYPES = [".vtk", ".vtp", ".trk", ".tck"]

# Gradio serves any file under `allowed_paths` from this route; run_web.py
# allows data/runs, which is where a job writes.
FILE_ROUTE = "/gradio_api/file="

# Height of the compact drop zones. Full-height file boxes push the model and
# the Run button below the fold (the lesson task 1's layout was built on).
# Shorter than task 1's, because the CSS below strips the zone down to its
# icon: with five of them on two rows, "Drop File Here - or - Click to Upload"
# was five times the same clipped sentence.
UPLOAD_BOX_HEIGHT = 70

CANVAS_HEIGHT = 620

# Where the console starts; the CSS below gives it a resize grip, so its final
# height is the reader's choice.
CONSOLE_LINES = 15

PAGE_CSS = """
<style>
/* Drop zones show the upload icon only. Gradio's UploadText renders the
   sentence as bare text nodes beside the icon, which no selector can reach,
   so the font is zeroed on the wrapper and the icon keeps its own width.
   :has() scopes it to the drop-zone wrap, not the block's status overlay. */
.anicond-upload .wrap:has(> .icon-wrap) {
    font-size: 0 !important;
    line-height: 0 !important;
    min-height: 0 !important;
    padding-top: 0 !important;
}
.anicond-upload .wrap:has(> .icon-wrap) .or { display: none !important; }
.anicond-upload .wrap:has(> .icon-wrap) .icon-wrap {
    margin-bottom: 0 !important;
    width: 26px !important;
}
/* The what-goes-where note under the inputs: quieter than the labels. */
.anicond-note { font-size: 0.3em;}
.anicond-note p { font-size: inherit !important; color: #6b7280 !important;  line-height: 0.5}
.anicond-console textarea {
    background: #101214 !important;
    color: #c9d1d9 !important;
    font-family: ui-monospace, "DejaVu Sans Mono", Menlo, Consolas, monospace !important;
    font-size: 10px !important;
    line-height: 1.35 !important;
    border: 1px solid #2b3137 !important;
    /* Drag the corner to make the console taller or shorter. Gradio ships
       resize:none on its textareas, hence the !important. */
    resize: vertical !important;
    min-height: 90px;
    overflow-y: auto !important;
}
</style>
"""

SEG_NOTE = (
    "No segmentation? Generate one on the "
    '<a href="/isocond">T1/T2 → conductivity map</a> page (CondNet-TART also '
    "writes a tissue segmentation) and bring it back here."
)

# Named roles (service.INTERMEDIATE_ROLES) rather than filenames: what a file
# IS was decided by the job that wrote it, not by how it happens to be spelled.
ROLE_LABELS = {
    "seg_dwi": "registered segmentation",
    "aniso_mask": "anisotropic tissue mask",
    "fa": "FA map",
    "d_tensor": "diffusion tensor (6×1)",
    "tracts": "whole-brain tractogram (.vtk)",
    "tracts_viz": "whole-brain tractogram, simplified for viewing (.vtk)",
    "eff_cond": "effective conductivity",
    "eff_mosaic": "effective conductivity mosaic (.png)",
    "eff_wmgm_mosaic": "effective conductivity mosaic, WM-GM (.png)",
    "eval_hist": "eigenvalue histogram",
}


# ============================================================================
# Download links
# ============================================================================
def _download_links(result: service.JobResult) -> str:
    """Markdown hyperlinks for what a run produced, the result first."""
    if result is None:
        return ""

    def link(path, text=None) -> str:
        return f"[{text or result.short_name(path)}]({FILE_ROUTE}{Path(path).resolve()})"

    blocks = []
    if result.zip_path and Path(result.zip_path).is_file():
        blocks.append(f"**[⬇ Download all (.zip)]"
                      f"({FILE_ROUTE}{Path(result.zip_path).resolve()})**")
    if result.tensors_6x1:
        blocks.append("**Result — conductivity tensor, 6×1 per voxel**  \n"
                      + " · ".join(link(p) for p in result.tensors_6x1))
    items = [link(p, ROLE_LABELS.get(role, result.short_name(p)))
             for role, p in result.intermediates if Path(p).is_file()]
    if items:
        blocks.append("**Intermediates**  \n" + " · ".join(items))
    return "\n\n".join(blocks) or "_(this run wrote no downloadable volume)_"


# ============================================================================
# Canvas
# ============================================================================
def _render(view_label, result):
    """What the canvas shows. Every view comes from a file the run wrote, so
    before a run there is nothing to draw."""
    try:
        return service.render_view(view_label, result)
    except Exception:
        logger.exception("Could not render %s", view_label)
        return None


def _freq_choices(cond_csv):
    """(choices, value) for the frequency box, from the table's columns."""
    try:
        choices = runctx.csv_frequencies(
            cond_csv or task_settings.DEFAULT_CONDUCTIVITY_CSV)
    except (OSError, ValueError):
        choices = []                    # build() will say what is wrong
    return choices, (runctx.default_freq(cond_csv) if choices else None)


def _on_csv_change(cond_csv):
    """A conductivity table was chosen: move the anisotropic default and the
    frequency list with it.

    Which ids are "white matter related", and which frequencies exist, are
    properties of the table, so both boxes follow it rather than a constant.
    """
    ids = runctx.default_aniso_ids(cond_csv)
    choices, value = _freq_choices(cond_csv)
    return (gr.update(value=runctx.ids_text(ids),
                      info=f"From CSV: {runctx.describe_ids(ids)}."),
            gr.update(choices=choices, value=value))


# ============================================================================
# The run
# ============================================================================
def _on_seg_change(seg_file):
    """A segmentation was chosen: propose the subject id its name carries."""
    return service.default_case_name(seg_file) if seg_file else gr.update()


def _run(dwi_files, bval_file, bvec_file, seg_file, cond_csv, tracts_file,
         aniso_ids, case_name, freq, seg_registered, t1_file, model_name,
         request: gr.Request = None):
    """Yields (console, downloads box, duration, links, canvas, view, state)."""
    hidden = gr.update(visible=False)
    idle = (hidden, "", "", gr.update(), gr.update(), gr.update())

    missing = [name for name, value in (
        ("DWI series", dwi_files), ("bval", bval_file),
        ("bvec", bvec_file), ("segmentation", seg_file)) if not value]
    if not seg_registered and not t1_file:
        missing.append("T1 (needed to register the segmentation)")
    if missing:
        yield (f"Missing input: {', '.join(missing)}.", *idle)
        return

    jobs.prune_old_runs("anicond")
    run = jobs.new_run("anicond")
    transcript = jobs.Transcript(log_file=run.log_file)

    def target():
        with jobs.capture_logs(transcript):
            with jobs.QUEUE.slot(report=transcript.report, run=run):
                return service.run_job(
                    dwi_paths=dwi_files,
                    bval_path=bval_file,
                    bvec_path=bvec_file,
                    seg_path=seg_file,
                    model_names=[model_name],
                    aniso_ids_text=aniso_ids,
                    # Not on the page: only directmapping reads it, and it is a
                    # property of the model, so it comes from the table in use.
                    iso_ref_ids_text=None,
                    conductivity_csv=cond_csv,
                    freq=freq,
                    seg_registered=bool(seg_registered),
                    t1_path=t1_file,
                    tracts_path=tracts_file,
                    case_name=case_name,
                    run=run,
                    report=transcript.report,
                )

    for _text, result, error in common.stream_job(target, transcript,
                                                  run=run, request=request):
        # The console gets the WHOLE transcript, not a tail: it scrolls, and a
        # user chasing an error wants to scroll back to the start.
        console = transcript.text()

        if error is not None:
            # Shown in the console, under the steps that led to it: the
            # transcript is the diagnosis, a bare banner would throw it away.
            logger.exception("anicond job %s failed", run.job_id, exc_info=error)
            transcript.report(f"FAILED: {type(error).__name__}: {error}")
            transcript.report(f"Full transcript: {run.log_file}")
            yield (transcript.text(), *idle)
            return

        if result is None:
            yield (console, *idle)
            continue

        for line in result.summary.splitlines():
            transcript.report(line)

        yield (transcript.text(),
               gr.update(visible=True),
               f"**Completed in {result.elapsed_s:.0f} s** "
               f"&nbsp;·&nbsp; {', '.join(result.ctx.model_names)} "
               f"&nbsp;·&nbsp; job `{run.job_id}`",
               _download_links(result),
               _render(service.RESULT_VIEW, result),
               gr.update(value=service.RESULT_VIEW),  # jump to what they came for
               result)


# ============================================================================
# Layout
# ============================================================================
def build_page() -> None:
    """Lay out this task's page. Called by app/web/main.py inside a route."""
    common.page_header("DTI → anisotropic conductivity tensor", INTRO)
    gr.HTML(PAGE_CSS)

    result_state = gr.State(None)
    default_aniso = runctx.default_aniso_ids()

    with gr.Row():
        # ----------------------------------------------------- controls --
        with gr.Column(scale=2):
            common.section_label("Input")
            gr.Markdown("### Inputs")
            # Labels stay short: the drop zones are narrow and a long label is
            # clipped mid-word; what each file is goes in the note below.
            gr.Markdown(
                f"<b>DWI</b>: one 4D file, or every 3D volume at once; "
                f"<b>Tissue table (CSV)</b>: maps tissue id → conductivity"
                f" [default CSV here]"
                f"({FILE_ROUTE}{task_settings.DEFAULT_CONDUCTIVITY_CSV.resolve()})"
                f"; <b>Tracts</b>: the whole-brain tractogram (will be calculated if not available).",
                elem_classes="anicond-note")

            with gr.Row(equal_height=True):
                dwi_files = gr.File(
                    label="DWI series", file_count="multiple",
                    file_types=NIFTI_TYPES, type="filepath",
                    height=UPLOAD_BOX_HEIGHT, elem_classes="anicond-upload")
                bval_file = gr.File(label="bval", type="filepath",
                                    height=UPLOAD_BOX_HEIGHT,
                                    elem_classes="anicond-upload")
                bvec_file = gr.File(label="bvec", type="filepath",
                                    height=UPLOAD_BOX_HEIGHT,
                                    elem_classes="anicond-upload")

            with gr.Row(equal_height=True):
                seg_file = gr.File(label="Tissue segmentation",
                                   file_types=NIFTI_TYPES, type="filepath",
                                   height=UPLOAD_BOX_HEIGHT,
                                   elem_classes="anicond-upload")
                cond_csv = gr.File(label="Tissue table (CSV, optional)",
                                   file_types=[".csv"], type="filepath",
                                   height=UPLOAD_BOX_HEIGHT,
                                   elem_classes="anicond-upload")
                tracts_file = gr.File(label="Tracts (VTK, optional)",
                                      file_types=TRACT_TYPES, type="filepath",
                                      height=UPLOAD_BOX_HEIGHT,
                                      elem_classes="anicond-upload")

            # The choices that shape the result, side by side: all are
            # text-height boxes with one line of info, so the row is level.
            with gr.Row(equal_height=True):
                model_name = gr.Dropdown(
                    label="Conductivity model", scale=3,
                    choices=runctx.model_labels(),
                    value=task_settings.DEFAULT_MODEL)
                aniso_ids = gr.Textbox(
                    label="Anisotropic tissue id(s)", scale=3,
                    value=runctx.ids_text(default_aniso),
                    info=f"From CSV: "
                         f"{runctx.describe_ids(default_aniso)}.")
                case_name = gr.Textbox(
                    label="Subject id (optional)", scale=2,
                    placeholder="auto")

            # The declaration, the frequency and the button share one boxed
            # row: gr.Group draws the frame around all three (a bare Button
            # would otherwise sit outside the form's box), and equal_height
            # lets the button fill its cell.
            freq_choices, freq_value = _freq_choices(None)
            with gr.Group(), gr.Row(equal_height=True):
                seg_registered = gr.Checkbox(
                    label="Segmentation is already in the DWI's space",
                    value=True, scale=5)
                freq = gr.Dropdown(
                    label="Frequency in use", scale=1, min_width=120,
                    choices=freq_choices, value=freq_value)
                run_btn = gr.Button("Run", variant="primary",
                                    scale=1, min_width=150)
            # Only asked for when it is needed, and it IS needed then: without
            # a T1 there is no transform to carry the labels by.
            t1_file = gr.File(label="T1 the segmentation was drawn on",
                              file_types=NIFTI_TYPES, type="filepath",
                              height=UPLOAD_BOX_HEIGHT, visible=False,
                              elem_classes="anicond-upload")

            # Above the console: once a run is done this is what the user came
            # for, and they should not have to scroll past a log to reach it.
            with gr.Group(visible=False) as downloads_box:
                common.section_label("Output")
                duration = gr.Markdown("")
                links = gr.Markdown("")

            console = gr.Textbox(
                label="Console", lines=CONSOLE_LINES, max_lines=CONSOLE_LINES,
                interactive=False, autoscroll=True,
                elem_classes="anicond-console",
                value="Idle. Upload a DWI series, its gradients and a "
                      "segmentation, then press Run.")

        # ------------------------------------------------------ display --
        with gr.Column(scale=3):
            view = gr.Dropdown(choices=service.view_labels(),
                               value=service.DEFAULT_VIEW, label="Show")
            canvas = gr.Image(label="Mosaic", type="pil", height=CANVAS_HEIGHT)

    common.page_footer()

    # -------------------------------------------------------- wiring --
    cond_csv.change(_on_csv_change, inputs=cond_csv, outputs=[aniso_ids, freq])
    seg_file.change(_on_seg_change, inputs=seg_file, outputs=case_name)
    seg_registered.change(lambda ok: gr.update(visible=not ok),
                          inputs=seg_registered, outputs=t1_file)
    view.change(_render, inputs=[view, result_state], outputs=canvas)

    run_btn.click(
        _run,
        inputs=[dwi_files, bval_file, bvec_file, seg_file, cond_csv,
                tracts_file, aniso_ids, case_name, freq, seg_registered,
                t1_file, model_name],
        outputs=[console, downloads_box, duration, links, canvas, view,
                 result_state],
    )

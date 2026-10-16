# ============================================================================
# tasks/tract_efield/web.py — the Gradio page for TMS field -> tracts
# ----------------------------------------------------------------------------
# Widgets and callbacks ONLY; the selection lives in service.py and every
# figure in preview.py, so a cli.py added later reproduces this page by
# calling the same run_job().
#
# Not on the home menu (registry: visible=False): reachable at /tract_efield
# by whoever is given the link. The layout is mesh_sim's:
#
#     left  | simulation archive + tractogram, one row (two drop zones)
#           | subject id + conductivity type, read out of the archive
#           | threshold row: percentile | absolute V/m (0 = use percentile)
#           |               | contact distance mm (any point counts, fixed)
#           | heal row: on/off | gap mm | angle, one line
#           | Run
#           | -> downloads + duration (hidden until a run finishes)
#           | -> console
#     right | small "Show" combobox
#           | -> large canvas
# ============================================================================
import logging
from pathlib import Path

import gradio as gr

from app import jobs
from app.web import common
from tasks.tract_efield import service
from tasks.tract_efield import settings as task_settings

logger = logging.getLogger(__name__)

INTRO = ("Upload the archive the Head Mesh → TMS Simulation page produced "
         "and the whole-brain tractogram of the same subject (from DTI → "
         "Conduc. Tensors), then Run. The task finds the white-surface "
         "vertices where |E| is in the top tail and keeps the streamlines "
         "that reach them.  \n"
         "Read from the uploaded simulation archive: "
         "`vtk/<id>_<type>_both.white.magn.vtk` — the white surface with "
         "|E| on it: the hot spot; also names the subject and the "
         "conductivity type; "
         "`subject_volumes/*_magnE.nii.gz` — |E| on the grid: the field "
         "along each tract; "
         "`subject_volumes/*_E.nii.gz` — the E vector: the activating "
         "function (optional — skipped without it); "
         "`subject_volumes/<id>_T1.nii.gz` — the pictures' background "
         "(optional — the field volume stands in); "
         "`<id>_job_spec.json` — the coil site named in the titles "
         "(optional). The head mesh, the surface overlays, the report and "
         "the other pictures in the archive are not read.")

ZIP_TYPES = [".zip"]
TRACT_TYPES = list(service.TRACT_TYPES)

UPLOAD_BOX_HEIGHT = 70
CANVAS_HEIGHT = 620
CONSOLE_LINES = 15

VIEW_LABELS = service.view_labels()
DEFAULT_VIEW = service.DEFAULT_VIEW

FILE_ROUTE = "/gradio_api/file="

PAGE_CSS = """
<style>
.tract-efield-upload .wrap:has(> .icon-wrap) {
    font-size: 0 !important; line-height: 0 !important;
    min-height: 0 !important; padding-top: 0 !important;
}
.tract-efield-upload .wrap:has(> .icon-wrap) .or { display: none !important; }
.tract-efield-upload .wrap:has(> .icon-wrap) .icon-wrap {
    margin-bottom: 0 !important; width: 26px !important;
}
.tract-efield-console textarea {
    background: #101214 !important; color: #c9d1d9 !important;
    font-family: ui-monospace, "DejaVu Sans Mono", Menlo, Consolas, monospace !important;
    font-size: 10px !important; line-height: 1.35 !important;
    border: 1px solid #2b3137 !important; resize: vertical !important;
    min-height: 90px; overflow-y: auto !important;
}
.tract-efield-guide table { font-size: 0.85em; }
.tract-efield-guide td, .tract-efield-guide th { padding: 2px 8px !important; }
.tract-efield-compact-number input { font-size: 0.85em !important; padding-right: 4px !important; }
/* The one-line "what was uploaded" note under the drop zones. */
.tract-efield-detected { font-size: 0.85em; }
.tract-efield-detected p { color: #4b5563 !important; margin: 0 !important; }
/* Heal row: gap/angle read as "label  box" on one line instead of Gradio's
   usual label-above-input (task 6's tract_simplify trick, same markup): drop
   each Number's own label and put a narrow Markdown beside it in the same
   Row). */
.tract-efield-inline-label { display: flex; align-items: center; }
.tract-efield-inline-label p {
    margin: 0 !important; white-space: nowrap; color: #374151 !important;
}
.tract-efield-inline-number { padding-top: 0 !important; }
</style>
"""


# ============================================================================
# Download links
# ============================================================================
def _link(path, text=None) -> str:
    return f"[{text or Path(path).name}]({FILE_ROUTE}{Path(path).resolve()})"


def _download_links(result) -> str:
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
    blocks.append("\n".join(rows))
    return "\n\n".join(blocks)


# ============================================================================
# Callbacks
# ============================================================================
def _on_archive(zip_path):
    """The subject id and the conductivity type are read from the archive's
    own name and index -- shown, not asked for."""
    if not zip_path:
        return gr.update(value="")
    subject, cond_type = service.identify_zip(zip_path)
    if not subject:
        return gr.update(value="⚠ not a TMS simulation archive "
                               "(no `<id>_tms_<type>` folder inside)")
    return gr.update(value=f"Archive: subject **{subject}** · conductivity "
                           f"type **{cond_type or '?'}**")


def _render(view_label, result):
    try:
        return service.render_view(view_label, result)
    except Exception:
        logger.exception("Could not render %s", view_label)
        return None


def _run(simulation_zip, tractogram, percentile, absolute, contact_mm,
         heal, join_mm, join_angle, request: gr.Request = None):
    """Yields (console, downloads box, duration, links, canvas, view, state)."""
    hidden = gr.update(visible=False)
    idle = (hidden, "", "", gr.update(), gr.update(), gr.update())
    missing = []
    if not simulation_zip:
        missing.append("simulation archive (.zip)")
    if not tractogram:
        missing.append("tractogram")
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
                    simulation_zip=simulation_zip, tractogram=tractogram,
                    percentile=float(percentile),
                    absolute_v_per_m=float(absolute or 0.0),
                    contact_mm=float(contact_mm),
                    follow_breaks_on=bool(heal), join_mm=float(join_mm),
                    join_max_angle_deg=float(join_angle),
                    run=run, report=transcript.report)

    for _text, result, error in common.stream_job(target, transcript,
                                                  run=run, request=request):
        console = transcript.text()
        if error is not None:
            logger.exception("tract_efield job %s failed", run.job_id,
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
        yield (transcript.text(),
               gr.update(visible=True),
               f"**Completed in {result.elapsed_s / 60:.1f} min** "
               f"&nbsp;·&nbsp; subject `{result.subject_id}` "
               f"&nbsp;·&nbsp; `{result.conductivity_type}` "
               f"&nbsp;·&nbsp; {result.n_selected:,} of "
               f"{result.n_streamlines:,} streamlines "
               f"&nbsp;·&nbsp; job `{run.job_id}`",
               _download_links(result),
               _render(service.RESULT_VIEW, result),
               gr.update(value=service.RESULT_VIEW),
               result)


# ============================================================================
# Layout
# ============================================================================
def build_page() -> None:
    common.page_header("TMS field → tracts", INTRO)
    gr.HTML(PAGE_CSS)
    result_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=2):
            common.section_label("Input")
            # Two drop zones on one row; what they contain is READ off the
            # archive and said in one line under them rather than asked
            # for in boxes of their own.
            with gr.Row(equal_height=True):
                simulation_zip = gr.File(
                    label="TMS simulation archive (.zip)",
                    file_types=ZIP_TYPES, type="filepath",
                    height=UPLOAD_BOX_HEIGHT,
                    elem_classes="tract-efield-upload")
                tractogram = gr.File(
                    label="Whole-brain tractogram (.vtk / .trk / .tck)",
                    file_types=TRACT_TYPES, type="filepath",
                    height=UPLOAD_BOX_HEIGHT,
                    elem_classes="tract-efield-upload")
            detected = gr.Markdown("", elem_classes="tract-efield-detected")

            # The selection, in two rows: what counts as the hot spot and
            # how close a streamline (any of its points, not only an
            # endpoint) must come to it; then whether to heal tracts that
            # tractography broke into pieces before selecting.
            with gr.Group():
                with gr.Row(equal_height=True):
                    percentile = gr.Number(
                        label="Hot spot: |E| percentile (white surface)",
                        value=task_settings.DEFAULT_PERCENTILE,
                        minimum=task_settings.PERCENTILE_MIN,
                        maximum=task_settings.PERCENTILE_MAX, step=0.1,
                        min_width=120,
                        elem_classes="tract-efield-compact-number")
                    absolute = gr.Number(
                        label="…or absolute V/m (0 = off)",
                        value=task_settings.DEFAULT_ABSOLUTE_V_PER_M,
                        minimum=0.0, min_width=110,
                        elem_classes="tract-efield-compact-number")
                    contact_mm = gr.Number(
                        label="Contact distance (mm)",
                        value=task_settings.DEFAULT_CONTACT_MM,
                        minimum=task_settings.CONTACT_MM_MIN,
                        maximum=task_settings.CONTACT_MM_MAX, step=0.5,
                        info="any point along the streamline within the "
                             "distance",
                        min_width=110,
                        elem_classes="tract-efield-compact-number")
                with gr.Row(equal_height=True):
                    heal = gr.Checkbox(
                        label="Heal broken tracts",
                        value=task_settings.DEFAULT_FOLLOW_BREAKS,
                        info="continue along pieces head-to-tail",
                        scale=3, min_width=170)
                    gr.Markdown("Gap ≤ (mm)",
                               elem_classes="tract-efield-inline-label",
                               min_width=70)
                    join_mm = gr.Number(
                        value=task_settings.DEFAULT_JOIN_MM,
                        minimum=task_settings.JOIN_MM_MIN,
                        maximum=task_settings.JOIN_MM_MAX, step=0.5,
                        show_label=False, container=False,
                        scale=1, min_width=55,
                        elem_classes=("tract-efield-compact-number",
                                     "tract-efield-inline-number"))
                    gr.Markdown("Angle ≤ (°)",
                               elem_classes="tract-efield-inline-label",
                               min_width=70)
                    join_angle = gr.Number(
                        value=task_settings.JOIN_MAX_ANGLE_DEG,
                        minimum=task_settings.JOIN_ANGLE_MIN,
                        maximum=task_settings.JOIN_ANGLE_MAX, step=5,
                        show_label=False, container=False,
                        scale=1, min_width=50,
                        elem_classes=("tract-efield-compact-number",
                                     "tract-efield-inline-number"))
            run_btn = gr.Button("Run", variant="primary")

            with gr.Group(visible=False) as downloads_box:
                common.section_label("Output")
                duration = gr.Markdown("")
                links = gr.Markdown("", elem_classes="tract-efield-guide")

            console = gr.Textbox(
                label="Console", lines=CONSOLE_LINES, max_lines=CONSOLE_LINES,
                interactive=False, autoscroll=True,
                elem_classes="tract-efield-console",
                value="Idle. Upload the simulation archive and the "
                      "tractogram, then press Run.")

        with gr.Column(scale=3):
            view = gr.Dropdown(choices=VIEW_LABELS, value=DEFAULT_VIEW,
                               label="Show")
            canvas = gr.Image(label="Mosaic", type="pil", height=CANVAS_HEIGHT)

    common.page_footer()

    simulation_zip.change(_on_archive, inputs=simulation_zip,
                          outputs=detected)
    view.change(_render, inputs=[view, result_state], outputs=canvas)
    run_btn.click(
        _run,
        inputs=[simulation_zip, tractogram, percentile, absolute, contact_mm,
                heal, join_mm, join_angle],
        outputs=[console, downloads_box, duration, links, canvas, view,
                 result_state],
    )

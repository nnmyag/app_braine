# ============================================================================
# tasks/tract_simplify/web.py — the Gradio page for tractogram -> visual tracts
# ----------------------------------------------------------------------------
# Widgets and callbacks ONLY; the work is service.py's, the canvas task 5's
# preview.py, so a cli.py added later reproduces this page through the same
# run_job() + apply().
#
# Not on the home menu (registry: visible=False): reachable at /tract_simplify.
# The layout is tract_efield's, with one difference in the wiring: Run does
# the loading, healing and clustering ONCE through the shared queue, and the
# three sliders then act on the cached arrays on release -- milliseconds of
# selection plus a redraw, no queue -- rewriting the download file each
# time, so the link is always current. The thinning slider's description
# line shows the count its position would keep while it is being dragged.
#
# The 3D viewer is NOT live: reloading NiiVue with a new .trk on every
# slider release was the heaviest thing on the page, so the sliders only
# mark it out of date and a "Refresh 3D viewer" button reloads it on
# demand (Run loads it once, when the first result is in).
#
#     left  | tractogram + reference image, one row (two drop zones)
#           | subject id, read off the tractogram's name
#           | heal row: on/off | gap mm | angle, one line  (take effect at Run)
#           | length · thinning · smoothing, one row of three sliders (live)
#           | Run
#           | -> downloads + numbers (hidden until a run finishes)
#           | -> console
#     right | one figure: original tracts (left) vs. revised (right), same
#           |   middle slice -- no view picker, nothing to choose
#           | -> Refresh 3D viewer button + its status line
#           | -> the 3D viewer (NiiVue in an iframe; a note when absent)
# ============================================================================
import logging
from pathlib import Path

import gradio as gr

from app import jobs
from app.web import common
from tasks.tract_simplify import service
from tasks.tract_simplify import settings as task_settings

logger = logging.getLogger(__name__)

INTRO = ("Upload a whole-brain tractogram (from DTI → Conduc. Tensors, or "
         "any .vtk / .trk / .tck) and an image in its space (the DWI's b0), "
         "then Run. Broken tracts are joined end to end and the streamlines "
         "clustered once; the sliders then cut short fragments, thin every "
         "bundle to a fraction and smooth what is left — for pictures only, "
         "never as an input to the tensor models.")

TRACT_TYPES = list(service.TRACT_TYPES)
IMAGE_TYPES = [".nii", ".gz"]

UPLOAD_BOX_HEIGHT = 70
CANVAS_HEIGHT = 620
CONSOLE_LINES = 12

FILE_ROUTE = service.FILE_ROUTE

PAGE_CSS = """
<style>
.tract-simplify-upload .wrap:has(> .icon-wrap) {
    font-size: 0 !important; line-height: 0 !important;
    min-height: 0 !important; padding-top: 0 !important;
}
.tract-simplify-upload .wrap:has(> .icon-wrap) .or { display: none !important; }
.tract-simplify-upload .wrap:has(> .icon-wrap) .icon-wrap {
    margin-bottom: 0 !important; width: 26px !important;
}
.tract-simplify-note { font-size: 0.3em; }
.tract-simplify-note p { font-size: inherit !important; color: #6b7280 !important; line-height: 0.5 }
.tract-simplify-console textarea {
    background: #101214 !important; color: #c9d1d9 !important;
    font-family: ui-monospace, "DejaVu Sans Mono", Menlo, Consolas, monospace !important;
    font-size: 10px !important; line-height: 1.35 !important;
    border: 1px solid #2b3137 !important; resize: vertical !important;
    min-height: 90px; overflow-y: auto !important;
}
.tract-simplify-guide table { font-size: 0.85em; }
.tract-simplify-guide td, .tract-simplify-guide th { padding: 2px 8px !important; }
.tract-simplify-compact-number input { font-size: 0.85em !important; padding-right: 4px !important; }
.tract-simplify-detected { font-size: 0.85em; }
.tract-simplify-detected p { color: #4b5563 !important; margin: 0 !important; }
.tract-simplify-viewer iframe { width: 100%; border: 1px solid #2b3137; border-radius: 6px; }
.tract-simplify-viewer-note p { color: #6b7280 !important; font-size: 0.85em; }
.tract-simplify-viewer-status p { color: #6b7280 !important; font-size: 0.85em; margin: 0 !important; }
/* Heal row: gap/angle read as "label  box" on one line instead of Gradio's
   usual label-above-input, by dropping each Number's own label and putting
   a narrow Markdown beside it in the same Row (see the heal row below). */
.tract-simplify-inline-label { display: flex; align-items: center; }
.tract-simplify-inline-label p {
    margin: 0 !important; white-space: nowrap; color: #374151 !important;
}
.tract-simplify-inline-number { padding-top: 0 !important; }
</style>
"""

NOTE = ("Healing and its two numbers take effect at Run (they come before "
        "the clustering); the three sliders act as soon as they are "
        "released and rewrite the download; the 3D viewer reloads only "
        "when you press Refresh.")


# ============================================================================
# Download links and the viewer
# ============================================================================
def _link(path, text=None) -> str:
    return f"[{text or Path(path).name}]({FILE_ROUTE}{Path(path).resolve()})"


def _download_links(result) -> str:
    if result is None:
        return ""
    blocks = []
    if result.zip_path and Path(result.zip_path).is_file():
        size = Path(result.zip_path).stat().st_size / 1e6
        blocks.append(f"**[⬇ Download all — both files below (.zip, "
                      f"{size:.1f} MB)]"
                      f"({FILE_ROUTE}{Path(result.zip_path).resolve()})**")
    rows = ["| file | what it is |", "|---|---|"]
    for _role, path, meaning in result.guide():
        rel = Path(path).relative_to(result.run.outputs)
        rows.append(f"| {_link(path, str(rel))} | {meaning} |")
    blocks.append("\n".join(rows))
    return "\n\n".join(blocks)


def _numbers_line(result) -> str:
    if result is None:
        return ""
    return (f"**{result.n_kept:,} of {result.n_original:,} streamlines "
            f"kept** &nbsp;·&nbsp; {result.settings_line} &nbsp;·&nbsp; "
            f"subject `{result.subject_id}` &nbsp;·&nbsp; "
            f"job `{result.run.job_id}`")


VIEWER_CURRENT = "3D viewer shows the current sliders."
VIEWER_STALE = ("3D viewer shows an earlier slider position — press "
                "**Refresh 3D viewer** to reload it.")


def _viewer_html(result, reload=0) -> str:
    ok, reason = service.viewer_status()
    if not ok:
        return (f'<div class="tract-simplify-viewer-note"><p>3D viewer not '
                f'available on this server: {reason}. The canvas above '
                f'is unaffected.</p></div>')
    url = service.viewer_url(result, reload=reload)
    if not url:
        return ""
    return (f'<div class="tract-simplify-viewer"><iframe src="{url}" '
            f'height="{task_settings.VIEWER_HEIGHT_PX}" '
            f'title="3D view of the revised tracts"></iframe></div>')


# ============================================================================
# Callbacks
# ============================================================================
def _on_tractogram(path):
    if not path:
        return gr.update(value="")
    return gr.update(value=f"Tractogram: `{Path(path).name}` → subject "
                           f"**{service.default_case_name(path)}**")


THINNING_INFO = (f"per cluster ({task_settings.QB_THRESHOLD_MM:g} mm), "
                 f"≥ 1 kept — live")


def _thinning_info(result, min_length, keep_percent):
    """The thinning slider's description line: how many streamlines its
    current position keeps, over the original count. Cheap enough to run
    while the slider is dragged; before Run it explains the slider."""
    if result is None:
        return gr.update(info=THINNING_INFO)
    try:
        n = service.count_kept(result, float(min_length), float(keep_percent))
    except Exception:
        logger.exception("Could not count the thinning")
        return gr.update(info=THINNING_INFO)
    return gr.update(info=f"{n:,}/{result.n_original:,} tracts — live")


def _refresh_viewer(result, reload):
    """The Refresh button: reload the viewer with the current .trk.
    Returns (viewer, status, reload counter)."""
    if result is None:
        return gr.update(), "", reload
    reload = int(reload or 0) + 1
    return _viewer_html(result, reload), VIEWER_CURRENT, reload


def _render(result):
    try:
        return service.render_canvas(result)
    except Exception:
        logger.exception("Could not render the canvas")
        return None


def _run(tractogram, reference, heal, heal_gap, heal_angle,
         min_length, keep_percent, smooth, request: gr.Request = None):
    """Yields (console, downloads box, numbers, links, canvas, viewer,
    viewer status, thinning info, state)."""
    hidden = gr.update(visible=False)
    idle = (hidden, "", "", gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update())
    missing = []
    if not tractogram:
        missing.append("tractogram")
    if not reference:
        missing.append("reference image (.nii.gz)")
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
                    tractogram=tractogram, reference=reference,
                    heal=bool(heal), heal_gap_mm=float(heal_gap),
                    heal_angle_deg=float(heal_angle),
                    min_length_mm=float(min_length),
                    keep_percent=float(keep_percent),
                    smooth_mm=float(smooth),
                    run=run, report=transcript.report)

    for _text, result, error in common.stream_job(target, transcript,
                                                  run=run, request=request):
        console = transcript.text()
        if error is not None:
            logger.exception("tract_simplify job %s failed", run.job_id,
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
        transcript.report(f"Done in {result.elapsed_s:.0f} s. Move a slider "
                          f"to revise; the download follows. Press Refresh "
                          f"to reload the 3D viewer.")
        yield (transcript.text(),
               gr.update(visible=True),
               _numbers_line(result),
               _download_links(result),
               _render(result),
               _viewer_html(result),
               VIEWER_CURRENT,
               _thinning_info(result, result.min_length_mm,
                              result.keep_percent),
               result)


def _apply(result, min_length, keep_percent, smooth):
    """A slider was released: re-cut, re-thin, re-smooth, rewrite, redraw.
    The 3D viewer is left alone and marked out of date -- reloading it is
    the Refresh button's job. Returns (numbers, links, canvas, viewer
    status, thinning info, state)."""
    if result is None:
        return (gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), None)
    try:
        service.apply(result, float(min_length), float(keep_percent),
                      float(smooth))
    except Exception as exc:              # noqa: BLE001 - shown, not raised
        logger.exception("tract_simplify apply failed")
        return (f"⚠ could not apply the sliders: {type(exc).__name__}: {exc}",
                gr.update(), gr.update(), gr.update(), gr.update(), result)
    return (_numbers_line(result), _download_links(result),
            _render(result), VIEWER_STALE,
            _thinning_info(result, result.min_length_mm, result.keep_percent),
            result)


# ============================================================================
# Layout
# ============================================================================
def build_page() -> None:
    common.page_header("Tractogram → visual tracts", INTRO)
    gr.HTML(PAGE_CSS)
    result_state = gr.State(None)
    viewer_reloads = gr.State(0)

    with gr.Row():
        with gr.Column(scale=2):
            common.section_label("Input")
            with gr.Row(equal_height=True):
                tractogram = gr.File(
                    label="Whole-brain tractogram (.vtk / .trk / .tck)",
                    file_types=TRACT_TYPES, type="filepath",
                    height=UPLOAD_BOX_HEIGHT,
                    elem_classes="tract-simplify-upload")
                reference = gr.File(
                    label="Reference image in its space — DWI b0 (.nii.gz)",
                    file_types=IMAGE_TYPES, type="filepath",
                    height=UPLOAD_BOX_HEIGHT,
                    elem_classes="tract-simplify-upload")
            detected = gr.Markdown("", elem_classes="tract-simplify-detected")
            gr.Markdown(NOTE, elem_classes="tract-simplify-note")

            with gr.Group():
                with gr.Row(equal_height=True):
                    heal = gr.Checkbox(
                        label="Heal broken tracts",
                        value=task_settings.DEFAULT_HEAL,
                        info="join pieces head-to-tail (at Run)",
                        scale=3, min_width=170)
                    gr.Markdown("Gap ≤ (mm)",
                               elem_classes="tract-simplify-inline-label",
                               min_width=70)
                    heal_gap = gr.Number(
                        value=task_settings.DEFAULT_HEAL_GAP_MM,
                        minimum=task_settings.HEAL_GAP_MIN,
                        maximum=task_settings.HEAL_GAP_MAX, step=0.5,
                        show_label=False, container=False,
                        scale=1, min_width=55,
                        elem_classes=("tract-simplify-compact-number",
                                     "tract-simplify-inline-number"))
                    gr.Markdown("Angle ≤ (°)",
                               elem_classes="tract-simplify-inline-label",
                               min_width=70)
                    heal_angle = gr.Number(
                        value=task_settings.DEFAULT_HEAL_ANGLE_DEG,
                        minimum=task_settings.HEAL_ANGLE_MIN,
                        maximum=task_settings.HEAL_ANGLE_MAX, step=5,
                        show_label=False, container=False,
                        scale=1, min_width=50,
                        elem_classes=("tract-simplify-compact-number",
                                     "tract-simplify-inline-number"))
                with gr.Row(equal_height=True):
                    min_length = gr.Slider(
                        label="Min length (mm)",
                        value=task_settings.DEFAULT_MIN_LENGTH_MM,
                        minimum=task_settings.MIN_LENGTH_MIN,
                        maximum=task_settings.MIN_LENGTH_MAX, step=1,
                        info="after healing — live")
                    keep_percent = gr.Slider(
                        label="Thinning (%)",
                        value=task_settings.DEFAULT_KEEP_PERCENT,
                        minimum=task_settings.KEEP_PERCENT_MIN,
                        maximum=task_settings.KEEP_PERCENT_MAX, step=1,
                        info=THINNING_INFO)
                    smooth = gr.Slider(
                        label="Smoothing σ (mm)",
                        value=task_settings.DEFAULT_SMOOTH_MM,
                        minimum=task_settings.SMOOTH_MIN_MM,
                        maximum=task_settings.SMOOTH_MAX_MM, step=0.1,
                        info="0 = off — live")
            run_btn = gr.Button("Run", variant="primary")

            with gr.Group(visible=False) as downloads_box:
                common.section_label("Output")
                numbers = gr.Markdown("")
                links = gr.Markdown("", elem_classes="tract-simplify-guide")

            console = gr.Textbox(
                label="Console", lines=CONSOLE_LINES, max_lines=CONSOLE_LINES,
                interactive=False, autoscroll=True,
                elem_classes="tract-simplify-console",
                value="Idle. Upload the tractogram and its reference image, "
                      "then press Run.")

        with gr.Column(scale=3):
            canvas = gr.Image(label="Original (left) · Revised (right)",
                              type="pil", height=CANVAS_HEIGHT)
            with gr.Row(equal_height=True):
                refresh_btn = gr.Button("Refresh 3D viewer", scale=0,
                                        min_width=160)
                viewer_status = gr.Markdown(
                    "", elem_classes="tract-simplify-viewer-status")
            viewer = gr.HTML(_viewer_html(None))

    common.page_footer()

    tractogram.change(_on_tractogram, inputs=tractogram, outputs=detected)
    run_btn.click(
        _run,
        inputs=[tractogram, reference, heal, heal_gap, heal_angle,
                min_length, keep_percent, smooth],
        outputs=[console, downloads_box, numbers, links, canvas,
                 viewer, viewer_status, keep_percent, result_state],
    )
    for slider in (min_length, keep_percent, smooth):
        slider.release(
            _apply,
            inputs=[result_state, min_length, keep_percent, smooth],
            outputs=[numbers, links, canvas, viewer_status, keep_percent,
                     result_state],
        )
    # While the length or thinning slider is dragged, the thinning slider's
    # description line follows: the count is a mask over cached arrays.
    for slider in (min_length, keep_percent):
        slider.input(
            _thinning_info,
            inputs=[result_state, min_length, keep_percent],
            outputs=keep_percent, show_progress="hidden",
        )
    refresh_btn.click(
        _refresh_viewer,
        inputs=[result_state, viewer_reloads],
        outputs=[viewer, viewer_status, viewer_reloads],
    )

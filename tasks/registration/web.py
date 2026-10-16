# ============================================================================
# tasks/registration/web.py — register a moving NIfTI onto a fixed one
# ----------------------------------------------------------------------------
# Widgets and callbacks ONLY; the work is in service.py. A utility: a tile on
# the home page, not in the navbar (app/registry.py, utility=True).
# ============================================================================
import logging

import gradio as gr

from app import jobs
from app.web import common
from app.web import slice_viewer as sv
from tasks.registration import service
from tasks.registration import settings as task_settings

logger = logging.getLogger(__name__)

TITLE = "Affine registration"
INTRO = ("Register a moving NIfTI image onto a fixed one (rigid or affine, mutual "
         "information). The result is resampled onto the fixed image's grid; a "
         "segmentation of the moving image can be registered with the same transform.")
NIFTI_TYPES = [".nii", ".gz"]
RESULT_TTL_SECONDS = 2 * 3600


def _run(fixed, moving, seg, mode, request: gr.Request = None):
    """Yields (console, status, slider, viewer, files, result)."""
    keep = gr.update()
    missing = [n for n, f in (("fixed", fixed), ("moving", moving)) if not f]
    if missing:
        yield ("", f"**Missing:** the {' and '.join(missing)} image.", keep, keep, keep, keep)
        return
    transcript = jobs.Transcript()

    def target():
        with jobs.capture_logs(transcript):
            with jobs.QUEUE.slot(report=transcript.report):
                return service.register(fixed, moving, seg, mode, report=transcript.report)

    yield ("", "Registering…", keep, keep, None, None)
    for _text, result, error in common.stream_job(target, transcript, request=request):
        if error is not None:
            logger.exception("registration failed", exc_info=error)
            transcript.report(f"FAILED: {type(error).__name__}: {error}")
            yield (transcript.text(), f"**Failed:** {error}", keep,
                   sv.placeholder_html(f"Registration failed: {error}"), None, None)
            return
        if result is None:
            yield (transcript.text(), keep, keep, keep, keep, keep)
            continue
        k = result.default_slice()
        yield (transcript.text(), "**Done.** " + result.summary.split(". Matrix")[0] + ".",
               gr.update(minimum=0, maximum=max(result.depth - 1, 1), value=k,
                         visible=result.depth > 1),
               service.render_slice(result, k), result.files, result)


def _on_slice(result, k):
    try:
        return service.render_slice(result, k)
    except Exception as exc:                       # noqa: BLE001 - shown on the page
        logger.exception("registration: render failed")
        return sv.placeholder_html(f"Could not draw this slice: {exc}")


def build_page() -> None:
    common.page_header(TITLE, INTRO)
    gr.HTML(sv.CSS)
    result_state = gr.State(None, time_to_live=RESULT_TTL_SECONDS)
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row(equal_height=True):
                fixed = gr.File(label="Fixed image", file_types=NIFTI_TYPES,
                                type="filepath", height=100)
                moving = gr.File(label="Moving image", file_types=NIFTI_TYPES,
                                 type="filepath", height=100)
            seg = gr.File(label="Segmentation of the moving image (optional)",
                          file_types=NIFTI_TYPES, type="filepath", height=100)
            with gr.Row(equal_height=True):
                mode = gr.Radio(list(task_settings.MODES), value=task_settings.DEFAULT_MODE,
                                show_label=False, container=False, scale=3)
                run_btn = gr.Button("Run", variant="primary", scale=1)
            status = gr.Markdown("")
            files = gr.File(label="Registered files", file_count="multiple",
                            interactive=False)
            console = gr.Textbox(label="Console", lines=8, max_lines=8,
                                 interactive=False, autoscroll=True)
        with gr.Column(scale=3):
            slice_k = gr.Slider(label="Slice", minimum=0, maximum=1, step=1, value=0,
                                visible=False)
            viewer = gr.HTML(service.render_slice(None, 0), js_on_load=sv.HOVER_JS,
                             apply_default_css=False, padding=False)
    common.page_footer()
    run_btn.click(_run, inputs=[fixed, moving, seg, mode],
                  outputs=[console, status, slice_k, viewer, files, result_state])
    slice_k.change(_on_slice, inputs=[result_state, slice_k], outputs=viewer,
                   show_progress="hidden", trigger_mode="always_last")

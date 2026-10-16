# ============================================================================
# tasks/resample/web.py — resample a volume to an isotropic voxel size
# ----------------------------------------------------------------------------
# Widgets and callbacks ONLY; the work is in service.py. A utility: a tile on
# the home page, not in the navbar (app/registry.py, utility=True).
# ============================================================================
import logging
from pathlib import Path

import gradio as gr

from app import jobs
from app.web import common
from app.web import slice_viewer as sv
from tasks.resample import service
from tasks.resample import settings as task_settings

logger = logging.getLogger(__name__)

TITLE = "Resample voxel size"
INTRO = ("Resample NIfTI volumes to a desired voxel size.")
NIFTI_TYPES = [".nii", ".gz"]

PAGE_CSS = """
<style>
/* Download list: compact rows. */
.rs-files tr.file { padding: 2px 6px !important; }
.rs-files tr.file td { padding-top: 2px !important; padding-bottom: 2px !important; }
/* "Interpolation" label beside its dropdown, not above it. */
.rs-inline-label { display: flex; align-items: center; flex: 0 0 6rem !important; min-width: 6rem !important; max-width: 6rem !important; }
.rs-inline-label p { margin: 0 !important; white-space: nowrap; }
/* Console: black, near-white monospace text. */
.rs-console textarea {
    background: #000 !important; color: #e8e8e8 !important;
    font-family: ui-monospace, "DejaVu Sans Mono", Menlo, Consolas, monospace !important;
    font-size: 11px !important; line-height: 1.35 !important;
}
/* Before/after table: 1.5x the slice viewer's 11 px. */
.rs-table table.sv-table { font-size: 16.5px; }
.rs-table .sv-note { font-size: 15px; }
</style>
"""


def _on_upload(paths):
    """One line per uploaded file."""
    lines = []
    for path in paths or []:
        try:
            info = service.describe(path)
            kind = "labels" if service.is_labels(path) else "image"
            lines.append(f"- {info.short_line()} ({kind})")
        except Exception as exc:                   # noqa: BLE001 - shown on the page
            lines.append(f"- **{Path(path).name}: cannot read this file:** {exc}")
    return "\n".join(lines)


def _show(batch, name):
    if batch is None:
        return service.render(None)
    return service.render(batch.by_name(name))


def _run(paths, voxel, method, reorient, request: gr.Request = None):
    """Yields (console, status, picker, canvas, files, batch)."""
    keep = gr.update()
    if not paths:
        yield "", "**Upload a volume first.**", keep, keep, None, None
        return
    transcript = jobs.Transcript()

    def target():
        with jobs.capture_logs(transcript):
            with jobs.QUEUE.slot(report=transcript.report):
                return service.resample_many(paths, voxel, method, bool(reorient),
                                             report=transcript.report)

    yield "", f"Resampling {len(paths)} file(s)…", keep, keep, None, None
    for _text, result, error in common.stream_job(target, transcript, request=request):
        if error is not None:
            if not isinstance(error, ValueError):
                logger.exception("resample failed", exc_info=error)
            transcript.report(f"FAILED: {error}")
            yield transcript.text(), f"**Failed:** {error}", keep, keep, None, None
            return
        if result is None:
            yield transcript.text(), keep, keep, keep, keep, keep
            continue
        names = result.names()
        files = ([result.zip_path] if result.zip_path else []) + [r.path for r in result.results]
        yield (transcript.text(),
               f"**Done** · {len(names)} file(s) at {float(voxel):g} mm"
               + (" · the zip holds them all" if result.zip_path else ""),
               gr.update(choices=names, value=names[0], visible=len(names) > 1),
               service.render(result.results[0]), files, result)


def build_page() -> None:
    common.page_header(TITLE, INTRO)
    gr.HTML(sv.CSS + PAGE_CSS)
    batch_state = gr.State(None, time_to_live=2 * 3600)
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row(equal_height=True):
                volume = gr.File(label="Volumes (.nii / .nii.gz; one or several)",
                                 file_types=NIFTI_TYPES, file_count="multiple",
                                 type="filepath", height=160, scale=3)
                voxel = gr.Number(value=task_settings.DEFAULT_VOXEL_MM,
                                  label="Voxel size (mm, isotropic)",
                                  minimum=task_settings.MIN_VOXEL_MM,
                                  maximum=task_settings.MAX_VOXEL_MM, step=0.1, scale=1,
                                  min_width=110)
            info = gr.Markdown("")
            with gr.Row(equal_height=True):
                gr.Markdown("Interpolation", elem_classes="rs-inline-label",
                            scale=0, min_width=96)
                method = gr.Dropdown(list(task_settings.METHODS), value=task_settings.DEFAULT_METHOD,
                                     show_label=False, container=False, scale=1, min_width=110)
                reorient = gr.Checkbox(value=task_settings.DEFAULT_REORIENT_RAS,
                                       label="To RAS", scale=1, min_width=90)
                run_btn = gr.Button("Run", variant="primary", scale=1, min_width=70)
            status = gr.Markdown("")
            output = gr.File(label="Resampled volumes", file_count="multiple",
                             interactive=False, elem_classes="rs-files")
            console = gr.Textbox(label="Console", lines=6, max_lines=6,
                                 interactive=False, autoscroll=True, elem_classes="rs-console")
        with gr.Column(scale=3):
            picker = gr.Dropdown(label="Show", choices=[], visible=False)
            canvas = gr.HTML(service.render(None), apply_default_css=False, padding=False)
    common.page_footer()
    volume.change(_on_upload, inputs=volume, outputs=info)
    run_btn.click(_run, inputs=[volume, voxel, method, reorient],
                  outputs=[console, status, picker, canvas, output, batch_state])
    picker.input(_show, inputs=[batch_state, picker], outputs=canvas)

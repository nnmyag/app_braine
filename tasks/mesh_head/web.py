# ============================================================================
# tasks/mesh_head/web.py — the Gradio page for T1/T2 -> head mesh (charm)
# ----------------------------------------------------------------------------
# Widgets and callbacks ONLY. The pipeline lives in service.py and every
# figure in preview.py, so a cli.py added later reproduces this page's
# behaviour by calling the same run_job().
#
# The layout is tasks 1 and 2's, because the pages are read the same way:
#
#     left  | inputs: T1 + T2 (+ b0 when the volumes are not in its space)
#           |         subject id / "already in b0 space" / Run
#           | -> downloads + what each file is (hidden until a run finishes)
#           | -> console (charm's own lines stream through it)
#     right | small "Show" combobox
#           | -> large canvas
#
# Before anything is laid out the page asks service.simnibs_status(): with no
# usable SimNIBS on the server the form is shown disabled under a notice,
# because a Run button that fails a minute later is worse than none.
# ============================================================================
import logging
from pathlib import Path

import gradio as gr

from app import jobs
from app.web import common
from tasks.mesh_head import service

logger = logging.getLogger(__name__)

INTRO = ("Upload a T1 (and a T2) as NIfTI files and Run. SimNIBS's "
         "charm segments the head, reconstructs the cortical surfaces and "
         "builds the tetrahedral head mesh. Typically 30–60 minutes.")

NIFTI_TYPES = [".nii", ".gz"]

# Gradio serves any file under `allowed_paths` from this route; run_web.py
# allows data/runs, which is where a job writes.
FILE_ROUTE = "/gradio_api/file="

# Compact drop zones (task 2's height and icon-only styling): three of them
# fit one row when the b0 box is revealed.
UPLOAD_BOX_HEIGHT = 70
CANVAS_HEIGHT = 620
CONSOLE_LINES = 15

PAGE_CSS = """
<style>
.mesh-head-upload .wrap:has(> .icon-wrap) {
    font-size: 0 !important;
    line-height: 0 !important;
    min-height: 0 !important;
    padding-top: 0 !important;
}
.mesh-head-upload .wrap:has(> .icon-wrap) .or { display: none !important; }
.mesh-head-upload .wrap:has(> .icon-wrap) .icon-wrap {
    margin-bottom: 0 !important;
    width: 26px !important;
}
.mesh-head-note { font-size: 0.3em; }
.mesh-head-note p { font-size: inherit !important; color: #6b7280 !important; line-height: 0.5 }
.mesh-head-unsupported {
    border: 1px solid #e0a0a0; background: #fff4f4; color: #7a1f1f;
    border-radius: 8px; padding: 0.6em 0.9em; margin: 0.2em 0 0.6em;
}
@media (prefers-color-scheme: dark) {
    .mesh-head-unsupported { background: #3a1e1e; color: #ffb3b3; border-color: #7a3b3b; }
}
.mesh-head-console textarea {
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
/* The file guide: a dense two-column table, not a paragraph per file. */
.mesh-head-guide table { font-size: 0.85em; }
.mesh-head-guide td, .mesh-head-guide th { padding: 2px 8px !important; }
</style>
"""


# ============================================================================
# Download links + file guide
# ============================================================================
def _link(path, text=None) -> str:
    return f"[{text or Path(path).name}]({FILE_ROUTE}{Path(path).resolve()})"


def _download_links(result: service.JobResult) -> str:
    """The archive first, then every guided file with what it is."""
    if result is None:
        return ""
    blocks = []
    if result.zip_path and Path(result.zip_path).is_file():
        size = Path(result.zip_path).stat().st_size / 1e6
        blocks.append(f"**[⬇ Download all — the m2m folder and every file "
                      f"below (.zip, {size:.0f} MB)]"
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
# Canvas
# ============================================================================
def _render(view_label, t1_path, t2_path, b0_path, result):
    """The one rule for what the canvas shows: an input view is drawn from the
    uploaded FILE whenever selected, everything else from the finished run."""
    slot = service.is_input_view(view_label)
    try:
        if slot is not None:
            path = {"t1": t1_path, "t2": t2_path, "b0": b0_path}[slot]
            label = {"t1": "T1", "t2": "T2", "b0": "DTI b0"}[slot]
            return service.render_input_preview(path, label) if path else None
        return service.render_result_view(view_label, result)
    except Exception:
        logger.exception("Could not render %s", view_label)
        return None


def _on_upload(path, slot, view_label, t1_path, t2_path, b0_path, result):
    """A volume was dropped in: show it if the selector already points at it,
    so a file can be sanity-checked before spending an hour on charm."""
    if not path or service.is_input_view(view_label) != slot:
        return gr.update()
    return _render(view_label, t1_path, t2_path, b0_path, result)


def _on_t1_change(path, view_label, t1_path, t2_path, b0_path, result):
    """The T1 also proposes the subject id its name carries."""
    canvas = _on_upload(path, "t1", view_label, t1_path, t2_path, b0_path, result)
    subject = service.default_subject_id(path) if path else gr.update()
    return canvas, subject


# ============================================================================
# The run
# ============================================================================
def _run(t1_file, t2_file, subject_id, aligned, b0_file,
         request: gr.Request = None):
    """Yields (console, downloads box, duration, links, canvas, view, state)."""
    hidden = gr.update(visible=False)
    idle = (hidden, "", "", gr.update(), gr.update(), gr.update())

    status = service.simnibs_status()
    if not status.available:
        yield (f"This function is not supported on this server: "
               f"{status.message}", *idle)
        return
    missing = []
    if not t1_file:
        missing.append("T1")
    if not aligned and not b0_file:
        missing.append("DTI b0 (needed to register the T1/T2 into its space)")
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
                    t1_path=t1_file,
                    t2_path=t2_file,
                    subject_id=subject_id,
                    aligned_to_b0=bool(aligned),
                    b0_path=b0_file,
                    run=run,
                    report=transcript.report,
                )

    for _text, result, error in common.stream_job(target, transcript,
                                                  run=run, request=request):
        # The console gets the WHOLE transcript, not a tail: it scrolls, and a
        # user chasing a charm error wants to scroll back to where it began.
        console = transcript.text()

        if error is not None:
            logger.exception("mesh_head job %s failed", run.job_id,
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
               f"&nbsp;·&nbsp; {'registered to b0' if result.registered else 'no registration'} "
               f"&nbsp;·&nbsp; job `{run.job_id}`",
               _download_links(result),
               _render(service.RESULT_VIEW, t1_file, t2_file, b0_file, result),
               gr.update(value=service.RESULT_VIEW),  # jump to what they came for
               result)


# ============================================================================
# Layout
# ============================================================================
def build_page() -> None:
    """Lay out this task's page. Called by app/web/main.py inside a route."""
    common.page_header("T1/T2 → head mesh (charm)", INTRO)
    gr.HTML(PAGE_CSS)

    status = service.simnibs_status()
    result_state = gr.State(None)

    with gr.Row():
        # ----------------------------------------------------- controls --
        with gr.Column(scale=2):
            common.section_label("Input")
            if not status.available:
                gr.Markdown(
                    f"⚠ **This function is not supported on this server.** "
                    f"{status.message} — set `SIMNIBS_ROOT` in `settings.py` "
                    f"to a SimNIBS 4 install and restart the app.",
                    elem_classes="mesh-head-unsupported")
            gr.Markdown("### Inputs")
            gr.Markdown(
                "<b>T1</b>: required; <b>T2</b>: optional, improves "
                "the skull/CSF boundary.",
                elem_classes="mesh-head-note")

            with gr.Row(equal_height=True):
                t1_file = gr.File(label="T1", file_types=NIFTI_TYPES,
                                  type="filepath", height=UPLOAD_BOX_HEIGHT,
                                  elem_classes="mesh-head-upload")
                t2_file = gr.File(label="T2 (optional)", file_types=NIFTI_TYPES,
                                  type="filepath", height=UPLOAD_BOX_HEIGHT,
                                  elem_classes="mesh-head-upload")
                # Only asked for when it is needed, and it IS needed then:
                # without a b0 there is nothing to register the T1 to.
                b0_file = gr.File(label="DTI b0", file_types=NIFTI_TYPES,
                                  type="filepath", height=UPLOAD_BOX_HEIGHT,
                                  visible=False, elem_classes="mesh-head-upload")

            # The declaration, the id and the button share one boxed row.
            with gr.Group(), gr.Row(equal_height=True):
                # Short labels on purpose: this row is equal_height, so a label
                # that wraps stretches the Run button with it.
                subject_id = gr.Textbox(
                    label="Subject id", scale=1, min_width=110,
                    placeholder="from T1 name")
                aligned = gr.Checkbox(
                    label="T1/T2 already in DTI b0 space",
                    value=True, scale=3)
                run_btn = gr.Button("Run", variant="primary", scale=1,
                                    min_width=150,
                                    interactive=status.available)

            with gr.Group(visible=False) as downloads_box:
                common.section_label("Output")
                duration = gr.Markdown("")
                links = gr.Markdown("", elem_classes="mesh-head-guide")

            console = gr.Textbox(
                label="Console", lines=CONSOLE_LINES, max_lines=CONSOLE_LINES,
                interactive=False, autoscroll=True,
                elem_classes="mesh-head-console",
                value=("Idle. Upload a T1 (and a T2), then press Run."
                       if status.available else
                       f"Not supported on this server: {status.message}"))

        # ------------------------------------------------------ display --
        with gr.Column(scale=3):
            view = gr.Dropdown(choices=service.view_labels(),
                               value=service.DEFAULT_VIEW, label="Show")
            canvas = gr.Image(label="Mosaic", type="pil", height=CANVAS_HEIGHT)

    common.page_footer()

    # -------------------------------------------------------- wiring --
    preview_inputs = [view, t1_file, t2_file, b0_file, result_state]
    t1_file.change(_on_t1_change, inputs=[t1_file, *preview_inputs],
                   outputs=[canvas, subject_id])
    t2_file.change(_on_upload, inputs=[t2_file, gr.State("t2"), *preview_inputs],
                   outputs=canvas)
    b0_file.change(_on_upload, inputs=[b0_file, gr.State("b0"), *preview_inputs],
                   outputs=canvas)
    aligned.change(lambda ok: gr.update(visible=not ok),
                   inputs=aligned, outputs=b0_file)
    view.change(_render, inputs=preview_inputs, outputs=canvas)

    run_btn.click(
        _run,
        inputs=[t1_file, t2_file, subject_id, aligned, b0_file],
        outputs=[console, downloads_box, duration, links, canvas, view,
                 result_state],
    )

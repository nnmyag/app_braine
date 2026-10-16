# ============================================================================
# tasks/isocond/web.py — the Gradio page for T1/T2 -> conductivity map
# ----------------------------------------------------------------------------
# Widgets and callbacks ONLY; the pipeline lives in service.py, and below that
# in vendor/isocond/pipeline/session.py.
# ============================================================================
import logging
from pathlib import Path

import gradio as gr

from app import jobs
from app.web import common
from app.web import slice_viewer as sv
from tasks.isocond import service
from tasks.isocond import settings as task_settings

logger = logging.getLogger(__name__)

INTRO = ("Upload T1 & T2 (.nii.gz), pick a model, and Run. They need not be aligned")

NIFTI_TYPES = [".nii", ".gz"]

# Gradio serves any file under `allowed_paths` from this route; run_web.py
# allows data/runs, which is where a job writes.
FILE_ROUTE = "/gradio_api/file="

# Height of the two drop zones. Full-height file boxes pushed everything else
# below the fold in the original, which is what this number is for.
UPLOAD_BOX_HEIGHT = 100

CANVAS_HEIGHT = 620

# The console. A plain Textbox with a fixed line count scrolls internally and
# stays selectable, so a user can copy a traceback out of it. Fixed rather than
# auto-growing: a long run must not stretch the column. The CSS below adds the
# browser's native resize grip, so the height is the reader's choice -- drag the
# bottom-right corner. CONSOLE_LINES is only where it starts.
#
# Sized to leave the downloads panel above it on screen without scrolling.
# Height is rows x font size, so the two shrink together: 15 rows at 10px is
# ~0.83 x 0.91 = 0.76 of the old 18 rows at 11px, i.e. about a quarter shorter.
CONSOLE_LINES = 15

CONSOLE_CSS = """
<style>
.isocond-console textarea {
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


# ============================================================================
# Download links
# ============================================================================
def _download_links(result: service.JobResult) -> str:
    """Markdown hyperlinks for what a run produced, the result first.

    Link text drops the "<case>_<model>_" prefix every row repeats — the full
    names survive in the downloaded files and in the archive. Review mosaics
    are not listed individually: they are the images already on the canvas, and
    they ship inside the archive.
    """
    if not result or not result.written_paths:
        return ""

    def link(path: Path) -> str:
        return f"[{result.short_name(path)}]({FILE_ROUTE}{Path(path).resolve()})"

    final, intermediate = [], []
    for key, path in sorted(result.written_paths.items()):
        path = Path(path)
        if not path.is_file():
            continue
        if key.endswith(service._PREVIEW_SUFFIX) or key == "segmentation_legend_path":
            continue                        # mosaics and the legend live in the zip
        # Grouped by the ROLE the runner recorded, not by filename. Note the
        # tissue table's key is deliberately free of "conductivity" -- it
        # documents the segmentation, it is not a result.
        (final if "conductivity" in key else intermediate).append(link(path))

    blocks = []
    if result.zip_path and Path(result.zip_path).is_file():
        blocks.append(f"**[⬇ Download all, including review images]"
                      f"({FILE_ROUTE}{Path(result.zip_path).resolve()})**")
    for title, items in (("Result", final), ("Intermediates", intermediate)):
        if items:
            blocks.append(f"**{title}**  \n" + " · ".join(items))
    return "\n\n".join(blocks) or "_(this run wrote no downloadable volume)_"


# ============================================================================
# Canvas
# ============================================================================
def _render(view_label, t1_path, t2_path, result, k=None):
    """The one rule for what the display shows, as updates for
    (mosaic image, slice viewer, slice slider).

    The slice viewer (hover values) and the mosaics are two widgets; one is
    visible at a time. The viewer keeps slice `k` when it is in range, else
    starts at service.viewer_default_slice() (mid white matter).
    """
    if view_label == service.VIEWER_VIEW:
        depth = service.viewer_depth(result)
        if depth:
            k = (int(k) if k is not None and 0 <= int(k) < depth
                 else service.viewer_default_slice(result))
            slider = gr.update(visible=depth > 1, minimum=0, maximum=max(depth - 1, 1),
                               value=k)
        else:
            k, slider = 0, gr.update(visible=False)
        try:
            html = service.render_viewer(result, k)
        except Exception:
            logger.exception("Could not render the slice viewer")
            html = sv.placeholder_html("The per-slice check failed (see the server log).")
        return gr.update(visible=False), gr.update(value=html, visible=True), slider
    hide = gr.update(visible=False)
    return gr.update(value=_render_mosaic(view_label, t1_path, t2_path, result),
                     visible=True), hide, hide


def _render_mosaic(view_label, t1_path, t2_path, result):
    """What the mosaic image shows.

    An input view is drawn from the uploaded FILE whenever one is selected,
    even after a run — otherwise replacing the T1 and looking at "T1 loaded"
    would still show the previous run's volume. Every other view comes from the
    result, which is the only place it exists.
    """
    slot = service.is_input_view(view_label)
    try:
        if slot is not None:
            path = t1_path if slot == "t1" else t2_path
            if path:
                return service.render_input_preview(path, slot.upper())
        return service.render_result_view(view_label, result)
    except Exception:
        logger.exception("Could not render %s", view_label)
        return None


def _on_upload(path, slot, view_label, t1_path, t2_path, result):
    """A volume was dropped in: show it if the selector is already pointing at
    it, so a file can be sanity-checked before spending minutes on inference."""
    if not path or service.is_input_view(view_label) != slot:
        return gr.update(), gr.update(), gr.update()   # not the view on screen
    return _render(view_label, t1_path, t2_path, result)


def _on_slice(k, result):
    """The slider moved: redraw the viewer only."""
    try:
        return service.render_viewer(result, k)
    except Exception:
        logger.exception("Could not render slice %s", k)
        return gr.update()


# ============================================================================
# The run
# ============================================================================
def _model_choices(uploads):
    """The model list: the folder's models, this session's uploads, and the
    upload option last."""
    return service.model_labels() + list(uploads or []) + [
        (service.UPLOAD_LABEL, service.UPLOAD_OPTION)]


def _refresh_models(current, uploads):
    """The model list as it is on disk now, keeping the selection if it is
    still there."""
    choices = _model_choices(uploads)
    keys = [key for _label, key in choices]
    return gr.update(choices=choices,
                     value=current if current in keys else service.default_model_key())


def _on_model(model_key):
    """Show the upload box only while the upload option is selected."""
    return gr.update(visible=model_key == service.UPLOAD_OPTION, value=None)


def _on_model_upload(paths, uploads):
    """Register an uploaded pair and select it. Yields (model, box, uploads)."""
    if not paths:
        return gr.update(), gr.update(), uploads
    try:
        label, key = service.register_upload(paths)
    except ValueError as exc:
        gr.Warning(str(exc))
        return gr.update(), gr.update(), uploads
    uploads = [u for u in (uploads or []) if u[1] != key] + [(label, key)]
    return (gr.update(choices=_model_choices(uploads), value=key),
            gr.update(visible=False, value=None), uploads)


def _run(t1_file, t2_file, model_key, save_intermediate, view_label,
         request: gr.Request = None):
    """Yields (console, downloads box, duration, links, mosaic, viewer, slider,
    view, state)."""
    hidden = gr.update(visible=False)
    keep = (gr.update(),) * 5               # mosaic, viewer, slider, view, state

    missing = [n for n, v in (("T1", t1_file), ("T2", t2_file)) if not v]
    if model_key == service.UPLOAD_OPTION:
        missing.append("the model files (or pick a listed model)")
    if missing:
        yield (f"Missing input: {', '.join(missing)}.",
               hidden, "", "", *keep)
        return

    jobs.prune_old_runs("isocond")
    run = jobs.new_run("isocond")
    transcript = jobs.Transcript(log_file=run.log_file)

    def target():
        with jobs.capture_logs(transcript):
            with jobs.QUEUE.slot(report=transcript.report, run=run):
                return service.run_job(
                    t1_path=t1_file,
                    t2_path=t2_file,
                    model_key=model_key,
                    save_intermediate=bool(save_intermediate),
                    # No case-name box: the snapshot derives the output prefix
                    # from the T1 filename, which is what a user recognises.
                    run=run,
                    report=transcript.report,
                )

    for _text, result, error in common.stream_job(target, transcript,
                                                  run=run, request=request):
        # The console gets the WHOLE transcript, not a tail: it scrolls, and a
        # user chasing an error wants to be able to scroll back to the start.
        console = transcript.text()

        if error is not None:
            logger.exception("isocond job %s failed", run.job_id, exc_info=error)
            transcript.report(f"FAILED: {type(error).__name__}: {error}")
            transcript.report(f"Full transcript: {run.log_file}")
            yield (transcript.text(), hidden, "", "", *keep)
            return

        if result is None:
            yield (console, hidden, "", "", *keep)
            continue

        # The original ended its log with the run report -- output shape and
        # value range -- which is the quickest way to spot a map that came out
        # empty or in the wrong units. Worth keeping in the console.
        for line in result.summary.splitlines():
            transcript.report(line)

        yield (transcript.text(),
               gr.update(visible=True),
               f"**Completed in {result.elapsed_s:.0f} s** "
               f"&nbsp;·&nbsp; {result.model_name} "
               f"&nbsp;·&nbsp; job `{run.job_id}`",
               _download_links(result),
               # Jump to what they came for: the slice viewer, at the middle
               # of the white-matter slices (the middle slice without a
               # segmentation); the mosaics stay a click away, for exporting.
               *_render(service.VIEWER_VIEW, t1_file, t2_file, result),
               gr.update(value=service.VIEWER_VIEW),
               result)


# ============================================================================
# Layout
# ============================================================================
def build_page() -> None:
    """Lay out this task's page. Called by app/web/main.py inside a route."""
    common.page_header("T1/T2 → conductivity map", INTRO)
    gr.HTML(CONSOLE_CSS)
    gr.HTML(sv.CSS)

    result_state = gr.State(None)
    uploads_state = gr.State([])          # [(label, key)] uploaded this session

    with gr.Row():
        # ----------------------------------------------------- controls --
        with gr.Column(scale=2):
            common.section_label("Input")
            # Everything needed to start a run on one line: two compact drop
            # zones and the model. Full-height file boxes pushed the rest of
            # the column below the fold in the original.
            with gr.Row(equal_height=True):
                t1_file = gr.File(label="T1", file_types=NIFTI_TYPES,
                                  type="filepath", height=UPLOAD_BOX_HEIGHT)
                t2_file = gr.File(label="T2", file_types=NIFTI_TYPES,
                                  type="filepath", height=UPLOAD_BOX_HEIGHT)
                model_key = gr.Dropdown(
                    label="Model",
                    choices=_model_choices([]),
                    value=service.default_model_key(),
                    info="CondNet-TART: S/m at the frequency in its name, "
                         "also segments. CondNet: normalized, 900 MHz. "
                         "Not comparable.")

            model_files = gr.File(
                label="condnettart_<freq>_stage0.pth + _stage1.pth (+ condnettart_<freq>.json)",
                file_count="multiple", file_types=[".pth", ".json"], type="filepath",
                height=UPLOAD_BOX_HEIGHT, visible=False)

            with gr.Row(equal_height=True):
                save_intermediate = gr.Checkbox(
                    label="Also save intermediates", container=False, scale=3,
                    value=task_settings.SAVE_INTERMEDIATE)
                run_btn = gr.Button("Run", variant="primary", scale=2)

            # Above the console, because once a run is done this is what the
            # user came for -- they should not have to scroll past a log to
            # reach it. Hidden until there is something to download.
            with gr.Group(visible=False) as downloads_box:
                common.section_label("Output")
                duration = gr.Markdown("")
                links = gr.Markdown("")

            console = gr.Textbox(
                label="Console", lines=CONSOLE_LINES, max_lines=CONSOLE_LINES,
                interactive=False, autoscroll=True,
                elem_classes="isocond-console",
                value="Idle. Upload a T1 and a T2, then press Run.")

        # ------------------------------------------------------ display --
        with gr.Column(scale=3):
            view = gr.Dropdown(choices=service.view_labels(),
                               value=service.DEFAULT_VIEW, label="Show")
            canvas = gr.Image(label="Mosaic", type="pil", height=CANVAS_HEIGHT)
            # The slice viewer: hover any panel to read every panel's value
            # at that point (app/web/slice_viewer.py).
            slice_k = gr.Slider(label="Slice", minimum=0, maximum=1, step=1, value=0,
                                visible=False)
            viewer = gr.HTML(visible=False, js_on_load=sv.HOVER_JS,
                             apply_default_css=False, padding=False)

    common.page_footer()

    # -------------------------------------------------------- wiring --
    preview_inputs = [view, t1_file, t2_file, result_state]
    display = [canvas, viewer, slice_k]
    t1_file.change(_on_upload,
                   inputs=[t1_file, gr.State("t1"), *preview_inputs],
                   outputs=display)
    t2_file.change(_on_upload,
                   inputs=[t2_file, gr.State("t2"), *preview_inputs],
                   outputs=display)
    # .input, not .change: only a USER's pick redraws here. A run sets the view
    # itself (and the slider with it); a .change would fire on that too and
    # redraw with the slider's old value, putting it back to slice 0.
    view.input(_render, inputs=[*preview_inputs, slice_k], outputs=display)
    # always_last: while the slider is dragged, only the newest slice is
    # drawn (each one is ~1 MB of images and values).
    slice_k.change(_on_slice, inputs=[slice_k, result_state], outputs=viewer,
                   show_progress="hidden", trigger_mode="always_last")
    # Rescan the model folder each time the list is opened, so a
    # CondNet-TART checkpoint pair dropped in while the app runs appears.
    model_key.focus(_refresh_models, inputs=[model_key, uploads_state],
                    outputs=model_key, show_progress="hidden")
    model_key.input(_on_model, inputs=model_key, outputs=model_files,
                    show_progress="hidden")
    model_files.upload(_on_model_upload, inputs=[model_files, uploads_state],
                       outputs=[model_key, model_files, uploads_state])

    run_btn.click(
        _run,
        inputs=[t1_file, t2_file, model_key, save_intermediate, view],
        outputs=[console, downloads_box, duration, links, *display, view,
                 result_state],
    )

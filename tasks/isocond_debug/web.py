# ============================================================================
# tasks/isocond_debug/web.py — check a conductivity result, slice by slice
# ----------------------------------------------------------------------------
# Widgets and callbacks ONLY; the work is in service.py. Unlisted
# (app/registry.py, visible=False): reachable at /isocond_debug.
#
#   left   T1, T2 (optional), Cond, Seg (optional) -- e.g. task 1's
#          conductivity_map.nii.gz / segmentation.nii.gz -- with, for a Seg,
#          its tissue CSV (names + expected conductivity), and Run: T2 is
#          registered onto T1, the model and frequency are read from the file
#          names. Nothing is estimated. Then the slice slider, the console,
#          and the architecture of the checkpoints the file name points to.
#   right  the slice in a 2 x 2 (or 4 x 1) grid you hover over to read every
#          panel's value at one point, and a per-tissue table for that slice.
# ============================================================================
import logging

import gradio as gr

from app import jobs
from app.web import common
from app.web import slice_viewer as sv
from tasks.isocond_debug import service

logger = logging.getLogger(__name__)

TITLE = "Isocond result check"
INTRO = ("Unlisted debug page. Upload T1 and a conductivity map (T2 and a "
         "segmentation are optional; a segmentation needs its tissue CSV: "
         "tissue_id, tissue_name and one conductivity column per frequency, "
         "e.g. 10khz), e.g. task 1's outputs, and press Run: T2 "
         "is registered onto T1, the model and frequency are read from the file "
         "names, nothing is estimated. Then hover over any image: every panel "
         "shows its value at that point.")

NIFTI_TYPES = [".nii", ".gz"]
CSV_TYPES = [".csv"]
UPLOAD_BOX_HEIGHT = 100
CONSOLE_LINES = 10

# How long a loaded case stays in memory after its last change (it holds the
# volumes, a few hundred MB), so an abandoned tab does not keep one for ever.
CASE_TTL_SECONDS = 2 * 3600

PAGE_CSS = """
<style>
/* Architecture diagram (service._diagram_svg): drawn at its own size, scaled
   down only when the column is narrower. */
.arch-diagram { margin: 4px 0 18px; }
.arch-diagram svg { max-width: 100%; height: auto; }
.arch-diagram text { font-family: ui-sans-serif, system-ui, sans-serif; font-size: 11px; }
.arch-title { font-weight: 600; font-size: 12px !important; fill: #111827; }
.arch-sub { fill: #6b7280; font-size: 10px !important; }
.arch-box { stroke: #3b82f6; stroke-width: 1; }
.arch-io { fill: #f3f4f6; stroke: #6b7280; stroke-dasharray: 3 2; }
.arch-name { font-weight: 600; fill: #111827; }
.arch-kind { fill: #374151; }
.arch-shape { fill: #1e3a8a; font-family: ui-monospace, Menlo, Consolas, monospace !important; }
.arch-flow { stroke: #6b7280; stroke-width: 1.2; }
.arch-head { fill: #6b7280; }
.arch-skip { fill: none; stroke: #f59e0b; stroke-width: 1.3; stroke-dasharray: 4 3; }
.arch-head-skip { fill: #f59e0b; }
.arch-stage { font-weight: 600; fill: #111827; }
.arch-band { fill: #f8fafc; stroke: #e5e7eb; }
.arch-skip-label { fill: #b45309; font-size: 10px !important; }
.arch-feat { fill: none; stroke: #0d9488; stroke-width: 1.3; }
.arch-head-feat { fill: #0d9488; }
.arch-feat-label { fill: #0f766e; font-size: 10px !important; font-weight: 600; }
.dbg-console textarea {
    font-family: ui-monospace, "DejaVu Sans Mono", Menlo, Consolas, monospace !important;
    font-size: 10px !important; line-height: 1.35 !important;
}
@media (prefers-color-scheme: dark) {
    .arch-title { fill: #f3f4f6; }
    .arch-sub, .arch-flow { fill: #9ca3af; stroke: #9ca3af; }
    .arch-head { fill: #9ca3af; }
    .arch-stage { fill: #f3f4f6; }
    .arch-band { fill: #111827; stroke: #374151; }
    .arch-skip-label { fill: #fbbf24; }
    .arch-feat-label { fill: #5eead4; }
}
</style>
"""


# ============================================================================
# Run: load, register, place
# ============================================================================
def _run(t1_file, t2_file, cond_file, seg_file, csv_file, request: gr.Request = None):
    """Yields (console, status, slider, canvas, case, model key)."""
    keep = gr.update()
    idle_slider = gr.update(interactive=False)
    missing = [n for n, f in (("T1", t1_file), ("Cond", cond_file)) if not f]
    if seg_file and not csv_file:
        missing.append("the tissue CSV (needed with a Seg)")
    if missing:
        yield (f"Missing: {' and '.join(missing)}.", f"**Missing:** {' and '.join(missing)}.",
               idle_slider, service.render_slice(None, 0), None, None)
        return

    transcript = jobs.Transcript()

    def target():
        with jobs.capture_logs(transcript):
            with jobs.QUEUE.slot(report=transcript.report):
                return service.prepare(t1_file, t2_file, cond_file, seg_file,
                                       csv_file, report=transcript.report)

    yield ("", "Loading" + (" and registering T2 (~15 s)…" if t2_file else "…"),
           idle_slider, keep, None, None)
    for _text, case, error in common.stream_job(target, transcript, request=request):
        if error is not None:
            logger.exception("isocond_debug: loading failed", exc_info=error)
            transcript.report(f"FAILED: {type(error).__name__}: {error}")
            yield (transcript.text(), f"**Failed:** {error}", idle_slider,
                   service.placeholder_html(f"Could not load the files: {error}"),
                   None, None)
            return
        if case is None:
            yield (transcript.text(), keep, keep, keep, keep, keep)
            continue
        k = case.default_slice()
        yield (transcript.text(),
               f"**{case.case_id}** · {case.info.describe()} · {case.depth} slices "
               f"on the {case.grid}",
               gr.update(minimum=0, maximum=max(case.depth - 1, 1), value=k,
                         interactive=case.depth > 1, label=f"Slice (0–{case.depth - 1})"),
               service.render_slice(case, k), case, case.info.model_key)


def _on_slice(case, k):
    try:
        return service.render_slice(case, k)
    except Exception as exc:                    # noqa: BLE001 - shown on the canvas
        logger.exception("isocond_debug: render failed")
        return service.placeholder_html(f"Could not draw this slice: {exc}")


# ============================================================================
# Checkpoint inspector
# ============================================================================
def _inspect_all(model_key):
    """(one diagram of both stages, then one module printout per checkpoint,
    in service.checkpoint_choices() order), for the model the Cond file name
    points to -- or the default before a Run."""
    roles = [role for _label, role in service.checkpoint_choices()]
    key = model_key or service.default_model_key()
    if key is None:
        return (service.placeholder_html("No CondNet-TART checkpoints in "
                                         "vendor/isocond/models."), *[""] * len(roles))
    try:
        r = service.inspect_architecture(key)
    except Exception as exc:                    # noqa: BLE001 - shown on the page
        logger.exception("isocond_debug: inspecting the checkpoints failed")
        return (service.placeholder_html(
            f"Could not load the checkpoints: {type(exc).__name__}: {exc}"),
            *[""] * len(roles))
    title = f'<div class="sv-note">Checkpoints: <b>{key}</b>' + (
        "" if model_key else " (no CondNet-TART frequency read from a file name yet: "
                             "the default)") + "</div>"
    return (title + r["diagram"], *[r["printouts"].get(role, "") for role in roles])


# ============================================================================
# Layout
# ============================================================================
def build_page() -> None:
    common.page_header(TITLE, INTRO)
    gr.HTML(PAGE_CSS)
    gr.HTML(sv.CSS)

    case_state = gr.State(None, time_to_live=CASE_TTL_SECONDS)
    model_key = gr.State(None)

    with gr.Row():
        # ----------------------------------------------------- controls --
        with gr.Column(scale=2):
            common.section_label("Input")
            with gr.Row(equal_height=True):
                t1_file = gr.File(label="T1", file_types=NIFTI_TYPES,
                                  type="filepath", height=UPLOAD_BOX_HEIGHT)
                t2_file = gr.File(label="T2 (optional)", file_types=NIFTI_TYPES,
                                  type="filepath", height=UPLOAD_BOX_HEIGHT)
            with gr.Row(equal_height=True):
                cond_file = gr.File(label="Cond", file_types=NIFTI_TYPES,
                                    type="filepath", height=UPLOAD_BOX_HEIGHT)
                seg_file = gr.File(label="Seg (optional)", file_types=NIFTI_TYPES,
                                   type="filepath", height=UPLOAD_BOX_HEIGHT)
                csv_file = gr.File(label="Tissue CSV (with Seg)", file_types=CSV_TYPES,
                                   type="filepath", height=UPLOAD_BOX_HEIGHT)
            run_btn = gr.Button("Run", variant="primary")
            status = gr.Markdown("Upload T1 and Cond, then press Run.")
            slice_slider = gr.Slider(0, 1, value=0, step=1, label="Slice",
                                     interactive=False)
            console = gr.Textbox(
                label="Console", lines=CONSOLE_LINES, max_lines=CONSOLE_LINES,
                interactive=False, autoscroll=True, value="Idle.",
                elem_classes="dbg-console")

            # The checkpoints of the model the Cond file name points to,
            # loaded when the panel is opened (and after each Run).
            with gr.Accordion("Network architecture", open=False) as arch_box:
                arch_diagram = gr.HTML(apply_default_css=False)
                printouts = []
                with gr.Tabs():
                    for label, _role in service.checkpoint_choices():
                        with gr.Tab(f"Printout · {label}"):
                            printouts.append(gr.Code(
                                language="python", lines=18, max_lines=30,
                                interactive=False, show_label=False))

        # ------------------------------------------------------ display --
        with gr.Column(scale=3):
            canvas = gr.HTML(service.render_slice(None, 0), js_on_load=sv.HOVER_JS,
                             apply_default_css=False, padding=False)

    common.page_footer()

    # -------------------------------------------------------- wiring --
    run_btn.click(_run, inputs=[t1_file, t2_file, cond_file, seg_file, csv_file],
                  outputs=[console, status, slice_slider, canvas, case_state, model_key]
                  ).then(_inspect_all, inputs=model_key, outputs=[arch_diagram, *printouts])
    slice_slider.change(_on_slice, inputs=[case_state, slice_slider], outputs=canvas,
                        trigger_mode="always_last", show_progress="hidden")
    arch_box.expand(_inspect_all, inputs=model_key, outputs=[arch_diagram, *printouts])

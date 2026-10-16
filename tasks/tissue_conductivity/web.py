# ============================================================================
# tasks/tissue_conductivity/web.py — conductivity per tissue at one frequency
# ----------------------------------------------------------------------------
# Widgets and callbacks ONLY; the work is in service.py. Unlisted
# (app/registry.py, visible=False): reachable at /tissue_conductivity.
# ============================================================================
import logging

import gradio as gr

from app.web import common
from app.web import slice_viewer as sv
from tasks.tissue_conductivity import service
from tasks.tissue_conductivity import settings as task_settings

logger = logging.getLogger(__name__)

TITLE = "Tissue conductivity at a frequency"
INTRO = ("Conductivity of the 15 head tissue classes at a chosen frequency, from "
         "the four-dispersion Cole-Cole model of Gabriel, Lau & Gabriel, Phys. Med. "
         "Biol. 41:2271–2293 (1996). Each class keeps its 10 kHz value from the "
         "anisotropic-conductivity task's default table and follows its Gabriel "
         "tissue's change from 10 kHz: σ(f) = σ<sub>table</sub>(10 kHz) + "
         "σ<sub>Gabriel</sub>(f) − σ<sub>Gabriel</sub>(10 kHz). Generate writes "
         "&lt;freq&gt;.yaml (frequency, normalization and σ per class) and tissues.csv "
         "(class names).")

HEADERS = ["id", "class", "σ at f (S/m)"]
# A compact table (app/web/slice_viewer's), a little larger than its 11 px.
TABLE_CSS = ("<style>.tc-table table.sv-table { font-size: 12.5px; }"
             ".tc-quick button { min-width: 0 !important; padding: 2px 6px !important; }"
             ".tc-table table.sv-table :is(th, td):nth-child(2) { text-align: left; }"
             "</style>")


def _generate(value, unit):
    """(status, warning, table HTML, files)."""
    try:
        f_hz = service.to_hz(value, unit)
        result = service.write_outputs(f_hz)
    except Exception as exc:                    # noqa: BLE001 - shown on the page
        if not isinstance(exc, ValueError):     # a bad frequency is not a fault
            logger.exception("tissue_conductivity failed")
        return (f"**Failed:** {exc}", gr.update(visible=False), gr.update(),
                gr.update(value=None))
    rows = [[r.class_id, r.name, (f"{r.sigma:.4g}", "tc-result")] for r in result.rows]
    table = sv.table_html(HEADERS, rows, f"<b>{result.tag}</b>")
    warning = result.warning
    return (f"**{result.tag}** ({result.f_hz:.6g} Hz) · files: `{result.tag}.yaml`, "
            f"`tissues.csv`",
            gr.update(value=warning or "", visible=bool(warning)),
            table,
            gr.update(value=[result.yaml_path, result.csv_path]))


def build_page() -> None:
    common.page_header(TITLE, INTRO)
    gr.HTML(sv.CSS + TABLE_CSS + "<style>.tc-result { font-weight: 600; }</style>")
    with gr.Row():
        with gr.Column(scale=2):
            common.section_label("Frequency")
            # One click = that frequency, generated at once.
            with gr.Row(elem_classes="tc-quick"):
                quick = [gr.Button(f"{v:g} {u}", size="sm", min_width=0)
                         for v, u in task_settings.QUICK_FREQUENCIES]
            # Frequency, unit and Generate on one row, across the column.
            with gr.Row(equal_height=True):
                value = gr.Number(value=task_settings.DEFAULT_VALUE, minimum=0,
                                  show_label=False, container=False, scale=2,
                                  min_width=70)
                unit = gr.Dropdown(choices=list(task_settings.UNITS),
                                   value=task_settings.DEFAULT_UNIT,
                                   show_label=False, container=False, scale=1,
                                   min_width=70)
                run_btn = gr.Button("Generate", variant="primary", scale=2,
                                    min_width=90)
            status = gr.Markdown(
                f"Range {service.frequency_tag(task_settings.MIN_HZ)} – "
                f"{service.frequency_tag(task_settings.MAX_HZ)}.")
            warning = gr.Markdown(visible=False)
            files = gr.File(label="Download", file_count="multiple", interactive=False)
        with gr.Column(scale=3):
            table = gr.HTML(sv.placeholder_html("Pick a frequency and press Generate."),
                            apply_default_css=False, elem_classes="tc-table")
    common.page_footer()
    outputs = [status, warning, table, files]
    run_btn.click(_generate, inputs=[value, unit], outputs=outputs)
    for btn, (v, u) in zip(quick, task_settings.QUICK_FREQUENCIES):
        btn.click(lambda v=v, u=u: (v, u), outputs=[value, unit]).then(
            _generate, inputs=[value, unit], outputs=outputs)

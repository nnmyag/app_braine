# ============================================================================
# tasks/condnettart_train/web.py — train CondNet-TART at a new frequency
# ----------------------------------------------------------------------------
# Widgets and callbacks ONLY; the work is the condnettart_training project's
# own scripts, run by service.train() while this page waits. Unlisted
# (app/registry.py, visible=False): reachable at /condnettart_train.
# ============================================================================
import logging

import gradio as gr

from app.web import common
from tasks.condnettart_train import service

logger = logging.getLogger(__name__)

TITLE = "CondNet-TART training at a frequency"
INTRO = ("Upload a conductivity table (&lt;freq&gt;.yaml from the Tissue Conductivity "
         "utility) and Start: CondNet-TART is trained on the default dataset in the "
         "condnettart_training project, and the best model files are offered here.")
CAUTION = ("⚠️ **A training takes about 40 minutes.** Keep this page open until it "
           "finishes: closing the page, or pressing Stop, ends the training.")


def _train(table, smoke):
    """Yields (status, log, files) while the training runs."""
    if not table:
        yield "**Upload a &lt;freq&gt;.yaml first.**", gr.update(), gr.update()
        return
    try:
        for prog in service.train(table, smoke=bool(smoke)):
            yield prog.summary(), prog.tail(), (prog.files or None)
    except (ValueError, FileNotFoundError) as exc:
        yield f"**Not started:** {exc}", gr.update(), gr.update()
    except Exception as exc:                     # noqa: BLE001 - shown on the page
        logger.exception("condnettart_train failed")
        yield f"**Failed:** {exc}", gr.update(), gr.update()


def build_page() -> None:
    common.page_header(TITLE, INTRO)
    with gr.Row():
        with gr.Column(scale=2):
            table = gr.File(label="<freq>.yaml", file_types=[".yaml", ".yml"],
                            type="filepath", height=100)
            gr.Markdown(CAUTION)
            with gr.Row(equal_height=True):
                smoke = gr.Checkbox(label="Quick test (2 batches)", value=False,
                                    container=False, scale=2)
                start_btn = gr.Button("Start", variant="primary", scale=2)
                stop_btn = gr.Button("Stop", variant="stop", scale=1)
            status = gr.Markdown("")
            files = gr.File(label="Model files", file_count="multiple", interactive=False)
        with gr.Column(scale=3):
            console = gr.Textbox(label="Log", lines=24, max_lines=24, interactive=False,
                                 autoscroll=True)
    common.page_footer()

    run = start_btn.click(_train, inputs=[table, smoke], outputs=[status, console, files],
                          concurrency_limit=1)
    stop_btn.click(lambda: "**Stopped.**", outputs=status, cancels=[run])

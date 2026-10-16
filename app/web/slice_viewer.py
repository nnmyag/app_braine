# ============================================================================
# app/web/slice_viewer.py — slices you can hover over to read values
# ----------------------------------------------------------------------------
# A grid of panels that all show the SAME slice on the SAME pixel grid. Hover
# over any one: every panel draws a crosshair at that pixel and labels it
# with its own value (a tissue name for a segmentation). Used by the isocond
# page's "Slice viewer"; the mechanics were first built and measured on the
# /isocond_debug page.
#
# Why each image is drawn 1:1 (or at an exact integer zoom) and is one PNG
# pixel per data pixel: a browser that scales an image by a non-integer
# factor drops or doubles pixel rows, so the pixel under the cursor can be a
# neighbour of the one whose value is shown (measured: ~20% of positions at
# 250 px for a 256 px slice, none at 1:1 or 2:1).
#
# Usage: a gr.HTML(..., js_on_load=HOVER_JS, apply_default_css=False) whose
# value is canvas_html([...panel_html(...)...]); put CSS on the page once.
# ============================================================================
import base64
import html
import io
import json
import uuid
from typing import Dict, List, Optional

import numpy as np
from matplotlib.colors import Normalize
from PIL import Image

CSS = """
<style>
/* 2 x 2, or 4 x 1 when the panel is wide enough for four 256 px slices
   (4 x 256 + 3 x 14 gap = 1066 px). A container query, so it follows the
   width of the page's display column, not of the window. A "wide" panel
   (conductivity with no segmentation) spans two cells at 2x in 2 x 2, and
   takes one cell at 1x in 4 x 1 (--w: its 1x width). */
.sv-canvas { container-type: inline-size; }
.sv-grid { display: grid; grid-template-columns: repeat(2, max-content);
           gap: 10px 14px; align-items: start; }
.sv-wide { grid-column: span 2; }
@container (min-width: 1066px) {
    .sv-grid { grid-template-columns: repeat(4, max-content); }
    .sv-wide { grid-column: auto; }
    .sv-wide img.sv-slice { width: var(--w); height: auto; }
    .sv-wide .sv-bar { width: var(--w) !important; }
}
.sv-imgwrap { position: relative; display: inline-block; line-height: 0; }
img.sv-slice {
    image-rendering: pixelated; cursor: crosshair; background: #000;
    /* An outline, not a border: it takes no room, so the image's box is
       exactly its pixels and the cursor -> pixel arithmetic stays exact. */
    outline: 1px solid #d1d5db;
}
/* The panel's name, written onto the image. */
.sv-tag { position: absolute; top: 5px; left: 7px; z-index: 1;
          pointer-events: none; color: #fff; line-height: 1;
          font: 600 13px ui-sans-serif, system-ui, sans-serif;
          text-shadow: 0 0 3px #000, 0 0 2px #000; }
.sv-cross { position: absolute; pointer-events: none; display: none;
            background: rgba(255, 255, 255, .55); }
.sv-h { left: 0; right: 0; height: 1px; }
.sv-v { top: 0; bottom: 0; width: 1px; }
.sv-linked .sv-cross { background: #ff2d55; }
.sv-tip { position: absolute; pointer-events: none; display: none; z-index: 2;
          white-space: nowrap; padding: 2px 6px; border-radius: 4px;
          font: 600 12px/1.35 ui-monospace, Menlo, Consolas, monospace;
          color: #fff; background: rgba(17, 24, 39, .85); }
.sv-linked .sv-tip { background: rgba(255, 45, 85, .92); }
.sv-bar { font-size: 11px; color: #4b5563; margin-top: 4px; }
.sv-bar-grad { height: 8px; border: 1px solid #9ca3af; }
.sv-bar-ticks { display: flex; justify-content: space-between; }
.sv-note { font-size: 11px; color: #6b7280; margin: 6px 0 2px; }
.sv-empty { padding: 48px 24px; color: #6b7280; text-align: center;
            border: 1px dashed #d1d5db; border-radius: 8px; }
table.sv-table { border-collapse: collapse; font-size: 11px; margin-top: 8px; }
table.sv-table th, table.sv-table td { padding: 1px 10px 1px 0; text-align: right; }
table.sv-table th:first-child, table.sv-table td:first-child { text-align: left; }
table.sv-table th { border-bottom: 1px solid #d1d5db; font-weight: 600; }
.sv-bad { color: #dc2626; font-weight: 600; }
.sv-low { color: #2563eb; font-weight: 600; }
@media (prefers-color-scheme: dark) {
    .sv-bar { color: #d1d5db; }
    .sv-note, .sv-empty { color: #9ca3af; }
    .sv-bad { color: #f87171; }
    .sv-low { color: #60a5fa; }
}
</style>
"""

# Runs once, when the canvas mounts. The canvas's content is replaced on every
# slider step; Gradio PATCHES the existing elements rather than replacing
# them, so a value cache on an element would outlive its render -- each render
# stamps its panels with a fresh data-rev and the cache is keyed on it.
HOVER_JS = r"""
const decode = (b64, dtype) => {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  if (dtype === 'f4') return new Float32Array(bytes.buffer);
  return bytes;
};
const channel = (panel) => {
  if (!panel._ch || panel._chRev !== panel.dataset.rev) {
    panel._ch = JSON.parse(panel.dataset.channel);
    panel._ch.arr = decode(panel._ch.b64, panel._ch.dtype);
    if (panel._ch.b64b) panel._ch.arr2 = decode(panel._ch.b64b, 'f4');
    panel._chRev = panel.dataset.rev;
  }
  return panel._ch;
};
const valueText = (panel, row, col) => {
  const c = channel(panel), i = row * +panel.dataset.w + col, raw = c.arr[i];
  if (c.arr2) {                     // two values, e.g. the two images of an RGB overlay
    const f = (v) => Number.isFinite(v) ? v.toFixed(c.digits) : '—';
    return c.labels[0] + ' ' + f(raw) + ' · ' + c.labels[1] + ' ' + f(c.arr2[i]);
  }
  if (c.names) return c.names[raw] || 'class ' + raw;
  if (!Number.isFinite(raw)) return '—';
  const s = raw.toFixed(c.digits);
  return c.unit ? s + ' ' + c.unit : s;
};
const place = (panel, row, col, linked) => {
  const w = +panel.dataset.w, h = +panel.dataset.h;
  const hl = panel.querySelector('.sv-h'), vl = panel.querySelector('.sv-v');
  const tip = panel.querySelector('.sv-tip');
  panel.classList.toggle('sv-linked', linked);
  const x = (col + 0.5) / w * 100, y = (row + 0.5) / h * 100;
  hl.style.top = y + '%';
  vl.style.left = x + '%';
  tip.textContent = valueText(panel, row, col);
  tip.style.left = x + '%';
  tip.style.top = y + '%';
  tip.style.transform = 'translate(' + (x > 55 ? 'calc(-100% - 8px)' : '8px') +
                        ', ' + (y > 80 ? 'calc(-100% - 8px)' : '8px') + ')';
  hl.style.display = vl.style.display = tip.style.display = 'block';
};
const reset = () => {
  element.querySelectorAll('.sv-cross, .sv-tip').forEach((x) => { x.style.display = 'none'; });
};
element.addEventListener('mousemove', (e) => {
  const img = e.target && e.target.closest ? e.target.closest('img.sv-slice') : null;
  if (!img) { reset(); return; }
  const panel = img.closest('.sv-panel'), holder = img.closest('.sv-canvas');
  const w = +panel.dataset.w, h = +panel.dataset.h;
  // The image pixel DRAWN under the cursor: snap the box to device pixels as
  // the browser does, and take the centre of the device pixel under the
  // cursor (getBoundingClientRect alone put values one row off).
  const dpr = window.devicePixelRatio || 1, snap = (v) => Math.round(v * dpr) / dpr;
  const r = img.getBoundingClientRect();
  const left = snap(r.left), top = snap(r.top);
  const width = snap(r.right) - left, height = snap(r.bottom) - top;
  const cx = (Math.floor(e.clientX * dpr) + 0.5) / dpr - left;
  const cy = (Math.floor(e.clientY * dpr) + 0.5) / dpr - top;
  const col = Math.floor(cx / width * w), row = Math.floor(cy / height * h);
  if (col < 0 || row < 0 || col >= w || row >= h) { reset(); return; }
  // Every panel shares the grid: the same pixel in all of them.
  holder.querySelectorAll('.sv-panel').forEach((p) => place(p, row, col, p !== panel));
});
element.addEventListener('mouseleave', reset);
"""


def colorize(values: np.ndarray, norm: Normalize, cmap) -> np.ndarray:
    """(H, W) -> (H, W, 4) uint8; NaN pixels transparent (the page's black
    background shows through)."""
    v = np.asarray(values, dtype=np.float32)
    rgba = cmap(norm(np.nan_to_num(v, nan=0.0)), bytes=True)
    rgba[..., 3] = np.where(np.isnan(v), 0, 255)
    return rgba


def _png_data_uri(rgba: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def channel(values: np.ndarray, digits: int = 3, unit: str = "",
            names: Optional[Dict[int, str]] = None, second: Optional[np.ndarray] = None,
            labels=("", "")) -> dict:
    """A panel's values for its cursor label: float32 (NaN reads as "—"), or
    uint8 class ids read out as `names`. With `second`, two float values per
    pixel, shown as "<labels[0]> a · <labels[1]> b" (an RGB overlay of two
    images)."""
    ch = {"digits": digits, "unit": unit}
    if second is not None:
        ch["b64b"] = base64.b64encode(np.ascontiguousarray(second, dtype="<f4").tobytes()).decode()
        ch["labels"] = list(labels)
    if names is not None:
        data = np.ascontiguousarray(np.rint(values), dtype=np.uint8)   # 7.9999 -> 8
        ch["dtype"] = "u1"
        ch["names"] = {str(k): v for k, v in names.items()}
    else:
        data, ch["dtype"] = np.ascontiguousarray(values, dtype="<f4"), "f4"
    ch["b64"] = base64.b64encode(data.tobytes()).decode()
    return ch


def bar_html(norm: Normalize, cmap, label: str, width: int,
             top_label: Optional[str] = None) -> str:
    """A slim horizontal colour bar under a panel."""
    stops = ", ".join("rgb({},{},{}) {:.0f}%".format(
        *(int(255 * c) for c in cmap(t)[:3]), 100 * t) for t in np.linspace(0, 1, 13))
    # Decimals to suit the range: 0 / 339 / 679, not 0.007493 / 339.3 / 678.7.
    span = float(norm.vmax - norm.vmin) or 1.0
    decimals = max(0, 2 - int(np.floor(np.log10(span))))
    ticks = [f"{t:.{decimals}f}" for t in np.linspace(norm.vmin, norm.vmax, 3)]
    if top_label:
        ticks[-1] = top_label
    return (f'<div class="sv-bar" style="width:{width}px">'
            f'<div class="sv-bar-grad" style="background:linear-gradient(to right, {stops})"></div>'
            f'<div class="sv-bar-ticks">{"".join(f"<span>{t}</span>" for t in ticks)}</div>'
            f'<div>{html.escape(label)}</div></div>')


def panel_html(values: np.ndarray, norm: Normalize, cmap, value_channel: dict,
               tag: str, zoom: int = 1, footer: str = "", wide: bool = False,
               rgb: Optional[np.ndarray] = None) -> str:
    """One image (drawn at an integer `zoom`), its name written onto it, its
    values attached for the cursor label. `rgb`: an (H, W, 3|4) uint8 image
    drawn instead of `values` coloured by `cmap` (values then only give the
    grid size)."""
    h, w = values.shape
    attrs = (f'data-w="{w}" data-h="{h}" data-rev="{uuid.uuid4().hex[:12]}" '
             f'data-channel="{html.escape(json.dumps(value_channel), quote=True)}"')
    cls = "sv-panel sv-wide" if wide else "sv-panel"
    return (f'<div class="{cls}" style="--w:{w}px" {attrs}><div class="sv-imgwrap">'
            f'<img class="sv-slice" src="{_png_data_uri(rgb if rgb is not None else colorize(values, norm, cmap))}" '
            f'width="{w * zoom}" height="{h * zoom}" draggable="false" alt="{html.escape(tag)}"/>'
            f'<div class="sv-tag">{html.escape(tag)}</div>'
            f'<div class="sv-cross sv-h"></div><div class="sv-cross sv-v"></div>'
            f'<div class="sv-tip"></div></div>{footer}</div>')


def table_html(headers: List[str], rows: List[list], caption: str = "") -> str:
    """A compact table. A cell is text, or (text, css class) -- e.g. "sv-bad"
    for a value drawn red. `caption` is HTML."""
    def td(c):
        if isinstance(c, tuple):
            return f'<td class="{c[1]}">{html.escape(str(c[0]))}</td>'
        return f"<td>{html.escape(str(c))}</td>"
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = "".join("<tr>" + "".join(td(c) for c in r) + "</tr>" for r in rows)
    cap = f'<div class="sv-note">{caption}</div>' if caption else ""
    return f'{cap}<table class="sv-table"><tr>{head}</tr>{body}</table>'


def canvas_html(panels: List[str], below: str = "", note: str = "") -> str:
    note_html = f'<div class="sv-note">{note}</div>' if note else ""
    return (f'<div class="sv-canvas"><div class="sv-grid">{"".join(panels)}</div></div>'
            f'{note_html}{below}')


def placeholder_html(message: str) -> str:
    return f'<div class="sv-empty">{message}</div>'

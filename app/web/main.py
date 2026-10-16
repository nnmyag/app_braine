# ============================================================================
# app/web/main.py — the shell: home menu + one mounted page per task
# ----------------------------------------------------------------------------
# Everything on this page is generated from app/registry.py.
#
# Gradio's multipage routing (demo.route) gives each task its own URL and nav 
# entry inside that single process.
# ============================================================================
import logging
from pathlib import Path
from typing import Optional

import gradio as gr

import settings
from app import registry
from app.web import common

logger = logging.getLogger(__name__)

# If a page is broken, log the errors for showing it
BUILD_ERRORS: dict = {}

# Gradio serves any file under `allowed_paths` from this route; run_web.py
# allows tasks/*/assets, which is where a card's demo image lives.
FILE_ROUTE = "/gradio_api/file="

# ============================================================================
# Menu card styling
# ----------------------------------------------------------------------------
# Cards are built as one raw HTML block, not a loop of gr.Markdown widgets --
# a Markdown component can't give a whole card a hover state or lay text and
# an image out side by side, and gr.HTML renders arbitrary markup untouched.
#
# Palette: a quiet, desaturated "dusty slate-blue" -- an off-white/graphite
# card on a neutral ground, shifting to a soft muted blue on hover rather than
# a saturated accent. @media (prefers-color-scheme) mirrors the browser's own
# light/dark setting; this app has no separate theme toggle to key off.
# ============================================================================
MENU_CSS = """
<style>
.app-title h1, .app-title p { text-align: center; font-family: Arial;}
.task-menu {
    display: flex; flex-direction: column; gap: 14px;
    max-width: 700px; margin: 18px auto 28px;
}
.task-card {
    display: flex; align-items: center; gap: 22px;
    padding: 18px 24px;
    border-radius: 14px;
    border: 1px solid #dde3ea;
    background: #fafbfc;
    text-decoration: none !important;
    transition: background-color .18s ease, border-color .18s ease,
                box-shadow .18s ease;
}
.task-card:hover {
    background: #e7edf6;
    border-color: #b9c9de;
    box-shadow: 0 2px 10px rgba(46, 74, 112, 0.10);
}
.task-card-text { flex: 1 1 auto; min-width: 0; }
.task-card-title {
    font-size: 1.15em; font-weight: 600; margin: 0 0 6px;
    color: #22303f;
    transition: font-weight .15s ease, font-style .15s ease, color .15s ease;
}
.task-card:hover .task-card-title {
    font-weight: 700; font-style: italic;
    color: #2e5a8f;
}
.task-card-blurb { font-size: 0.92em; line-height: 1.55; color: #6b7280; }
/* Every icon in one fixed 4:3 box, a little under the three lines of text
   beside it, so the cards line up whatever each PNG's own size is. The
   icons' aspect ratios sit between 1.15 and 1.35, so `cover` crops at most
   a sliver from the squarer ones rather than letterboxing them. The icons
   carry their own backgrounds, so the frame is a hairline border only. */
.task-card-image {
    flex: 0 0 auto; width: 120px; height: 100px;
    box-sizing: border-box; padding: 8px;
}
.task-card-image img {
    width: 100%; height: 100%;
    object-fit: cover; object-position: center; display: block;
}

/* Utilities: small tools, as compact tiles in a grid under the task cards. */
.utility-section { max-width: 700px; margin: 0 auto 28px; }
/* "Tasks" / "Utilities" over the two sections. */
.task-menu .menu-heading { margin-bottom: -6px; }
.menu-heading {
    font-size: 0.8em; font-weight: 600; letter-spacing: .06em;
    text-transform: uppercase; color: #6b7280; margin: 0 0 8px 2px;
}
.utility-grid {
    display: grid; gap: 10px;
    grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
}
.utility-tile {
    display: block; padding: 10px 14px; border-radius: 10px;
    border: 1px solid #dde3ea; background: #fafbfc;
    text-decoration: none !important;
    transition: background-color .18s ease, border-color .18s ease;
}
.utility-tile:hover { background: #e7edf6; border-color: #b9c9de; }
.utility-title { font-size: 0.95em; font-weight: 600; color: #22303f; margin-bottom: 3px; }
.utility-tile:hover .utility-title { color: #2e5a8f; }
.utility-blurb { font-size: 0.82em; line-height: 1.45; color: #6b7280; }

@media (prefers-color-scheme: dark) {
    .utility-tile { background: #1c2128; border-color: #333a44; }
    .utility-tile:hover { background: #223349; border-color: #3d5674; }
    .utility-title { color: #e6e9ef; }
    .utility-tile:hover .utility-title { color: #9dbbe6; }
    .utility-blurb, .menu-heading { color: #9ca3af; }
    .task-card { background: #1c2128; border-color: #333a44; }
    .task-card:hover {
        background: #223349; border-color: #3d5674;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.35);
    }
    .task-card-title { color: #e6e9ef; }
    .task-card:hover .task-card-title { color: #9dbbe6; }
    .task-card-blurb { color: #9ca3af; }
    .task-card-image { border-color: #333a44; }
}
</style>
"""


def _card_image_src(task: registry.TaskSpec) -> Optional[str]:
    """URL for a card's optional thumbnail, or None if it has none / is missing.

    A bare filename is resolved against tasks/<slug>/assets/ and served through
    the same allowed_paths route every task's own downloads use; a full URL is
    passed through as-is. Checked against disk so a typo'd filename quietly
    drops the image instead of showing a broken-image icon on the home page.
    """
    if not task.image:
        return None
    if task.image.startswith(("http://", "https://", "/")):
        return task.image
    path = settings.TASKS_DIR / task.slug / "assets" / task.image
    if not path.is_file():
        logger.warning("Task %r: image %r not found at %s",
                       task.slug, task.image, path)
        return None
    return f"{FILE_ROUTE}{path.resolve()}"


def _menu_html(tasks) -> str:
    """The whole card list as one HTML block."""
    cards = []
    for task in tasks:
        image_src = _card_image_src(task)
        image_html = (f'<div class="task-card-image">'
                      f'<img src="{image_src}" alt="{task.title}"></div>'
                      if image_src else "")
        cards.append(
            f'<a class="task-card" href="/{task.slug}">'
            f'<div class="task-card-text">'
            f'<div class="task-card-title">{task.title}</div>'
            f'<div class="task-card-blurb">{task.blurb}</div>'
            f'</div>'
            f'{image_html}'
            f'</a>'
        )
    return (f'<div class="task-menu"><div class="menu-heading">Tasks</div>'
            f'{"".join(cards)}</div>')


def _utilities_html(utilities) -> str:
    """The utilities as a grid of compact tiles; nothing if there are none."""
    if not utilities:
        return ""
    tiles = "".join(
        f'<a class="utility-tile" href="/{w.slug}">'
        f'<div class="utility-title">{w.title}</div>'
        f'<div class="utility-blurb">{w.blurb}</div></a>'
        for w in utilities)
    return (f'<div class="utility-section"><div class="menu-heading">Utilities</div>'
            f'<div class="utility-grid">{tiles}</div></div>')


def build_app() -> gr.Blocks:
    """The whole web app: a menu page plus every registered task page.

    Two lists, on purpose: the home menu and the navbar show only the VISIBLE
    tasks, but every ENABLED task gets its route -- a hidden one is served at
    /<slug> for whoever is given the link (registry.TaskSpec.visible). A
    visible UTILITY is a tile on the home menu and stays out of the navbar.
    """
    common.reset_page_chrome()

    with gr.Blocks(title=settings.APP_TITLE) as demo:
        gr.Markdown(f"# {settings.APP_TITLE}", elem_classes="app-title")
        gr.HTML(MENU_CSS)
        gr.HTML(_menu_html(registry.visible_tasks()) + _utilities_html(registry.utilities()))
        common.page_footer()

    # One route per task. A task that fails to build loses its own page and
    # nothing else. What it must NOT do is fail silently, so the BUILD_ERRORS records it.
    BUILD_ERRORS.clear()
    for task in registry.enabled_tasks():
        with demo.route(task.title, f"/{task.slug}",
                        show_in_navbar=task.visible and not task.utility):
            try:
                task.build_page()
            except Exception as exc:
                BUILD_ERRORS[task.slug] = exc
                logger.exception("Task %r failed to build its page", task.slug)
                _error_panel(task, exc)
            _hidden_links(task)

    return demo


# ============================================================================
# The hidden tasks' own menu
# ----------------------------------------------------------------------------
# A hidden task is off the home menu and out of the navbar, so from its page
# there is no way to reach the OTHER hidden tasks -- except this: a footer
# line, under the page's own footer, linking to them. Only a hidden page
# gets it, so the visible pages advertise nothing they should not; and it is
# added here, from the registry, rather than by each task's web.py, so a
# newly hidden task cannot forget to take part.
# ============================================================================
def _hidden_links(task: registry.TaskSpec) -> None:
    if task.visible:
        return
    others = [t for t in registry.hidden_tasks() if t.slug != task.slug]
    if not others:
        return
    gr.HTML(_hidden_links_html(others))


def _hidden_links_html(others) -> str:
    links = " &nbsp;·&nbsp; ".join(
        f'<a href="/{t.slug}" title="{t.blurb}">{t.title}</a>' for t in others)
    return (f'<div class="hidden-task-links" style="text-align:center;'
            f'color:#888;font-size:0.85em;padding:0 0 0.75em">'
            f'Other unlisted pages: {links}</div>')


def _error_panel(task: registry.TaskSpec, exc: Exception) -> None:
    gr.Markdown(
        f"## {task.title} failed to build.**\n\n"
        f"`{type(exc).__name__}: {exc}`\n\n"
        f"The traceback is in the server log. The other tasks are unaffected — "
        f"[back to the menu](/)."
    )

# ============================================================================
# app/web/common.py — widgets every task page shares
# ----------------------------------------------------------------------------
# Page chrome, the progress box, and the bridge between a Gradio handler (which
# must keep yielding) and a job (which blocks for minutes). A task page should
# never re-implement any of this.
# ============================================================================
import queue
import threading
import time
from typing import List

import gradio as gr
from gradio.context import Context

import settings
from app import jobs

# Visible rows of a console box. 
LOG_BOX_LINES = 16

# How often a running job's transcript is pushed to the browser.
POLL_SECONDS = 0.5

# How often each open tab reports in, and how long one may stay silent before
# it is assumed gone.
HEARTBEAT_SECONDS = 10
SESSION_TIMEOUT = 3 * HEARTBEAT_SECONDS


# ============================================================================
# Page chrome
# ============================================================================
# The intro under a title is read once and then ignored, so it is set small
# and grey; the Input / Output labels are the opposite, bold and black, the
# two landmarks a returning user scans for. One style block per page (each
# route has its own config) rather than app-wide CSS, so a page is
# self-contained.
CHROME_CSS = """
<style>
.page-intro p, .page-intro li { font-size: 0.75em !important; color: #6b7280 !important; font-weight: 400 !important; }
.page-intro strong, .page-intro b { font-weight: 400 !important; }
.page-intro code { font-size: inherit !important; }
.section-label p { color: #000 !important; font-weight: 700 !important; margin: 0 !important; }
</style>
"""


def page_header(title: str, blurb: str = "") -> None:
    """Title and, under it, the one-paragraph intro at 75 % in grey. The
    nav bar is the way back to the menu; there is no link here."""
    gr.HTML(CHROME_CSS)
    gr.Markdown(f"## {title}")
    if blurb:
        gr.Markdown(blurb, elem_classes="page-intro")


def section_label(text: str) -> None:
    """A bold black landmark: "Input" above the inputs, "Output" above
    the downloads."""
    gr.Markdown(f"**{text}**", elem_classes="section-label")


# Every footer built this session, for the wiring tests to check against.
# Gradio serves a SEPARATE config per page, containing only that page's
# dependencies -- so the load/tick/unload trio has to be registered inside each
# route, next to the footer it repaints. 
_footers: List[gr.HTML] = []


def reset_page_chrome() -> None:
    """Forget the footers of a previous build. """
    _footers.clear()


def page_footer() -> gr.HTML:
    """Footer replacing Gradio's own, carrying the live session count.

    SELF-WIRING, deliberately: a page gets a ticking count purely by calling
    this, so a new task cannot forget to hook one up. (It could before, and
    every task page silently showed a frozen count.)

    The initial value is rendered when the page is BUILT, i.e. once at server
    start, so it is only a placeholder -- the load event below replaces it the
    moment a tab connects, and the timer keeps it current after that.
    """
    footer = gr.HTML(footer_html(active_sessions()))

    # show_progress="hidden" throughout: the count refreshes in the background
    # and must not flash a spinner over the footer every few seconds.
    timer = gr.Timer(HEARTBEAT_SECONDS)
    timer.tick(_footer_tick, None, footer, show_progress="hidden")

    root = Context.root_block          # the app, while this route is open
    if root is not None:
        root.load(_footer_tick, None, footer, show_progress="hidden")
        root.unload(on_session_end)

    _footers.append(footer)
    return footer


def _footer_tick(request: gr.Request):
    """A tab loaded or ticked: renew its claim and repaint the count."""
    return footer_html(touch_session(getattr(request, "session_hash", None)))


def footer_html(active: int) -> str:
    """The footer markup. Session count first, as in the original app: it is
    the line that changes, and it is why anyone looks down here."""
    users = "session" if active == 1 else "sessions"
    return (
        f'<div style="text-align:center;color:#888;font-size:0.85em;'
        f'padding:0.75em 0 0.25em">'
        f'Active: <b>{active}</b> {users}.<br/>'
        f'{settings.DEVELOPED_BY} '
        f'&nbsp;|&nbsp; Updated {settings.LAST_UPDATE}</div>'
    )


# ============================================================================
# Active-tab count
# ----------------------------------------------------------------------------
# Everyone on the LAN shares one process and one job queue, so it is worth
# knowing whether somebody else is on the app before starting a run that will
# queue behind theirs. A tab is counted from load, refreshes its claim on every
# timer tick, and is dropped when its Gradio session ends. The timestamps are
# the backstop for the exits `unload` misses (a killed browser, a dropped link),
# so a tab that stops reporting cannot inflate the count for ever.
# ============================================================================
_seen_lock = threading.Lock()
_last_seen: dict = {}          # gradio session hash -> monotonic time


def touch_session(session_hash) -> int:
    now = time.monotonic()
    with _seen_lock:
        if session_hash:
            _last_seen[session_hash] = now
        for key, seen in list(_last_seen.items()):
            if now - seen > SESSION_TIMEOUT:
                del _last_seen[key]
        return len(_last_seen)


def drop_session(session_hash) -> int:
    with _seen_lock:
        _last_seen.pop(session_hash, None)
        return len(_last_seen)


def active_sessions() -> int:
    """How many tabs are currently counted"""
    with _seen_lock:
        return len(_last_seen)


def on_session_end(request: gr.Request) -> None:
    """A tab closed. Nothing to repaint -- the remaining tabs pick the new
    count up on their next tick. Registered once, on the app (see web/main)."""
    drop_session(getattr(request, "session_hash", None))


# ============================================================================
# Running a job from a Gradio handler
# ============================================================================
def _session_id(request) -> str:
    """The browser tab's session hash, off gradio's Request; '' without."""
    return str(getattr(request, "session_hash", "") or "")


def _client_ip(request) -> str:
    """The requester's address: the first X-Forwarded-For hop when the app
    sits behind a proxy, else the socket's peer; '' without a request."""
    if request is None:
        return ""
    forwarded = ""
    try:
        forwarded = (request.headers or {}).get("x-forwarded-for", "")
    except Exception:                          # noqa: BLE001 - best effort
        forwarded = ""
    if forwarded:
        return forwarded.split(",")[0].strip()
    client = getattr(request, "client", None)
    return str(getattr(client, "host", "") or "")


def stream_job(target, transcript, run=None, request=None):
    """Run `target()` on a worker thread, yielding (text, result, error).

    `run` (the job's RunDir) and `request` (the handler's `gr.Request`,
    which gradio injects into a handler that declares a parameter of that
    type) add the job to the usage log: its row is appended here with the
    session and the IP, and completed when the thread ends. The queue
    stamps started_at itself (`jobs.QUEUE.slot(run=run)`).

    Gradio handlers must keep yielding to stay responsive, while a job blocks
    for minutes -- so the job goes to a thread and this generator pushes the
    growing transcript to the page. `result` and `error` stay None until the
    thread is done; exactly one of them is set on the final yield.

    The exception is handed BACK rather than raised, so the page can show it in
    the progress box next to the steps that led up to it instead of replacing
    the whole run with a red banner.
    """
    outcome = {}

    def _run():
        try:
            outcome["result"] = target()
        except BaseException as exc:            # noqa: BLE001 - reported, not swallowed
            outcome["error"] = exc

    if run is not None:
        jobs.usage_start(run, session_id=_session_id(request),
                         ip=_client_ip(request))
    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    while worker.is_alive():
        yield transcript.text(LOG_BOX_LINES), None, None
        time.sleep(POLL_SECONDS)
    worker.join()
    if run is not None:
        jobs.usage_finish(run)
    yield (transcript.text(LOG_BOX_LINES),
           outcome.get("result"), outcome.get("error"))


def sink_queue():
    """A (sink, drain) pair for feeding a Transcript from a worker thread.

    `sink` is safe to call from the job thread; `drain` returns whatever has
    arrived since the last call. Only needed by a page that wants the lines
    themselves -- most pages just read `transcript.text()`.
    """
    q: queue.Queue = queue.Queue()

    def drain():
        out = []
        while True:
            try:
                out.append(q.get_nowait())
            except queue.Empty:
                return out

    return q.put, drain


def progress_box(label: str = "Console") -> gr.Textbox:
    return gr.Textbox(label=label, lines=LOG_BOX_LINES, max_lines=LOG_BOX_LINES,
                      interactive=False, autoscroll=True,
                      value="Idle. Fill the form and press Run.")

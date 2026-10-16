#!/usr/bin/env python3
# ============================================================================
# run_web.py — the simplified calls to app/web/main.py for running the web app
# ----------------------------------------------------------------------------
# Every task on ONE port:
#     python run_web.py --host 0.0.0.0 --port 7777
#
# Other forms
#     python run_web.py                 # http://127.0.0.1:7777 (local only)
#     python run_web.py --share         # temporary public gradio.live link
#
#   1. --host 0.0.0.0 binds all interfaces, so the app is reachable at
#      http://<server-LAN-IP>:7777 from other machines. `hostname -I` finds it.
#   2. Ports 1024-65535 need no root on Linux.
#   3. With ufw enabled, open the port first: `sudo ufw allow 7777`.
#   4. Check the port is free: `ss -tulpn | grep 7777` (no output = free).
#   5. To survive a logout or reboot, run under systemd, tmux or screen.
#
# INTERPRETER: use the env that has gradio + dipy + nibabel + vtk, e.g.
#     ~/anaconda3/envs/myenv/bin/python run_web.py --host 0.0.0.0 --port 7777
# Tasks 3 and 4 do NOT need simnibs in this env; they shell out to SimNIBS's
# own Python. See tasks/mesh_head/ and tasks/mesh_sim/.
#
# A task's cli.py / gui.py are deliberately NOT wired in here -- run those
# directly from their own folder.
# ============================================================================

import os as _os, sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
for _p in (_root, _os.path.join(_root, "vendor", "anicond")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse
import logging

import settings


# Must be set before gradio is imported, or it reports usage on startup.
_os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")



def main() -> None:
    parser = argparse.ArgumentParser(description="Run the APP_BRAINE task server.")
    parser.add_argument("--host", default=settings.HOST,
                        help="interface to bind (0.0.0.0 for LAN access)")
    parser.add_argument("--port", type=int, default=settings.PORT)
    parser.add_argument("--share", action="store_true",
                        help="also expose a temporary public gradio.live link")
    parser.add_argument("--log-level", default=settings.LOG_LEVEL)
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level,
                        format="%(levelname)s:%(name)s:%(message)s")

    from app.web.main import build_app #import a func from app/web/main.py

    _os.makedirs(settings.RUNS_DIR, exist_ok=True) #for saving running files

    demo = build_app()
    demo.queue()          # long jobs must not block other tabs' page loads
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        # Results are served out of data/runs/, so Gradio must be allowed to
        # read from there for the download links to work; tasks/*/assets holds
        # the files a page hands out ready-made (the default tissue table).
        allowed_paths=[str(settings.RUNS_DIR), *(str(p) for p in settings.ASSET_DIRS)],
        footer_links=[],      # the pages carry their own footer
        show_error=True,
    )


if __name__ == "__main__":
    main()

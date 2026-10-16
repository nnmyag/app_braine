## Auto-Deploy Braine Web App (systemd user service)

This app runs as a systemd user service named `braine.service`, so it
starts automatically and restarts on failure — no manual command needed.

### Setup (one-time)

1. Create the service file:
   ~/.config/systemd/user/braine.service

   [Unit]
   Description=Braine Web App
   After=network.target

   [Service]
   WorkingDirectory=/home/yourusername/path/to/your/folder
   ExecStart=/home/my/anaconda3/bin/python3 run_web.py --host 0.0.0.0 --port 7777
   Restart=on-failure
   RestartSec=5

   [Install]
   WantedBy=default.target

2. Enable and start it:
   systemctl --user daemon-reload
   systemctl --user enable braine.service
   systemctl --user start braine.service

3. Allow the service to start at boot without logging in first:
   loginctl enable-linger $USER
   (If this fails due to permissions, ask an admin to run it once.)

### Everyday commands

   systemctl --user status braine.service     # check if running
   systemctl --user restart braine.service    # restart manually
   systemctl --user stop braine.service       # stop
   journalctl --user -u braine.service -f     # view live logs

### Updating after code changes

When you pull new code, add features, or edit files in the project folder:

1. Make/pull your changes as usual in the project folder.
2. Restart the service to load the new code:
   systemctl --user restart braine.service
3. Confirm it came back up cleanly:
   systemctl --user status braine.service
   journalctl --user -u braine.service -n 50 --no-pager

Notes:
- If you only changed Python source files, a restart is enough — no
  need to touch the .service file.
- If you change the WorkingDirectory, the python interpreter path, or
  the command-line arguments (e.g. a different port), edit the
  .service file itself, then run:
    systemctl --user daemon-reload
    systemctl --user restart braine.service
- If you add new Python dependencies, install them into the same
  environment the service uses (the anaconda3 env at
  /home/my/anaconda3/bin/python3) before restarting.

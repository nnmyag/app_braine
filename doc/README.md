# APP_BRAINE

Web front-end for the tasks related to brain-conductivity. 

    python run_web.py --host 0.0.0.0 --port 7777

| Task                                  | Page          | Note              |
| T1/T2 → conductivity map              | `/isocond`    |                   |
| DTI → anisotropic conductivity tensor | `/anicond`    |                   |
| T1/T2 → head mesh (charm)             | `/mesh_head`  |                   |
| head mesh → TMS simulation            | `/mesh_sim`   |                   |
| TMS field → tracts                    | `/tract_efield` | hidden: not on the menu or navbar, reachable by URL |
| tractogram → visual tracts            | `/tract_simplify`  | hidden; for pictures only |
| CondNet-TART training at a frequency  | `/condnettart_train` | hidden; runs condnettart_training's scripts (03 --table --dated, 04, 05) while the page stays open |

The home page lists the **Tasks** above as cards (and in the navbar), and the
**Utilities** below as compact tiles (not in the navbar):

| Utility                               | Page          | Note              |
| tissue conductivity at a frequency    | `/tissue_conductivity` | Gabriel 1996 Cole-Cole model; YAML + CSV |
| affine registration                   | `/registration` | rigid or affine (SimpleITK, mutual information) onto the fixed grid; optional segmentation with the same transform |
| resample voxel size                   | `/resample` | isotropic voxel size, same field of view; labels nearest, images linear (anti-aliased); RAS by default |

## Layout

    run_web.py       the ONLY exposed entry point
    settings.py      server, data root

    app/             the shell. Knows about pages, nothing about science.
      registry.py      TASKS -> the home menu and the mounted routes
                       (enabled = has a page; visible = listed on the menu)
      jobs.py          run folders, the shared FIFO queue, progress + logging
      mosaic.py        the one rule for a review mosaic's grid (rows x cols)
      report.py        the .docx run report's one style (document + table)
      web/main.py      MAIN HOMEPAGE + one demo.route() per task
      web/common.py    page chrome, session count, progress box, job streaming

    tasks/<task>/    one folder per task
      service.py       THE seam: UI-agnostic. Every front-end calls only this.
      web.py           Gradio page — reached from the menu
      settings.py      task-local knobs
      (cli.py, gui.py) run directly from here; not exposed by run_web.py

    vendor/          frozen library snapshots the tasks compute with.
    data/runs/<task>/<job id>/    every run: inputs/, outputs/, run.log, .zip

**The rule:** one file per front-end, all thin, all calling `service.py` and
nothing else. `service.py` returns paths and values, never widgets — that is
what lets a `cli.py` added later reproduce a web run byte for byte.

**`__init__.py` under `vendor/`:** what makes a subfolder a real Python
package, so its own files can import from each other with `from .. import
settings`-style relative paths (`vendor/isocond/__init__.py` is empty and
exists for exactly this). Where it also re-exports a few names — `from
.simplify import simplify_tracts` in `vendor/tractviz/__init__.py`,
similarly `vendor/gii2vtk/__init__.py` — a task can then write the short
`from vendor import tractviz; tractviz.simplify_tracts(...)` instead of
naming the submodule. It is never where the actual code lives; that stays
in the sibling files (`simplify.py`, `convert.py`, …).

**Adding a task** = one folder under `tasks/` + one line in `app/registry.py`. 

## Running it

Use the environment that has gradio, dipy, nibabel, torch and vtk:

    python run_web.py --host 0.0.0.0 --port 7777

`--host 0.0.0.0` binds every interface, so the app is reachable at
`http://<server-LAN-IP>:7777`; `hostname -I` finds the address. Check the port
is free with `ss -tulpn | grep 7777`, and open it if ufw is on
(`sudo ufw allow 7777`). To survive a logout, run under systemd, tmux or screen.

Every page footer shows how many browser tabs are currently connected, so you
can see whether a run you start will queue behind somebody else's.

**How the count works.** Gradio's browser-side client stamps every request a
tab sends — page load, periodic heartbeat, button clicks — with a random
`session_hash` it generates once when the page's JavaScript starts running (so
a refresh gets a new one; it is not a stable per-browser or per-user id).

A task's `cli.py` / `gui.py`, when they exist, are run straight from their own
folder — `python tasks/anicond/cli.py …` — and are deliberately not wired into
`run_web.py`. 

## Jobs

One heavy job at a time across **all** tasks (`MAX_CONCURRENT_JOBS` in
`settings.py`). 

Every run gets its own folder under `data/runs/<task>/<job id>/`:

    inputs/     the uploads, staged server-side — a complete record of what
                the run computed from
    outputs/    everything produced
    run.log     the full transcript

Folders older than `KEEP_RUNS_DAYS` are swept when a new job starts.

**Usage log.** Every calculation request, from every task, adds one row to
`data/usage_log.csv` (`USAGE_LOG` in `settings.py`): `job_id`, `task`,
`session_id` (the browser tab's Gradio session hash — see "How the count
works" above; it changes on refresh), `ip` (the first `X-Forwarded-For` hop
when the app sits behind a proxy, else the socket's peer), `requested_at`
(the request reached the server), `started_at` (the job took its queue
slot — the gap is the queue wait), `completed_at` (time of day only; its
date is `requested_at`'s) and `duration_s` (from `started_at`). No status
column: the columns say it — no `started_at` is still queued, no
`completed_at` is still running or a job the server died under; whether
it succeeded is in the run's own `run.log`. The row is appended when the
request arrives and filled in place when the job ends. A request refused
before a run folder exists (a missing input) is not a row. The file is
rewritten whole under a lock and nothing depends on it: a job never fails
over its log; a log with an older column set is set aside as
`usage_log.<stamp>.old.csv` rather than appended to.



## Tissue conductivity tables

Several tasks assign a conductivity (S/m) to each tissue label. The tables
live in five files across three projects, and **must agree** (at 10 kHz the
reference is the table CondNet-TART is trained on,
`PROJ_CONDNETS/condnettart_training/config/10kHz.yaml`):

| File | Read by |
|---|---|
| `vendor/isocond/pipeline/tissue_labels.csv` (columns `10khz`, `1hz`) | isocond: the per-slice check's Expected column, the `tissue_conductivity_<freq>.csv` saved with a result |
| `tasks/anicond/assets/default_target_conductivity.csv` | anicond (tensor scaling), mesh_sim (rows with a `simnibs_tissue`), the Tissue Conductivity utility (its 10 kHz anchor) |
| `vendor/anicond/core/config.yaml` (`labels.conductivity_iso_10khz`) | the vendored anicond pipeline |
| `PROJ_CONDNETS/model_runner/code/pipeline/tissue_labels.csv` | the standalone model_runner (upstream copy of the first file) |
| `PROJ_DCONDNET/scripts/core/config.yaml` (`labels.conductivity_iso_10khz`) | the DCond project |

Decisions behind the current values:

| Tissue (label) | 10 kHz | 1 Hz | Since |
|---|---|---|---|
| csf_meninges (6) | **0.5** (was 2.0) | **0.16** (was 1.8) | 2026-10-08 |
| brainstem & cortical nuclei (14) | **0.07 = white matter** (0.179111 on 2026-10-08, 0.07 before) | 0.14 = white matter | 2026-10-09 |
| muscle (10) | 0.34 (unchanged) | 0.16 | — |

Brainstem & cortical nuclei is treated as **white matter** everywhere: no
measured value exists for it (Gabriel et al. 1996 list none), so the Tissue
Conductivity utility also gives it white matter's frequency dependence --
it equals white matter at every frequency. Thalamus (label 15, anicond/DCond
only) is 0.07 as well.

**Changing a value**: edit all five files (and `config/10kHz.yaml` in
condnettart_training), then re-run what reads them. Per-change notes, with
the list of what to re-run, are in `doc/20261008_conductivity_table_fix.md`.

## Customizing pages

Everything a visitor reads is a Python string in one of these files

| What | Where |
|---|---|
| Home page heading and the "Pick a task…" paragraph; how each menu card is drawn (`_menu_card`) | `app/web/main.py` |
| Menu card title (also the nav-bar entry) and one-line blurb per task | `app/registry.py` — `TASKS` |
| Page title line and the intro under it (75 % size, grey), on every page; the bold **Input** / **Output** section labels | `app/web/common.py` — `page_header()`, `section_label()`, `CHROME_CSS` |
| Footer (session count, app title, lab, date) | `app/web/common.py` — `footer_html()`; the values come from `APP_TITLE`, `DEVELOPED_BY`, `LAST_UPDATE` in the root `settings.py` |
| A task page's own text: the intro sentence, notes under the inputs, widget labels and `info=` lines, download-link names, the console's idle line | that task's `tasks/<task>/web.py` (`INTRO`, `SEG_NOTE`, `ROLE_LABELS`, the `gr.Markdown(...)` and widget calls in `build_page()`) |
| Numbers a user should not have to know (defaults, thresholds, file switches) | `tasks/<task>/settings.py` |

**HTML.** Any `gr.Markdown` block accepts inline HTML (that is how task 2's note
links to `/isocond`), and `gr.HTML("...")` places a raw block. To hand-design
the home page, replace the `_menu_card` loop in `build_app()` with one
`gr.HTML` built from `registry.enabled_tasks()`, so the registry stays the
single source of what is listed.

**CSS.** Two scopes:

- *one page* — a `gr.HTML("<style>…</style>")` block at the top of that page's
  `build_page()`. Task 2's `PAGE_CSS` in `tasks/anicond/web.py` is the model:
  it restyles the console and strips the drop zones down to their icon.
- *the whole app* — `gr.Blocks(title=..., css="...")` or `css_paths=[...]` in
  `build_app()` (`app/web/main.py`); this reaches the home page and every task
  route.

Target widgets with `elem_id=` / `elem_classes=` on the component, and expect to
need `!important` on properties Gradio already sets (its own stylesheet is
scoped and specific). Gradio serves a *separate* config per page, so anything
that must run on page load — a `load`, a `Timer` — is registered inside each
route; `common.page_footer()` shows how.

After editing, restart `run_web.py`: pages are laid out once, at start-up.

## Adding a task

A task is a folder and a registry line. For a task called `mytask`:

1. **Create `tasks/mytask/`** with:
   - `service.py` — the pipeline behind a `run_job(...)` that takes paths and
     values and returns paths and values. Stage every upload with `jobs.stage_upload()`
     so the run folder is a complete record. Start the file with the 4-line
     `sys.path` bootstrap the other tasks carry.
   - `web.py` — `build_page()`, widgets and callbacks only. Copy
     `tasks/anicond/web.py` (or the shorter `tasks/isocond/web.py`) and keep
     its shape: `common.page_header()`, inputs, `common.page_footer()`, then
     the wiring. The run handler wraps `service.run_job` in
     `jobs.capture_logs()` and `jobs.QUEUE.slot()` and streams through
     `common.stream_job()`, which is what keeps the page updating while the
     job blocks.
   - `settings.py` — task-local constants. Anything a user should change per
     run is a widget instead.
   - `__init__.py` (empty), and `assets/` if the page hands out a file
     (`run_web.py` serves every `tasks/*/assets` folder).

2. **Register it** in `app/registry.py`:

       TaskSpec(slug="mytask", title="My task", blurb="One line for the menu.",
                module="tasks.mytask.web"),

   `slug` is the URL (`/mytask`), the run folder (`data/runs/mytask/`) and
   part of the archive's name: a task that offers a "download all" zips
   its `outputs/` with `run.zip_outputs(subject_id=..., model=...)`, which
   names the file `<subject>_<slug>[_<model>]___output.zip`
   (`jobs.output_zip_name`; `model` only for a task tied to one
   conductivity model — `IXI258_mesh_sim_tbe___output.zip`). That is the
   convention every new task follows, so a downloads folder holding runs
   of several tasks and subjects reads by itself. The files sit at the
   archive's root: it unpacks into a folder of its own name, and a second
   folder inside would only repeat it.
   `title` is the nav entry and the menu card. Two switches: `enabled=False`
   removes the page altogether; `visible=False` keeps the page (it is served
   at `/mytask` for whoever has the link) but takes it off the home menu. Hidden
   pages do see each other: under its own footer, every hidden page lists
   "Other unlisted pages" with a link to each of the other hidden tasks —
   added by `app/web/main.py` from the registry, not by the task's own
   `web.py`, so a hidden task takes part just by being registered. Task 5 (`tract_efield`) is registered that way, not something every visitor should be offered.
   A third switch, `utility=True`, is for a small self-contained tool (a
   calculator, a converter) rather than a full task: it is shown as a tile
   under **Utilities** on the home page instead of a card under **Tasks**,
   and stays out of the navbar (e.g. `tissue_conductivity`).

3. **Restart and check.** The card and route appear on their own. If the page
   is blank, the task raised while building: `app/web/main.py` records it in
   `BUILD_ERRORS`, and shows it on that page.

4. If the service both `logger.info`s and `report`s the same lines, add its
   logger name to `UNCAPTURED_LOGGERS` in the root `settings.py`, or every
   line appears twice in the console.

Things a new task should not do: import gradio in `service.py`; edit anything
under `vendor/` (fix it in the task layer); expose a `cli.py`/`gui.py` through
`run_web.py` — those are run from their own folder.

## How a page's URL reaches its code

Each entry in `app/registry.py`'s `TASKS` tuple is a `TaskSpec` with two plain
string fields — `slug` and `module`:

    TaskSpec(slug="isocond", ..., module="tasks.isocond.web")

- `slug` becomes the URL. `app/web/main.py` loops over the registered tasks
  and calls Gradio's `demo.route(task.title, f"/{task.slug}")`, which mounts a
  page at `/isocond` inside the one running app.
- `module` becomes the code that page runs. Still inside that same route,
  `TaskSpec.build_page()` calls `importlib.import_module(self.module)` —
  turning the string `"tasks.isocond.web"` into a real import at runtime — and
  then calls that module's `build_page()`. That function lays out the widgets
  and wires its Run button to `service.py` in the same task folder.

So the chain for a task is: **registry entry → route mounted at `/slug` →
`module.build_page()` imported and called → that page calls into its
`service.py`.** The pairing between a slug and the folder that answers it is a
naming convention chosen when the `TaskSpec` is written.

## Task 1 — T1/T2 → conductivity map  (`/isocond`)

**In:** a T1 and a T2 of the same head (they need not be aligned — T2 is
registered to T1 first) and a network: CondNet-TART or CondNet. A 2D image works
too and gives a 2D result.

**Out:** the conductivity map as `.nii.gz` plus a review mosaic. With *save
intermediates* on — and for CondNet-TART, which has a segmentation head — also
the 15-class tissue segmentation with its mosaic and colour legend, the exact
normalized volumes the model saw, and
`<case>_<model>_tissue_conductivity_<freq>.csv`, the
`tissue_id, tissue_name, conductivity_s_per_m` mapping at the frequency that run
used, so a downloaded segmentation's integers still mean something months later.
All zipped.

CondNet-TART predicts S/m at 10 kHz; CondNet predicts normalized values at
900 MHz. Each mosaic is labelled with its own model's unit — **the numbers are
not comparable between the two.**

Checkpoints stay loaded between jobs (CondNet-TART's stage 0 is 107 MB), which
is safe because the queue runs one job at a time.

**Two runs of the same input do not give identical output.** The T2→T1 step is a
Mattes mutual-information registration with `RANDOM` metric sampling; it passes a
fixed seed, but ITK's sampling still depends on the thread count, so repeated
runs converge to slightly different transforms and the maps that follow differ
(correlation ~0.85 between runs on the same case). If bit-reproducibility ever
matters, pin `SetGlobalDefaultNumberOfThreads(1)` in the registration and expect
it to be slower.

The `.pth` checkpoints in `vendor/isocond/models/` are **symlinks**, to avoid a
second 122 MB copy on disk. `cp -L` over them makes the app fully standalone;
`vendor/VERSION.md` records where they point.

**Adding a CondNet-TART model**: put `condnettart_<freq>_stage0.pth` and
`_stage1.pth` in `vendor/isocond/models/` (e.g. `condnettart_10Hz_…`); the
page lists it the next time its model list is opened, no restart. The app
also needs the pair's **output scale** (σ = output × σ_max / (1 − τ)):
from its model card `condnettart_<freq>.json` beside it (written by
condnettart_training's `05_export_for_app.py`; preferred), else from the entry
for `<freq>` in **`vendor/isocond/models/output_scales.json`** (shipped with
50 Hz, 10 kHz, 1 MHz, 100 MHz; one entry per frequency, editable). A pair with
neither is skipped, with a warning in the log.

**Manual check** — what to read by hand if the AI-written code is in doubt:

- `vendor/isocond/pipeline/session.py` + `runner.py` — the actual inference:
  `InferenceSession.run()` loads the T1/T2 and calls
  `InferenceRunner._infer_batch()`, the forward pass — de-normalized to S/m
  for CondNet-TART, left normalized for CondNet (`CondNetTart` / `CondNetModel`
  in `nets/`), which is where the two units in the intro come from.
  Checkpoints load once and are cached (`InferenceSession.get_runner()`).
- `vendor/isocond/pipeline/tissues.py` — `load_tissue_conductivity()`: the
  literature-value {tissue id: S/m} map `write_tissue_table()`'s CSV
  documents (a label with no row maps to 0.0 — `segmentation_to_conductivity()`
  is the same lookup applied to a whole segmentation, task 1's own reference
  when comparing a predicted map to the classical piecewise-constant one).
- `vendor/isocond/pipeline/preprocess.py` — `_configure_sitk_registration()`:
  `RANDOM` sampling and the seed (the non-reproducibility above).
- `tasks/isocond/service.py` — `run_job()`: the model gate (`ENABLED_MODELS`),
  what is staged and what is written; `write_tissue_table()`: the frequency in
  the CSV's name is the model's own; `capped_conductivity_mosaic()`:
  `CONDUCTIVITY_VMAX` applies only to S/m models.
- `tasks/isocond/settings.py` — `DEVICE`, `CACHE_LOADED_MODELS`,
  `ANISOTROPIC_TISSUE_NAMES`.

## Task 2 — DTI → conductivity tensor  (`/anicond`)

**In:** a DWI series (one 4D file, or every per-direction 3D volume at once),
its `bval` and `bvec`, and a tissue segmentation. Optionally: the target
conductivity table (leave empty for the shipped default, downloadable from the
page to edit and re-upload; its conductivity columns are named
`conductivity_iso_<freq>`, one per frequency, and the page's Frequency box
lists every `<freq>` the table carries, starting from the config's
`conductivity_freq_in_use`; its optional `simnibs_tissue` column names the
SimNIBS tissue each row stands for — `Blood_in` → `Blood`, `Cancellous_bone`
→ `Spongy_bone`, `Cortical_bone` → `Compact_bone`, `Eye_humour` →
`Eye_balls`, … — and is read by task 4, not by this task), a whole-brain
tractogram a previous run
of the *same DWI* wrote (skips the minutes of tracking), and — when the
segmentation is not yet in the DWI's space — the T1 it was drawn on. The
anisotropic tissue ids default to the WM-related ids of the table in use; the
subject id defaults to the first six characters of the segmentation's name.

**Out:** `outputs/<model>/<subject>_c_tensor6x1.nii.gz` — the anisotropic
conductivity tensor in FSL's 6-component layout (`xx, xy, xz, yy, yz, zz`).
Intermediates: the segmentation in DWI space, the anisotropic tissue mask, the
FA map, the diffusion tensor in the same 6×1 layout, the whole-brain
tractogram (`.vtk`, tract-based models only) with a simplified twin for
viewing beside it (`<case>_wb_tracts_viz.vtk`: `vendor/tractviz` at its
defaults — healed, ≥ 100 mm, 10 % of each bundle, smoothed σ 1.5 mm; never
an input to a model), the effective conductivity
G_eff and the eigenvalue histogram over the anisotropic tissue. The canvas
shows any of these as a mosaic (the histogram as itself).

Models: Direct Mapping (DM), Volume Normalized (VN), Tract-based effective
conductivity (TBe). Only the tract-based models need tractography, so a DM/VN
job skips that step entirely — seconds instead of minutes. A TBe run on a
128×128×64 case takes about 3½ minutes end to end, or ~2 with an uploaded
tractogram.

**Segmentation space.** Tick *already in the DWI's space* and it is only
resampled onto the b0 grid (nearest-neighbour). Untick it and the page asks for
the T1 the segmentation was drawn on, registers that T1 to the b0 rigidly and
carries the labels along — the study's own `b4` step. Either way the job
measures how much of the anisotropic tissue falls inside the b0 brain mask and
reports it in the console; a low number means the result is wrong.

Tractogram files are read and written by `tasks/anicond/tractio.py` on the
`vtk` package alone (dipy's own `.vtk` path needs `fury`, whose version pin
breaks on upgrade); the convention is dipy's, so the study's existing `.vtk`
files load unchanged.

The per-job conductivity table and tissue ids are passed to each model through
its constructor, never written into a global — so two queued jobs can never see
each other's settings.

**Manual check** — what to read by hand if the AI-written code is in doubt:

- `vendor/anicond/cond_models/` — `DirectMappingModel`, `VolumeNormalizedModel`,
  `TractBasedGeoModel` / `TractBasedEffModel` (`eff_cond_spherical_mean()` in
  `core/cond/lib_tractbased_core.py`): the model behind each of DM / VN / TBe,
  one `__call__(d_evals, d_evecs, segs, streamlines, affine)` per name in
  `ctx.model_names` — the actual conductivity estimate.
- `vendor/anicond/core/med/lib_tract.py` — `run_tractography()`: the CSA
  tracking `whole_brain_tracts()` calls, `SEED_DENSITY`.
- `vendor/anicond/core/tens/lib_tsio.py` — `to_upper_triangular6()` /
  `from_upper_triangular6()`: the 6×1 layout the deliverable and task 4's
  tensor upload both use.
- `tasks/anicond/service.py` — `run_job()`: which step the *already in the
  DWI's space* box selects, `align_segmentation()` (resample only) or
  `register_segmentation_via_t1()` (rigid; `REG_ITERATIONS`,
  `REG_LEARNING_RATE`), and that the table and tissue ids go into the model's
  constructor; `check_segmentation()`: `MIN_SEG_BRAIN_OVERLAP`;
  `whole_brain_tracts()`: `FA_THR`; `load_uploaded_tracts()`:
  `MIN_TRACT_GRID_OVERLAP`; `restrict_to_region()`: `RESTRICT_TO_MODEL_REGION`;
  `simplified_tracts()`: the `_wb_tracts_viz.vtk` twin, `vendor/tractviz` at
  its defaults.
- `tasks/anicond/runctx.py` — `parse_conductivity_csv()` (the
  `conductivity_iso_<freq>` column), `default_aniso_ids()`.
- `tasks/anicond/tractio.py` — `save_tractogram()` / `load_tractogram()`: the
  RAS↔LPS convention, since tasks 4 and 5 read these files.

## Task 3 — T1/T2 → head mesh  (`/mesh_head`)

**In:** a T1 (required) and a T2 (optional), `.nii.gz`. A subject id (default:
the first six characters of the T1's filename) names `m2m_<id>/` and
`<id>.msh`. If the T1/T2 are **not** already in the DTI b0's space, untick the
box and upload the b0: the T1 is registered to it rigidly (task 2's parameters)
and the transform is written into the T1/T2 *headers* — no resampling — so
charm builds the head model at full resolution but in the b0's world
coordinates, where task 2's tensors and task 4's simulation live. An
alignment-check mosaic (T1 contours over the b0) is written and shown.

**Out:** charm's `m2m_<id>/` folder — the tetrahedral head mesh `<id>.msh`,
`final_tissues.nii.gz` (+ LUT), the central/pial/white cortical surfaces as
gifti, the T1/T2 charm used, MNI warps, EEG positions and charm's own
`charm_report.html` — plus a `vtk/` folder with those three surfaces as
`both.<surface>.vtk` (lh + rh merged into one model, one colour, charm's
per-vertex sigma attached, title line tagged `SPACE=RAS` so 3D Slicer places
them on the T1 without a transform; `vendor/gii2vtk`), the copies handed to
charm (`<id>_T1_charm.nii.gz`, qform = sform), the `.tfm` transform when
registration ran, two review mosaics and `<id>_run_report.txt`. Everything is
zipped; the page lists each file with one line on what it is.

**Canvas views:** T1 / T2 / b0 as uploaded; alignment check; tissue
segmentation in SimNIBS's colours; white/central/pial cortical surfaces cut by
each axial slice over the T1. The same two mosaics are saved into `outputs/`
(the surfaces one with thicker lines), next to `<id>_run_report.txt`, which
lists the inputs, timing, completion time and what every file and folder is.

**How it runs:** `service.run_job()` shells out to `$SIMNIBS_ROOT/bin/charm`
with `cwd = outputs/`, `--forcerun --forceqform` and a `charm_settings.ini`
derived from SimNIBS's `charm.ini` with the task's thread count
(`CHARM_THREADS`, 0 = all cores). charm's stdout streams into the console.
No `simnibs` import anywhere: charm is already a program. At page build
`service.simnibs_status()` probes the install (`SIMNIBS_ROOT`, `charm`,
`import simnibs` under `SIMNIBS_PYTHON`, version ≥ 4); if the probe fails the
page shows a "not supported on this server" notice with the reason and the
Run button is disabled. A run takes 30–60 min on a 1 mm T1; the shared queue
means a charm run and a tractography run never overlap.

**Manual check** — what to read by hand if the AI-written code is in doubt:

- `tasks/mesh_head/service.py` — `run_job()`: the header-only rewrite,
  `register_t1_to_b0()` → `b0_space_affine()` → `write_with_affine()` (no
  resampling, qform = sform); `run_charm()`: `--forcerun --forceqform`,
  `CHARM_TIMEOUT_S`; `_write_charm_settings()`: `CHARM_THREADS`;
  `_probe_simnibs()`: `MIN_SIMNIBS_VERSION`, `SIMNIBS_ROOT`.
- `vendor/gii2vtk/convert.py` — `convert_m2m_surfaces()`: scans charm's
  `surfaces/` folder and calls `merge_surfaces()` (lh + rh, per-vertex sigma
  as the active scalar) and `write_vtk_polydata()` (the `SPACE=RAS` tag) —
  the whole `vtk/both.<surface>.vtk` step in one call.
- `tasks/mesh_head/settings.py` — the `REG_*` knobs are a copy of task 2's;
  keep the two in step.

## Task 4 — head mesh → TMS simulation  (`/mesh_sim`)

**In:** the `.zip` task 3 produces — the page reads the subject id, the cap
files and the electrode coordinates straight out of the archive's index, so
its dropdowns fill in instantly without unpacking several hundred megabytes.
Then a coil (figure-eight models only, scanned live from the install), the cap
to place it by, the centre electrode and the electrode its handle points
toward, the coil-to-scalp gap and dI/dt. Optionally an anisotropic
conductivity tensor for white and gray matter — `anicond`'s own 6-component
output. No control for SimNIBS's own tensor mapping (`vn` / `mc` / `dir`,
what SimNIBS does with the tensor FILE, not which model produced it): it is
fixed to `dir`, "use the tensor's own values", because every tensor this app
hands SimNIBS is already a real, computed conductivity tensor — never a bare
diffusion tensor that still needs SimNIBS to invent a magnitude for it, which
is the only case `vn` or `mc` would be right for (`settings.py` still
documents what each of the three actually does). Plus a **conductivity
type** label (`iso` / `dm` / `vn` / `tbe`) that is written verbatim into the
results CSV so a cohort can be grouped by it; it is a label, not a switch,
because `dm` / `vn` / `tbe` are *anicond*'s model names and the tensor they
produced is already baked into the file you upload. The page warns if the
label and the run contradict each other (`iso` with a tensor attached, or a
tensor name that says `DM` while the label says `VN`).

**Conductivities.** SimNIBS's built-in per-tissue values (Wagner 2004: WM
0.126, GM 0.275, CSF 1.654 S/m, …) are *not* used. Every run takes its
isotropic values from task 2's target table
(`tasks/anicond/assets/default_target_conductivity.csv`, the column
`mesh_sim/settings.CONDUCTIVITY_FREQ` selects, default the table's own), joined
to SimNIBS's tissues through the table's `simnibs_tissue` column — so an `iso`
run and a `dm` / `vn` / `tbe` run of one subject differ only by the tensor,
never by two reference tables. An isotropic run applies them to every tissue;
an anisotropic run applies them to every tissue *outside* the tensor. A SimNIBS
tissue the table has no row for (with the shipped table: `Bone`, tag 4, and
`Cartilage`, tag 11 — neither is written by charm) keeps SimNIBS's value, and
the console and run report say so. The run report lists every value the solver
ran with, tag by tag, as the worker read them back from SimNIBS.

With a tensor attached the tensor is used **as-is** (`TENSOR_AS_IS`): the
worker calls SimNIBS's `cond2elmdata` with `correct_intensity=False` and the
eigenvalue caps (`aniso_maxcond`, `aniso_maxratio`) lifted, because stock
`dir` first rescales the whole tensor field by one global factor so its mean
matches the isotropic reference and then clips eigenvalues — a correction meant
for diffusion-derived tensors that still need a conductivity magnitude, which
anicond's output already has. SimNIBS still interpolates the tensor onto the
WM/GM elements, rotates its components into world space, and fills any
all-zero or negative-semi-definite voxel with the tissue's table value; no
diffusion-to-conductivity conversion happens anywhere in this task.

**Out:** the induced E-field on the whole head mesh (`.msh`, a few hundred
MB), the field on the **central, pial and white** surfaces as FreeSurfer
overlays (central: magnitude, normal, tangential, angle — SimNIBS's own;
pial and white: magnitude and normal, computed by the worker on the mesh,
since SimNIBS's `depth` argument is a no-op in 4.6 and it only ever writes
the central one), the field resampled onto a NIfTI grid in gray *and* white
matter (`tissues_in_niftis = [1, 2]`), a `vtk/` folder for 3D Slicer, four
review pictures, `<id>_field_stats.csv` and `<id>_run_report.docx`.
SimNIBS's `fields_summary.txt` is *not* kept as a file: its two tables are
parsed into the report's Result section and the file is removed. Everything
is zipped as `<id>_mesh_sim_<conductivity type>___output.zip` —
`IXI034_mesh_sim_vn___output.zip`, the archive convention below — with the
files at the archive's root, so it unpacks into a folder of that name. The
model in the name is what keeps four runs of one subject apart: the file
names *inside* are identical from run to run. Task 5 reads the subject and
the model off the white VTK's name inside.

The pial surface reads the field from the gray-matter tetrahedra (as
SimNIBS's central interpolation does); the white surface reads it from the
**white-matter** tetrahedra — it is the white matter's surface, and what a
tract reaching it sees. A vertex a hair outside its tissue takes the nearest
element's value.

The `vtk/` folder holds the surfaces carrying the field as `SPACE=RAS`-tagged
polydata (both hemispheres merged, ~25 MB each): the central surface with
magnitude and, separately, the normal component, plus the pial and white
surfaces with magnitude (`<id>_<type>_both.{central,pial,white}.magn.vtk`).
Each carries the field **twice**: as a named float array a viewer can remap,
and as explicit RGB in an unsigned-char `Colors` array set as the active
scalars, so the file opens in the colour scale the mosaics use — jet for a
magnitude, coolwarm about zero for the signed normal component — instead
of the viewer's default grey (3D Slicer maps a three-component
unsigned-char scalar directly). The window is the mosaics' own (2nd to
99.5th percentile of the values, symmetric for the normal component) and
is written into the header line, e.g. `... SPACE=RAS jet 2.56..99.8 V/m`,
so each file states its own scale. And `<id>_<conductivity type>_coil.vtk`:
the coil's **real casing**, read out
of SimNIBS's own binary `.stl` and transformed by this run's `matsimnibs`,
with its axis drawn as a 60 mm line pointing into the head. Load it beside
the T1 and the field NIfTI and the coil sits exactly where it sat in the
solve — the casing lies wholly on the away-from-head side of the contact
plane, so the gap you see is the coil-to-scalp distance you asked for.

The pictures are `<id>_field_cortex_with_coil_position_<conductivity type>.png`
(jet, with the percentile table, and the coil ring and its axis drawn on
top), `<id>_field_gray_white_matter_volume_<conductivity type>.png` (the
field NIfTI over the T1 in both tissues, the white surface's cut drawn in
white as the gray/white boundary), `<id>_field_white_matter_surface_...png`
(the white-surface overlay), `<id>_coil_position_<conductivity type>.png`
(one axial slice — the one through the coil centre, not a mosaic), and —
anisotropic runs only —
`<id>_conductivity_tensor_orientation_<conductivity type>.png`. Mosaics are
rendered at `FIGURE_DPI` (200) — about 2600 px across — with slice labels at
`SLICE_FONTSIZE`.

**The run report is a `.docx`** (python-docx), because everything in it is a
table: the stimulation settings, every conductivity the solver ran with
(tissue, tag, table row, value, source), the app's per-tissue percentiles
(p99 / p99.5 / p99.9, max, mean, volume), SimNIBS's own percentile and
focality tables, the tensor's mean anisotropy for an anisotropic run, and a
file guide limited to what is read, viewed or analysed (no job spec, gmsh
side files or logs — those stay in the folder and in the page's downloads
table). Job ids, staged input paths, archive and log paths and the solver's
eigenvalue QC line are deliberately not in it.

**Every mosaic, VTK filename and figure title carries the conductivity
type**, not just the archive's own name: these are also offered as loose,
individually-downloaded files (the downloads table links straight into
`outputs/`, not just the zip), and two runs of the same subject under a
different model would otherwise both write `IXI034_field_cortex.png` and
overwrite each other in a downloads folder — the same reasoning the
archive's own `<id>_mesh_sim_<conductivity type>___output.zip` name
already applied. The
VTK files' internal header (the legacy-format title line every viewer,
Slicer included, can show) inherits the same tag, since it is built from the
now-tagged filename rather than set separately. On the figure itself the tag
is on the TITLE, appended as `[vn]` — deliberately not folded into the
subtitle text, since every mosaic's subtitle already says something
different (coil settings, the tensor's own name, nothing at all for the
coil-position picture before this), so the title is the one place any
picture from this task can be traced back to which conductivity model made
it, subtitle wording aside (`service._titled()`).

**Canvas views** (six): the head model's T1; the axial slice the coil sits
on, with its centre marked and axis drawn; the field on the cortex with the
coil overlaid; the field on the white surface; the volume field across gray +
white matter with the gray/white boundary drawn; and the conductivity
tensor's own orientation (anisotropic runs only, empty otherwise). Every
canvas view is produced by the *same* function as the saved PNG — only the
figure size and DPI differ (`_figure_kwargs(saved)`), so what you review on
screen is what lands in the zip.

**Checking the tensor itself, not just its effect.** An anisotropic run
consumes a tensor as a black box — nothing about a wrong result would tell
you whether the *tensor* was the problem. `<id>_conductivity_tensor_orientation.png`
is a QC picture of the upload, not of the simulation: the tensor's principal
eigenvector, direction-encoded the way every DTI viewer does it (red =
left–right, green = anterior–posterior, blue = superior–inferior), with
opacity carrying the tensor's own anisotropy index (0 = isotropic, fades into
the T1, so a near-degenerate voxel's numerically noisy direction is not drawn
at full strength). A flipped axis or a mislabelled tissue mask upstream shows
up here as fibre directions that do not look like fibres — the corpus
callosum should read red, the corticospinal tract blue — long before it
would be visible in the E-field result. It is drawn on the tensor's OWN grid
(task 2's DWI/b0 space, not the head model's T1 grid — the T1 and the tissue
mask are resampled onto it, the cheaper direction since it is usually the
coarser of the two), and only over voxels the model actually assigned a
tensor to (by default just white matter — gray matter draws nothing unless
the model touched it, rather than being read as falsely isotropic).

**Percentiles and the results CSV.** After the solve the worker re-reads the
mesh and computes, per tissue (gray matter tag 2, white matter tag 1), the
99th / 99.5th / 99.9th percentile of `|E|` and the distribution of everything
*above* each threshold — min, q1, median, q3, max, mean, element count and
volume. Percentiles are **volume-weighted** the way SimNIBS weights them
(cumulative element volume), which is why the gray-matter p99 reproduces
`fields_summary.txt` to the digit; a tetrahedral mesh has wildly varying
element sizes and an unweighted percentile would over-count wherever the mesh
is fine. Those numbers are annotated in the corner of the cortex, volume and
white-surface figures and tabulated in the run report; `<id>_field_stats.csv`
carries just the threshold, in long form — five columns, `subject_id`,
`conductivity_type`, `tissue`, `percentile`, `threshold_v_per_m`, one row per
(tissue × percentile). Concatenate the CSVs across subjects and each plot in
the analysis plan is a `groupby` away. The tail summaries and the tensor's
anisotropy index (voxel-based and unweighted: a NIfTI's voxels are all the
same size, unlike a FEM mesh's elements) are in the report, not the CSV.

**dI/dt and the exposure guideline.** The default dI/dt is 6 × 10⁶ A/s
(`DEFAULT_DIDT`), anchored to ICNIRP 2010 (Health Phys 99(6):818–836)
rather than to a stimulator's ceiling. The simulation stands at the
frequency the conductivity table is tabulated at (10 kHz, shown under the
10-20 picture on the page); ICNIRP's basic restriction on the internal
E-field in CNS tissue of the head is 2.7 × 10⁻⁴ f V/m rms for occupational
and 1.35 × 10⁻⁴ f for general-public exposure in 3 kHz–10 MHz (Table 2) —
2.7 and 1.35 V/m at 10 kHz — evaluated as the 99th percentile of the
tissue's field averaged over 2 mm cubes, and for a pulse as a weighted
peak, √2 × the rms value (appendix). On the reference subject IXI258
(MCF-B65 at C3, any conductivity model) the gray-matter p99 is ~90 V/m at
the coil's 100 % MSO (147.86 × 10⁶ A/s), so the occupational restriction
sits at ~6.2 × 10⁶ A/s, rounded to 6 × 10⁶ (about 4 % MSO). Occupational
rather than public because that level is ICNIRP's "no myelinated-nerve
stimulation" level (a 4 V/m threshold with a reduction factor of 5), a
physical anchor for a TMS field; the public level adds one more policy
factor of 2 (3.1 × 10⁶ A/s). Because |E| is linear in dI/dt, every run
reports its **own** figure: the console, the summary and the report's
"Exposure guideline" section give this run's p99 in gray and white matter
as a multiple of each restriction and the dI/dt at which it would exactly
meet it. Two things the guideline says that matter here: medical exposure
is excluded from it by definition, and a source a few centimetres from the
body is to be assessed dosimetrically against the basic restrictions,
never against the reference levels (which assume a uniform field).

**ICNIRP exceedance mosaic.** `<id>_icnirp_exceedance_<type>.png` (and the
"ICNIRP exceedance" Show option) draws |E| in gray and white matter divided
by the ICNIRP 2010 occupational weighted-peak restriction at the table's
frequency (3.82 V/m at 10 kHz), only where that ratio is above 1, in a
"hot" ramp starting at 1 and saturating at the 99.5th percentile of the
ratio, with the gray/white boundary in white and the fraction of the GM+WM
volume over the restriction annotated. It is a map of the margin between
the intended stimulation and the guideline's incidental-exposure level,
not a compliance statement — ICNIRP excludes this exposure.

**Four more files** (2026-09-16). `vtk/<id>_<type>_coil_axis.vtk`: the
coil's position without its casing — a 4 mm sphere at the coil centre, a
60 mm tube of 1.5 mm radius along the coil axis into the head and a 40 mm
tube of 1 mm radius along the handle, with a `part` scalar (0/1/2) to
colour them apart. Solid geometry rather than a line on purpose: a VTK
line has no thickness and 3D Slicer draws it one pixel wide at any zoom.
The handle tube is drawn along *minus* `matsimnibs`'s 2nd column
(`coil_axis_vtk()`, fixed 2026-09-17): SimNIBS's own coil convention
(`coil_axesorientation.pdf`, `calc_matsimnibs` in `mesh_io.py`: `y =
pos_ydir - centre`) has the coil's +y axis point *away* from the physical
handle, not along it, so drawing the raw column pointed the tube into the
coil casing instead of out along the handle a reader would hold.
`subject_volumes/<id>_icnirp_exceedance_<type>.nii.gz`: the exceedance
ratio as a volume on the field grid, 0 where the ratio is at or below 1.
For an anisotropic run, `subject_volumes/<id>_gm_wm_labels.nii.gz`: white
(1) and gray (2) matter from charm's `final_tissues` as a uint8 label map
on the field grid; and `subject_volumes/<id>_conductivity_tensor_orientation_rgb_<type>.nii.gz`:
the tensor's principal eigenvector direction-encoded and scaled by the
anisotropy index, as an RGB24 NIfTI (datatype 128) on the tensor's own
grid — the QC mosaic as a volume, which Slicer opens as a colour volume.

**dI/dt is capped at the coil's own ceiling.** Every coil file in the
Drakaki 2022 folder carries its measured 100% MSO `dIdtmax`; the page reads
it for the selected coil (`service.coil_didtmax`), shows it in the box's
label — `dI/dt (A/s) ≤ 1.479e+08 (MagVenture MCF-B65 at 100% MSO)` — and
enforces it as the box's maximum. Choosing another coil relabels the box
and pulls the value down if it is now above the new ceiling. `run_job`
applies the same cap with a console warning, so a CLI caller cannot ask
for a dI/dt no stimulator reaches with that coil.

**How it runs:** `service.py` unpacks the archive, writes a plain-JSON job
spec, and runs `worker.py` under SimNIBS's own interpreter
(`SIMNIBS_PYTHON`), streaming its stdout into the console. The worker builds a
`SESSION` + `TMSLIST` + `POSITION`, turns the two electrode names into the 4×4
`matsimnibs` that places the coil, solves, and lets SimNIBS map the result onto
the cortex and onto a volume. It imports nothing from this repository — only
two JSON files and stdout cross between the interpreters, because `simnibs`
cannot be imported from the environment this app runs in.

**The stimulation frequency is on the page.** SimNIBS's quasi-static
solve has no frequency of its own; the one frequency the simulation stands
on is the frequency the tissue conductivities are tabulated at (the
table's `conductivity_iso_<freq>` column, 10 kHz with the shipped table).
`service.conductivity_frequency()` reads it and the page prints it under
the 10-20 reference picture, so a run's field values can be read against
an exposure limit at the right frequency.

**Handle angles are computed, not tabulated.** For the chosen centre, each
nearby electrode's direction is projected onto the scalp's tangent plane there
and measured against the nasion→inion axis, so the dropdown reads
`CP5 (≈43° posterolateral)` for this subject rather than a figure from a
reference head. Medial versus lateral comes from the coordinates rather than
the angle's sign, so it stays right on either hemisphere (`C3 → CP5` and
`C4 → CP6` both read "posterolateral"), and a midline centre says left/right
instead. A run takes about 3 minutes: the FEM solve is ~7 seconds, and the
post-processing is the rest.

**One SimNIBS gap worth knowing:** `simnibs.mesh_io.write_vtk` needs `pyvista`,
which this install does not ship, so the *volumetric* result is not converted
to VTK. It stays a `.msh`, which is gmsh's own format and what SimNIBS's viewer
opens. The Slicer-ready artefacts are the surfaces above, built in this
app's environment from the gifti surfaces and the curv overlays
(`vendor/gii2vtk`).

**Manual check** — what to read by hand if the AI-written code is in doubt:

- `tasks/mesh_sim/worker.py` (runs under SimNIBS's Python) —
  `make_tmslist()`: `correct_intensity=False` and the caps lifted when
  `TENSOR_AS_IS`; `apply_conductivities()`: table values replace SimNIBS's,
  a tag with no row keeps SimNIBS's; `tissue_field_stats()`: volume-weighted
  percentiles; `surface_overlays()`: pial reads tag 2, white reads tag 1.
- `tasks/mesh_sim/service.py` — `run_job()`: the dI/dt cap
  (`coil_didtmax()`) and the label-vs-tensor warning; `conductivity_table()`:
  `CONDUCTIVITY_FREQ` and the `simnibs_tissue` join; `handle_choices()` /
  `_direction_word()`: the angle and the medial/lateral wording;
  `icnirp_basic_restriction()` / `guideline_check()`: the ICNIRP numbers;
  `coil_vtk()`: the casing under this run's `matsimnibs`;
  `write_surface_vtk()` / `field_window()`: the RGB `Colors` array and the
  2–99.5 % window.
- `vendor/gii2vtk/convert.py` — `read_gii_surface()`: the vertices/triangles
  `write_surface_vtk()` builds a fresh polydata from (this task writes its
  own VTK per surface rather than calling `write_vtk_polydata()`, since it
  attaches the field, not sigma, as the scalar).
- `tasks/mesh_sim/settings.py` — `DEFAULT_DIDT`, `DEFAULT_DISTANCE_MM`,
  `DEFAULT_ANISOTROPY_TYPE` (`dir`), `TENSOR_AS_IS`, `FIELD_PERCENTILES`,
  `ICNIRP_CNS_HEAD_RMS`.

## Task 5 — TMS field → tracts  (`/tract_efield`, hidden)

**Not on the menu**: registered with `visible=False`, so it is reachable
only by its URL. It is the analysis step behind the study's anisotropic
question — *which tracts reach the part of the white-matter surface the coil
drives hardest?* — and it runs in the app's own environment on files the
other tasks already produced; SimNIBS is not involved.

**In:** the archive task 4 produces
(`<id>_mesh_sim_<conductivity type>___output.zip`) and
the whole-brain tractogram of the same subject (task 2's
`<case>_wb_tracts.vtk`, or any `.vtk` / `.vtp` / `.trk` / `.tck` in the same
space). From the archive it reads the white surface carrying `|E|`
(`vtk/<id>_<type>_both.white.magn.vtk`), the field NIfTI and the T1
(`subject_volumes/`). The subject id and the conductivity type are read
off the archive's own name and index the moment it is chosen and shown in
one line under the drop zones — there is nothing to type or pick. Archives
from before 2026-09-14 lack the white VTK and are refused with a message
saying so.

**The space question.** The tractogram is in the DWI's space (task 2) and
the simulation in the head model's — which task 3 rewrote into the DWI's b0
space, so the two overlay with no transform *provided the head model was
built with its b0 given*. The run does not assume it: it reports the fraction
of streamline points inside the field grid and the fraction of streamline
endpoints within 5 mm of the white surface (a tractogram in the wrong space
has its endpoints anywhere), and warns below 90 % / 50 %
(`MIN_GRID_OVERLAP`, `MIN_ENDPOINT_NEAR_FRACTION`). On IXI258: 100 % and
65 %, median endpoint distance 2.8 mm — the 35 % are streamlines ending in
the brainstem, cerebellum and deep gray, which have no white surface.

**Hot spot.** The white-surface vertices whose `|E|` is at or above the
chosen percentile (default 99.5, `DEFAULT_PERCENTILE`) of the surface's own
distribution — **area-weighted**, each vertex carrying a third of its
triangles' area, by the same cumulative-weight rule SimNIBS and task 4's
worker use for volume percentiles. An absolute threshold in V/m can replace
the percentile (for a cohort compared at one field strength). Note this is
a percentile of the *surface*, not of the white-matter volume task 4's CSV
reports: on IXI258 `tbe`, 98.7 V/m on the surface against 92.4 V/m in the
volume.

**Reaching it.** A streamline is selected when any point along it (default)
or either endpoint lies within the contact distance (default 3 mm,
`DEFAULT_CONTACT_MM`) of a hot-spot vertex — a KD-tree query, so the whole
selection takes a second. "Any point" counts a fibre passing beneath the
spot as well as one ending there; "either endpoint" keeps only tracts that
terminate at it (CSA streamlines stop where FA drops, just inside the
gray/white boundary, which is why 3 mm reaches their endpoints). On IXI258
with C3 → CP1: 3,623 of 98,989 streamlines by the any-point rule (2,667 by
the endpoint rule), fanning from the left motor hot spot toward the
internal capsule.

**Following broken streamlines** (off by default, a checkbox). Tractography
breaks a long tract into pieces — a stopping criterion fires mid-tract and
the next seed picks it up a step later — so the piece that touches the hot
spot may say nothing about where the tract goes. With the option on, a
streamline whose endpoint lies within the gap (default 2 mm) of an endpoint
of a selected one **and continues it head-to-tail** (the two exit tangents
within 45°, `JOIN_MAX_ANGLE_DEG`) is added; repeated from the new pieces up
to the hop limit (default 3). The direction test is what keeps this from
swallowing the cortex: near the gray/white boundary thousands of streamlines
end side by side, and two neighbours that both terminate there leave in the
*same* direction, not head-to-tail, so they are not joined. On IXI258: +746
streamlines over three hops (607, 107, 32). The hop at which each
streamline was reached (0 = touches the hot spot) is a cell scalar in the
VTK, so Slicer can colour or hide the followed pieces.

**Out.** Two sets of the selection, and the split between them is the
rule of this task: **the numbers and the pictures use the selection as
tracked; only the coloured VTKs carry a simplified copy.** The selection
as tracked — every streamline
that reaches the hot spot, plus the healed pieces — is what the counts,
`<id>_tracts_stats.csv`, `<id>_tracts_activation.csv` and the |E| /
activating-function statistics are computed on, and it is written once, as
`vtk/<id>_<type>_tracts_coloredby_orientation.vtk` (legacy polydata, LPS on
disk like every tractogram this app writes, header-tagged `SPACE=LPS`;
the local segment direction as the RGB `Colors` array — red left–right,
green anterior–posterior, blue superior–inferior — with `|E|` per point and
the `hop` per streamline as plain arrays), the file the CSVs refer to line
for line. The same selection is then simplified for a viewer through
`vendor/tractviz` at this task's own settings (`SIMPLIFY_*` in
`settings.py`: healed at gap ≤ 2 mm, ≤ 45°; fragments under 20 mm dropped;
50 % of every bundle kept; smoothed σ 1 mm — gentler than task 6's page
defaults, because a selection is already a small set of tracts and a tenth
of it was too few to look at) and that copy is what the three coloured
files carry, each with a `___simplified` suffix:
`vtk/<id>_<type>_tracts_coloredby_orientation___simplified.vtk`, the same
direction colouring; `..._tracts_coloredby_efield___simplified.vtk` — jet by
`|E|` re-sampled on the simplified points, from 0 to the 99th percentile of
|E| along the selection *as tracked* (`TRACT_COLOR_PCTL`; red is the
strongest, one outlier cannot compress the scale, and the window is the
pictures'); and `..._tracts_coloredby_activation___simplified.vtk` (below).
The simplified files carry the `pieces` each streamline was healed from
rather than a `hop`, which a healed streamline no longer has one of. There
is no white-surface VTK from this task: task 4's archive already holds the
white surface with `|E|` on it, and the hot spot is only a threshold on that
scalar. `<id>_field_with_tracts_<type>.png` — the selection as tracked
over a darkened T1, each segment coloured by direction and fading with its
distance from the slice plane inside a 5 mm slab; the gray/white boundary
as a thin grey line and the hot spot as the same line in white; healed
pieces drawn thinner. The pictures never use the simplified copy: at any
useful thinning it is too few streamlines to make a picture of a small
selection. No field map on this picture — that is
task 4's white-surface mosaic — so a hue can only mean a direction. The
title is subject · coil site (`C3 (left M1, hand area)`, read from the
archive's job spec) · threshold · "connected fiber tracts"; the numbers
sit in a strip under the tiles where they cover nothing. Strokes are thin
on purpose: at 320 dpi a tile is ~2.75 px per mm, and a 1 pt stroke
(~1.6 mm) is wider than the gap between neighbouring streamlines, so
opaque neighbours fused into filled patches; 0.45 pt with the depth fade
keeps them apart and lets overlaps add up instead of covering.
`<id>_field_with_tracts_by_efield_<type>.png` is the same mosaic with the
streamlines coloured by |E| along them instead (jet over the E-field VTK's
own window, 0 to the 99th percentile, red strongest). No colour bar: the
two pictures have identical geometry so they can be flipped between, and
the window is stated in the legend line. Saved at 320 dpi
(~4100 px across) so one slice can be read on its own. And
`<id>_tracts_stats.csv`, one row per run with nine columns (`subject_id`,
`conductivity_type`, `percentile`, `threshold_v_per_m`, `threshold_source`,
`contact_mm`, `n_streamlines`, `n_selected`, `fraction_selected`). No run
report and no job spec for this task. Zipped as
`<id>_tract_efield_<type>___output.zip`, flat. Selections larger than
`MAX_DRAWN_STREAMLINES` (4000) are subsampled for drawing only — the VTKs
and the numbers use all of them.

**Activating function.** What a fibre in the field "feels" is not |E| but
Rattay's activating function: with `E_s = E · t` the field's component
along the streamline, `f = −dE_s/ds` (V/m²) is the cable equation's source
term, positive depolarising. It is invariant to the streamline's arbitrary
direction (`E_s` and `s` flip together), so task 2's streamlines serve as
they are. At a fibre's *end* the driver is the field itself: the terminal
field `E · t_out`, `t_out` the tangent pointing out of the fibre through
that end, positive depolarising that end. Both are computed from the E
**vector** volume in task 4's archive (`subject_volumes/*_E.nii.gz`),
sampled along each selected streamline with a mask-aware interpolation
(`E·mask` and `mask` interpolated separately and divided, so a point near
the gray/white boundary reads the tissue side's field rather than a blend
with the zero the NIfTI carries outside GM+WM), `E_s` smoothed along the
fibre with a 1.5 mm Gaussian (`ACTIVATION_SMOOTH_MM`) before
differentiating, since the derivative of a 1 mm grid's interpolant is
otherwise a staircase. Outputs: `vtk/<id>_<type>_tracts_coloredby_activation.vtk`
(blue hyperpolarising, grey zero, red depolarising, symmetric window at
the 95th percentile of |f| — f is spiky, and a window at the maximum would
leave nearly every segment grey; `f`, `E_s` and |E| per point; the terminal
field at both ends and the hop per streamline),
`<id>_field_with_tracts_by_activation_<type>.png` (same geometry as the
other two mosaics, plus a dot at each fibre end coloured by its terminal
field), `<id>_tracts_activation.csv` (one row per selected streamline:
hop, length, peak `f` and where along the fibre it is, the terminal field
at both ends, max |E|), and three columns on the run's stats row
(`activation_max_v_per_m2`, `activation_p95_abs_v_per_m2`,
`e_terminal_max_v_per_m`). The field is one instant of a biphasic pulse,
so the sign of `f` reverses over the pulse: use |f| for thresholds and
read the signed map as a picture. On IXI258: f within ±2.7e4 V/m², p95 |f|
5.6e3 V/m², terminal field up to 103 V/m.

**Canvas views** (four): the T1, the connected tracts coloured by
direction, by |E|, and by the activating function. Same function as the saved PNG, as in task 4, at 150 dpi on the
canvas.

**Manual check** — what to read by hand if the AI-written code is in doubt:

- `tasks/tract_efield/service.py` — `identify_zip()` / `unpack_simulation()`:
  the archive-name pattern and the refusal of pre-2026-09-14 archives;
  `space_check()`: `MIN_GRID_OVERLAP`, `MIN_ENDPOINT_NEAR_FRACTION`;
  `hotspot_mask()` + `vertex_areas()`: the area-weighted percentile;
  `select_streamlines()`: `DEFAULT_CONTACT_MM`, any-point vs endpoint;
  `follow_breaks()` + `_exit_tangents()`: `JOIN_MAX_ANGLE_DEG`,
  `DEFAULT_JOIN_MM`, `DEFAULT_MAX_HOPS`; `activation_along()` +
  `sample_vector_field()`: the sign of `f`, the mask-aware interpolation,
  `ACTIVATION_SMOOTH_MM`; `write_tracts_vtk()`: `SPACE=LPS` and the `Colors`
  array.
- `vendor/tractviz` — `simplify_tracts()`: the selection simplified for a
  viewer at `settings.SIMPLIFY_*` (healed gap ≤ 2 mm / 45°, ≥ 20 mm, 50 %
  of each bundle, σ 1 mm); `run_job()` step 5b writes the three
  `___simplified.vtk` files from it, with |E| and f re-computed on its
  points — never an input to the selection, the numbers, the CSVs or the
  pictures.
- `tasks/tract_efield/settings.py` — `DEFAULT_PERCENTILE`, `TRACT_COLOR_PCTL`,
  `ACTIVATION_COLOR_PCTL`, `MAX_DRAWN_STREAMLINES`, `SIMPLIFY_*`.


## Task 6 — Tractogram → visual tracts  (`/tract_simplify`, hidden)

**Not on the menu**, like task 5. A tractogram as task 2 writes it is honest
about tractography — long tracts broken into pieces, thousands of
near-identical streamlines per bundle, short fragments everywhere — and it
makes a poor picture. This task writes a copy that reads well in a viewer.
**For visualization only**: the thinned copy must never go back into task 2's
tract-based model; a healed-only copy is a fair input for task 5, which does
the same join outward from its own selection.

**In:** a whole-brain tractogram (task 2's `<case>_wb_tracts.vtk`, or any
`.vtk` / `.vtp` / `.trk` / `.tck`) and a NIfTI in its space — the DWI's b0,
or task 3's T1 when the head model was built with its b0. The VTK file keeps
its coordinates in mm, so the space itself is not lost; the image is needed
for the grid the reader and writer work on and for the mosaic's background.
The subject id is the first six characters of the tractogram's name.

**Healing** (`heal_breaks`; on by default, gap 2 mm, angle 45°). Two
endpoints are candidates when they lie within the gap, belong to different
streamlines and their *outward* exit tangents (measured over 4 points in
from the end) are within the angle of head-to-tail. Candidates are scored —
distance over the gap plus angle over its limit — and matched one partner
per endpoint, best first, then one pass of length-3 augmenting paths
(`_match_endpoints`), which recovers what plain greedy loses in a dense
bundle where a piece grabs its neighbour's continuation and strands its own.
The direction test is what keeps the cortex out of it: thousands of
streamlines end side by side at the gray/white boundary, leaving in the
*same* direction, and are never joined. Chains are concatenated with points
laid across each gap; a chain that closes on itself (rare) is cut open. The
number of pieces each healed streamline was built from travels with it.

**Length cut.** Absolute, in mm (default 100), applied *after* healing.
Fragments of step-limited tracking are short in absolute terms whatever the
subject, so one threshold gives comparable pictures across a cohort; a
percentile cut would remove a fixed fraction whether or not those are
fragments, and would shift its meaning after healing.

**Thinning.** QuickBundles (dipy, already required) clusters the healed
streamlines by shape **once**, at `QB_THRESHOLD_MM` (12 mm, MDF on 12
resampled points). The slider keeps a percentage of *every* cluster (default
10 %), rounded up and never below one, so a small bundle survives thinning
that a random subsample would empty. Which members survive is fixed by one
seeded random key per streamline: the same slider position always keeps the
same streamlines, and lowering it only removes, never swaps.

**Smoothing.** A 1D Gaussian along each kept streamline, σ in mm
(default 1.5 mm; 0 = off), converted to points with the streamline's own
step.

**Run, then live sliders.** Loading, healing and clustering do not depend on
the sliders, so they run once behind Run, through the shared queue and into
a run folder like every other task. The three sliders then act on the
cached arrays on *release* (not while dragging): milliseconds of selection,
the download rewritten under the same name, the canvas redrawn. The healing
switch and its two numbers come before the clustering, so they take effect
at the next Run.

**Out:** `<case>_tracts_viz.vtk` — the kept streamlines, LPS on disk like
every tractogram this app writes, coloured by direction in an explicit RGB
`Colors` array (task 5's writer), with `pieces`, `cluster` and `length_mm`
per streamline so 3D Slicer can colour by any of them; `<case>_tracts_viz.trk`
— the same streamlines as TrackVis `.trk` on the reference image's grid,
what the 3D viewer loads (TrackVis, Slicer, DSI Studio and DIPY read it
too). Two files and their archive, nothing else: no settings dump, no
report, no saved PNG — the values are in the page's numbers line and the
run log. The archive, `<case>_tract_simplify___output.zip`, is rewritten
with the two files on every slider release, so its link is always current.

**Canvas.** One figure, nothing to pick: the original tractogram on the
left and the current revision on the right, both at the same slice — the
middle of the reference image's occupied slices (`service.middle_slice()`)
— through `preview.tract_pair_view()`, drawing at most
`MAX_DRAWN_STREAMLINES` (4000) per side. No slice number over a tile and no
separate legend: both live in the one three-line title instead —
`<subject> Processed Tracts` / the length cut, thinning and kept-of-original
count / the slice index and the colour key. Redrawn on Run and on every
slider release. Lines are drawn 50 % thicker than task 5's canvas default
(`TRACT_LINEWIDTH_CANVAS`), since two tiles side by side are narrower than
one.

**The 3D viewer** under the canvas is [NiiVue](https://github.com/niivue/niivue)
(BSD-2), one JavaScript bundle in `tasks/tract_simplify/assets/niivue.js` next to
`viewer.html`, opened in an iframe: a separate document, so its script runs
(a script inside a `gr.HTML` block does not). Both are served by Gradio's
file route because `tasks/*/assets` is in `allowed_paths`, and it serves an
allowed file inline with its own type. It loads the reference NIfTI and the
`.trk` and shows three slices plus a 3D render, streamlines coloured by
direction. **Nothing Python-side depends on it**: there is no new package
in the environment, and if `niivue.js` is missing `service.viewer_status()`
puts a one-line notice where the viewer would be and the canvas above
carries on. To install or update it:

    curl -L -o tasks/tract_simplify/assets/niivue.js \
         https://unpkg.com/@niivue/niivue@0.69.0/dist/index.js

**Manual check** — what to read by hand if the AI-written code is in doubt:

- `vendor/tractviz/simplify.py` — the algorithms, shared with tasks 2 and
  5's simplified exports: `heal_breaks()` + `_match_endpoints()`: the
  head-to-tail test (`dot ≤ −cos(angle)`), one partner per endpoint, the
  cycle cut; `cluster_streamlines()`: `QB_THRESHOLD_MM`, `QB_POINTS`;
  `thin_mask()`: ceil of the fraction, never below one, seeded key;
  `prepare()` / `select()` are the two halves, `simplify_tracts()` both in
  one call with the module's defaults (the page's defaults).
- `tasks/tract_simplify/service.py` — `run_job()`: `prepare()` once behind Run;
  `apply()`: `select()` on release, length cut before thinning, the VTK and
  `.trk` rewritten each time; `render_canvas()`: the three-line title,
  `preview.tract_pair_view()` for the two panels; `viewer_url()`: the two
  file URLs and the cache-busting query.
- `tasks/tract_simplify/web.py` — the sliders are bound to `release`, Run to the
  queued `run_job()`.
- `tasks/tract_simplify/settings.py` — the slider ranges and the canvas; the
  defaults are read from `vendor/tractviz`.

REQUIREMENTS
gradio>=5.6            # demo.route() multipage needs 5.6+; developed on 6.22
numpy
scipy
pandas
PyYAML
nibabel
SimpleITK
dipy                   # tensor fit, CSA tractography, tractogram IO
vtk                    # .vtk tractogram reader/writer (tasks/anicond/tractio.py)
matplotlib             # preview mosaics (Agg backend, no display needed)
# Browser-side only, no Python package: tasks/tract_simplify/assets/niivue.js
# (NiiVue 0.69, BSD-2) for task 6's 3D viewer; optional, see its section.

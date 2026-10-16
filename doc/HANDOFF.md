# HANDOFF — APP_BRAINE

Context dump for continuing this work in a fresh chat. Written 2026-09-02.
`README.md` is the user-facing doc; this file is the "what you need to know
before touching the code" that does not belong there.

---

## 1. What this is

`/home/my/Documents/APP_BRAINE/` — a Gradio web app that puts four brain
conductivity pipelines behind one menu on **one port**. Built from scratch in
this conversation.

| Task | Page | State |
|---|---|---|
| T1/T2 → conductivity map | `/isocond` | working, verified |
| DTI → anisotropic conductivity tensor | `/anicond` | working, verified |
| T1/T2 → head mesh (charm) | `/mesh_head` | working (charm subprocess) |
| head mesh → TMS simulation | `/mesh_sim` | working (SimNIBS subprocess) |
| TMS field → tracts | `/tract_efield` | working; HIDDEN (visible=False) |
| tractogram → visual tracts | `/tract_simplify` | working on synthetic data; HIDDEN (visible=False) |

Run: `~/anaconda3/envs/myenv/bin/python run_web.py --host 0.0.0.0 --port 7778`

**There is no test suite in the repo.** One existed (44 stdlib-`unittest` tests
under `tests/`) and the user removed it deliberately after asking whether it was
needed — do not re-add one unasked. The verification numbers in §5 were measured
before it went, and remain valid as facts about the code as it stands.

## 2. Environment facts

- **Use `~/anaconda3/envs/myenv/bin/python`** (3.12). It has gradio 6.22, dipy,
  nibabel, SimpleITK, torch, vtk, matplotlib. The base anaconda python (3.13)
  does **not** have dipy and cannot run the app.
- **pytest is NOT installed** in that env — relevant if you are ever asked to
  add tests. Use stdlib `unittest`, or install pytest first.
- **`simnibs` cannot be imported from `myenv`.** SimNIBS 4.6 ships its own
  Python 3.11 at `~/SimNIBS-4.6/simnibs_env`. Tasks 3 and 4 must therefore subprocess
  into `~/SimNIBS-4.6/bin/simnibs_python`, never merge the dependency trees.
- The user runs their own server from this repo on **port 7777**. Don't kill it.
  Use another port (7899 was used for testing) and clean up after yourself.
- `gr.Textbox` in this gradio has no `show_copy_button` kwarg.

## 3. Architecture and the rules

```
run_web.py        the ONLY exposed entry point
settings.py       global: server, data root, SIMNIBS paths, logging
app/              the shell — knows pages, knows no science
  registry.py       TASKS list -> home menu + mounted routes
                    (enabled: page exists; visible: on the menu + navbar)
  jobs.py           run folders, FIFO queue, Transcript/log capture,
                    the usage log (data/usage_log.csv, one row per request;
                    web/common.stream_job appends + completes it, the
                    queue stamps started_at)
  report.py         .docx run report style (new_document / add_table)
  web/main.py       home menu (visible tasks) + one demo.route() per
                    enabled task, show_in_navbar=task.visible; a hidden
                    page also gets a footer line linking the OTHER hidden
                    tasks (_hidden_links), visible pages never do
  web/common.py     page chrome, session count, progress box, stream_job
tasks/<task>/     service.py (UI-agnostic seam) + web.py (widgets only)
                  + settings.py; cli.py/gui.py may be added later
                  anicond also has runctx.py (per-job config) and preview.py
                  (figures; returns PIL, never widgets)
vendor/           frozen library snapshots (see §4)
data/runs/<task>/<job id>/{inputs,outputs,run.log,*.zip}
```

Rules that were agreed with the user and should be preserved:

1. **One file per front-end, all thin, all calling `service.py` and nothing
   else.** `service.py` returns paths and values, never widgets. This is what
   would let a `cli.py` reproduce a web run byte for byte.
2. `app/jobs.py` sits **above** `app/web/`, not inside it, because a future
   `cli.py` needs the same run-folder layout and progress reporting.
3. Adding a task = one folder under `tasks/` + one line in `app/registry.py`.
4. `cli.py` / `gui.py` are **not** exposed by `run_web.py` — run directly from
   their folder. The user was explicit about this.
5. Every runnable script carries the 4-line `sys.path` bootstrap (the same
   convention the user's research repos use), so cwd never matters.
6. Never edit `vendor/` to fix a task; fix it in the task layer.

**sys.path bootstrap is exactly:** repo root + `vendor/anicond`. See §4.

## 4. vendor/ — two snapshots, imported two different ways

`vendor/VERSION.md` is the authoritative record of what was copied and every
change made. Read it before re-syncing. Summary:

| folder | contains | on sys.path? |
|---|---|---|
| `vendor/anicond/` | `core/`, `cond_models/` for task 2 | **yes** — added directly, so its files keep their original `from core import general as gnr` imports untouched (smallest diff = cheapest re-sync) |
| `vendor/isocond/` | `nets/`, `pipeline/`, `settings.py` for task 1 | **no** — it is a package imported as `vendor.isocond` |

**Why isocond could not go on sys.path:** its modules say `import settings`,
`from nets…`, `from pipeline…`. `settings` would collide head-on with this app's
own root `settings.py`. Its 20 intra-project import lines were rewritten to
relative form (`from .. import settings`, `from .base import …`). Nothing else
in those files changed. There is a test asserting the two `settings` modules
stay distinct.

Other notable vendoring decisions:

- `anicond/core/general.py` was **rewritten**: the original is a BIDS cohort
  driver. Only the ~25 dataset-free constants the libs read off `gnr` survive,
  fed by a trimmed `core/config.yaml`.
- `anicond/core/med/lib_dwi.py`: `fit_dti_tensor` (folder-scanning) replaced by
  `fit_dti_tensor_from_arrays`. The fit itself is unchanged.
- `anicond/core/cond/lib_tractbased_core.py`: dropped the `TractBasedDebugMixin`
  base (861 lines of research plotting).
- `isocond/settings.py` was trimmed to the algorithm; operator choices live in
  `tasks/isocond/settings.py`.
- **`vendor/isocond/models/*.pth` are symlinks**, not copies (122 MB). `cp -L`
  makes the app standalone. If those files ever go missing, this is why.

## 5. Verification already done — do not redo blindly

**Task 2** reproduces its reference derivatives on case IXI015 to float32
storage precision:

| output | vs reference |
|---|---|
| `d_evals`, `d_fa` | max Δ 1e-8 / 4e-7, corr 1.000000 |
| DM / VN 6×1 tensors | 0.000% of max, corr 1.00000000 |
| TBe 6×1 tensor | 0.038% of max, corr 1.000000, zero voxels off by >1% |

Timings: DM+VN job 19 s (tractography skipped). TBe job 211 s total
(121 561 streamlines in 63 s, TBe 98 s).

**Task 1 is NOT run-to-run reproducible**, and this was measured, not assumed.
The T2→T1 Mattes-MI registration uses `RANDOM` sampling with `seed=42`, but ITK's
sampling is thread-count dependent. Three runs of the *same* code gave three
different metrics (−0.4698, −0.4586, −0.4897). On the conductivity map:

| compared | corr |
|---|---|
| this app vs the stored reference output | 0.991 |
| the unmodified source code vs that same reference | 0.847 |
| this app vs a fresh run of the source code | 0.850 |

i.e. the source disagrees with its own stored output more than this app does.
The spread is the pipeline's, not the port's. Fix if ever needed:
`SetGlobalDefaultNumberOfThreads(1)`. A CondNet-TART run takes ~96 s.

Test data used: `PROJ_CONDNETS/model_runner/data/input/IXI015/` (T1/T2) and
`PROJ_DCONDNET/IXI/data_ixi/IXI015/` + `.../data_ixi_derivatives/IXI015/`.

## 6. Gotchas that cost real debugging time

1. **Headless matplotlib.** On a server, matplotlib picks an interactive
   backend — Qt, reached through the plugins `cv2` bundles — and the job dies
   with *"Could not load the Qt platform plugin xcb"* when it draws a mosaic.
   `matplotlib.use("Agg")` is forced in `vendor/isocond/pipeline/imaging.py` and
   `vendor/anicond/core/med/lib_mri.py`. **The upstream code has this bug too**
   (confirmed by running it); `MPLBACKEND=Agg` works around it there.
2. **Gradio serves a SEPARATE config per page**, containing only that page's
   dependencies. So `load` / `Timer.tick` / `unload` must be registered *inside
   each route*, next to the component they update. Registering one app-wide set
   looks equivalent in the combined config and is not: re-entering `with demo:`
   after the route loop attaches to whichever page was built **last**, leaving
   every other page with no load and no unload. This silently broke the footer
   session count once already.
3. **`demo.load` refuses to register outside a Blocks context; `demo.unload`
   does not care.** Easy to get a misleading result from this asymmetry.
4. **Double-logged progress lines.** A service that both `logger.info(...)`s and
   `report(...)`s a line puts it in the transcript twice. `settings.UNCAPTURED_LOGGERS`
   lists the loggers to drop from the page box (terminal still gets them). Add
   any new double-reporting service module to that tuple.
5. **A page that fails to build used to mount as a blank route** — the app
   started, the nav entry appeared, and nothing said why the page was empty.
   Now `app/web/main.py` records the exception in `BUILD_ERRORS` and renders an
   error panel on that page. Check `BUILD_ERRORS` after `build_app()` when
   something looks missing.
6. **Extra DWI volumes.** A DWI upload can carry more volumes than the gradient
   table describes (a trailing QC volume). `load_dwi_series` drops them **by
   filename index**, never by position.
7. Task 2's segmentation reaches the b0 grid one of two ways, and the page's
   "already in the DWI's space" checkbox decides which: ticked, it is only
   resampled (nearest-neighbour); unticked, the page asks for the T1 it was
   drawn on, registers that to the b0 rigidly (b4's parameters) and carries the
   labels by that transform. Either way the job reports the anisotropic
   tissue's overlap with the brain mask, so a segmentation still in the wrong
   space is loud rather than silent.

## 7. Open items

- **Task 3 (`mesh_head`) is implemented** (2026-09-08): T1 (+T2) → charm →
  `m2m_<id>/`, optional rigid registration into the DTI b0's space done by
  rewriting the NIfTI headers (no resampling), charm run as a subprocess with
  `--forceqform` and a patched `charm.ini` (thread count). Verified end to end
  on IXI034 with T2 and an unaligned b0. charm's qform/sform check is strict
  (1e-6): the copies handed to charm always have qform = sform written from
  one matrix, and `--forceqform` absorbs quaternion rounding.
- **Task 4 (`mesh_sim`) is implemented** (2026-09-10): the head-model zip from
  task 3 → SimNIBS TMS simulation → field on the cortex and on a NIfTI grid,
  plus a Slicer-ready cortex VTK. Verified end to end on IXI034 (C3 → CP5,
  MagVenture MCF-B65): peak |E| 1.16 V/m in gray matter at dI/dt 1e6 A/s, the
  hot spot correctly on the left hemisphere. `simnibs.mesh_io.write_vtk` needs
  pyvista (not installed — the volume stays a .msh).
- **Task 4 outputs extended for cohort analysis** (2026-09-11):
  - **Percentiles.** `worker.tissue_field_stats()` computes p99 / p99.5 /
    p99.9 of `|E|` in gray matter (tag 2) and white matter (tag 1) plus the
    full distribution of the tail above each threshold. Weighting is by
    element volume, replicating SimNIBS's own cumulative-weight rule in
    `_weighted_quantile()` — **verified** against `fields_summary.txt` (GM p99
    = 97.31 V/m both ways), and internally consistent (each tail's volume is
    exactly 1 % / 0.5 % / 0.1 % of tissue volume, and the median of the p99
    tail equals the p99.5 threshold).
  - **`<id>_field_stats.csv`**, long format, one row per (tissue ×
    percentile), 26 columns, led by `subject_id` and `conductivity_type`.
    Designed to be concatenated across subjects and box-plotted with a
    `groupby` — that is the whole point of the long format.
  - **`conductivity_type`** (`iso` / `dm` / `vn` / `tbe`) is an **explicit
    dropdown**, the user's choice: `dm` / `vn` / `tbe` are *anicond*'s model
    short names (`DM`, `VN`, `TBe`), which collide confusingly with SimNIBS's
    own `vn` anisotropy mode, so nothing is inferred. `run_job` only warns
    when the label and the run disagree.
  - **Archive naming** (changed 2026-09-17, twice): the file is
    `<id>_mesh_sim_<conductivity type>___output.zip` —
    `jobs.output_zip_name(slug, subject_id, model)`, the
    `<subject>_<slug>[_<model>]___output` convention mesh_sim, tract_efield
    and tract_simplify follow and every new task should (isocond / anicond /
    mesh_head still pass their own `archive_name`) — and FLAT inside: no
    nested folder (the earlier `<id>_tms_<type>/` nesting is gone; the
    `nest` / `nest_folder` options of `RunDir.zip_outputs` remain for an
    archive some reader still expects nested). tract_efield's
    `identify_zip()` reads the subject and the model off the white VTK's
    name inside (`<id>_<type>_both.white.magn.vtk`) and still honours the
    old nested folder. `run_job` pins `conductivity_type` to a known tag
    before it reaches the archive's name.
  - **`vtk/<id>_coil.vtk`**: the coil's real casing from SimNIBS's binary
    `.stl` (`read_binary_stl`), transformed by `matsimnibs`, plus a 60 mm axis
    line, written as one legacy polydata with `SPACE=RAS`. Verified: the STL
    is already in mm (extent 172×210×53, same units as `matsimnibs`, no
    scaling), and the whole casing projects 0 to −53 mm along the into-head
    normal — i.e. wholly on the away-from-head side, face exactly on the
    contact plane.
  - **Figures**: `FIGURE_DPI = 200` (≈2600 px wide, was ≈1400),
    `SLICE_FONTSIZE = 10.5` (+50 %), `FIELD_CMAP = "jet"`, plus
    `_field_cortex_with_coil_position.png` and the white-matter volume and
    surface mosaics. Canvas and saved PNG come from the *same* function;
    `_figure_kwargs(saved)` is the only difference between them, so a change
    to one is a change to both.
  - Two things that had to be fixed by looking at the pictures: doubling the
    coil stroke filled an 8 pt ring into a solid green dot (marker *size* now
    scales with stroke width), and the white-matter volume mosaic floated on
    white with no anatomy (`field_volume_mosaic()` now overlays the masked
    field on a dimmed T1, like every other mosaic in the app).
- **Conductivities no longer SimNIBS's defaults** (2026-09-11): every
  `mesh_sim` run now takes its isotropic per-tissue values from anicond's
  target table (`tasks/anicond/assets/default_target_conductivity.csv`), so
  an `iso` run and a tensor run of one subject share one reference set.
  - The table gained a `simnibs_tissue` column (`Blood_in`→`Blood`,
    `Cancellous_bone`→`Spongy_bone`, `Cortical_bone`→`Compact_bone`,
    `Eye_humour`→`Eye_balls`, `GM`, `WM`, `CSF`, `Scalp`, `Muscle`, `Fat`;
    blank for Cerebellar / Brainstem / Thalamus / CSF_Meninges /
    Mucous_membrane, which charm does not label). anicond's parser ignores
    it; `runctx.parse_simnibs_tissues()` reads it and
    `mesh_sim.service.conductivity_table()` joins it to SimNIBS's tags
    (`settings.SIMNIBS_TISSUE_TAGS`). `tasks/anicond/__init__.py` now does
    the `core` sys.path bootstrap so `runctx` imports from anywhere.
  - The values travel in the job spec (`"conductivities"`), the worker sets
    `tmslist.cond[tag-1].value` and reports back every slot it ran with
    (`result["conductivities"]`, source `table` / `simnibs_default`) — the
    run report lists them. `Bone` (tag 4) and `Cartilage` (tag 11) have no
    table row and keep SimNIBS's value; the console says so.
  - **Tensor as-is** (`settings.TENSOR_AS_IS = True`): under `dir` the
    worker's `AsIsTMSLIST` subclass calls `cond_utils.cond2elmdata` with
    `correct_intensity=False` and `max_cond = max_ratio = inf`, so the
    uploaded conductivity tensor is neither rescaled to SimNIBS's isotropic
    reference nor eigenvalue-clipped. Verified on IXI034's mesh and DM
    tensor: the as-is WM tensors reproduce the file's own values, stock
    `dir` rescaled them (see README, Task 4, "Conductivities").
- **Anisotropic-run QC added** (2026-09-11): before this, an uploaded
  conductivity tensor was consumed as a black box — nothing in the output
  ever showed what SimNIBS actually did with it, so a flipped axis or a
  mislabelled tissue mask upstream would only be visible (if at all) as an
  E-field result that looked subtly wrong.
  - **`<id>_conductivity_tensor_orientation.png`** (new file role
    `tensor_orientation_png`, new canvas view): the tensor's principal
    eigenvector, direction-encoded the DTI-viewer way (R=left-right,
    G=anterior-posterior, B=superior-inferior), opacity = the tensor's own
    anisotropy index so a near-isotropic voxel's numerically noisy direction
    fades into the T1 instead of being drawn at full strength.
    `preview.load_tensor6()` / `tensor_eigen()` (batched `np.linalg.eigh` on
    the reconstructed 3x3 per voxel) / `tensor_orientation_mosaic()`.
    Verified on a synthetic tensor (an anisotropic stripe with a known
    x-aligned axis inside an isotropic background): principal eigenvector
    resolves to exactly `[1,0,0]` in the stripe, anisotropy index 0.0 in the
    background vs 0.88 in the stripe. Verified again on the real IXI034 TBe
    tensor, which is the check that matters: corpus callosum reads red,
    corticospinal tract and corona radiata blue, SLF and cingulum green,
    brainstem blue -- and the anisotropy index splits 0.48 (WM) against
    0.076 (GM), so the tissue masking is landing where it should.
  - **Frame**: the six numbers in a voxel are a matrix in the FILE's voxel
    frame, and `as_closest_canonical` reorients the array's axes without
    touching them -- so `load_tensor6()` does both: reorients the array AND
    rotates the components into world RAS with SimNIBS's own rule
    (`_fsl_world_rotation`, copied from `cond_utils.cond2elmdata`'s
    `correct_FSL` branch; anicond's eigenvectors come straight from dipy's
    fit, in the DWI voxel frame, which is what that rule expects). Verified
    against an independent computation of SimNIBS's frame on the real
    IXI034 TBe tensor: max RGB difference 2e-14. Without the rotation the
    map is still right for a near-axis-aligned acquisition -- a pure axis
    flip cancels under the `abs()` a direction-encoded colour takes -- and
    drifts with the obliquity otherwise, which is a QC picture trustworthy
    only when it is not needed.
  - Drawn on the **tensor's own grid** (task 2's DWI/b0 space), not the head
    model's T1 grid — the two are never the same grid, and resampling the T1
    and `final_tissues.nii.gz` onto the tensor's (usually coarser) grid is
    the cheaper direction. New generic helper
    `preview.resample_to_grid(volume, affine, dst_shape, dst_affine)`, built
    on the existing `sample_volume_at()` rather than a second
    implementation.
  - Masking is by **tensor presence**, not just tissue tag: a tissue the
    model never assigned a tensor to (typically gray matter — anicond's own
    default anisotropic tissue is white matter) draws nothing there rather
    than reading as falsely isotropic.
  - **Stats**: `preview.tensor_tissue_stats()` — mean / median / p95 of the
    anisotropy index per tissue, voxel-based and unweighted (a NIfTI's
    voxels are all one size, unlike a FEM mesh's elements, so there is
    nothing for a volume weight to correct for here, unlike
    `worker.tissue_field_stats`). Three new columns on the *same*
    `<id>_field_stats.csv` row (`tensor_anisotropy_mean/_median/_p95`)
    rather than a second file, blank for an isotropic run or a
    tensor-absent tissue.
- **dI/dt default corrected** (2026-09-11): SimNIBS's own library default,
  1e6 A/s, is a nominal per-unit placeholder its FAQ explicitly says not to
  read as a real pulse — ~100x below what any commercial stimulator puts
  through a coil. Every coil file in `FIGURE8_COIL_DIR` carries its own
  measured `dIdtmax` (the Drakaki et al. 2022 dataset SimNIBS cites), 84.4-
  249.6 A/µs across the 25 coils, median 147.9. `DEFAULT_DIDT` is now
  `DEFAULT_COIL`'s own value, 147.86e6 A/s (100% MSO), which happens to equal
  that median almost exactly. Still a static, coil-independent number in
  `settings.py`, though — the honest fix is reading `dIdtmax` from whichever
  coil is actually selected, which nothing here does yet.
- **dI/dt default re-anchored to ICNIRP 2010** (2026-09-15): the user
  asked for the guideline to be investigated (PDF read page by page --
  the fetcher cannot parse it) and a default proposed. `DEFAULT_DIDT` is
  now 6e6 A/s (was the MCF-B65's 100% MSO, 147.86e6, kept as
  `DIDT_100_MSO_DEFAULT_COIL`). Anchor: Table 2, CNS tissue of the head,
  3 kHz-10 MHz: 2.7e-4 f (occupational) / 1.35e-4 f (public) V/m rms; p99
  of the tissue over 2 mm cubes; weighted peak = sqrt 2 x rms for a pulse.
  IXI258 GM p99 ~90 V/m at 147.86e6 -> occupational met at ~6.2e6.
  `settings.ICNIRP_CNS_HEAD_RMS` holds the whole Table 2 column (all bands,
  as rules in f); `service.icnirp_basic_restriction(f)` evaluates it,
  `guideline_check()` compares a run's p99 (GM, WM) with the weighted-peak
  restriction and gives the dI/dt at which the larger would meet it;
  `_guideline_lines` feeds console + summary, and the report has an
  "Exposure guideline (ICNIRP 2010)" table. The user had already added
  the frequency note under the 10-20 picture (`web._frequency_note`,
  `service.conductivity_frequency`); the ICNIRP line was appended to it.
  Caveats stated in the code: ICNIRP excludes medical exposure; reference
  levels do not apply to a coil on the head (p827: assess dosimetrically);
  SimNIBS's p99 is volume-weighted over tets, not 2 mm cubes.
- **dI/dt capped at the coil's dIdtmax** (2026-09-15): the user kept the
  raw "dI/dt (A/s)" box (a %-MSO control was proposed and declined) and
  asked for the limit shown after the label. `service.coil_didtmax(coil)`
  parses `dIdtmax=<A/us>` from the .ccd header (x1e6 -> A/s; None for a
  .tcd or an unknown file, then no cap). `web._didt_label` /
  `_on_coil_change` relabel + re-cap on the coil dropdown and clamp the
  value; `run_job` clamps too, with a WARNING line. DEFAULT_DIDT (6e6)
  left as the user set it.
- **ICNIRP exceedance mosaic** (2026-09-15): `service._exceedance_view`,
  role `exceedance_png`, view `VIEW_EXCEEDANCE`; ratio = magnE NIfTI /
  `occupational_peak()` (rms x ICNIRP_PEAK_FACTOR at the table's
  frequency), masked to ratio > 1, `preview.field_volume_mosaic` gained
  `vmin`/`vmax`/`cbar_extend` for the pinned window (1 .. p99.5). Cmap
  `settings.EXCEEDANCE_CMAP` ("hot"), group `EXCEEDANCE_GROUP`. On IXI258
  at 100% MSO 95.7% of GM+WM is above 1, max ratio 43.6 (the NIfTI's
  max, not the mesh's 449 V/m peak, which the grid does not resolve).
- **Four exports added to mesh_sim** (2026-09-16): `coil_axis_vtk`
  (vtk package: sphere + two vtkTubeFilter tubes, `part` scalar, 4.2
  layout, RAS header; a plain LINES cell is 1 px in Slicer, hence
  solids), `write_exceedance_nifti` (magnE / occupational_peak, 0 where
  <= 1), and for anisotropic runs `write_gm_wm_labels` (final_tissues
  masked to tags 1, 2, uint8) and `write_tensor_rgb` (RGB24 structured
  dtype -- nibabel needs `header.set_data_dtype(rgb_dtype)`; datatype
  128; on the tensor's canonical grid, |principal| x anisotropy index,
  masked to `_tensor_grid_arrays`' combined mask). Roles
  `coil_axis_vtk`, `exceedance_nii`, `gm_wm_labels_nii`, `tensor_rgb_nii`;
  written in run_job step 6a, each guarded.
- **`coil_axis_vtk`'s handle tube was backwards** (fixed 2026-09-17): drawn
  along raw `matsimnibs[:3, 1]`, which is SimNIBS's coil +y axis --
  `pos_ydir - centre` in `calc_matsimnibs` (mesh_io.py) -- and per SimNIBS's
  own `coil_axesorientation.pdf` that axis points AWAY from the physical
  handle, not along it. Now drawn along `-matsimnibs[:3, 1]`. No other code
  in this task derives a direction from that column (checked), so nothing
  else needed the same fix.
- **Concurrency**: I chose a single FIFO queue across all tasks. Never
  explicitly confirmed; the user has not objected.
- Task 2 → task 4 coupling is now **wired but manual**: task 4 accepts
  anicond's 6-component tensor as an upload, and records a
  `conductivity_type` label, but nothing hands a task 2 output to task 4
  automatically. The tasks stay independent, as asked.
- **Anisotropy-mapping control removed** (2026-09-11): the "Anisotropy
  mapping" dropdown (SimNIBS's `vn` / `mc` / `dir` -- what SimNIBS does with
  the tensor FILE, a different axis from `conductivity_type`, which model
  produced it) is gone from the page at the user's request: every tensor
  this app ever hands SimNIBS is a real, already-computed conductivity
  tensor from task 2, never a bare diffusion tensor SimNIBS would need to
  invent a magnitude for, so `dir` is the only correct choice and always
  was. `run_job(anisotropy_type=...)` still exists and still defaults to
  `task_settings.DEFAULT_ANISOTROPY_TYPE` (now `"dir"`, was `"vn"`) — a
  future CLI or another caller can still override it, but the web page no
  longer exposes it. `ANISOTROPY_TYPES` and the comment explaining what each
  of the three options actually does (from
  `simnibs.utils.cond_utils.cond2elmdata`) stay in `settings.py`, since the
  reasoning for ruling out `vn`/`mc` is worth keeping even unused.
- **Conductivity type tagged onto every mosaic, VTK file and figure title**
  (2026-09-11): before this, only the zip archive's own name carried it.
  - **Filenames**: all six saved PNGs (cortex, cortex+coil, WM volume/surface,
    tensor orientation, coil position) and all three VTKs this task writes
    (cortex magnitude, cortex normal, coil) now carry `_<conductivity type>`
    -- `IXI034_field_cortex_tbe.png`, `IXI034_tbe_coil.vtk`. Same reasoning as
    the archive naming two entries up: these are also offered as loose,
    individually-downloaded files (the downloads table links into
    `outputs/`, not just the zip), so two runs of one subject under a
    different model would otherwise both write `IXI034_field_cortex.png` and
    silently overwrite each other in a downloads folder. NOT applied to
    files SimNIBS itself names (`field_mesh`, `field_vec_nii`,
    `cortex_mesh`, `coil_geo`, `fields_summary`, `sim_log`) -- those come
    from SimNIBS's own naming convention and are riskier to rename.
  - The VTK internal header title (the legacy-format comment line every
    viewer can show) inherits the tag for free: both writers already set
    `title=<path>.name`, so retagging the filename retagged the header too,
    no separate edit needed. Verified on a fresh `tbe` run: `strings` on
    both `.vtk` files shows `IXI034_tbe_coil.vtk SPACE=RAS` and
    `IXI034_tbe_both.central.magn.vtk SPACE=RAS`.
  - **Figure titles**: new `service._titled(result, text)` appends
    `  [<conductivity type>]` to the TITLE (not the subtitle) of every
    mosaic this task draws -- T1, tissue segmentation, coil position, cortex
    (plain / +coil / normal), WM volume/surface, combined volume, tensor
    orientation. Deliberately the title rather than the subtitle: every
    mosaic's subtitle already said something different (coil settings, the
    tensor's own filename, nothing at all for the coil-position picture),
    so it was never one consistent place to trace a picture back to its
    model. Two real gaps this fixed, not just a cosmetic pass:
    `field_cortex_with_coil_position.png` REPLACED its subtitle with
    coil-specific text and dropped the conductivity type entirely, and
    `coil_position.png` never carried it anywhere. Now-redundant mentions
    inside subtitles (`· {conductivity_type}` tacked onto coil/placement
    text) were removed since the title always carries it. Verified visually
    on the re-rendered `coil_position_tbe.png`: title reads `IXI034: coil at
    C3, handle toward CP1  [tbe]`.
- **Task 4 outputs revised** (2026-09-14), per the user's list:
  - **Pial + white overlays** are computed by the worker
    (`worker.surface_overlays`) with SimNIBS's own
    `ElementData.interpolate_to_surface`, GM tets for pial, WM tets for
    white, `out_fill='nearest'`; SimNIBS's `depth` argument to
    `middle_gm_interpolation` is accepted and ignored in 4.6 (checked in
    the source), so this is the only way to get them. Named
    `<hemi>.<sim>.{pial,white}.E.{magn,normal}` beside the central ones,
    so `service._overlay_for(..., surface=)` reads all three.
    `SURFACE_VTKS` then writes four surface VTKs (central magn/normal,
    pial magn, white magn). The white-surface mosaic uses the white
    overlay and falls back to NIfTI sampling only for a run without one.
    **2026-09-14 (evening)**: the surface VTKs now carry explicit RGB
    (`write_surface_vtk`, vtk package, uchar 'Colors' as active scalars,
    the field as a named float array beside it; jet for magnitude,
    coolwarm symmetric for normal; window = `preview.FIELD_PCTL`, written
    into the header). The user had noticed viewers opened them in a
    different map from the mosaics. vendor/gii2vtk's writer is no longer
    used here (it takes 1-D arrays only; never edit vendor). Both this
    writer and tract_efield' tract writer set `SetFileVersion(42)`: the
    writer's default 5.1 layout stores cells as int64 and made each
    surface 41 MB instead of ~25. `tract_efield.read_vtk_surface` reads
    the NAMED `E_magnitude_V_per_m` array, falling back to the active
    scalars for a pre-change file.
  - **Report is `.docx`** (`write_run_report`, python-docx 1.2 in myenv).
    `fields_summary.txt` is parsed (`_parse_fields_summary`) into
    `JobResult.field_summary`, the peak is read from that, and the file is
    deleted from outputs/. Written ONCE, before the zip (the report no
    longer mentions the archive). `REPORT_ROLES` is the file-table subset.
  - **CSV cut to 5 columns** (`STATS_COLUMNS`): subject_id,
    conductivity_type, tissue, percentile, threshold_v_per_m. The user's
    list did not name `percentile`; kept because three thresholds per
    tissue are otherwise indistinguishable. Everything dropped is in the
    report.
  - **Pictures**: `field_cortex` (no coil) removed; `field_white_matter_volume`
    became `field_gray_white_matter_volume` (role `volume_png`) -- GM+WM
    field with the white surface's cut drawn in white
    (`preview.field_volume_mosaic(boundaries=, affine=)`); coil position
    is now one axial slice (`preview.coil_position_slice`, the mosaic is
    gone). Canvas views cut to six (`VIEW_LABELS`), `RESULT_VIEW` is the
    cortex + coil view.
  - mesh_sim now also copies the head model's T1 into
    `subject_volumes/<id>_T1.nii.gz` (role `t1_nii`), so its archive is
    self-sufficient for task 5 (no head-model re-upload).
- **Task 5 (`tract_efield`) implemented** (2026-09-14) as the dedicated
  task recommended for the streamline question. Inputs: mesh_sim's archive +
  task 2's whole-brain tractogram. Reads the white VTK with the `vtk`
  package (`service.read_vtk_surface`), loads tracts through
  `tasks.anicond.tractio` (vox -> RAS mm), area-weighted surface percentile
  (`hotspot_mask`), KD-tree contact test (`select_streamlines`, endpoint /
  any), writes a streamline VTK with |E| along it (`write_tracts_vtk`, LPS,
  `SPACE=LPS` in the header), the white surface with a hotspot flag
  (gii2vtk writer), two mosaics (`preview.tract_mosaic`, in-slab segments
  as one LineCollection per slice), a one-row CSV, a .docx report, a
  nested zip. Verified on IXI258 tbe: 2,667 / 98,989 streamlines, 10 s.
  - **Space check** (`space_check`): grid overlap + endpoints within 5 mm
    of the white surface; warns below `MIN_GRID_OVERLAP` /
    `MIN_ENDPOINT_NEAR_FRACTION`. On IXI258 65 % of endpoints are near the
    white surface -- the rest end in brainstem/cerebellum/deep GM, which is
    why the threshold is 0.5, not 0.9.
  - **Percentile is of the SURFACE, not the WM volume**: the report and
    CSV carry both (`wm_volume_threshold_v_per_m` from mesh_sim's CSV).
  - **Hidden task mechanism** added for this: `TaskSpec.visible`,
    `registry.visible_tasks()/hidden_tasks()`, `main.build_app()` mounts
    every enabled task with `show_in_navbar=task.visible` and lists only
    visible ones on the menu. `enabled` still means "page exists".
    Hidden pages link each other from a footer line
    (`main._hidden_links`, appended after `build_page()`, so it sits
    under the task's own footer); verified per page through the config's
    `page` -> components map (the combined `/config` cannot tell pages
    apart -- use that map, not a grep over all components).
  - `app/report.py` now holds the .docx style; mesh_sim's report uses it
    (task 5's own .docx report was removed on 2026-09-14 at the user's
    request, so mesh_sim is its only user now).
  - **Revised 2026-09-14 (later)**: outputs cut to the streamline VTK
    (now with a `hop` cell scalar), the white-surface VTK, ONE mosaic and
    a 9-column CSV -- no report, no job spec, no T1-only mosaic. The
    mosaic draws |E| on the WHITE SURFACE's cut (not the volume), window
    vmax = the threshold, tracts direction-coloured per segment
    (`preview.direction_colors`), T1 dimmed harder (`BACKGROUND_DIM`
    3.2), saved at 320 dpi / canvas at 150. The subtitle is wrapped
    (`SUBTITLE_WRAP`) and the title band reserved by height
    (`TITLE_LINE_IN`): an unwrapped long subtitle made bbox_inches="tight"
    widen the PNG to 6300 px with the tiles small in the middle.
  - **Follow broken streamlines** (`service.follow_breaks`): endpoint
    KD-tree, join if gap <= `join_mm` AND outward exit tangents
    head-to-tail (dot <= -cos 45°), BFS up to `max_hops`. The angle test is
    essential: without it the endpoint clusters at the gray/white boundary
    chain the whole cortex. Default off; IXI258 +746 over 3 hops.
  - Page: the subject-id / conductivity-type boxes were replaced by one
    Markdown line filled from the archive's name; three numbers per row,
    inline-label dropdown, checkbox + gap + hops on one row.
  - **Colour clash fixed (2026-09-14, evening)**: the user found jet on
    the surface + direction-coloured tracts confusing (both use every
    hue). Chosen fix ("proposal 1"): surface ACHROMATIC (`SURFACE_RAMP`
    grey -> white, vmax = threshold, hot-spot segments in pure white at
    `HOTSPOT_LINEWIDTH_FACTOR` x), tracts the only colour; followed pieces
    at `FOLLOWED_LINEWIDTH_FACTOR`. Alternatives offered and declined: hot
    spot outline only (no field), and a two-panel-per-slice layout.
  - **Mosaic reworked again (2026-09-15)**: no field map at all (only
    tracts + dark T1 + thin grey boundary, hot spot white), legend moved
    to a strip UNDER the tiles (`legend=` in `tract_mosaic`, reserved by
    height like the title), title = subject · coil site · threshold ·
    "connected fiber tracts" (coil site from the archive's
    `<id>_job_spec.json`, region name via `mesh_sim.service.ELECTRODE_
    REGIONS`; `Simulation.coil_center`). The "filled regions" look was
    diagnosed as stroke width (1 pt = 1.6 mm at 320 dpi, wider than the
    inter-streamline spacing) + 0.95 alpha: now 0.45 pt, alpha fading
    0.9 -> 0.25 with depth in a 5 mm slab, near segments drawn last.
  - **Second mosaic (2026-09-15)**: `field_with_tracts_by_efield` -- the
    same `tract_mosaic` call with `point_values=` (|E| per point, kept on
    `JobResult.selected_e`, split from the VTK's `along`) and
    `value_range=(0, e_color_top)`, so the picture and the E-field VTK
    share one scale. NO colour bar (removed at the user's request so the
    two mosaics align pixel-for-pixel when flipped; the window is in the
    legend text). `_drawn()` returns the same subsample for both.
  - **Activating function (2026-09-15)**: `service.activation_along`
    (E_s = E·t, Gaussian-smoothed along arc length, f = -dE_s/ds with s in
    METRES -> V/m^2; terminal field E·t_out at both ends) on the E vector
    NIfTI via `sample_vector_field` (mask-aware: E*mask / mask, weight <
    `ACTIVATION_MIN_MASK_WEIGHT` = outside). Third VTK
    (`tracts_coloredby_activation`, colours from `preview.ACTIVATION_CMAP`
    -- a diverging map with a DARK middle so zero fades into the T1
    instead of turning white like the hot spot), third mosaic (endpoint
    dots = terminal field), `<id>_tracts_activation.csv` per streamline,
    3 columns on the stats row. Verified: |E_s| <= |E| everywhere, red
    rises with f in the VTK, magnitudes ~1e4 V/m^2 for a 100 V/m field
    over cm, as expected. Sign convention derived, not copied: for a
    finite fibre in a uniform field the end the field points TOWARD
    depolarises, hence t_out (E·t_out > 0 = depolarising). Not done:
    any membrane model / threshold (Aberra-style NEURON models) -- the
    E_s(s) profile this writes is exactly their input.
  - **VTKs**: the white-surface VTK was dropped (mesh_sim's
    `both.white.magn.vtk` already carries |E|; the hot spot is a threshold
    on it). The tract VTK became TWO files with explicit RGB in an
    unsigned-char `Colors` point array as the active scalars (Slicer maps
    uchar x3 directly): `tracts_coloredby_efield` (jet, 0 ..
    p`TRACT_COLOR_PCTL` of |E| along the tracts) and
    `tracts_coloredby_orientation` (local tangent, `point_directions`).
    Both keep `E_magnitude_V_per_m` (point) and `hop` (cell). Verified by
    reading back: corr(|E|, R-B) = 0.88 in the efield file.
- **Stimulation frequency shown on mesh_sim's page** (2026-09-15):
  `service.conductivity_frequency()` (table column label -> "10 kHz", SI
  prefix case kept) under the 10-20 picture (`web._frequency_note`). Asked
  for together with an ICNIRP investigation; the dI/dt default was NOT
  changed -- a proposal was given (see the 2026-09-15 conversation).
- `cli.py` / `gui.py` for any task: wanted "later", not now.
- For CondNet (no segmentation head), the tissue-table CSV is still written when
  intermediates are on. Offered to gate it on a segmentation existing; no answer.
- **Task 6 (`tract_simplify`) implemented** (2026-09-17), hidden like task 5:
  a tractogram healed / length-cut / thinned / smoothed FOR PICTURES ONLY
  (README §Task 6 has the reasoning behind each choice). The algorithms
  moved to `vendor/tractviz` the same day (`prepare()` = heal + cluster,
  `select()` = cut + thin + smooth, `simplify_tracts()` = both with the
  page's defaults); task 2 calls `simplify_tracts()` to write a
  `*_wb_tracts_viz.vtk` twin beside its tractogram, and task 5 writes its
  `*___simplified.vtk` files from it at its own gentler `SIMPLIFY_*`
  settings (≥ 20 mm, 50 %, σ 1 mm; the pictures, numbers and CSVs stay on
  the selection as tracked -- the simplified copy was too few streamlines
  to draw). Shape:
  - `service.run_job()` (queued) = load via task 5's `load_tracts` against
    the uploaded reference NIfTI, `heal_breaks()` (task 5's exit-tangent
    test made global: KD-tree `query_pairs` on all endpoints, greedy
    best-first one-partner-per-endpoint, then ONE pass of length-3
    augmenting paths in `_match_endpoints` -- on a dense synthetic bundle
    greedy alone healed 387/500 planted breaks, with the pass 466/500;
    no union-find, a closed chain is cut when walked), then
    `cluster_streamlines()` = dipy QuickBundles once at 12 mm.
  - `service.apply()` (NOT queued, bound to the three sliders' `release`)
    = length cut, `thin_mask()` (ceil(fraction x cluster size), min 1,
    seeded key so lowering only removes), `smooth_streamlines()`,
    rewrite `<case>_tracts_viz.vtk` (task 5's `write_tracts_vtk`, LPS,
    `Colors` by direction, cell arrays pieces/cluster/length_mm) and a
    `.trk` twin (nibabel, header from the reference image) for the browser
    viewer; no settings json. The JobResult
    with its arrays lives in `gr.State`, like task 5's.
  - Canvas: task 5's `preview.tract_mosaic` with no surfaces, ONE slice
    (`slices=[middle_slice(bg)]`, `tile_in=CANVAS_TILE_IN`), two views
    (original / revised). No saved PNG, no zip, no report.
  - 3D viewer: NiiVue 0.69 (`assets/niivue.js`, 3.1 MB, BSD-2, licence
    text next to it) + `assets/viewer.html` in an iframe under the canvas.
    Served through `/gradio_api/file=` -- Gradio's `file_fetch` serves a
    file under `allowed_paths` INLINE with its guessed type (checked:
    viewer.html text/html, niivue.js application/javascript), which is
    what lets the module script import next to itself. No Python
    dependency; `service.viewer_status()` -> a note instead of the iframe
    when the bundle is missing. Re-fetch line in `settings.py`.
  - Verified: synthetic run (5 arcs x 300 streamlines, every 3rd cut in
    two, 200 fragments) under both the base and `myenv` interpreters;
    `build_app()` with empty `BUILD_ERRORS`; the served files above. The
    viewer's WebGL rendering was NOT exercised in a real browser during
    development -- open /tract_simplify once with a real run and look.
  - New: `settings.UNCAPTURED_LOGGERS` has `tasks.tract_simplify.service`;
    `registry.TASKS` has the `tract_simplify` entry (visible=False).

## 8. Conventions to match

- Comments explain **why**, not what; module docstrings carry a layout or
  pipeline sketch. Match the density of the surrounding code.
- The user cares about compact, purposeful UI layout and has iterated on it
  several times — check the current `web.py` before assuming a layout.
- The user edits files between turns. **Read a file before editing it**; more
  than once the requested change was already partly applied.

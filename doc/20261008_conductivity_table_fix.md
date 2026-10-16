# Conductivity table fixes (2026-10-08/09): 10 kHz and 1 Hz

## Why

Two tissues had a different 10 kHz conductivity in the app and DCond tables
than in the table CondNet-TART was trained on
(`PROJ_CONDNETS/condnettart_training/config/10kHz.yaml`). The training values
are taken as correct:

| Tissue (label) | Old (S/m) | New (S/m) |
|---|---|---|
| csf_meninges (6) | 2.0 | **0.5** |
| brainstem (& cortical nuclei) (14) | 0.07 | **0.179111** -- superseded 2026-10-09: now 0.07, see the last section |
| muscle (10) | 0.34 | 0.34 -- already correct, unchanged |

Not changed in this fix: thalamus (label 15, anicond/DCond scheme only,
0.07). The `1hz` column has its own fix, below.

## What was changed

| # | File | Change |
|---|---|---|
| 1 | `APP_BRAINE/vendor/isocond/pipeline/tissue_labels.csv` | rows 6 and 14, `10khz` column |
| 2 | `PROJ_CONDNETS/model_runner/code/pipeline/tissue_labels.csv` | rows 6 and 14, `10khz` column |
| 3 | `APP_BRAINE/tasks/anicond/assets/default_target_conductivity.csv` | rows 6 (CSF_Meninges) and 14 (Brainstem) |
| 4 | `APP_BRAINE/vendor/anicond/core/config.yaml` | `labels.conductivity_iso_10khz` "6" and "14" |
| 5 | `PROJ_DCONDNET/scripts/core/config.yaml` | `labels.conductivity_iso_10khz` "6" and "14" (old values kept in a comment) |

`APP_BRAINE/vendor/VERSION.md` records the change to files 1 and 4.

What the change reaches through these files:

- **isocond**: the "Expected" column of the per-slice check, and the
  `tissue_conductivity_10khz.csv` saved with each result. The conductivity
  map itself comes from the network and does not change.
- **anicond**: the isotropic base conductivity its tensors are scaled to.
- **Tissue Conductivity utility**: its 10 kHz anchor (it reads file 3), so
  every `<freq>.yaml` it generates.
- **mesh_sim**: not directly. It skips the CSF_Meninges and Brainstem rows.
- **DCond** (`PROJ_DCONDNET`): the isotropic base table (`core.general.dict_cond_10khz`)
  every conductivity model starts from. Brainstem is in its WM group.

## What to re-run

### APP_BRAINE

Restart the app first so it reads the new tables.

| Affected | Re-run | Note |
|---|---|---|
| isocond results | isocond, for the subjects you keep | only the Expected column and the saved tissue CSV change |
| anicond results | anicond (page or `tasks/anicond/cli.py`), every subject | brainstem tensors scale by ~2.6× (0.07 → 0.179) |
| mesh_sim results | only runs that used an anicond tensor (`dm` / `vn` / `tbe`), after anicond | `iso` runs are unaffected |
| `<freq>.yaml` files from the Tissue Conductivity utility | regenerate each one | they were anchored to the old 10 kHz values |
| models trained from those YAMLs | retrain on `/condnettart_train` | e.g. the 1 Hz quick tests |

### PROJ_DCONDNET (from `scripts/`, in order)

1. `p2_condmodel/c1_diff_to_cond_estimation.py` (or `c1 (gpu)_…`) --
   conductivity tensors for all models.
2. `p2_condmodel/c2_diff_to_cond_model_scoring.py`, then
   `p2_condmodel/c3_diff_n_cond_model_score_stats.py` -- scores and statistics.
3. `p2_condmodel/c4_visualize_model_validation.py`,
   `p2_condmodel/c6_visualize_cohort_eigenvalue_distribution.py` -- figures, if used.
4. `p3_dlmodel/d1_export_ml_tractproj_dataset.py` -- the DL dataset, under a
   **new dataset tag** (e.g. `…_v3`; the project never overwrites a build).
   Then retrain the DL model on it.

Not needed: `p1_common/b1`–`b7` (preprocessing, diffusion tensors,
tractography, tract reference) do not use conductivity. Data built before
the fix -- e.g. `IXI/dl/ds_tractbasedg_r3_224_v1/groundtruth/*_cond.csv`,
`IXI/data_ixi_derivatives/` -- keeps the old values until regenerated.

### PROJ_CONDNETS

Nothing. `condnettart_training` already used the correct values; the
model_runner copy (file 2) only feeds its own legend and table outputs.

---

## 1 Hz: csf_meninges (second fix, same day)

| Tissue (label) | Column | Old (S/m) | New (S/m) |
|---|---|---|---|
| csf_meninges (6) | `1hz` | 1.8 | **0.16** |

The column is labelled `1hz` (1 Hz) in the files; it is the only
low-frequency column in any of the three projects, so it is the one meant.
No other tissue or column changed.

### What was changed

| # | File | Change |
|---|---|---|
| 1 | `APP_BRAINE/vendor/isocond/pipeline/tissue_labels.csv` | row 6, `1hz` column |
| 2 | `PROJ_CONDNETS/model_runner/code/pipeline/tissue_labels.csv` | row 6, `1hz` column |

Files 3-5 (anicond, DCond) have only a 10 kHz table, so nothing to change
there. `APP_BRAINE/vendor/VERSION.md` records the change to file 1.

### What to re-run

Only results that used the 1 Hz column, i.e. isocond runs with a CondNet-TART
**1 Hz** model (`condnettart_1Hz_…`): their "Expected" column and their
saved `tissue_conductivity_1hz.csv`. Re-run those subjects after restarting
the app. The conductivity map itself does not change. Nothing else reads the
`1hz` column: not the Tissue Conductivity utility (anchored at 10 kHz),
anicond, mesh_sim, DCond or condnettart_training.

---

## 2026-10-09: brainstem & cortical nuclei = white matter

The 2026-10-08 value 0.179111 came from Kubota's June 2025 training table
with no recorded source, and Gabriel et al. (1996) have no brainstem tissue.
Decision: treat brainstem & cortical nuclei as **white matter** everywhere.

| Tissue (label) | 10 kHz | 1 Hz |
|---|---|---|
| brainstem & cortical nuclei (14) | 0.179111 → **0.07** (= WM) | 0.14 (= WM, unchanged) |

### What was changed

| File | Change |
|---|---|
| `APP_BRAINE/vendor/isocond/pipeline/tissue_labels.csv` | row 14, `10khz` |
| `PROJ_CONDNETS/model_runner/code/pipeline/tissue_labels.csv` | row 14, `10khz` |
| `APP_BRAINE/tasks/anicond/assets/default_target_conductivity.csv` | row 14 (Brainstem) |
| `APP_BRAINE/vendor/anicond/core/config.yaml` | `conductivity_iso_10khz` "14" |
| `PROJ_DCONDNET/scripts/core/config.yaml` | `conductivity_iso_10khz` "14" |
| `PROJ_CONDNETS/condnettart_training/config/10kHz.yaml` | `sigma` 14 → 0.07; Kubota's June value kept in `reference_normalized`, with a new `reference_exceptions` entry for class 14 |
| `PROJ_CONDNETS/condnettart_training/script/check_frequency_table.py` | lists `reference_exceptions` instead of failing on them |

Unchanged on purpose: the Tissue Conductivity utility already follows white
matter's frequency dependence for this class (`classes.csv`), so with the
new anchor it now equals white matter at every frequency. Records of past
runs keep the value they were made with: `model/10kHz/training_record.json`,
the model card `condnettart_10kHz.json`, `evaluation/*/tissue_metrics.csv`.

### What to re-run

| Project | Re-run | Why |
|---|---|---|
| condnettart_training | `03_train_conductivity.py --overwrite` (or `--dated`), then `04_evaluate.py`, `05_export_for_app.py`; deploy the new files to `APP_BRAINE/vendor/isocond/models/` | the deployed 10 kHz model learned brainstem = 0.179 |
| APP_BRAINE | isocond, for the subjects you keep (after the new model is deployed) | Expected column / saved tissue CSV; the map itself only changes with the retrained model |
| APP_BRAINE | anicond, every subject; mesh_sim runs that used an anicond tensor | brainstem tensors scale back by 0.07/0.179 |
| APP_BRAINE | regenerate every `<freq>.yaml` from the Tissue Conductivity utility; retrain models made from them | their brainstem row followed the 0.179 anchor |
| PROJ_DCONDNET | `c1` → `c2`/`c3` → `c4`/`c6` → `d1` (new dataset tag) → DL retraining, as in the first section | brainstem is in DCond's WM group |

If the 2026-10-08 re-runs were not done yet, do only these: the net effect
of the two days on brainstem at 10 kHz is no change for anicond and DCond
(0.07 → 0.179 → 0.07); csf_meninges remains changed (2.0 → 0.5).


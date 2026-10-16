# ============================================================================
# tasks/mesh_sim/settings.py — how the SimNIBS simulation task runs
# ----------------------------------------------------------------------------
# The global settings.py already holds where SimNIBS lives (SIMNIBS_ROOT /
# SIMNIBS_PYTHON), because that is a machine fact rather than a task choice.
#
# The coil / cap / electrode lists below are FALLBACKS, not the catalogue.
# service.py scans the install for coils and reads the uploaded head model's
# own cap files and electrode coordinates; these lists are what the page
# shows before anything has been uploaded, and what it falls back to if a
# scan comes up empty.
# ============================================================================
from pathlib import Path

import settings as global_settings

TASK_DIR = Path(__file__).resolve().parent
ASSETS_DIR = TASK_DIR / "assets"

# --- The worker -------------------------------------------------------------
# Run by SIMNIBS_PYTHON, never imported: `simnibs` cannot be imported from
# this app's environment. See service.py and worker.py.
WORKER = TASK_DIR / "worker.py"

# Threads for the FEM solve. The solve itself takes seconds on this size of
# mesh; the post-processing that follows is single-threaded anyway, so there
# is little to gain from taking the whole machine, and jobs already run one
# at a time across every task (MAX_CONCURRENT_JOBS).
SIM_CPUS = 8

# Hard ceiling on one simulation. A healthy run is a few minutes; anything
# past this is a hang, and the queue should be given back.
SIM_TIMEOUT_S = 2 * 3600

# --- Coil model ---------------------------------------------------------
# SimNIBS's coil catalog has three folders: Drakaki_BrainStim_2022 (standard
# clinical figure-eight / "butterfly" coils from the major manufacturers --
# MagVenture, Magstim, Deymed, MagMore), flexible_coils (Brainsway H1/H4/H7
# deep-TMS arrays and the MST-Twin bilateral coil -- not figure-eight
# geometry) and legacy_and_other (a mix, not individually vetted here). Per
# the user's request to offer figure-eight coils only, the catalog is scoped
# to the Drakaki_BrainStim_2022 folder, which is unambiguous: every file in
# it is a standard double/butterfly coil model.
COIL_DIR = global_settings.SIMNIBS_ROOT / "resources" / "coil_models"
FIGURE8_COIL_DIR = COIL_DIR / "Drakaki_BrainStim_2022"

# A snapshot of that folder's contents (24 files, checked 2026-09-10), used
# only until the live scan exists. (label shown in the dropdown, filename).
COIL_CHOICES = (
    ("MagVenture MCF-B65", "MagVenture_MCF-B65.ccd"),
    ("MagVenture MCF-B65 (new)", "MagVenture_MCF-B65_new.ccd"),
    ("MagVenture MCF-75", "MagVenture_MCF-75.ccd"),
    ("MagVenture Cool-B65", "MagVenture_Cool-B65.ccd"),
    ("MagVenture Cool-B70", "MagVenture_Cool-B70.ccd"),
    ("MagVenture Cool-B35", "MagVenture_Cool-B35.ccd"),
    ("MagVenture Cool-D-B80", "MagVenture_Cool-D-B80.ccd"),
    ("MagVenture C-B60", "MagVenture_C-B60.ccd"),
    ("MagVenture C-B70", "MagVenture_C-B70.ccd"),
    ("MagVenture C-100", "MagVenture_C-100.ccd"),
    ("MagVenture MC-125", "MagVenture_MC-125.ccd"),
    ("MagVenture MC-125 (new)", "MagVenture_MC-125_new.ccd"),
    ("MagVenture MC-B70", "MagVenture_MC-B70.ccd"),
    ("MagVenture MC_B65_HO8", "MagVenture_MC_B65_HO8.ccd"),
    ("MagVenture MMC-140-II", "MagVenture_MMC-140-II.ccd"),
    ("MagVenture MRI-B91", "MagVenture_MRI-B91.ccd"),
    ("MagVenture MST Twin", "MagVenture_MST_Twin.ccd"),
    ("Magstim D70", "MagStim_D70.ccd"),
    ("Magstim DCC", "MagStim_DCC.ccd"),
    ("Deymed 50BF", "Deymed_50BF.ccd"),
    ("Deymed 70BF", "Deymed_70BF.ccd"),
    ("Deymed 120BFV", "Deymed_120BFV.ccd"),
    ("MagMore PMD70", "MagMore_PMD70.ccd"),
    ("MagMore PMD45-EEG", "MagMore_PMD45-EEG.ccd"),
    ("MagMore PMD25-decentral", "MagMore_PMD25-decentral.ccd"),
)
DEFAULT_COIL = "MagVenture_MCF-B65.ccd"     # a common research figure-eight coil

# --- EEG cap / coil position ---------------------------------------------
# Every m2m folder ships its own eeg_positions/*.csv; these are the names
# seen in a typical charm output, used as the dropdown's starting choices
# until an upload lets the page scan the real folder.
EEG_CAP_CHOICES = (
    "EEG10-10_UI_Jurak_2007.csv",
    "EEG10-10_Cutini_2011.csv",
    "EEG10-10_Neuroelectrics.csv",
    "EEG10-20_extended_SPM12.csv",
    "EEG10-20_Okamoto_2004.csv",
    "easycap_BC_TMS64_X21.csv",
)
DEFAULT_EEG_CAP = "EEG10-10_UI_Jurak_2007.csv"      # SimNIBS's own default cap

# Coil center: named targets, each annotated with the region it sits over so
# the choice is readable without leaving the page. C3 over left motor cortex
# (the classic single-pulse / motor-mapping target) is the default.
COIL_CENTER_CHOICES = (
    ("C3 (left M1, hand area)", "C3"),
    ("C4 (right M1, hand area)", "C4"),
    ("Cz (vertex: M1 leg area / SMA)", "Cz"),
    ("F3 (left DLPFC)", "F3"),
    ("F4 (right DLPFC)", "F4"),
    ("P3 (left parietal)", "P3"),
    ("P4 (right parietal)", "P4"),
)
DEFAULT_COIL_CENTER = "C3"

# Handle direction: the immediate 10-10 neighbours of C3, each annotated with
# the approximate angle picking it would give the coil handle -- 0 deg =
# pointing straight posterior, positive = rotated toward the midline
# (medial), negative = rotated away from it (lateral); the label spells this
# out in words so nobody has to hold the sign convention in their head.
#
# This particular list is a fallback in a stronger sense than the others
# above: service.handle_choices() computes the real thing (these angles, for
# the uploaded subject's own scalp) live, so this static version is only ever
# seen before a head model is uploaded. It comes from one reference subject
# (m2m_IXI258's EEG10-10_UI_Jurak_2007.csv, checked 2026-09-10): the vector
# from C3 to each neighbour, projected onto the scalp tangent plane at C3
# (normal approximated as the C3 point itself, i.e. the coordinate origin
# standing in for the head centre) and measured against the tangent-plane
# projection of the global nasion->inion axis. A real subject's angles will
# differ by some degrees -- note the medial/lateral wording flips if the coil
# centre is ever changed from a left-hemisphere site (C3) to a right-
# hemisphere one (C4); service.handle_choices() gets this right live by
# reading the coordinates rather than assuming a side.
COIL_YDIR_CHOICES = (
    ("CP1 (≈45° posteromedial)", "CP1"),
    ("CP5 (≈45° posterolateral — classic M1 angle)", "CP5"),
    ("CP3 (≈0°, straight posterior)", "CP3"),
    ("C5 (≈90° lateral)", "C5"),
    ("C1 (≈90° medial)", "C1"),
    ("FC5 (≈145° anterolateral)", "FC5"),
    ("FC1 (≈135° anteromedial)", "FC1"),
    ("FC3 (≈175°, near-anterior)", "FC3"),
)
# CP1: the closest named 10-10 position to 45 degrees on the postero-medial
# side. The textbook M1 handle orientation cited in the literature is
# actually postero-LATERAL (CP5, kept above as the other ≈45° option); CP1 is
# set here by explicit choice, not because it is the more commonly cited one.
DEFAULT_COIL_YDIR = "CP1"

# How far from the coil centre an electrode may sit and still be offered as a
# handle reference. Only the direction matters to SimNIBS, but the tangent
# plane the angle is measured in only approximates the scalp near the centre,
# and a reference on the far side of the head tells a reader nothing.
HANDLE_MIN_MM = 15.0
HANDLE_MAX_MM = 60.0

# --- Stimulation parameters -----------------------------------------------
# dI/dt: SimNIBS's own library default is 1e6 A/s (1 A/us), but that is a
# nominal per-unit placeholder, not a realistic pulse -- SimNIBS's FAQ says so
# explicitly ("What is the dI/dt in TMS simulations?"), and its own tutorial
# examples use 1e6-20e6 purely to illustrate the API, not as a stimulation
# level. It is ~100x below what any real stimulator puts through a coil.
#
# Real numbers are embedded in the coil files themselves. Every .ccd in
# FIGURE8_COIL_DIR carries a measured `dIdtmax=<value in A/us>` for its
# manufacturer-paired stimulator at 100% MSO (maximum stimulator output) --
# the Drakaki et al. 2022 (Brain Stimulation) dataset SimNIBS cites in its own
# coil documentation. Checked across all 25 coils in that folder
# (2026-09-11): dIdtmax ranges 84.4-249.6 A/us (median 147.9, mean 153.1).
# DEFAULT_COIL's own value, 147.86 A/us, happens to equal that catalog
# median almost exactly -- a genuine coincidence, not a chosen anchor, that
# makes it a reasonable single number to default to before a coil is picked.
#
# 100% MSO is the rated CEILING, not necessarily what a given protocol uses
# (single-pulse and rTMS studies commonly run at 80-120% of an individual's
# resting motor threshold, itself well below 100% MSO for a modern biphasic
# stimulator) -- but it is the one value that is measured, sourced and
# reproducible without assuming a specific threshold or protocol, unlike a
# fraction-of-threshold figure this app has no way to know per subject. The
# field scales exactly linearly with dI/dt (both the run report and the page
# say so), so scaling this down for a lower-intensity protocol is a plain
# multiplication.
#
# NOTE: since dIdtmax varies up to ~3x across the coil catalog (84-250 A/us),
# this one static number is only exactly right for DEFAULT_COIL. The more
# correct design reads dIdtmax from whichever coil is actually selected and
# updates the suggested dI/dt live, the way the handle-direction angles
# already update when the coil centre changes -- not done here; flagged for
# whoever next touches coil selection.
#
# DEFAULT (2026-09-15): anchored to the ICNIRP 2010 guideline rather than
# to a stimulator's ceiling. The study reads the simulation at the
# frequency the conductivity table is tabulated at (10 kHz); ICNIRP's basic
# restriction on the internal E-field in CNS tissue of the head is
# 2.7e-4 f V/m rms for occupational and 1.35e-4 f for general public
# exposure in 3 kHz-10 MHz (Table 2), evaluated as the 99th percentile of
# the tissue's field, and for a pulse as a weighted peak (sqrt 2 x the rms
# limit, appendix): at 10 kHz, 3.82 V/m peak occupational. The reference
# subject IXI258 (MCF-B65 at C3, any conductivity model) reaches a gray-
# matter p99 of ~90 V/m at the coil's 147.86e6 A/s, so the occupational
# restriction sits at ~6.2e6 A/s -- rounded to 6e6 (about 4% of that
# coil's 100% MSO). Occupational rather than public because that level is
# ICNIRP's "no myelinated-nerve stimulation" level (a 4 V/m threshold with
# a reduction factor of 5), a physical anchor for a TMS field; the public
# level adds one more policy factor of 2 (3.1e6 A/s). Every run reports
# the exact dI/dt at which ITS OWN p99 would meet each restriction (run
# report, "Exposure guideline"), since the number is subject-specific.
# Note ICNIRP excludes medical exposure by definition, and says that a
# source a few centimetres from the body is assessed dosimetrically
# against the basic restrictions, never against the reference levels.
DEFAULT_DIDT = 73.93e6        # A/s -- see above; MCF-B65's 100% MSO is 147.86e6
# The coil's own 100% MSO value, kept for the page's hint and the report.
DIDT_100_MSO_DEFAULT_COIL = 147.86e6
# Coil-to-scalp gap. SimNIBS's own POSITION default is 4 mm (roughly a coil
# casing's thickness against bare scalp), but its own scripting tutorial
# (documentation/tutorial/scripting.html) uses 5 mm in its worked example --
# closer to practice, where hair, a cap or padding commonly add a millimetre
# or two beyond the casing itself. 5 mm is used here for that reason.
DEFAULT_DISTANCE_MM = 5.0

# --- Anisotropic conductivity ----------------------------------------------
# SimuList.anisotropy_type -- what SimNIBS does with the tensor FILE it is
# given, not which model produced that file (that is CONDUCTIVITY_TYPES
# below). SimNIBS accepts exactly four: 'scalar', 'vn', 'mc', 'dir'. Read off
# simnibs.utils.cond_utils.cond2elmdata, which is the only place that says
# what they really do:
#
#   vn   normalize=True -- each tensor rescaled so its geometric mean equals
#        the isotropic conductivity for that tissue. Direction and relative
#        shape kept, the tensor's own magnitudes discarded.
#   mc   excentricity_scaling=0.0 -- which _adjust_excentricity turns into
#        EQUAL eigenvalues, i.e. isotropic. SimNIBS's own log line for this
#        branch reads "Using isotropic mean conductivities". It keeps the
#        tensor's magnitude and throws the direction away; it is not an
#        anisotropic simulation, whatever the name suggests.
#   dir  the tensor's own directions and relative shape, as given.
#
# Stock 'dir' is NOT a pure passthrough: correct_intensity=True by default
# (Rullmann et al. 2009) scales the WHOLE tensor field by one global scalar
# so the volume-weighted mean conductivity in the anisotropic tissues
# matches the isotropic reference, and _fix_eigv caps eigenvalues at
# aniso_maxcond (2 S/m) with aniso_maxratio (10). That correction exists
# for tensors that came out of a DIFFUSION fit and still need a
# conductivity magnitude invented for them. Every tensor this app hands
# SimNIBS is already a CONDUCTIVITY tensor (anicond's DM / VN / TBe output,
# in S/m), so the worker bypasses it -- see TENSOR_AS_IS.
#
# Not a page control: 'dir' is the only correct choice for a ready-made
# conductivity tensor. 'vn' would discard the tensor's own computed
# magnitude, and 'mc' throws away direction entirely and forces the tensor
# ISOTROPIC; both exist only for the case this app does not have, an upload
# that is a raw diffusion tensor. Kept as a tuple (rather than inlining
# "dir") so `run_job`'s default and any future CLI read the same one place.
ANISOTROPY_TYPES = (
    ("Direct mapping (dir) — the tensor's own values (use for a ready-made "
     "conductivity tensor: DM / VN / TBe)", "dir"),
    ("Volume normalized (vn) — direction from the tensor, isotropic magnitude",
     "vn"),
    ("Mean conductivity (mc) — magnitude only, tensor forced ISOTROPIC",
     "mc"),
)
DEFAULT_ANISOTROPY_TYPE = "dir"

# Use the uploaded conductivity tensor AS-IS under 'dir': the worker calls
# cond2elmdata with correct_intensity=False (no global rescaling to
# SimNIBS's isotropic reference) and lifts the eigenvalue caps
# (aniso_maxcond / aniso_maxratio) so no component is clipped either. What
# SimNIBS still does, and should: interpolate the tensor onto the WM/GM
# tetrahedra, rotate the components from the file's voxel frame into world
# space (correct_FSL, the same rule preview.load_tensor6 uses), and replace
# a tensor that is all-zero or negative semi-definite (a voxel the tensor
# never covered) with the tissue's isotropic value from the table below.
# Nothing in this path converts diffusion to conductivity: that happened in
# anicond, and SimNIBS's own dwi2cond is never run. False restores stock
# SimNIBS behaviour (rescale + clip), for comparison only.
TENSOR_AS_IS = True

# Optional caps applied to the as-is tensor, WITHOUT the rescaling: the
# largest eigenvalue allowed (S/m) and the largest ratio between a tensor's
# largest and smallest eigenvalue. None = no cap, the tensor's own values
# go through untouched. Measured on IXI034's tract-based tensor
# (2026-09-11): 24 of 389k voxels exceed 2 S/m (max 46 S/m) and a handful
# have a near-zero smallest eigenvalue (ratio up to ~5e7); stock SimNIBS
# clipped those at 2 S/m and ratio 10. Set e.g. 2.0 and 10.0 to get the
# clipping back while keeping the magnitudes unscaled. Either way the
# worker prints, and the run report records, the eigenvalue range the
# solver actually saw per anisotropic tissue.
TENSOR_MAX_COND = None
TENSOR_MAX_RATIO = None

# --- Isotropic conductivity table ------------------------------------------
# The per-tissue isotropic values every run uses -- for ALL tissues of an
# isotropic run, and for every tissue outside the tensor (everything but WM
# and GM) of an anisotropic one. They are NOT SimNIBS's built-in values
# (Wagner 2004: WM 0.126, GM 0.275, CSF 1.654, ...) but the same target
# table anicond computes its tensors against, so an 'iso' run and a 'dm' /
# 'vn' / 'tbe' run of one subject differ only by the tensor, not by two
# different sets of reference conductivities.
#
# The table's `simnibs_tissue` column says which SimNIBS tissue each row
# stands for (charm's own names: WM, GM, CSF, Scalp, Compact_bone, ...);
# rows without one (Cerebellar, Brainstem, Thalamus, CSF_Meninges,
# Mucous_membrane: tissues charm does not label separately) are not used
# here. A SimNIBS tissue the table has no row for keeps SimNIBS's built-in
# value, and the run report says which -- with the shipped table that is
# Bone (tag 4, only in pre-charm head models) and Cartilage (tag 11).
# CONDUCTIVITY_FREQ picks the conductivity_iso_<freq> column; None means
# the table's default (anicond's runctx.default_freq: the config's
# conductivity_freq_in_use, else the first column).
from tasks.anicond import settings as _anicond_settings
CONDUCTIVITY_CSV = _anicond_settings.DEFAULT_CONDUCTIVITY_CSV
CONDUCTIVITY_FREQ = None

# SimNIBS's tissue names -> element tags, copied from
# simnibs.utils.mesh_element_properties (ElementTags / tissue_names), which
# this app's environment cannot import. Matched case-insensitively against
# the table's `simnibs_tissue` column.
SIMNIBS_TISSUE_TAGS = {
    "WM": 1, "GM": 2, "CSF": 3, "Bone": 4, "Scalp": 5, "Eye_balls": 6,
    "Compact_bone": 7, "Spongy_bone": 8, "Blood": 9, "Muscle": 10,
    "Cartilage": 11, "Fat": 12,
}

# --- Exposure guideline ----------------------------------------------------
# ICNIRP 2010 (Health Phys 99(6):818-836), Table 2: basic restrictions on
# the internal electric field, rms, as (f_low_Hz, f_high_Hz, value or
# coefficient of f) for CNS tissue of the head. The evaluation rule (p825):
# vector average over a 2x2x2 mm cube, the 99th percentile of the tissue.
# Pulsed fields (appendix, Table 6): the peak of the waveform is compared
# with sqrt(2) x the rms restriction ("weighted peak").
ICNIRP_CNS_HEAD_RMS = {
    "occupational": ((1, 10, "0.5/f"), (10, 25, 0.05), (25, 400, "2e-3*f"),
                     (400, 3000, 0.8), (3000, 10e6, "2.7e-4*f")),
    "general_public": ((1, 10, "0.1/f"), (10, 25, 0.01), (25, 1000, "4e-4*f"),
                       (1000, 3000, 0.4), (3000, 10e6, "1.35e-4*f")),
}
ICNIRP_PEAK_FACTOR = 2 ** 0.5
# The exceedance mosaic: |E| / the occupational weighted-peak restriction
# on the GM+WM field volume, drawn only where the ratio is above 1. A
# sequential "heat" ramp rather than the field's jet: the picture answers
# "how many times over", and its lower end is the restriction itself.
EXCEEDANCE_CMAP = "hot"
EXCEEDANCE_GROUP = "occupational"

# --- Conductivity type, for the results CSV --------------------------------
# WHICH conductivity model produced the tensor this run was given -- the
# label a cohort's box plots are grouped by. This is anicond's taxonomy
# (its own model short names: DM, VN, TBe), NOT SimNIBS's ANISOTROPY_TYPES
# above, which is a different axis entirely: SimNIBS's 'vn' is how IT
# rescales whatever tensor it is handed, while anicond's 'vn' is the model
# that computed the tensor in the first place. The two can be combined
# freely, and they share the spelling 'vn' while meaning different things --
# hence one explicit field rather than trying to derive one from the other.
#
# Set by hand on the page: a tensor's filename is not a reliable witness to
# which model wrote it once files have been copied or renamed.
CONDUCTIVITY_TYPES = (
    ("iso — isotropic, the table's per-tissue values (no tensor)", "iso"),
    ("dm — anicond directmapping", "dm"),
    ("vn — anicond volumenormalized", "vn"),
    ("tbe — anicond tractbased_e", "tbe"),
)
DEFAULT_CONDUCTIVITY_TYPE = "iso"

# --- Field statistics -------------------------------------------------------
# The percentiles of |E| extracted per tissue, and the tails above them that
# the cohort box plots are drawn from. Computed on the mesh, volume-weighted
# (see worker.tissue_field_stats) so they agree with SimNIBS's own summary.
FIELD_PERCENTILES = (99.0, 99.5, 99.9)

# Tissues the statistics and the white/gray views are computed over, as
# (label, SimNIBS element tag). 1 = white matter, 2 = gray matter.
STAT_TISSUES = (("gray_matter", 2), ("white_matter", 1))

# --- Figures ----------------------------------------------------------------
# Saved PNGs are rendered at a higher DPI than the canvas needs: they are
# opened and zoomed into, where the canvas is read at ~600 px in a browser.
# mesh_head's own figures keep its 110 -- this is mesh_sim's alone.
FIGURE_DPI = 200
# Slice index labels, 50% up from the 7 pt the other tasks' mosaics use, so
# they stay readable once a figure is scaled down into a document.
SLICE_FONTSIZE = 10.5

# The coil marker: one colour, and thick enough to find on a busy mosaic.
COIL_COLOR = "#00e676"
COIL_LINEWIDTH = 2.4          # 2x the 1.2 it started at
COIL_MARKER_EDGEWIDTH = 3.2   # 2x the 1.6 it started at

# The field's colour ramp on the cortex figures. 'jet' by request: it is what
# the study's own figures use, so a served picture and a paper figure are
# read the same way.
FIELD_CMAP = "jet"

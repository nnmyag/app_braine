# ============================================================================
# tasks/mesh_sim/worker.py — the half that runs under SimNIBS's own Python
# ----------------------------------------------------------------------------
# NOT importable from the app. This file is executed as a script by
# SIMNIBS_PYTHON (see service.py), because `simnibs` cannot be imported from
# the environment the web app runs in and the two never share a dependency
# tree. It therefore imports NOTHING from this repository -- not settings,
# not app/, not its own package -- only the standard library and simnibs.
#
#     simnibs_python worker.py <spec.json> <result.json>
#
# `spec.json` is written by service.py and holds plain values only (paths,
# names, numbers). Progress goes to stdout, one line at a time, which
# service.py streams into the page's console. `result.json` is the answer:
# the files that were produced, plus the coil placement that was actually
# used, so the app can draw it without a second SimNIBS call.
#
# What one run does, in SimNIBS's own terms:
#     SESSION            the head model, where to write, what to post-process
#       .map_to_surf     interpolate the field onto the central cortical
#                        surface (this is what msh2cortex does on the command
#                        line: subject_overlays/, FreeSurfer curv per hemi)
#       + pial / white   the same interpolation, done here, onto the other
#                        two surfaces the head model ships (see
#                        surface_overlays) -- SimNIBS's own `depth` argument
#                        is a no-op in 4.6, so the central one is all it does
#       .map_to_vol      resample the field onto a NIfTI grid
#                        (subject_volumes/, what msh2nii does)
#       TMSLIST          one coil model, the per-tissue isotropic
#                        conductivities from the app's table (spec
#                        "conductivities"), optionally an anisotropic
#                        conductivity tensor and how to map it
#         POSITION       centre + handle-direction electrode, coil-scalp
#                        distance, dI/dt -- SimNIBS turns these into the 4x4
#                        `matsimnibs` that places the coil (see calc_matsimnibs)
# ============================================================================
import json
import os
import sys
import traceback


def say(message):
    """One progress line to stdout, flushed: service.py is reading this live."""
    print(message, flush=True)


# ============================================================================
# Field statistics per tissue
# ----------------------------------------------------------------------------
# The E-field analysis plan: the 99th / 99.5th / 99.9th percentile of |E| in
# each tissue mask, and enough of the distribution ABOVE each of those
# thresholds -- the "percentile-to-maximum" tail -- to draw a box plot of it
# later without keeping a million element values around.
#
# Computed here, on the mesh, rather than downstream on the resampled NIfTI:
# a tetrahedral mesh has wildly varying element sizes, so a percentile is only
# meaningful weighted by element VOLUME. That is what SimNIBS's own
# fields_summary does (ElementData.get_percentiles walks the cumulative
# weights), and matching it means the numbers here and the numbers in
# fields_summary.txt agree instead of quietly differing by a few percent.
# ============================================================================
TISSUE_TAGS = (("gray_matter", 2), ("white_matter", 1))    # ElementTags.GM/.WM


def _weighted_quantile(values, weights, q):
    """Volume-weighted quantile, by SimNIBS's own cumulative-weight rule."""
    import numpy as np
    order = np.argsort(values)
    v = values[order]
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    return float(v[np.argmin(np.abs(cumulative - q))])


def tissue_field_stats(mesh, tag, percentiles, field_name="magnE"):
    """Percentiles of |E| in one tissue, plus the tail above each of them."""
    import numpy as np
    try:
        sub = mesh.crop_mesh(tags=[int(tag)])
    except Exception:                       # noqa: BLE001 - tissue absent, not fatal
        return None
    if field_name not in sub.field:
        return None
    field = sub.field[field_name]
    values = np.asarray(field.value, dtype=float).reshape(-1)
    volumes = np.asarray(sub.elements_volumes_and_areas()[:], dtype=float)
    if values.size == 0 or values.size != volumes.size:
        return None

    thresholds = field.get_percentiles(list(percentiles))
    out = {
        "n_elements": int(values.size),
        "volume_mm3": float(volumes.sum()),
        "max": float(values.max()),
        "mean": float(np.average(values, weights=volumes)),
        "percentiles": {},
    }
    for percentile, threshold in zip(percentiles, np.atleast_1d(thresholds)):
        selected = values >= float(threshold)
        tail, tail_w = values[selected], volumes[selected]
        if tail.size == 0:
            continue
        # min is the threshold by construction; it is written out anyway so a
        # box plot can be drawn straight from the row without re-deriving it.
        out["percentiles"][f"{percentile:g}"] = {
            "threshold": float(threshold),
            "n_elements": int(tail.size),
            "volume_mm3": float(tail_w.sum()),
            "min": float(tail.min()),
            "q1": _weighted_quantile(tail, tail_w, 0.25),
            "median": _weighted_quantile(tail, tail_w, 0.50),
            "q3": _weighted_quantile(tail, tail_w, 0.75),
            "max": float(tail.max()),
            "mean": float(np.average(tail, weights=tail_w)),
        }
    return out


# ============================================================================
# The field on the pial and white surfaces
# ----------------------------------------------------------------------------
# SimNIBS's map_to_surf interpolates onto the CENTRAL surface only (its
# `depth` parameter is accepted and ignored in 4.6). The m2m folder ships the
# pial and white surfaces too, and the same call --
# ElementData.interpolate_to_surface, the field looked up in the tetrahedra
# under each vertex -- puts the field on them just as well. Done here, on the
# mesh, rather than downstream by sampling the resampled NIfTI: a vertex on a
# tissue boundary would otherwise read a blend of the two sides and of the
# zero the NIfTI carries outside the tissues it was written for.
#
# Which tetrahedra each surface reads from: the pial surface is the GM/CSF
# boundary and reads GM (as SimNIBS's central interpolation does); the white
# surface is the GM/WM boundary and reads WM -- it is the white MATTER's
# surface, and what a tract reaching it sees. A vertex that lands a hair
# outside its tissue gets the nearest element's value (out_fill='nearest').
# Written in SimNIBS's own naming, `<hemi>.<sim>.<surface>.E.<quantity>`,
# beside the central overlays, so one reader serves all three.
# ============================================================================
SURFACE_OVERLAYS = (("pial", (2,)), ("white", (1,)))     # surface -> tags read
SURFACE_QUANTITIES = ("magn", "normal")


def surface_overlays(mesh, sub_files, mesh_path, out_dir):
    """The E-field on the pial and white surfaces, as FreeSurfer curv files.
    Returns the paths written."""
    import numpy as np
    from simnibs.mesh_tools import mesh_io

    if "E" not in mesh.field:
        say("Surfaces: no vector field E on the mesh, pial/white skipped")
        return []
    # WM + GM + CSF, as middle_gm_interpolation crops it: convex enough that
    # a surface vertex is never far from a tetrahedron it may read.
    cropped = mesh.crop_mesh(tags=[1, 2, 3])
    field = cropped.field["E"]
    sim_name = "." + os.path.splitext(os.path.basename(mesh_path))[0]
    written = []
    for surface, tags in SURFACE_OVERLAYS:
        try:
            surfaces = mesh_io.load_subject_surfaces(sub_files, surface)
        except Exception as exc:            # noqa: BLE001 - surface absent, not fatal
            say(f"Surfaces: no {surface} surface in the head model, skipped "
                f"({type(exc).__name__}: {exc})")
            continue
        th_indices = cropped.elm.get_tags(list(tags),
                                          return_element_numbers=True)
        for hemi, surf in surfaces.items():
            interpolated = field.interpolate_to_surface(
                surf, out_fill="nearest", th_indices=th_indices)
            for quantity in SURFACE_QUANTITIES:
                if quantity == "magn":
                    values = interpolated.norm().value
                else:
                    # SimNIBS's own sign flip (calc_quantities in
                    # transformations.py): positive = out of the cortex.
                    values = -interpolated.normal().value
                out = os.path.join(out_dir,
                                   f"{hemi}{sim_name}.{surface}.E.{quantity}")
                mesh_io.write_curv(out, np.asarray(values).squeeze(),
                                   surf.elm.nr)
                written.append(out)
        say(f"Surfaces: E on the {surface} surface "
            f"({', '.join(SURFACE_QUANTITIES)}) -> subject_overlays/")
    return written


# ============================================================================
# Conductivity
# ----------------------------------------------------------------------------
# Two departures from stock SimNIBS, both because the app knows what it is
# feeding the solver better than SimNIBS's defaults assume:
#
#   * The isotropic values per tissue come from the spec ("conductivities":
#     [{tag, value, table_tissue}, ...]), i.e. the same target table anicond
#     computes its tensors against -- not SimNIBS's built-in Wagner 2004 set.
#     A tag the spec does not list keeps SimNIBS's value; the result file
#     reports every value actually used, so the run report can show them.
#
#   * Under anisotropy_type 'dir' with spec "tensor_as_is" true, the tensor
#     file is used AS-IS. Stock TMSLIST.cond2elmdata calls
#     cond_utils.cond2elmdata with its default correct_intensity=True, which
#     rescales the whole tensor field by one global factor so its mean
#     matches the isotropic reference, then clips eigenvalues at
#     aniso_maxcond / aniso_maxratio. That is for diffusion-derived tensors
#     that still need a conductivity magnitude; ours are already conductivity
#     tensors in S/m. The subclass below makes that one call itself with
#     correct_intensity=False and the caps lifted. Everything else in that
#     function still runs: interpolation onto WM/GM elements, the FSL
#     voxel->world rotation of the components, and the replacement of
#     all-zero / negative-semi-definite tensors with the tissue's isotropic
#     value. No diffusion-to-conductivity conversion happens anywhere here.
# ============================================================================
def apply_conductivities(tmslist, entries):
    """Overwrite tmslist.cond[tag-1].value for every {tag, value} given.
    Returns one dict per tissue slot that carries a value afterwards, so the
    caller can report exactly what the solver saw."""
    given = {}
    for entry in entries or []:
        tag = int(entry["tag"])
        if tag < 1 or tag > len(tmslist.cond):
            raise ValueError(f"conductivity for tag {tag}: no such tissue "
                             f"slot in SimNIBS (1..{len(tmslist.cond)})")
        tmslist.cond[tag - 1].value = float(entry["value"])
        given[tag] = entry
    used = []
    for index, cond in enumerate(tmslist.cond):
        if cond.value is None:
            continue
        tag = index + 1
        used.append({
            "tag": tag,
            "simnibs_name": cond.name or "",
            "value": float(cond.value),
            "source": "table" if tag in given else "simnibs_default",
            "table_tissue": (given[tag].get("table_tissue", "")
                             if tag in given else ""),
        })
    return used


def tensor_qc(cond, mesh, tags):
    """Eigenvalue range of the conductivity the solver will use, per tissue
    tag: what actually went in, after interpolation and any fixing."""
    import numpy as np
    out = {}
    values = np.asarray(cond.value, dtype=float)
    if values.ndim != 2 or values.shape[1] != 9:
        return out
    tensors = values.reshape(-1, 3, 3)
    for tag in tags:
        select = mesh.elm.tag1 == int(tag)
        if not select.any():
            continue
        eig = np.linalg.eigvalsh(tensors[select])
        ratio = eig[:, 2] / np.maximum(eig[:, 0], 1e-12)
        out[str(int(tag))] = {
            "n_elements": int(select.sum()),
            "eig_min": float(eig.min()),
            "eig_p1": float(np.percentile(eig, 1)),
            "eig_median": float(np.median(eig)),
            "eig_p99": float(np.percentile(eig, 99)),
            "eig_max": float(eig.max()),
            "mean_trace3": float(np.trace(tensors[select], axis1=1,
                                          axis2=2).mean() / 3),
            "ratio_median": float(np.median(ratio)),
            "ratio_max": float(ratio.max()),
            "n_eig_above_2": int((eig.max(axis=1) > 2.0 * (1 + 1e-6)).sum()),
        }
    return out


def make_tmslist(sim_struct, tensor_as_is, max_cond=None, max_ratio=None):
    """A TMSLIST that (a) reports the tensor it hands the solver and (b),
    when `tensor_as_is`, hands a 'dir' tensor through without SimNIBS's
    intensity correction, capped only by `max_cond` / `max_ratio` (None =
    uncapped)."""
    from simnibs.utils import cond_utils

    class AppTMSLIST(sim_struct.TMSLIST):
        # SimNIBS names its output files and .mat fields by this, and tests
        # it against the literal "TMSLIST"; the base class derives it from
        # the class name, which a subclass must therefore pin back.
        @property
        def type(self):
            return "TMSLIST"

        def cond2elmdata(self, mesh=None, excentricity_scale=None,
                         logger_level=20):
            if mesh is None:
                mesh = self.mesh
            if self.anisotropy_type == "dir" and tensor_as_is:
                if mesh is None:
                    raise ValueError("The mesh for this simulation is not set")
                image, affine = self._get_vol_info()
                say("Conductivity: tensor used as-is (correct_intensity off; "
                    f"eigenvalue cap {max_cond if max_cond else 'none'} S/m, "
                    f"ratio cap {max_ratio if max_ratio else 'none'})")
                cond = cond_utils.cond2elmdata(
                    mesh, [c.value for c in self.cond],
                    anisotropy_volume=image, affine=affine,
                    aniso_tissues=self.anisotropic_tissues,
                    correct_intensity=False,
                    max_cond=float(max_cond) if max_cond else float("inf"),
                    max_ratio=float(max_ratio) if max_ratio else float("inf"),
                    excentricity_scaling=excentricity_scale)
            else:
                cond = super().cond2elmdata(mesh, excentricity_scale,
                                            logger_level)
            if self.anisotropy_type != "scalar" and mesh is not None:
                self.tensor_qc = tensor_qc(cond, mesh, self.anisotropic_tissues)
                for tag, qc in self.tensor_qc.items():
                    # "Conductivity:" is one of service.py's console
                    # markers; keep it verbatim so this line is shown.
                    say(f"Conductivity: QC tag {tag} -- eigenvalues "
                        f"{qc['eig_min']:.3g} .. {qc['eig_max']:.3g} S/m "
                        f"(median {qc['eig_median']:.3g}, p99 "
                        f"{qc['eig_p99']:.3g}), mean trace/3 "
                        f"{qc['mean_trace3']:.3g}, ratio max "
                        f"{qc['ratio_max']:.3g}, {qc['n_eig_above_2']} "
                        f"element(s) above 2 S/m")
            return cond

    return AppTMSLIST()


def main(spec_path, result_path):
    with open(spec_path) as fh:
        spec = json.load(fh)

    # Imported here rather than at module scope so a failure to import
    # simnibs is reported through the same result file as any other error.
    from simnibs import SubjectFiles, run_simnibs
    from simnibs.simulation import sim_struct

    m2m_dir = spec["m2m_dir"]
    out_dir = spec["out_dir"]
    sub_files = SubjectFiles(subpath=m2m_dir)
    say(f"SimNIBS: head mesh {os.path.basename(sub_files.fnamehead)}")

    session = sim_struct.SESSION()
    session.subpath = m2m_dir
    session.fnamehead = sub_files.fnamehead
    session.pathfem = out_dir
    session.eeg_cap = spec["eeg_cap"]
    # Headless server: gmsh must never be launched, or the run blocks forever
    # waiting for a window nobody can see.
    session.open_in_gmsh = False
    # The two post-processing steps that make the result readable: the field
    # on the cortical surface (what the page draws) and on a NIfTI grid (what
    # other tools read). Both need subpath, set above.
    session.map_to_surf = True
    session.map_to_vol = True
    # SimNIBS writes only gray matter into the NIfTI by default (tag 2). The
    # white-matter views and the white-matter percentiles both need tag 1 as
    # well, so ask for both -- the volume is the same size either way, it is
    # only a question of which tissues are left non-zero in it.
    session.tissues_in_niftis = list(spec.get("tissues_in_niftis", [1, 2]))
    session.fields = spec.get("fields", "eE")

    anisotropy = spec.get("anisotropy_type", "scalar")
    tensor_as_is = bool(spec.get("tensor_as_is", False)) and anisotropy == "dir"
    tmslist = session.add_tmslist(make_tmslist(
        sim_struct, tensor_as_is, spec.get("tensor_max_cond"),
        spec.get("tensor_max_ratio")))
    tmslist.fnamecoil = spec["coil_file"]

    conductivities = apply_conductivities(tmslist, spec.get("conductivities"))
    from_table = [c for c in conductivities if c["source"] == "table"]
    if from_table:
        say("Conductivity: isotropic values from the app's table -- " +
            ", ".join(f"{c['simnibs_name']} {c['value']:g}" for c in from_table)
            + " S/m")
    else:
        say("Conductivity: WARNING -- no table values given, SimNIBS's "
            "built-in isotropic values in use")

    if anisotropy != "scalar":
        # fname_tensor is set on the SESSION and copied down to the poslist
        # in SESSION._prepare; anisotropy_type belongs to the poslist itself.
        session.fname_tensor = spec["tensor_file"]
        tmslist.anisotropy_type = anisotropy
        say(f"Conductivity: anisotropic ({anisotropy}) from "
            f"{os.path.basename(spec['tensor_file'])} in WM and GM"
            + ("; tensor used as-is" if tensor_as_is else
               "; SimNIBS intensity correction and eigenvalue caps apply"))
    else:
        say("Conductivity: isotropic in every tissue (no tensor)")

    position = tmslist.add_position()
    position.centre = spec["coil_center"]
    position.pos_ydir = spec["coil_ydir"]
    position.distance = float(spec["distance"])
    position.didt = float(spec["didt"])
    say(f"Coil: {os.path.basename(spec['coil_file'])} at {spec['coil_center']}, "
        f"handle toward {spec['coil_ydir']}, {position.distance:g} mm off the "
        f"scalp, dI/dt {position.didt:g} A/s")

    say("Solving the FEM — this is the long step …")
    run_simnibs(session, cpus=int(spec.get("cpus", 1)))

    # -- what came out ------------------------------------------------------
    # SESSION.run names the result <meshname>_TMS_1-<n>_<coil>_scalar.msh and
    # puts the post-processing beside it; rather than re-deriving those names
    # here, look at what is actually on disk.
    field_meshes = sorted(
        os.path.join(out_dir, f) for f in os.listdir(out_dir)
        if f.endswith(".msh"))
    overlays_dir = os.path.join(out_dir, "subject_overlays")
    volumes_dir = os.path.join(out_dir, "subject_volumes")

    result = {
        "ok": True,
        "field_meshes": field_meshes,
        "overlays": (sorted(os.path.join(overlays_dir, f)
                            for f in os.listdir(overlays_dir))
                     if os.path.isdir(overlays_dir) else []),
        "volumes": (sorted(os.path.join(volumes_dir, f)
                           for f in os.listdir(volumes_dir))
                    if os.path.isdir(volumes_dir) else []),
        "head_mesh": sub_files.fnamehead,
        "hemispheres": list(sub_files.hemispheres),
        "conductivities": conductivities,
        "tensor_as_is": tensor_as_is,
        "tensor_qc": getattr(tmslist, "tensor_qc", {}),
    }

    # The coil placement SimNIBS actually used. matsimnibs is filled in during
    # the run (POSITION.calc_matsimnibs); its 4th column is the coil centre in
    # world mm and its 3rd column the direction pointing into the head, which
    # is all the app needs to draw where the coil sat.
    matsimnibs = getattr(position, "matsimnibs", None)
    if matsimnibs is not None:
        matsimnibs = [[float(v) for v in row] for row in matsimnibs]
        result["matsimnibs"] = matsimnibs
        result["coil_centre"] = [row[3] for row in matsimnibs[:3]]
        result["coil_normal"] = [row[2] for row in matsimnibs[:3]]

    # -- the field on the pial and white surfaces ---------------------------
    mesh = None
    if field_meshes:
        from simnibs.mesh_tools import mesh_io
        say("Surfaces: interpolating E onto the pial and white surfaces …")
        try:
            mesh = mesh_io.read_msh(field_meshes[0])
            os.makedirs(overlays_dir, exist_ok=True)
            surface_overlays(mesh, sub_files, field_meshes[0], overlays_dir)
            result["overlays"] = sorted(os.path.join(overlays_dir, f)
                                        for f in os.listdir(overlays_dir))
        except Exception as exc:        # noqa: BLE001 - an export, not the result
            say(f"Surfaces: pial/white skipped ({type(exc).__name__}: {exc})")

    # -- per-tissue field statistics ----------------------------------------
    percentiles = [float(p) for p in spec.get("percentiles", [99, 99.5, 99.9])]
    if field_meshes:
        from simnibs.mesh_tools import mesh_io
        say("Statistics: percentiles of |E| per tissue, on the mesh …")
        try:
            if mesh is None:
                mesh = mesh_io.read_msh(field_meshes[0])
            stats = {}
            for name, tag in TISSUE_TAGS:
                tissue = tissue_field_stats(mesh, tag, percentiles)
                if tissue is None:
                    say(f"Statistics: no {name.replace('_', ' ')} elements, "
                        f"skipped")
                    continue
                stats[name] = tissue
                line = ", ".join(
                    f"p{p}={tissue['percentiles'][p]['threshold']:.3g}"
                    for p in tissue["percentiles"])
                say(f"Statistics: {name.replace('_', ' ')} — {line} V/m "
                    f"(max {tissue['max']:.3g}, "
                    f"{tissue['volume_mm3'] / 1000:.0f} mL)")
            result["tissue_stats"] = stats
            result["percentiles"] = percentiles
        except Exception as exc:        # noqa: BLE001 - statistics, not the result
            say(f"Statistics: skipped ({type(exc).__name__}: {exc})")

    # NOT converted to VTK here. simnibs.mesh_io.write_vtk needs pyvista,
    # which this install does not ship, and the volumetric result it would
    # write is a tetrahedral unstructured grid -- not the surface polydata
    # 3D Slicer loads as a Model anyway. The Slicer-ready artefact is the
    # cortical surface with the field on it, which service.py builds in the
    # app's own environment from the gifti surfaces and the curv overlays
    # below (vendor/gii2vtk). The volume field stays a .msh, which is gmsh's
    # own format and what SimNIBS's own viewer opens.
    coil_geo = [os.path.join(out_dir, f) for f in os.listdir(out_dir)
                if f.endswith("_coil_pos.geo")]
    if coil_geo:
        result["coil_geo"] = sorted(coil_geo)
    summary = os.path.join(out_dir, "fields_summary.txt")
    if os.path.isfile(summary):
        result["fields_summary"] = summary
        # Worth echoing: it is the one place the run states its own peak
        # field, and a user watching the console should not have to open a
        # file to see whether the simulation produced anything sensible.
        with open(summary) as fh:
            for line in fh:
                line = line.rstrip()
                if line:
                    say(f"summary | {line}")

    with open(result_path, "w") as fh:
        json.dump(result, fh, indent=2)
    say("Worker finished.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: worker.py <spec.json> <result.json>", file=sys.stderr)
        raise SystemExit(2)
    try:
        main(sys.argv[1], sys.argv[2])
    except Exception as exc:                # noqa: BLE001 - reported, not swallowed
        # The traceback goes to stdout so it lands in the page's console next
        # to the steps that led to it, AND into the result file so service.py
        # can raise it as a real error rather than guessing from an exit code.
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        try:
            with open(sys.argv[2], "w") as fh:
                json.dump({"ok": False,
                           "error": f"{type(exc).__name__}: {exc}",
                           "traceback": traceback.format_exc()}, fh, indent=2)
        except OSError:
            pass
        raise SystemExit(1)

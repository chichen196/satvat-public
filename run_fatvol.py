#!/usr/bin/env python3
"""
Driver for abdominal fat volume estimation — replaces EXECESTFATVOLMRIV2.

What changed from the IDL driver, and why
-----------------------------------------
* No CD mounting, no rsync-to-scratch, no `spawn` shell-outs, no /tmp text file
  used to pass directory listings back into the program. Directories are given
  on the command line; pathlib does the walking.
* No interactive "hit y to continue" loop. Point it at a directory tree and it
  processes everything it finds.
* The raw-data deletion at the end of each patient (`rm -Rf .../im[1-9]*`) is
  gone. It made sense when scratch space was scarce; it is a bad default now.
* Results accumulate into ONE CSV rather than a text file per series, which is
  what "spreadsheet-friendly" should have meant.
* Output volumes are NIfTI rather than ANALYZE .hdr/.img pairs.

Usage
-----
    python run_fatvol.py --self-test              # no data needed
    python run_fatvol.py --input /path/to/dicom --output results/
    python run_fatvol.py --input /path/to/one/series --single --output results/
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

from fatvol import est_fat_vol, SUBCUTANEOUS, TOTAL
from fatvol.io import (index_dicom_tree, is_target_series, make_phantom,
                       read_series_from_files, read_tags, save_nifti)


def split_merged_series(series, verbose=True):
    """Split any indexed series whose files span more than one directory.

    Some exports (e.g. (X)MedCon 0.8.13) give different series of a study the
    SAME SeriesInstanceUID while SOPInstanceUIDs stay unique, so grouping by UID
    alone merges them into one stack. When each series sits in its own folder,
    the containing directory is the reliable series identity; description and
    series number are re-read from the first file of each part.
    """
    import copy
    try:
        import pydicom
    except ImportError:            # pragma: no cover
        pydicom = None
    out, nsplit = [], 0
    for s in series:
        groups = {}
        for f in s.files:
            groups.setdefault(Path(f).parent, []).append(f)
        if len(groups) <= 1:
            out.append(s)
            continue
        nsplit += 1
        for d, files in sorted(groups.items()):
            t = copy.copy(s)
            t.files = list(files)
            try:
                t.directory = type(s.directory)(d) if s.directory is not None else str(d)
            except Exception:
                t.directory = str(d)
            try:
                t.n_slices = len(files)
            except AttributeError:
                pass
            if pydicom is not None:
                try:
                    ds = pydicom.dcmread(files[0], stop_before_pixels=True)
                    t.description = str(getattr(ds, "SeriesDescription", "") or s.description)
                    t.series_number = str(getattr(ds, "SeriesNumber", "") or s.series_number)
                except Exception:
                    pass
            out.append(t)
    if nsplit and verbose:
        print(f"note: {nsplit} series UIDs spanned more than one directory (MedCon "
              f"export shares SeriesInstanceUIDs across series); split by directory "
              f"-> {len(out)} series")
    return out

FIELDS = ["series_key", "patient_id", "study", "series", "n_slices", "pxsize", "pysize",
          "sspacing", "total_fat_ml", "subcutaneous_fat_ml", "visceral_fat_ml",
          "is_abd_up", "source", "notes"]


# --------------------------------------------------------------------------
def analyse_volume(images, pxsize, pysize, sspacing, *, args, verbose=False):
    """Run the total and subcutaneous passes; visceral is the difference.

    This mirrors the IDL driver, which computed tfatvol and pfatvol and then set
    vfatvol = tfatvol - pfatvol rather than running a third segmentation.
    """
    total = est_fat_vol(images, pxsize, pysize, sspacing, fat_loc=TOTAL,
                        thresh_method=args.thresh_method, nobins=args.nobins,
                        is_vol_thresh=args.vol_thresh, is_arm_in=args.arms,
                        idl_integer_division=not args.clean_division,
                        thresh_factor=args.thresh_factor,
                        threshold_rule=args.threshold_rule,
                        thresh_mask_slice=args.thresh_mask_slice,
                        verbose=verbose)
    abd = None if args.abd_up == "auto" else int(args.abd_up)
    sub = est_fat_vol(images, pxsize, pysize, sspacing, fat_loc=SUBCUTANEOUS,
                      thresh_method=args.thresh_method, nobins=args.nobins,
                      is_vol_thresh=args.vol_thresh, is_arm_in=args.arms,
                      idl_integer_division=not args.clean_division,
                      thresh_factor=args.thresh_factor,
                      threshold_rule=args.threshold_rule,
                      abd_up=abd,
                      fit_rule=args.fit_rule,
                      visceral_jump_test=not args.no_visceral_jump_test,
                      thresh_mask_slice=args.thresh_mask_slice,
                      verbose=verbose)
    visceral = total.volume_ml - sub.volume_ml
    return total, sub, visceral


def overlay(images, edge):
    """The IDL `(images + edge*max) < max` display volume.

    Kept for continuity with the original, but note this BAKES the edge into the
    pixel data — once written, the mask cannot be separated again. Prefer
    --save-masks and load the labels as a layer in a viewer.
    """
    m = float(images.max())
    return np.minimum(images + edge * m, m)


def label_volume(sub_result, visceral_mask=None):
    """Combined label map: 0 background, 1 subcutaneous, 2 visceral.

    One file, two labels, so a viewer shows both compartments in one layer with
    an indexed colour table. Visceral is written second and wins any overlap,
    which cannot occur anyway since the two masks are disjoint by construction.
    """
    lab = np.zeros(sub_result.fmask.shape, np.uint8)
    lab[sub_result.fmask.astype(bool)] = 1
    if visceral_mask is not None:
        lab[visceral_mask.astype(bool)] = 2
    return lab


# --------------------------------------------------------------------------
def self_test(args) -> int:
    """Run the whole pipeline on a synthetic phantom. No patient data needed."""
    print("=" * 68)
    print("SELF TEST — synthetic abdominal phantom, no DICOM required")
    print("=" * 68)
    ok = True

    # Oct 2026: the row-run erosion/dilation used for large structuring elements must give
    # exactly what scipy gives on this machine (same centre, reflection and border rules)
    from scipy import ndimage
    from fatvol import idlcompat as ic
    rng = np.random.default_rng(7)
    nmis = ncmp = 0
    for side in (16, 17, 24, 31, 40, 63, 64):
        for r in (side // 2 - 1, side // 2, side):
            s = ic.bdisk(r, side)
            for _ in range(3):
                img = ndimage.gaussian_filter(rng.random((int(rng.integers(40, 120)), int(rng.integers(40, 120)))), 3) > 0.5
                for erosion in (True, False):
                    ref = (ndimage.binary_erosion(img, structure=s.astype(bool), border_value=1) if erosion
                           else ndimage.binary_dilation(img, structure=s.astype(bool), border_value=0))
                    ncmp += 1
                    nmis += not np.array_equal(ic._row_morph(img, s, erosion), ref)
    print(f"\n  {'PASS' if nmis == 0 else 'FAIL'}  large-element morphology = scipy "
          f"({ncmp} comparisons, {nmis} mismatches)")
    ok &= nmis == 0

    for label, kw, want_abd_up in (("abdomen up", dict(abdomen_up=True), 1),
                                   ("abdomen down", dict(abdomen_up=False), 0)):
        vol, truth, meta = make_phantom(nx=args.phantom_size, ny=args.phantom_size,
                                        nz=args.phantom_slices, **kw)
        px, py, ss = meta["pxsize"], meta["pysize"], meta["sspacing"]
        voxel_ml = px * py * ss / 1000.0
        true_sub = float((truth == 1).sum()) * voxel_ml
        true_vis = float((truth == 2).sum()) * voxel_ml

        t0 = time.time()
        total, sub, visceral = analyse_volume(vol, px, py, ss, args=args,
                                              verbose=args.verbose)
        dt = time.time() - t0

        print(f"\n[{label}]  {vol.shape}  {dt:.1f}s")
        print(f"  total fat        {total.volume_ml:9.1f} ml   "
              f"(phantom truth {true_sub + true_vis:.1f})")
        print(f"  subcutaneous fat {sub.volume_ml:9.1f} ml   "
              f"(phantom truth {true_sub:.1f})")
        print(f"  visceral fat     {visceral:9.1f} ml   "
              f"(phantom truth {true_vis:.1f})")
        print(f"  IsAbdUp detected {sub.is_abd_up}")
        for n in sub.notes[:5]:
            print(f"  note: {n}")

        # sanity checks — these test that the code RUNS correctly, not that the
        # segmentation is accurate on a phantom it was never designed for
        checks = [
            ("total fat > 0", total.volume_ml > 0),
            ("subcutaneous > 0", sub.volume_ml > 0),
            ("subcutaneous <= total", sub.volume_ml <= total.volume_ml * 1.001),
            ("visceral >= 0", visceral >= -1e-6),
            ("masks are the right shape", sub.fmask.shape == vol.shape),
            ("edge map is binary", set(np.unique(sub.maskedge)) <= {0, 1}),
            ("thresholds finite", np.all(np.isfinite(sub.thresholds))),
            ("subcutaneous mask is inside the fat mask",
             not (sub.fmask.astype(bool) & ~sub.fatmask.astype(bool)).any()),
            ("total >= subcutaneous + visceral",
             abs(total.volume_ml - (sub.volume_ml + visceral)) < 1e-6),
            (f"IsAbdUp detected correctly (expected {want_abd_up})",
             sub.is_abd_up == want_abd_up),
            ("zone map present, right shape, values 0/1/2",
             sub.zones is not None and sub.zones.shape == vol.shape
             and set(np.unique(sub.zones)) <= {0, 1, 2}),
            ("zone map: body part = body mask",
             sub.zones is not None
             and np.array_equal(sub.zones > 0, sub.bdmask.astype(bool))),
            ("visceral fat lies inside the visceral zone",
             sub.zones is not None and sub.vfmask is not None
             and not (sub.vfmask.astype(bool) & (sub.zones != 2)).any()),
        ]
        if sub.zones is not None:
            fat = sub.fatmask.astype(bool)
            pf = sub.fmask.astype(bool)
            vz = sub.zones == 2
            print(f"  zone vs method: fat in zone 2 called subcut {int((fat & pf & vz).sum())} vox, "
                  f"fat in zone 1 in neither {int((fat & ~pf & ~vz).sum())} vox")
        for name, passed in checks:
            print(f"  {'PASS' if passed else 'FAIL'}  {name}")
            ok &= bool(passed)

    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        vol, truth, meta = make_phantom(nx=args.phantom_size, ny=args.phantom_size,
                                        nz=args.phantom_slices)
        total, sub, _ = analyse_volume(vol, meta["pxsize"], meta["pysize"],
                                       meta["sspacing"], args=args)
        px, py, ss = meta["pxsize"], meta["pysize"], meta["sspacing"]
        save_nifti(vol, out / "phantom_image.nii.gz", px, py, ss)
        save_nifti(label_volume(sub, sub.vfmask), out / "phantom_labels.nii.gz",
                   px, py, ss)
        if sub.zones is not None:
            save_nifti(sub.zones, out / "phantom_zones.nii.gz", px, py, ss)
        save_nifti(overlay(vol, sub.maskedge),
                   out / "phantom_subcutaneous.nii.gz", px, py, ss)
        save_nifti(overlay(vol, total.maskedge),
                   out / "phantom_total.nii.gz", px, py, ss)
        print(f"\nvolumes written to {out}")
        print("  phantom_image.nii.gz + phantom_labels.nii.gz  <- load as layers")
        print("  phantom_*fat.nii.gz                           <- burned-in edges")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, help="directory tree of DICOM files")
    p.add_argument("--output", type=Path, default=Path("fatvol_results"),
                   help="where results go (default: fatvol_results)")
    p.add_argument("--list", action="store_true",
                   help="list the series found and exit, without segmenting")
    p.add_argument("--series", default=None,
                   help="only series whose description contains this text "
                        "(case-insensitive)")
    p.add_argument("--exclude", default="T2",
                   help="comma-separated description tokens to skip "
                        "(default: T2; use '' to disable)")
    p.add_argument("--any-plane", action="store_true",
                   help="do not require the series to be axial")
    p.add_argument("--min-slices", type=int, default=3)
    p.add_argument("--no-dicomdir", action="store_true",
                   help="ignore the CD's DICOMDIR index and scan every file "
                        "(slower; use if the DICOMDIR is wrong or incomplete)")
    p.add_argument("--self-test", action="store_true",
                   help="run on a synthetic phantom; no data needed")
    p.add_argument("--all-series", action="store_true",
                   help="process every series, not just axial non-T2 ones")
    p.add_argument("--thresh-method", type=int, default=1, choices=(0, 1),
                   help="0 lowest-point, 1 centre-of-weight (default)")
    p.add_argument("--nobins", type=int, default=100)
    p.add_argument("--vol-thresh", action="store_true",
                   help="threshold over the whole volume instead of per slice")
    p.add_argument("--arms", type=int, default=0,
                   help="1 if arms are inside the axial field of view")
    p.add_argument("--thresh-factor", type=float, default=1.0,
                   help="multiplier on the threshold before the fat mask is taken "
                        "(1.0 = validated; other values for sensitivity runs only)")
    p.add_argument("--abd-up", choices=("auto", "0", "1"), default="auto",
                   help="IsAbdUp for the visceral pass: auto = the .pro's detection "
                        "(default); 0 or 1 forces it, noted in the CSV when it differs")
    p.add_argument("--threshold-rule", choices=("rc", "idl", "float"), default="rc",
                   help="fat threshold rule. rc (default): plain Ridler-Calvard on the unit-bin "
                        "body-masked histogram - the rule the 2006 paper describes. "
                        "idl: ESTFATVOLMRIV2.PRO lines 293-336 with IDL's integer-histogram bin "
                        "width. float: the same loop on a float-width histogram")
    p.add_argument("--fit-rule", choices=("strict", "fov", "any"), default="strict",
                   help="ellipse fit acceptance (port-corrections table only): strict (default) "
                        "= converged, in-field fits only; fov = any in-field fit; any = whatever "
                        "the optimiser returns, as the port did before 17 Sep 2026")
    p.add_argument("--no-visceral-jump-test", action="store_true",
                   help="drop the fit sanity check in the visceral pass, as in the .pro "
                        "(port-corrections table only)")
    p.add_argument("--thresh-mask-slice", type=int, default=None,
                   help="take every slice's threshold inside this slice's body outline, as the "
                        "2003/2008 threshold program did with 0 (port-corrections table only)")
    p.add_argument("--clean-division", action="store_true",
                   help="use true division in the ISODATA threshold instead of "
                        "reproducing IDL's integer division (changes results "
                        "slightly)")
    p.add_argument("--no-images", action="store_true",
                   help="write the CSV only, skip NIfTI overlays")
    p.add_argument("--save-masks", action="store_true",
                   help="also write the segmentation as separate label volumes "
                        "(1 = subcutaneous, 2 = visceral), for loading as an "
                        "overlay layer in MRIcroGL / FSLeyes / ITK-SNAP, plus "
                        "_zones (0 outside body, 1 body outside the visceral "
                        "zone, 2 visceral zone)")
    p.add_argument("--masks-only", action="store_true",
                   help="write the original image plus label volumes, and skip "
                        "the burned-in edge overlays entirely")
    p.add_argument("--any-modality", action="store_true",
                   help="do not reject series on the Modality tag ((X)MedCon "
                        "exports are accepted automatically; this covers anything else)")
    p.add_argument("--no-split", action="store_true",
                   help="do not split a SeriesInstanceUID that spans several "
                        "directories (default: split, needed for exports where "
                        "different series share a series UID)")
    p.add_argument("--keep-ids", action="store_true",
                   help="name outputs and fill the CSV with the DICOM PatientID and "
                        "StudyDate (default: series keys only, so outputs carry no "
                        "patient identifiers)")
    p.add_argument("--phantom-size", type=int, default=192)
    p.add_argument("--phantom-slices", type=int, default=4)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    if args.self_test:
        return self_test(args)
    if not args.input:
        p.error("either --input or --self-test is required")
    if not args.input.exists():
        p.error(f"input not found: {args.input}")

    series = index_dicom_tree(args.input, verbose=True,
                              use_dicomdir=not args.no_dicomdir)
    if not series:
        print("no DICOM series found")
        return 1
    if not args.no_split:
        series = split_merged_series(series)

    excl = tuple(x for x in args.exclude.split(",") if x) if args.exclude else ()
    decisions = []
    for s in series:
        ok, why = is_target_series(s, require_axial=not args.any_plane,
                                   exclude=excl, min_slices=args.min_slices,
                                   require_modality=not args.any_modality)
        if ok and args.series and args.series.lower() not in (s.description or "").lower():
            ok, why = False, f"description does not match {args.series!r}"
        decisions.append((s, ok, why))

    print(f"\n{len(series)} series found:")
    last_study = object()
    for s, ok, why in decisions:
        if s.study_uid != last_study:
            last_study = s.study_uid
            parts = sorted({x.body_part_display()
                            for x, _, _ in decisions if x.study_uid == s.study_uid})
            hdr = " | ".join(filter(None, [
                f"patient {s.patient_id}" if (s.patient_id and args.keep_ids) else "",
                s.study_date if args.keep_ids else "",
                s.study_description or "(no study description)",
                s.manufacturer]))
            print(f"\n  == {hdr}")
            print(f"     body part(s): {', '.join(parts)}")
        mark = "USE " if ok else "skip"
        print(f"  {mark} {s.label()}" + ("" if ok else f"   [{why}]"))
    print("\n  ('~' before a body part means it was guessed from the text, "
          "not read from BodyPartExamined; '?' means neither was available)")
    if args.list:
        return 0

    chosen = [s for s, ok, _ in decisions if ok]
    if not chosen:
        print("\nNothing matched. Try --any-plane, --exclude '' or --series TEXT.")
        return 1

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "fat_volumes.csv"
    rows = []
    t_start = time.time()

    pid = (lambda s: s.patient_id) if args.keep_ids else (lambda s: "")
    for idx, s in enumerate(chosen, 1):
        print(f"\nStudy:  {s.study_description}")
        print(f"Series: {s.description}   ({s.n_slices} slices, {s.plane})")
        try:
            images, meta = read_series_from_files(s.files)
        except Exception as exc:
            print(f"  READ FAILED: {exc}")
            rows.append({"series_key": f"series{idx:03d}", "patient_id": pid(s),
                         "study": s.study_description,
                         "series": s.description, "source": s.directory,
                         "notes": f"READ FAILED: {exc}"})
            continue

        px, py, ss = meta["pxsize"], meta["pysize"], meta["sspacing"]
        try:
            total, sub, visceral = analyse_volume(images, px, py, ss, args=args,
                                                  verbose=args.verbose)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            rows.append({"series_key": f"series{idx:03d}", "patient_id": pid(s),
                         "study": s.study_description,
                         "series": s.description, "source": s.directory,
                         "notes": f"FAILED: {exc}"})
            continue

        print(f"  tfatvol: {total.volume_ml:.2f} ml")
        print(f"  pfatvol: {sub.volume_ml:.2f} ml")
        print(f"  vfatvol: {visceral:.2f} ml")

        if args.keep_ids:
            stem = "_".join(filter(None, [s.patient_id or "unknown", s.study_date,
                                          s.series_number, s.description]))
        else:
            stem = "_".join(filter(None, [f"series{idx:03d}", s.series_number, s.description]))
        stem = stem.replace("/", "-").replace(" ", "_")
        if not args.no_images and not args.masks_only:
            save_nifti(overlay(images, total.maskedge), out / f"{stem}_tfat.nii.gz", px, py, ss)
            save_nifti(overlay(images, sub.maskedge), out / f"{stem}_pfat.nii.gz", px, py, ss)
            if sub.vmaskedge is not None:
                save_nifti(overlay(images, sub.vmaskedge), out / f"{stem}_vfat.nii.gz", px, py, ss)
        if (args.save_masks or args.masks_only) and not args.no_images:
            save_nifti(images, out / f"{stem}_image.nii.gz", px, py, ss)
            save_nifti(label_volume(sub, sub.vfmask), out / f"{stem}_labels.nii.gz", px, py, ss)
            save_nifti(total.fmask.astype(np.uint8),
                       out / f"{stem}_totalfat_mask.nii.gz", px, py, ss)
            if sub.zones is not None:
                # 0 outside body / 1 body outside visceral zone / 2 visceral zone
                save_nifti(sub.zones, out / f"{stem}_zones.nii.gz", px, py, ss)

        rows.append({
            "series_key": f"series{idx:03d}", "patient_id": pid(s),
            "study": s.study_description,
            "series": s.description, "n_slices": meta["n_slices"],
            "pxsize": px, "pysize": py, "sspacing": ss,
            "total_fat_ml": round(total.volume_ml, 3),
            "subcutaneous_fat_ml": round(sub.volume_ml, 3),
            "visceral_fat_ml": round(visceral, 3),
            "is_abd_up": sub.is_abd_up, "source": s.directory,
            "notes": "; ".join(sub.notes[:3]),
        })
        with csv_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)

    dt = time.time() - t_start
    h, rem = divmod(int(dt), 3600)
    m, s = divmod(rem, 60)
    print(f"\n{len(rows)} series processed in {h}h {m}m {s}s")
    if rows:
        print(f"results: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

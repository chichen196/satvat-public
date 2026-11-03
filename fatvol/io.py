"""
Replaces the IDL DICOM/ANALYZE helpers:
    get_dcm_value()  -> read_tags()
    MDCReadDCM()     -> read_series()
    MRIDcm2Anlz()    -> save_nifti()

ANALYZE .hdr/.img is superseded by NIfTI, which every modern viewer reads and
which carries the voxel geometry properly. If you specifically need ANALYZE for
an old tool, nibabel can still write it — say so and I'll switch it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import pydicom
except ImportError:                     # pragma: no cover
    pydicom = None

try:
    import nibabel as nib
except ImportError:                     # pragma: no cover
    nib = None


def read_tags(path) -> dict:
    """Port of get_dcm_value() for the tags the driver actually used."""
    if pydicom is None:
        raise ImportError("pydicom is required to read DICOM files")
    ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    g = lambda k, d="": str(getattr(ds, k, d)).strip()
    return {
        "PatientID": g("PatientID"),
        "StudyDescription": g("StudyDescription"),
        "SeriesDescription": g("SeriesDescription"),
        "Modality": g("Modality"),
        "SeriesNumber": g("SeriesNumber"),
        "StudyDate": g("StudyDate"),
    }


@dataclass
class SeriesInfo:
    """One DICOM series, identified by SeriesInstanceUID rather than by folder."""
    uid: str
    files: list
    description: str = ""
    modality: str = ""
    series_number: str = ""
    patient_id: str = ""
    study_description: str = ""
    study_date: str = ""
    plane: str = "unknown"
    rows: int = 0
    cols: int = 0
    directory: str = ""
    body_part: str = ""
    protocol: str = ""
    study_uid: str = ""
    patient_name: str = ""
    manufacturer: str = ""

    @property
    def n_slices(self) -> int:
        return len(self.files)

    def label(self) -> str:
        bp = self.body_part_display()
        return (f"{self.n_slices:4d} slices  {self.plane:9s}  {bp:12s}  "
                f"{self.description!r}")

    def body_part_display(self) -> str:
        """BodyPartExamined if the scanner filled it in, else a guess.

        BodyPartExamined (0018,0015) is optional and frequently blank on older
        GE studies. When it is, a keyword guess from the study/protocol/series
        text is shown prefixed with '~' so it is never mistaken for header data.
        """
        if self.body_part:
            return self.body_part.upper()
        text = " ".join([self.study_description, self.protocol,
                         self.description]).lower()
        for key, name in (("abdom", "ABDOMEN"), ("liver", "ABDOMEN"),
                          ("pelvi", "PELVIS"), ("brain", "BRAIN"),
                          ("head", "HEAD"), ("neuro", "BRAIN"),
                          ("spine", "SPINE"), ("lumbar", "LSPINE"),
                          ("cervic", "CSPINE"), ("thorac", "TSPINE"),
                          ("chest", "CHEST"), ("thorax", "CHEST"),
                          ("knee", "KNEE"), ("shoulder", "SHOULDER"),
                          ("hip", "HIP"), ("breast", "BREAST"),
                          ("cardiac", "HEART"), ("heart", "HEART"),
                          ("neck", "NECK"), ("body", "BODY"),
                          ("abd", "ABDOMEN"), ("l-spine", "LSPINE"),
                          ("c-spine", "CSPINE"), ("wb", "WHOLEBODY")):
            if key in text:
                return "~" + name
        return "?"


def plane_of(iop) -> str:
    """Anatomical plane from ImageOrientationPatient direction cosines.

    Far more reliable than parsing SeriesDescription for 'Ax' — the description
    is whatever the operator typed, while the cosines are geometry.
    """
    if not iop or len(iop) < 6:
        return "unknown"
    try:
        r = np.array([float(x) for x in iop[:3]])
        c = np.array([float(x) for x in iop[3:6]])
    except (TypeError, ValueError):
        return "unknown"
    n = np.cross(r, c)
    return ["sagittal", "coronal", "axial"][int(np.argmax(np.abs(n)))]


def index_dicom_tree(root, *, verbose: bool = False, use_dicomdir: bool = True):
    """Group every DICOM under `root` by (SeriesInstanceUID, directory).

    Many CD authoring tools write ALL series into one flat directory, so
    grouping by folder alone (as the IDL driver's fixed <patient>/st1/<series>/
    layout assumed) merges unrelated series into one and then fails on the shape
    mismatch. Grouping by UID alone fails the other way on some exports: (X)MedCon
    0.8.13 can give different series of a study the SAME SeriesInstanceUID while
    SOP UIDs stay unique, so real series collapse into one stack. Keying on the
    pair (UID, containing directory) handles both authoring styles: a flat disc
    still separates by UID, and a per-folder export separates by folder.

    Reads headers only — no pixel decoding — so it is fast even on a CD.
    """
    if pydicom is None:
        raise ImportError("pydicom is required to read DICOM files")
    root = Path(root)
    if use_dicomdir:
        dd = find_dicomdir(root)
        if dd is not None:
            try:
                series = index_from_dicomdir(dd, verbose=verbose)
                if series:
                    return series
            except Exception as exc:
                if verbose:
                    print(f"DICOMDIR unusable ({exc}); falling back to a full scan")
    out: dict[tuple, SeriesInfo] = {}
    n_files = n_bad = 0
    for f in sorted(root.rglob("*")):
        if not f.is_file():
            continue
        n_files += 1
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True,
                                 specific_tags=_INDEX_TAGS)
            uid = getattr(ds, "SeriesInstanceUID", None)
            if uid is None:
                n_bad += 1
                continue
        except Exception:
            n_bad += 1
            continue
        key = (uid, str(f.parent))
        s = out.get(key)
        if s is None:
            g = lambda k, d="": str(getattr(ds, k, d)).strip()
            s = SeriesInfo(uid=uid, files=[],
                           description=g("SeriesDescription"),
                           modality=g("Modality"),
                           series_number=g("SeriesNumber"),
                           patient_id=g("PatientID"),
                           study_description=g("StudyDescription"),
                           study_date=g("StudyDate"),
                           body_part=g("BodyPartExamined"),
                           protocol=g("ProtocolName"),
                           study_uid=g("StudyInstanceUID"),
                           patient_name=g("PatientName"),
                           manufacturer=g("Manufacturer"),
                           plane=plane_of(getattr(ds, "ImageOrientationPatient", None)),
                           rows=int(getattr(ds, "Rows", 0) or 0),
                           cols=int(getattr(ds, "Columns", 0) or 0),
                           directory=str(f.parent))
            out[key] = s
        s.files.append(str(f))
    if verbose:
        print(f"indexed {n_files} files, {len(out)} series, {n_bad} non-DICOM/unreadable")
    return sorted(out.values(),
                  key=lambda s: (s.study_date, s.series_number, s.description))


# tags needed for indexing — passing these to dcmread avoids parsing the rest
_INDEX_TAGS = ["SeriesInstanceUID", "StudyInstanceUID", "SeriesDescription",
               "StudyDescription", "ProtocolName", "BodyPartExamined",
               "Modality", "SeriesNumber", "PatientID", "PatientName",
               "StudyDate", "Manufacturer", "ImageOrientationPatient",
               "Rows", "Columns"]


def _probe(path, series):
    """Fill plane/body part/geometry on a SeriesInfo from one representative file."""
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True,
                             specific_tags=_INDEX_TAGS)
    except Exception:
        return series
    g = lambda k, d="": str(getattr(ds, k, d)).strip()
    series.plane = plane_of(getattr(ds, "ImageOrientationPatient", None))
    series.body_part = series.body_part or g("BodyPartExamined")
    series.protocol = series.protocol or g("ProtocolName")
    series.description = series.description or g("SeriesDescription")
    series.study_description = series.study_description or g("StudyDescription")
    series.patient_id = series.patient_id or g("PatientID")
    series.study_date = series.study_date or g("StudyDate")
    series.manufacturer = series.manufacturer or g("Manufacturer")
    series.modality = series.modality or g("Modality")
    series.rows = series.rows or int(getattr(ds, "Rows", 0) or 0)
    series.cols = series.cols or int(getattr(ds, "Columns", 0) or 0)
    return series


def _resolve(base, parts):
    """Join DICOMDIR path components, falling back to a case-insensitive match.

    ISO9660 and Joliet disagree about case, and how a CD is mounted decides
    which the kernel presents. DICOMDIR often stores upper case while the mount
    shows lower, so an exact join can miss files that are plainly there.
    """
    direct = base.joinpath(*parts)
    if direct.exists():
        return direct
    cur = Path(base)
    for part in parts:
        if not cur.is_dir():
            return direct
        match = next((c for c in cur.iterdir() if c.name.lower() == part.lower()), None)
        if match is None:
            return direct
        cur = match
    return cur


def find_dicomdir(root):
    """Locate a DICOMDIR at or just above `root`, if the CD has one."""
    root = Path(root)
    for cand in (root / "DICOMDIR", root.parent / "DICOMDIR",
                 root / "dicomdir", root.parent / "dicomdir"):
        if cand.is_file():
            return cand
    return None


def index_from_dicomdir(dicomdir, *, verbose: bool = False):
    """Build the series index from the CD's own DICOMDIR.

    Vastly faster than walking the tree: one file read plus one probe per
    series, instead of a header read for every image. On optical media, where
    each seek costs milliseconds, that is the difference between seconds and
    minutes.

    DICOMDIR carries the patient/study/series hierarchy and the file list, but
    not ImageOrientationPatient, so one representative file per series is still
    read to determine the plane.
    """
    if pydicom is None:
        raise ImportError("pydicom is required to read DICOM files")
    dicomdir = Path(dicomdir)
    base = dicomdir.parent
    ds = pydicom.dcmread(str(dicomdir), force=True)
    recs = getattr(ds, "DirectoryRecordSequence", None)
    if not recs:
        return []

    out, cur = [], {}
    g = lambda r, k, d="": str(getattr(r, k, d)).strip()
    for rec in recs:
        kind = g(rec, "DirectoryRecordType").upper()
        if kind == "PATIENT":
            cur["patient_id"] = g(rec, "PatientID")
            cur["patient_name"] = g(rec, "PatientName")
        elif kind == "STUDY":
            cur["study_uid"] = g(rec, "StudyInstanceUID")
            cur["study_description"] = g(rec, "StudyDescription")
            cur["study_date"] = g(rec, "StudyDate")
        elif kind == "SERIES":
            s = SeriesInfo(uid=g(rec, "SeriesInstanceUID") or f"series{len(out)}",
                           files=[],
                           description=g(rec, "SeriesDescription"),
                           modality=g(rec, "Modality"),
                           series_number=g(rec, "SeriesNumber"),
                           patient_id=cur.get("patient_id", ""),
                           patient_name=cur.get("patient_name", ""),
                           study_description=cur.get("study_description", ""),
                           study_date=cur.get("study_date", ""),
                           study_uid=cur.get("study_uid", ""),
                           body_part=g(rec, "BodyPartExamined"),
                           directory=str(base))
            out.append(s)
        elif kind == "IMAGE" and out:
            ref = getattr(rec, "ReferencedFileID", None)
            if ref:
                # ReferencedFileID is a pydicom MultiValue, not a list/tuple —
                # an isinstance check against (list, tuple) misses it and
                # stringifies the whole sequence into one bogus path component.
                parts = ([ref] if isinstance(ref, str)
                         else [str(x) for x in ref])
                out[-1].files.append(str(_resolve(base, parts)))

    out = [s for s in out if s.files]
    for s in out:
        _probe(s.files[0], s)
    if verbose:
        print(f"DICOMDIR: {len(out)} series, "
              f"{sum(s.n_slices for s in out)} images (index read, not scanned)")
    return sorted(out, key=lambda s: (s.study_date, s.series_number, s.description))


def read_series_from_files(files, *, sort_by: str = "InstanceNumber"):
    """Read one series' pixel data. Returns (volume (nx, ny, nz), meta)."""
    if pydicom is None:
        raise ImportError("pydicom is required to read DICOM files")
    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), force=True)
            ds.pixel_array          # force decode now so failures surface here
            slices.append(ds)
        except Exception:
            continue
    if not slices:
        raise ValueError("no decodable images in series")

    def key(ds):
        if sort_by == "InstanceNumber" and getattr(ds, "InstanceNumber", None) is not None:
            return float(ds.InstanceNumber)
        ipp = getattr(ds, "ImagePositionPatient", None)
        return float(ipp[2]) if ipp else 0.0

    slices.sort(key=key)
    shapes = {s.pixel_array.shape for s in slices}
    if len(shapes) > 1:
        raise ValueError(f"inconsistent slice dimensions within series: {shapes}")

    vol = np.stack([s.pixel_array for s in slices], axis=0)
    vol = np.transpose(vol, (2, 1, 0)).astype(np.float64)

    ds0 = slices[0]
    px, py = (float(v) for v in getattr(ds0, "PixelSpacing", (1.0, 1.0)))
    thickness = float(getattr(ds0, "SliceThickness", 1.0) or 1.0)
    spacing = getattr(ds0, "SpacingBetweenSlices", None)
    if spacing is not None:
        sspacing = abs(float(spacing))
    elif len(slices) > 1:
        z = [float(getattr(s, "ImagePositionPatient", [0, 0, i])[2])
             for i, s in enumerate(slices)]
        d = np.abs(np.diff(z))
        sspacing = float(np.median(d)) if d.size and np.median(d) > 0 else thickness
    else:
        sspacing = thickness

    g = lambda k, d="": str(getattr(ds0, k, d)).strip()
    meta = {"PatientID": g("PatientID"), "StudyDescription": g("StudyDescription"),
            "SeriesDescription": g("SeriesDescription"), "Modality": g("Modality"),
            "SeriesNumber": g("SeriesNumber"), "StudyDate": g("StudyDate"),
            "pxsize": px, "pysize": py, "sspacing": sspacing,
            "n_slices": len(slices), "shape": vol.shape,
            "plane": plane_of(getattr(ds0, "ImageOrientationPatient", None))}
    return vol, meta


def read_series(directory, *, sort_by: str = "InstanceNumber"):
    """Back-compatible: read one directory assumed to hold a single series."""
    series = index_dicom_tree(directory)
    if not series:
        raise ValueError(f"no readable DICOM images in {directory}")
    if len(series) > 1:
        raise ValueError(f"{len(series)} series found in {directory}; "
                         "use index_dicom_tree() and read_series_from_files()")
    return read_series_from_files(series[0].files, sort_by=sort_by)


def is_target_series(series, *, require_axial: bool = True,
                     exclude: tuple = ("T2",), min_slices: int = 3,
                     require_modality: bool = True) -> tuple:
    """Should this series be segmented? Returns (bool, reason).

    The IDL driver matched 'Ax'/'AX' in SeriesDescription and excluded 'T2'.
    That depends on how the operator typed the protocol name; the plane is now
    taken from ImageOrientationPatient instead, which is geometry rather than
    typing. The T2 exclusion is kept as a description match, since that IS a
    naming convention rather than something derivable from the header.
    """
    if isinstance(series, str):                     # legacy string call
        s = series
        return (("Ax" in s or "AX" in s) and "T2" not in s), "legacy string test"
    if require_axial and series.plane != "axial":
        return False, f"plane is {series.plane}"
    d = series.description or ""
    for token in exclude:
        if token and token.lower() in d.lower():
            return False, f"description contains {token!r}"
    if series.n_slices < min_slices:
        return False, f"only {series.n_slices} slice(s)"
    if require_modality and series.modality and series.modality not in ("MR", "CT"):
        # (X)MedCon 0.8.13 stamps every export "NM" regardless of source; MR data
        # exported through it carry that label, so MedCon output is let through
        # and reported as such rather than rejected on the label.
        if "medcon" in (series.manufacturer or "").lower():
            return True, f"ok (modality {series.modality} from MedCon export, treated as MR)"
        return False, f"modality {series.modality}"
    return True, "ok"


def save_nifti(volume, path, pxsize=1.0, pysize=1.0, sspacing=1.0):
    """Replaces MRIDcm2Anlz() + the raw writeu of the overlay volume."""
    if nib is None:
        raise ImportError("nibabel is required to write NIfTI files")
    affine = np.diag([pxsize, pysize, sspacing, 1.0])
    img = nib.Nifti1Image(np.asarray(volume), affine)
    nib.save(img, str(path))
    return Path(path)


# --------------------------------------------------------------------------
# synthetic data, so the pipeline can be exercised with no patient data present
# --------------------------------------------------------------------------
def make_phantom(nx=256, ny=256, nz=6, pxsize=0.86, pysize=0.86, sspacing=10.0,
                 abdomen_up=True, arms=False, seed=0):
    """An abdominal cross-section phantom: body, subcutaneous fat ring,
    visceral fat blobs, muscle and a bright spine marker.

    Intensities imitate T1: fat bright, muscle mid, air dark — which is what the
    ISODATA threshold expects. Returns (volume, truth, meta) where `truth` has
    labels 1 = subcutaneous fat, 2 = visceral fat.
    """
    rng = np.random.default_rng(seed)
    vol = np.zeros((nx, ny, nz))
    truth = np.zeros((nx, ny, nz), np.uint8)
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx))
    cx, cy = nx / 2.0, ny / 2.0

    for k in range(nz):
        a = nx * 0.36 * (1 + 0.03 * k)
        b = ny * 0.24 * (1 + 0.03 * k)
        body = (((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2) <= 1.0
        fat_t = 0.86
        inner = (((xx - cx) / (a * fat_t)) ** 2 + ((yy - cy) / (b * fat_t)) ** 2) <= 1.0
        sub_fat = body & ~inner

        img = np.zeros((nx, ny))
        img[body] = 300.0                      # muscle / soft tissue
        img[sub_fat] = 900.0                   # subcutaneous fat
        truth[:, :, k][sub_fat] = 1

        # visceral fat: a few blobs inside the peritoneal area
        vis = np.zeros((nx, ny), bool)
        for _ in range(7):
            r = rng.uniform(6, 16)
            ang = rng.uniform(0, 2 * np.pi)
            rad = rng.uniform(0, 0.55)
            bx = cx + np.cos(ang) * rad * a
            by = cy + np.sin(ang) * rad * b * 0.8
            vis |= ((xx - bx) ** 2 + (yy - by) ** 2) <= r ** 2
        vis &= inner
        img[vis] = 880.0
        truth[:, :, k][vis] = 2

        # Posterior block: spine plus paraspinal muscle. A uniform-muscle
        # phantom cannot test IsAbdUp at all, because the abdomen-up decision
        # keys on where NON-FAT tissue concentrates relative to the body centre.
        # Giving the posterior a real muscle mass makes that decision testable.
        sy = cy - b * 0.62 if abdomen_up else cy + b * 0.62
        spine = ((xx - cx) ** 2 + (yy - sy) ** 2) <= (nx * 0.05) ** 2
        paraspinal = ((((xx - cx) / (a * 0.55)) ** 2
                       + ((yy - sy) / (b * 0.42)) ** 2) <= 1.0) & inner
        img[paraspinal] = 330.0
        truth[:, :, k][paraspinal & (truth[:, :, k] == 2)] = 0
        img[spine] = 120.0
        truth[:, :, k][spine] = 0

        if arms:
            for sx in (cx - a * 1.28, cx + a * 1.28):
                arm = (((xx - sx) / (nx * 0.07)) ** 2 + ((yy - cy) / (ny * 0.09)) ** 2) <= 1
                img[arm] = 700.0

        img += rng.normal(0, 12, size=img.shape)
        vol[:, :, k] = np.clip(img, 0, None)

    meta = {"pxsize": pxsize, "pysize": pysize, "sspacing": sspacing,
            "PatientID": "PHANTOM", "StudyDescription": "SelfTest",
            "SeriesDescription": "Ax T1 phantom"}
    return vol, truth, meta

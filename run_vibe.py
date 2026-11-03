#!/usr/bin/env python3
"""run_vibe.py - VIBESegmentator on the CHAOS in-phase volumes (the reference for the MRI arm).

  prep  copy <root>/images{Tr,Ts}/chaos_NNNN_e2.nii.gz (dcm2niix's in-phase echo) to
        <root>/mri_arm/inputs/chaos_NNNN.nii.gz
  run   segment every input with VIBESegmentator model 100 into
        <root>/mri_arm/segs/chaos_NNNN_seg.nii.gz, as chaos_mri_arm.py expects

    python3 run_vibe.py prep --root /path/to/CHAOS
    /path/to/vibe-venv/bin/python run_vibe.py run --root /path/to/CHAOS --vibe-dir /path/to/VIBESegmentator

`run` must be started with the Python of the environment where VIBESegmentator is installed
(torch, nnunetv2, TPTBox); it calls the repository's own run_VIBESegmentator.py for each
volume, from inside the repository, exactly as in the paper (CPU by default):

    run_VIBESegmentator.py --img <input> --out_path <seg> --dataset_id 100 --ddevice cpu

Versions used for the paper: VIBESegmentator commit d72826e (28 Aug 2026), TPTBox 1.0.0
(which downloads the model 100 weights, folds 0-2), nnU-Net 2.8.1.  Run times go to
mri_arm/run_times.csv.
"""
import argparse, re, shutil, subprocess, sys, time
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('step', choices=['prep', 'run'])
ap.add_argument('--root', type=Path, required=True, help='CHAOS root (images{Tr,Ts}/ from dcm2niix)')
ap.add_argument('--arm', default='mri_arm')
ap.add_argument('--vibe-dir', type=Path, help='the cloned VIBESegmentator repository (run)')
ap.add_argument('--dataset-id', type=int, default=100)
ap.add_argument('--device', default='cpu', choices=('cpu', 'cuda', 'mps'))
ap.add_argument('--force', action='store_true', help='redo existing segmentations')
a = ap.parse_args()
root = a.root.expanduser().resolve()
arm = root / a.arm

if a.step == 'prep':
    (arm / 'inputs').mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(root.glob('images*/chaos_*_e2.nii*')):
        m = re.match(r'chaos_(\d+)_e2', p.name)
        if not m:
            continue
        dst = arm / 'inputs' / f'chaos_{int(m.group(1)):04d}.nii.gz'
        if not dst.exists():
            shutil.copy2(p, dst)
        n += 1
    print(f'{n} in-phase volumes in {arm / "inputs"}')
    sys.exit(0)

if not a.vibe_dir:
    sys.exit('--vibe-dir is required for run')
script = (a.vibe_dir.expanduser() / 'run_VIBESegmentator.py').resolve()
if not script.exists():
    sys.exit(f'not found: {script}')
(arm / 'segs').mkdir(parents=True, exist_ok=True)
for f in sorted((arm / 'inputs').glob('chaos_*.nii.gz')):
    key = f.name[:-7]
    out = arm / 'segs' / f'{key}_seg.nii.gz'
    if out.exists() and not a.force:
        print(f'  {key}: done already')
        continue
    cmd = [sys.executable, str(script), '--img', str(f), '--out_path', str(out),
           '--dataset_id', str(a.dataset_id), '--ddevice', a.device]
    print(f'  {key}: running ...', flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=script.parent, capture_output=True, text=True)
    dt = time.time() - t0
    ok = not p.returncode and out.exists()
    print(f'  {key}: {dt:.0f} s' if ok else f'  {key}: FAILED after {dt:.0f} s\n'
          + '\n'.join((p.stdout + p.stderr).strip().splitlines()[-15:]))
    with (arm / 'run_times.csv').open('a') as fh:
        fh.write(f'{key},{dt:.1f},{int(ok)},{a.device}\n')

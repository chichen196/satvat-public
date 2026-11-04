# satvat-public

Code and derived tables for

> Chen C-H. *Subcutaneous–visceral fat separation on public CT and MRI: a reproduced 2006 threshold
> method, a wall-following rule and deep learning.* (manuscript in preparation)

Everything here runs on public data only: the AATCT-IDS CT dataset and the CHAOS MRI dataset.
Neither dataset is redistributed; download them from their sources (below).

## Contents

| Part | Files |
|---|---|
| The 2006 method, ported from its IDL source | `fatvol/` (package), `run_fatvol.py` (DICOM driver and self-test) |
| Wall-following SAT/VAT rule | `wallsep.py` |
| CT arm (AATCT-IDS) | `aattct_sep.py` (2006 method, its zone and the wall rule on one fat mask), `aattct_unet.py` (2D U-Net, 5-fold CV), `unet_overlay.py` |
| MRI arm (CHAOS) | `chaos_labels.py`, `chaos_mri_screen.py`, `mri_body.py`, `mri_trunk.py`, `run_vibe.py`, `chaos_mri_arm.py` |
| Port corrections (Table 2) | `port_corrections.py` |
| Tables and figures | `tables_ct_mri.py` (Tables 3–4), `figs_mri_tables.py` (Figures 3–4), `fig1_pipeline.py` (Figure 1), `overlay_figs.py` (Figures 2 and 5) |
| Derived tables used in the paper | `results/` (see `results/README.md`) |

`fatvol` keeps the original IDL array order `(nx, ny, nz)` so that it can be read side by side with the
`.pro` source. Its defaults are the corrected method described in the paper. The switches
`threshold_rule='idl'`, `thresh_mask_slice`, `visceral_jump_test=False`, `fit_rule` and `abd_up=None`
undo one correction each (see `port_corrections.py`).

## Data

**AATCT-IDS** (Ma et al., *Comput Biol Med* 2024;177:108628): 300 subjects, 3,213 labelled CT slices.
Its authors publish it for non-commercial use (Ma et al., 2024); the figshare record itself carries no licence:
<https://figshare.com/articles/dataset/AATTCT-IDS/23807256>.
Expected layout under `--root`: `Image/Extracted/patient_NNN/FILEnnn.jpg`,
`Label/Subcutaneous/patient_NNN/FILEnnn.png`, `Label/Visceral/patient_NNN/FILEnnn.png`.

**CHAOS** (Kavur et al., *Med Image Anal* 2021;69:101950): 40 MR subjects, T1-DUAL in-phase series used.
CC BY-NC-SA 4.0: <https://doi.org/10.5281/zenodo.3431873>. Preparation:

1. Convert each case's `T1DUAL/DICOM_anon` folder with `dcm2niix -z y`; keep the in-phase echo (the `_e2` file)
   as `images{Tr,Ts}/chaos_NNNN_e2.nii.gz` (NNNN = CHAOS case number).
2. `python3 chaos_labels.py --root $CHAOS`: training-case organ labels to `labelsTr/` (optional; marks kidney slices).
3. `python3 run_vibe.py prep --root $CHAOS`, then, with the Python of a VIBESegmentator environment,
   `python run_vibe.py run --root $CHAOS --vibe-dir /path/to/VIBESegmentator`.

**VIBESegmentator** (Graf et al., *Eur Radiol* 2025, doi:10.1007/s00330-025-12035-9) is used as released,
without retraining: <https://github.com/robert-graf/VIBESegmentator> commit d72826e, model 100,
TPTBox 1.0.0, nnU-Net 2.8.1. It needs its own environment (PyTorch, nnunetv2, TPTBox) and is not included here.

## Environment

Python 3.11 or later. `pip install -r requirements.txt`. PyTorch is needed only for `aattct_unet.py`
(CPU, CUDA or Intel XPU). Run every script from this folder so that `fatvol` and the helper modules import.
Two runs write a relative `--out` under `--root`: `chaos_mri_arm.py` and `port_corrections.py chaos`
(so `mri_arm/results3` lands in `$CHAOS/mri_arm/results3`). Every other relative `--out` is taken from
this folder, and an absolute `--out` is always used as given.

```
python3 run_fatvol.py --self-test        # phantom checks, no data needed: ALL CHECKS PASSED
```

## Reproducing the paper

```
AATCT=/path/to/AATTCT-IDS; CHAOS=/path/to/CHAOS

# Table 3 (CT): the 2006 method, its zone and the wall rule (setting B), then the U-Net (f2m)
python3 aattct_sep.py --root $AATCT --n 0 --abd-up 0 --wall-open 0.78 --wall-close 9.4 --wall-open-post 3.1 \
        --png 0 --out runs/aattct_sep_wallB
python3 aattct_unet.py --root $AATCT --mode f2m --tta --fold all --folds-from results/ct/folds.csv --out runs/unet_f2m
python3 tables_ct_mri.py table3 --sep runs/aattct_sep_wallB/sep_slices.csv --unet runs/unet_f2m/sep_slices_unet.csv

# MRI arm (Table 4, Figures 3-4): screen, then the arm with the MRI and the CT wall settings
python3 chaos_mri_screen.py --root $CHAOS --out runs/chaos_screen
python3 chaos_mri_arm.py --root $CHAOS --out mri_arm/results3
python3 chaos_mri_arm.py --root $CHAOS --wall-set ct --out mri_arm/results_ct
python3 tables_ct_mri.py table4 --mri $CHAOS/mri_arm/results3/arm_cases.csv --ct $CHAOS/mri_arm/results_ct/arm_cases.csv
python3 figs_mri_tables.py --cases $CHAOS/mri_arm/results3/arm_cases.csv --out figs

# Table 2 (port corrections): synthetic phantoms, then CHAOS
python3 port_corrections.py phantom --out runs/table2_phantom --jobs 2
python3 port_corrections.py chaos --root $CHAOS --out mri_arm/table2 --jobs 2

# Figures 1, 2 and 5
python3 fig1_pipeline.py figs
python3 overlay_figs.py fig2 --root $AATCT --unet-run runs/unet_f2m --sep-csv runs/aattct_sep_wallB/sep_slices.csv --out figs
python3 overlay_figs.py fig5 --root $CHAOS --out figs --rows skin:38,leak:1,pass2:33
```

The U-Net was trained on an Intel Arc B580 (about an hour per fold at 512 px); the other steps run on a CPU
in minutes to an hour. `results/` holds the tables these commands produced for the paper, so Tables 2–4
and Figures 3–4 can be checked without rerunning anything, for example
`python3 tables_ct_mri.py table4 --mri results/mri/arm_cases_mri.csv --ct results/mri/arm_cases_ct_setting.csv`.

The CT commands reproduce `results/` exactly. The MRI commands reproduce it to within rounding: rerunning
three cases under both wall settings changed no volume by more than 0.4 mL and no percentage by more than
0.01 points, in the 2006 method and its zone as much as in the wall rule, which points to floating-point
variation upstream of all three (most likely the N4 bias correction) rather than to the method.

## Licence

Code: MIT (`LICENSE`). Derived tables in `results/`: CC BY-NC-SA 4.0 (`results/LICENSE.md`), as they derive
from CHAOS (CC BY-NC-SA 4.0) and from AATCT-IDS, which its authors publish for non-commercial use.
Cite the datasets and VIBESegmentator when you use them. How to cite this repository is in `CITATION.cff`.

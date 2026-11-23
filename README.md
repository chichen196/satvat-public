# satvat-public

Code and derived tables for

> Chen C-H. *Subcutaneous–visceral fat separation on public CT and MRI: a reproduced 2006 threshold
> method, a wall-following rule and deep learning.* (submitted)

Everything here runs on public data only: the AATCT-IDS CT dataset and the CHAOS MRI dataset.
Neither dataset is redistributed; download them from their sources (below).

Release 1.1.0 adds TotalSegmentator to both arms (a fourth boundary on CT, a second reference model on
MRI) and the fat-definition sensitivity analysis. Release 1.0.0 is the code and tables of the first
submitted version.

## Contents

| Part | Files |
|---|---|
| The 2006 method, ported from its IDL source | `fatvol/` (package), `run_fatvol.py` (DICOM driver and self-test) |
| Wall-following SAT/VAT rule | `wallsep.py` |
| CT arm (AATCT-IDS) | `aattct_sep.py` (2006 method, its zone and the wall rule on one fat mask), `aattct_unet.py` (2D U-Net, 5-fold CV), `unet_overlay.py`, `unet_variants.py` (Supplementary Tables S1–S2) |
| TotalSegmentator on the CT arm | `totalseg_aattct.py` (`tissue_types` task, scored on the same fat mask) |
| Fat-definition sensitivity (CT, Supplementary Table S3) | `aattct_sep.py --score-fat-hi`, `unet_rescore.py`, `totalseg_aattct.py score --fat-hi` |
| MRI arm (CHAOS) | `chaos_labels.py`, `chaos_mri_screen.py`, `mri_body.py`, `mri_trunk.py`, `run_vibe.py`, `chaos_mri_arm.py` (`--ref ts`: TotalSegmentator MRI as the reference in place of VIBESegmentator) |
| Port corrections (Table 2) | `port_corrections.py` |
| Tables and figures | `tables_ct_mri.py` (Tables 3–4), `fig1_pipeline.py` (Figure 1), `overlay_figs.py` (Figures 2 and 4), `fig3_threshold_content.py` (Figure 3), `figs_mri_tables.py` (Figure 5) |
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
`Label/Subcutaneous/patient_NNN/FILEnnn.png`, `Label/Visceral/patient_NNN/FILEnnn.png`; the
TotalSegmentator comparison also reads the full series, `Image/Initial/patient_NNN/FILEnnn.jpg`
(consecutive 1.25 mm slices, of which `Extracted` holds every fifth). File names are not uniform (three
files of patient_021 carry a `_0` suffix), so `totalseg_aattct.py` orders and matches slices by the first
number in the name.

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

**TotalSegmentator** (Wasserthal et al., *Radiol Artif Intell* 2023; MRI: Akinci D'Antonoli et al., *Radiology*
2025) is used as released, version 2.18.0: task `tissue_types` on CT and `tissue_types_mr` on MRI (subcutaneous
fat, torso fat, skeletal muscle). These tasks need TotalSegmentator's free
academic licence for non-commercial use (<https://backend.totalsegmentator.com/license-academic/>), set once
with `totalseg_set_license -l <your licence>`. Install it in its own environment (CPU PyTorch is enough, about
35 s per CT subject) and run `totalseg_aattct.py` with that environment's Python, which also needs `nibabel`.
AATCT-IDS ships 8-bit JPEGs, so TotalSegmentator sees CT values reconstructed from them: HU = −106 + grey ×
253/255 inside the body, −1000 outside. Fat below about −106 HU and bowel gas inside the body both come out
at −106 HU, and bone stops at +147 HU. Because the subcutaneous fat of some subjects is rendered at grey 0
like the surrounding air, the body outline of the input is completed with the annotators' labels of the
nearest annotated slice (`--with-labels`), as `aattct_sep.py` does for the other methods. Only fat inside the
annotators' fat mask is scored, so gas that TotalSegmentator labels as fat does not enter the comparison.
On MRI it runs on the CHAOS in-phase images as they are.

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

# Table 3, TotalSegmentator column (Python of the TotalSegmentator environment; resumes where it stopped)
python totalseg_aattct.py prep  --root $AATCT --out runs/totalseg --z down --body 1 --with-labels
python totalseg_aattct.py check --root $AATCT --out runs/totalseg --body 1 --with-labels   # optional: input pictures
python totalseg_aattct.py run   --out runs/totalseg
python totalseg_aattct.py score --root $AATCT --out runs/totalseg              # -> results/ct/ts_slices_fat79.csv
cp runs/totalseg/ts_slices.csv runs/totalseg/ts_slices_fat79.csv

# Supplementary Table S3 (fat-definition sensitivity): every method as above, scored only on fat pixels
# with grey <= 40 (about -66 HU)
python3 aattct_sep.py --root $AATCT --n 0 --abd-up 0 --wall-open 0.78 --wall-close 9.4 --wall-open-post 3.1 \
        --png 0 --score-fat-hi 40 --out runs/aattct_sep_wallB_fat40         # -> sep_slices_settingB_fat40.csv
python3 unet_rescore.py --root $AATCT --run runs/unet_f2m --fat-hi 40        # -> sep_subjects_unet_f2m_fat40.csv
python totalseg_aattct.py score --root $AATCT --out runs/totalseg --fat-hi 40  # -> ts_slices_fat40.csv
cp runs/totalseg/ts_slices.csv runs/totalseg/ts_slices_fat40.csv

# Supplementary Tables S1-S2: the other two U-Net variants on the same folds, then the comparison
python3 aattct_unet.py --root $AATCT --mode 3class --fold all --folds-from results/ct/folds.csv --out runs/unet_3class
python3 aattct_unet.py --root $AATCT --mode f2 --fold all --folds-from results/ct/folds.csv --out runs/unet_f2
python3 unet_variants.py --run 3class=runs/unet_3class --run f2=runs/unet_f2 --run f2m=runs/unet_f2m \
        --drop patient_179,patient_260 --out unet_variants.csv

# MRI arm against VIBESegmentator (Table 4, Figure 5): screen, then the arm with the MRI and the CT wall settings
python3 chaos_mri_screen.py --root $CHAOS --out runs/chaos_screen
python3 chaos_mri_arm.py --root $CHAOS --out mri_arm/results3
python3 chaos_mri_arm.py --root $CHAOS --wall-set ct --out mri_arm/results_ct
python3 tables_ct_mri.py table4 --mri $CHAOS/mri_arm/results3/arm_cases.csv --ct $CHAOS/mri_arm/results_ct/arm_cases.csv
python3 figs_mri_tables.py --cases $CHAOS/mri_arm/results3/arm_cases.csv --out figs

# MRI arm against TotalSegmentator MRI (Table 5): tissue_types_mr on the in-phase images, then the same arm
mkdir -p $CHAOS/ts_mr
for f in $CHAOS/images{Tr,Ts}/chaos_*_e2.nii.gz; do
  TotalSegmentator -i $f -o $CHAOS/ts_mr/$(basename $f _e2.nii.gz).nii.gz --ml -ta tissue_types_mr --device cpu
done
python3 chaos_mri_arm.py --root $CHAOS --ref ts --out mri_arm/results3_ts

# Table 2 (port corrections): synthetic phantoms, then CHAOS
python3 port_corrections.py phantom --out runs/table2_phantom --jobs 2
python3 port_corrections.py chaos --root $CHAOS --out mri_arm/table2 --jobs 2

# Figure 1 (study design)
python3 fig1_pipeline.py figs
# Figure 2 (the four boundaries on CT, subjects 074 and 138; --ts-dir adds the TotalSegmentator column)
python3 overlay_figs.py fig2 --root $AATCT --unet-run runs/unet_f2m --sep-csv runs/aattct_sep_wallB/sep_slices.csv \
        --ts-dir runs/totalseg --ts-csv runs/totalseg/ts_slices_fat79.csv --out figs
# Figure 3 (threshold VAT that each reference model does not call visceral fat)
python3 fig3_threshold_content.py $CHAOS/mri_arm/results3/arm_cases.csv $CHAOS/mri_arm/results3_ts/arm_cases.csv figs
# Figure 4 (why the wall rule needed an MRI setting; the fig5 subcommand writes the paper's Figure 4)
python3 overlay_figs.py fig5 --root $CHAOS --out figs --rows skin:38,leak:1,pass2:33
# Figure 5 (Bland-Altman plots) is written by figs_mri_tables.py in the MRI block above

# Supplementary Figure S1: TotalSegmentator against the labels (first, middle and last slice of each subject)
python totalseg_aattct.py png --root $AATCT --out runs/totalseg --subjects patient_076,patient_089,patient_150 --n 3
```

The U-Net was trained on an Intel Arc B580 (about an hour per fold at 512 px); TotalSegmentator takes about
three hours for the 300 CT subjects on a CPU and a few minutes per MRI case; the other steps run on a CPU in
minutes to an hour. `results/` holds the tables these commands produced for the paper, so Tables 2–5,
Supplementary Tables S1–S3 and Figures 3 and 5 can be checked without rerunning anything, for example

```
python3 tables_ct_mri.py table4 --mri results/mri/arm_cases_mri.csv --ct results/mri/arm_cases_ct_setting.csv
python3 unet_variants.py --run 3class=results/ct/sep_slices_unet_3class.csv --run f2=results/ct/sep_slices_unet_f2.csv \
        --run f2m=results/ct/sep_slices_unet_f2m.csv --drop patient_179,patient_260
python3 fig3_threshold_content.py results/mri/arm_cases_mri.csv results/mri/arm_cases_mri_totalseg.csv figs
```

In the commands above, `score` writes `ts_slices.csv` (and `ts_summary.txt`, the Table 3 measures by half) and
`unet_rescore.py` writes `unet_rescore_fh40_subjects.csv`; the `-> …` comments give the names those tables
carry in `results/ct/`. Table 3's TotalSegmentator column and Supplementary Table S3 apply the measures
defined under Table 3 (slices summed per subject) to those tables. Figures 2 and 4 and Supplementary Figure S1
are overlays drawn from the images and the saved segmentations, not from `results/`, so they need the datasets
and the runs above; Figure 2 also writes `fig2_check.txt`, the pixel counts of its two slices, which should
match those slices' rows in `results/ct/`. The 3class run behind the supplement was
made with an earlier version of `aattct_unet.py`, before `--mode` existed; `--mode 3class` is its equivalent.

`--score-fat-hi` (and `--fat-hi` of `score` and `unet_rescore.py`) changes only which pixels are scored. Every
method still runs on the fat mask grey <= 79, so this is an analysis of the evaluation, not of a stricter fat
threshold inside the methods. (`aattct_sep.py --fat-hi` would also change the wall rule's input, because it
takes everything that is not fat as wall material.)

The `aattct_sep.py` command reproduces its table in `results/` exactly. Retraining the U-Net reproduces its
tables only up to the run-to-run variation of training on a GPU, which was not measured. The MRI commands
reproduce `results/` to within rounding: rerunning three cases under both wall settings changed no volume by
more than 0.4 mL and no percentage by more than 0.01 points, in the 2006 method and its zone as much as in the
wall rule, which points to floating-point variation upstream of all three (most likely the N4 bias
correction) rather than to the method.

Script names and output files keep the figure numbers of an earlier draft, in which the paper's
Figures 4 and 5 were the other way round: `overlay_figs.py fig5` writes the paper's Figure 4, and
`figs_mri_tables.py` writes the Bland–Altman plots (the paper's Figure 5) under their earlier number,
together with a one-reference version of Figure 3 that `fig3_threshold_content.py` supersedes.

## Licence

Code: MIT (`LICENSE`). Derived tables in `results/`: CC BY-NC-SA 4.0 (`results/LICENSE.md`), as they derive
from CHAOS (CC BY-NC-SA 4.0) and from AATCT-IDS, which its authors publish for non-commercial use.
TotalSegmentator's segmentations and the volumes reconstructed from the JPEGs are not included; only the
per-slice and per-case counts derived from them are. Cite the datasets, VIBESegmentator and TotalSegmentator
when you use them. How to cite this repository is in `CITATION.cff`.

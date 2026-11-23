# Derived tables

Produced by the scripts in this repository for the paper; areas are in pixels, volumes in ml.
Case keys are the datasets' own (AATCT-IDS `patient_NNN`, CHAOS `chaos_NNNN`).

| File | Made by | Used for |
|---|---|---|
| `ct/sep_slices_settingB.csv` | `aattct_sep.py` (wall setting B) | Table 3: per slice, fat mask F, labels, and the SAT/VAT split of the 2006 method (`meth`), its zone (`zone`) and the wall rule (`wall`) |
| `ct/sep_slices_unet_f2m.csv` | `aattct_unet.py --mode f2m --tta` | Table 3: the U-Net's split of the same F, with its test fold |
| `ct/sep_slices_unet_3class.csv`, `ct/sep_slices_unet_f2.csv` | `aattct_unet.py --mode 3class`, `--mode f2` | Supplementary Tables S1–S2 (with `unet_variants.py`): the other two U-Net variants on the same F |
| `ct/folds.csv` | `aattct_unet.py` | the subject-level folds (pass with `--folds-from`) |
| `ct/ts_slices_fat79.csv` | `totalseg_aattct.py score` | Table 3, TotalSegmentator column, and Supplementary Table S3: per slice, F, labels (`S`, `V`), TotalSegmentator's split of F (`pred_S`, `pred_V`), and its errors (`sat_called_vat`, `vat_called_sat`, `neither`) |
| `ct/sep_slices_settingB_fat40.csv` | `aattct_sep.py` (wall setting B) `--score-fat-hi 40` | Supplementary Table S3: as `sep_slices_settingB.csv`, scored only on the F pixels with grey ≤ 40 (about −66 HU) |
| `ct/ts_slices_fat40.csv` | `totalseg_aattct.py score --fat-hi 40` | Supplementary Table S3: as `ts_slices_fat79.csv`, scored on grey ≤ 40 |
| `ct/sep_subjects_unet_f2m_fat40.csv` | `unet_rescore.py --fat-hi 40` | Supplementary Table S3: per subject, the U-Net (f2m) predictions scored on grey ≤ 40, with the half (`even`/`odd`) each subject belongs to |
| `mri/screen_volumes.csv` | `chaos_mri_screen.py` | Table 1 and screening (geometry, lung run, arms, edge cuts) |
| `mri/arm_cases_mri.csv`, `mri/arm_slices_mri.csv` | `chaos_mri_arm.py` (MRI wall setting) | Table 4, Table 5 (VIBESegmentator column), Figures 3a–b and 5; per case and per slice, original (`raw`) and N4 images |
| `mri/arm_cases_ct_setting.csv` | `chaos_mri_arm.py --wall-set ct` | Table 4, wall rule with its CT setting |
| `mri/arm_cases_mri_totalseg.csv`, `mri/arm_slices_mri_totalseg.csv` | `chaos_mri_arm.py --ref ts` | Table 5 (TotalSegmentator column), Figure 3c–d: the same arm with TotalSegmentator MRI as the reference; the case rows also compare TotalSegmentator with VIBESegmentator (`vibe_*` columns: volumes, % difference, Dice) |
| `port_corrections/corr_cases_chaos.csv`, `corr_summary_chaos.txt` | `port_corrections.py chaos` | Table 2, CHAOS columns |
| `port_corrections/corr_cases_phantom.csv`, `corr_summary_phantom.txt` | `port_corrections.py phantom` | Table 2, phantom column |

The `_fat40` tables change only which pixels are scored. The fat mask, the labels and every
method's input are the same as in the main tables, so the two sets differ only in the fat definition
(grey ≤ 79, the dataset's own, against grey ≤ 40).

The TotalSegmentator CT tables have 3,211 slices rather than 3,213. TotalSegmentator's input is built
from `Image/Initial`, which has no image for patient_121 FILE145 or FILE149. They replace the
`sep_slices_totalseg*.csv` tables of an earlier commit, in which three files of patient_021 had been
stacked out of order; only that subject's ten slices differ.

TotalSegmentator has no organ, bowel or bone labels, so in the `_totalseg` MRI tables the breakdown of
rejected threshold VAT has only model SAT, muscle and unlabelled.

Figures 3 and 5 are drawn from these tables. Figures 2 and 4 and Supplementary Figure S1 are image
overlays (`overlay_figs.py`, `totalseg_aattct.py png`) and need the datasets and the saved segmentations;
Figure 2 shows patient_074 FILE200 and patient_138 FILE126, whose pixel counts are those slices' rows in
`ct/sep_slices_settingB.csv`, `ct/sep_slices_unet_f2m.csv` and `ct/ts_slices_fat79.csv`. Figure 1 is a
schematic (`fig1_pipeline.py`).

Licence: CC BY-NC-SA 4.0 (see `LICENSE.md`).

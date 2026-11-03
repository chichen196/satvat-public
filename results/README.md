# Derived tables

Produced by the scripts in this repository for the paper; areas are in pixels, volumes in ml.
Case keys are the datasets' own (AATCT-IDS `patient_NNN`, CHAOS `chaos_NNNN`).

| File | Made by | Used for |
|---|---|---|
| `ct/sep_slices_settingB.csv` | `aattct_sep.py` (wall setting B) | Table 3: per slice, fat mask F, labels, and the SAT/VAT split of the 2006 method (`meth`), its zone (`zone`) and the wall rule (`wall`) |
| `ct/sep_slices_unet_f2m.csv` | `aattct_unet.py --mode f2m --tta` | Table 3: the U-Net's split of the same F, with its test fold |
| `ct/folds.csv` | `aattct_unet.py` | the subject-level folds (pass with `--folds-from`) |
| `mri/screen_volumes.csv` | `chaos_mri_screen.py` | Table 1 and screening (geometry, lung run, arms, edge cuts) |
| `mri/arm_cases_mri.csv`, `mri/arm_slices_mri.csv` | `chaos_mri_arm.py` (MRI wall setting) | Table 4, Figures 3–4; per case and per slice, original (`raw`) and N4 images |
| `mri/arm_cases_ct_setting.csv` | `chaos_mri_arm.py --wall-set ct` | Table 4, wall rule with its CT setting |
| `port_corrections/corr_cases_chaos.csv`, `corr_summary_chaos.txt` | `port_corrections.py chaos` | Table 2, CHAOS columns |
| `port_corrections/corr_cases_phantom.csv`, `corr_summary_phantom.txt` | `port_corrections.py phantom` | Table 2, phantom column |

Licence: CC BY-NC-SA 4.0 (see `LICENSE.md`).

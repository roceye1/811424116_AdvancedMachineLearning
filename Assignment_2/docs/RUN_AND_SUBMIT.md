# Running and submitting the coding project

This is a code-delivery guide, **not** the separate academic report.

## Recommended Kaggle workflow

1. Open the final executed notebook and attach the original **Dogs vs Cats** competition input providing `train.zip` under `/kaggle/input`. Enable a GPU accelerator if available.
2. To **inspect the already completed work**, open the notebook with saved outputs; do not rerun training just for viewing or exporting.
3. To **reproduce from scratch**, run from top to bottom in a clean Kaggle session with enough compute budget and disk space. The notebook builds the model/checkpoint folders in `/kaggle/working`; the dataset archive is required. The first execution of training cells trains fresh models; later executions verify existing checkpoint files when those artifacts are present.
4. For a final model-specific SHA audit without retraining, paste the contents of `src/model_specific_duplicate_audit.py` into a new code cell at the **end of your existing executed Kaggle session**, after the data selection cells have been run. Its output is saved in `/kaggle/working/final_integrity_audit`.
5. Save a **Kaggle version with outputs**, and download the executed `.ipynb` if you re-executed the final audit code. Keep generated `.keras` checkpoints, manifests, prediction tables and original 200-DPI PNG graphs in the Kaggle outputs or a separate artifact archive.
6. Copy the `Assignment 2` folder into the **same GitHub repository used for Assignment 1**. Commit README, notebook, source, figures, tables and audit. Do not commit the dataset archive or huge trained model files unless repository storage policies permit it.
7. Confirm the GitHub notebook renders, the CSV and figure links are correct, and the required separate rendered HTML/PDF/Word report is submitted when it is eventually prepared. Provide the repository's **actual** `.git` clone URL, not a fabricated URL.

## Verification checklist before final submission

- [ ] Repo contains the correct folder `Assignment 2` in the Assignment 1 repository.
- [ ] `notebooks/Assignment_2_CNN_Final_Executed.ipynb` renders its 21 executed code cells and 10 final figures.
- [ ] The post-run model-specific hash audit is documented and its CSVs are retained.
- [ ] Kaggle saved version preserves `.keras` model checkpoints, `training_history.csv`, sample manifests, prediction data and original full-resolution PNGs.
- [ ] Six required models and Q1/Q2/Q3/Q4 are straightforward to locate; larger experiments clearly labeled supplementary.
- [ ] README's reported numbers match the exported `results/04_all_eight_original_test.csv`.
- [ ] The separate academic report/rendered output is added at the appropriate later step.
- [ ] The final submission provides the correct GitHub repository clone URL.

## Notes about outputs in this package

- The 10 PNGs in `figures/` were **extracted from the saved executed notebook visualization outputs**. For original resolution, obtain the files from `/kaggle/working/assignment2_master_visuals` in the Kaggle run.
- `results/*.csv` were extracted directly from the notebook's displayed HTML data tables; they are not reconstruction estimates.
- `audit/*.csv` reflect the **separately executed Kaggle audit output supplied later**. Its exact original generated CSVs are not included in the uploaded notebook; replace these documented copies with Kaggle originals if you export those files.
- `src/assignment2_cnn.py` is a convenient code export and depends on Kaggle paths/setup. The executed notebook is the canonical submitted code.

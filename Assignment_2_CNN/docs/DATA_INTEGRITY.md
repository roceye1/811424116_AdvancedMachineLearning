# Data integrity — exact duplicate images and model-specific leakage audit

## Check method and scope

The original notebook checked distinct filenames across the complete 25,000-image Kaggle archive, followed by SHA-256 byte-content hashing. Its initial cross-split audit flagged three filenames with the same contents across extended training and dropout holdout 2. A **subsequent model-specific Kaggle audit** checked each model's **actual training selection** against only the evaluation sets used to assess that particular model.

### Final audited findings

| Measure | Observed result |
|---|---:|
| Model–evaluation pairs checked | **28** |
| Exact train–evaluation hash overlaps | **0** |
| Duplicate-content groups in all 25,000 files | **5** |
| Filenames involved in duplicate groups | **11** |
| Core scratch/pretrained training repeated-content occurrences (1K–2K) | **0** |
| Scratch 12K repeated training-content occurrences | **3** |
| Scratch 18K repeated training-content occurrences | **4** |
| Extra 2,000-image validation repeated-content occurrences | **1** |

The five groups include: (i) a duplicate pair entirely within extended validation; (ii) two duplicate pairs entirely within extended training; (iii) a duplicate pair shared between the core 2K train pool and extended training; and (iv) a three-way duplicate, two copies in extended training and one in dropout holdout 2. Exact filenames/hashes are listed in `audit/exact_duplicate_groups.csv`.

The **12K and 18K scratch models were not tested against dropout holdout 2**; the models evaluated against dropout holdout 2 were trained only on the original 2K pool. Therefore, the duplicate group crossing those partitions **did not create exact-content train–evaluation overlap in a reported comparison**.

**Final conclusion:** In all **28 evaluated model/dataset combinations**, zero byte-identical image hashes appeared in both training and evaluation. The originally reported performance values are preserved without post-hoc data changes. This conclusion is **not** a guarantee of absence of perceptual near-duplicate images or label noise.

### Source labeling issue

The user visually inspected `train/cat.4085.jpg`, which displays a dog although the original archive filename assigns it the class `cat`. The uploaded notebook verified that the image bytes matched the original Kaggle competition ZIP, and all four selected models predicted Dog. This is a **manually identified suspicious source-label error**, not a retroactive redefinition of the original benchmark. No labels or metrics were edited. An apparent pet-product logo classified as Cat was observed in exploratory galleries; its exact original filename has not been independently established in the uploaded executed notebook.

### Interpretation cautions

- **Duplicate training rows:** Effective distinct observations are slightly fewer than nominal rows in the extended configurations. Duplicates need not cause test leakage.
- **Duplicate evaluation rows:** A repeated photo in the extended validation set changes its effective distinct count from 2,000 to 1,999 and may slightly affect its accuracy estimate. The original 500-image test and other holdout metrics remain unchanged.
- **SHA-256 scope:** Different compressions or resized/cropped versions of the same photograph generally have different SHA-256 bytes; exact hashing cannot catch all visually similar images.
- **Different holdouts:** Model holdout 1 and dropout holdout 2 contain different images; comparing their scores as if they were the same cohort would be misleading.
- **Post-evaluation audit:** Because the dataset audit was performed after observing evaluation results, it is appropriately presented as an audit/limitation, not as a way to improve headline test accuracy.

### Reproducibility and source provenance

The 28-check table and the 11-row duplicate listing in `audit/` were transcribed from the Kaggle audit output supplied by the user in the conversation after the executed final notebook was uploaded. `src/model_specific_duplicate_audit.py` reproduces the checks in a running Kaggle session that already has the dataframes and SHA-256 table initialized; its new output goes to `final_integrity_audit/`.

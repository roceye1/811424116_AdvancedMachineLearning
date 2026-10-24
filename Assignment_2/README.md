# Assignment 2 — CNN Classification: Dogs vs Cats

**Advanced Data Mining | Reproducible Kaggle notebook and analysis code**

This repository folder contains the **executed, documented Python/Kaggle coding submission** for the Dogs vs Cats convolutional-network assignment. The professor-required six models are clearly separated from additional investigations into scaling, regularization, variability, and image/data quality. **This package is code and evidence, not the separate written report.**

## Start here

1. Open [`notebooks/Assignment_2_CNN_Final_Executed.ipynb`](notebooks/Assignment_2_CNN_Final_Executed.ipynb) to see the complete code, outputs, tables, figures and documentation. This is the **main notebook**.
2. Consult [`docs/RUN_AND_SUBMIT.md`](docs/RUN_AND_SUBMIT.md) before reproducing on Kaggle or publishing to GitHub.
3. Browse `results/` for CSVs extracted *directly* from the executed notebook outputs; see `figures/` for PNG versions of its ten final charts.
4. Consult [`docs/DATA_INTEGRITY.md`](docs/DATA_INTEGRITY.md) and `audit/` for the separately verified 28-pair SHA-256 image-overlap analysis.

## Professor-required experiments

All models were trained with 150 × 150 RGB images, 25 epochs, a fixed 500-image validation cohort, and a fixed 500-image original test cohort. Model checkpoints were chosen using the lowest validation loss; reported test numbers are descriptive outcomes from selected checkpoints.

| Model | Training images | Validation accuracy | Original test accuracy |
|---|---:|---:|---:|
| CNN from scratch | 1,000 | 76.80% | 74.20% |
| CNN from scratch | 1,500 | 77.00% | 73.20% |
| CNN from scratch | 2,000 | 79.20% | 77.00% |
| Pretrained MobileNetV2 | 1,000 | 97.40% | 96.20% |
| Pretrained MobileNetV2 | 1,500 | 97.40% | 96.60% |
| Pretrained MobileNetV2 | 2,000 | 97.60% | 96.40% |

### Supplementary work — beyond assignment requirements

- Scratch CNNs using **12,000 and 18,000** training images (original test accuracy **92.00%** and **93.20%**).
- A separate 2,000-image model holdout for scratch 2K/12K/18K and pretrained 2K.
- Dropout 0.0, 0.25, 0.5 evaluated on a different 2,000-image dropout holdout.
- Nine repeated runs across predetermined seeds 17, 42 and 123.
- Prediction-error overlap, confusion matrices, error galleries, and a manually identified suspicious source label (`train/cat.4085.jpg`).
- Exact SHA-256 duplicate-content audit: **28/28 model-specific train–evaluation comparisons had zero shared file-content hashes.** See limitations in the integrity notes.

## Repository layout

```text
Assignment 2/
├── notebooks/
│   └── Assignment_2_CNN_Final_Executed.ipynb  # main executed notebook
├── src/
│   ├── assignment2_cnn.py                      # synchronized Python code export
│   └── model_specific_duplicate_audit.py      # final optional Kaggle audit cell
├── results/                                    # measured tables, from notebook output
├── figures/                                    # charts / error galleries, embedded-output exports
├── audit/                                      # documented SHA-256 findings from later Kaggle audit
├── docs/
│   ├── RUN_AND_SUBMIT.md
│   ├── DATA_INTEGRITY.md
│   └── EVIDENCE_PROVENANCE.md
├── archive/
│   └── Original_Exploratory_Notebook.ipynb      # preserves exploratory code and outputs
├── requirements.txt
└── README.md
```

## Environment and data

Executed on **Kaggle Notebooks** using **Python 3.13.15**, **TensorFlow 2.20.0**, and **Pandas 2.3.3**, with GPU devices visible to TensorFlow. Dataset: the **Dogs vs Cats Kaggle competition `train.zip`** containing 25,000 labeled files. The source images and large `.keras` checkpoints are **not bundled** in this GitHub package; preserve them in a Kaggle saved version/output archive. The notebook will require its input competition to be attached when reproduced.

Pretrained MobileNetV2 accepts 150×150 RGB images with `include_top=False`; Keras warns that its standard pretrained 224-resolution ImageNet weight variant is loaded. This warning is expected and does not automatically resize input images to 224×224. Architecture differs between pretrained MobileNetV2 and the scratch CNN, so the performance difference cannot be attributed to pretraining alone.

## Important interpretation limits

The 12K/18K images exceed the assignment's professor-defined 2K pool and are labeled **supplementary**. Two independent 2,000-image holdouts serve **different studies** and are not interchangeable. The reported test results should **not** be used for further hyperparameter selection. Hashes rule out only *byte-identical* train–evaluation overlap, not all perceptual near-duplicates, and duplicate images remain inside extended training/validation. See [`docs/DATA_INTEGRITY.md`](docs/DATA_INTEGRITY.md).

**Do not silently replace the executed metrics with numbers from the earlier exploratory notebook.** The final executed Kaggle run is the source of truth for this package; the archive is retained for provenance.

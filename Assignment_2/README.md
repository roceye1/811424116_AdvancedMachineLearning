# Assignment 2 — CNN Classification of Dogs and Cats

**Advanced Machine Learning / Advanced Data Mining · TensorFlow · Kaggle**

This project compares convolutional neural networks (CNNs) trained from scratch with a pretrained MobileNetV2 classifier on the Kaggle Dogs vs. Cats dataset. It includes the required sample-size experiments, additional analyses of training scale and model stability, and a data-quality audit.

## Project files

- **[Academic report (PDF)](report/Assignment_2_CNN_Aamir_Sohaib.pdf)** — methodology, results, figures, interpretation, limitations, and conclusions.
- **[Executed notebook (IPYNB)](notebooks/Assignment_2_CNN_Final_Executed.ipynb)** — complete Kaggle code, execution outputs, and plots.
- **[Rendered notebook (HTML)](notebooks/Assignment_2_CNN_Results_and_Code.html)** — code and outputs in HTML format. Download the file to view it in a browser.
- **[Python source](src/assignment2_cnn.py)** — code export from the experiment notebook.
- **[Result tables](results/04_all_eight_original_test.csv)** — performance metrics for all eight evaluated configurations.
- **[Figures](figures/)** — comparison charts and misclassification examples.
- **[Data-integrity findings](docs/DATA_INTEGRITY.md)** and **[hash audit results](audit/model_specific_hash_audit.csv)** — duplicate-content and model-specific overlap checks.
- **[Run instructions](docs/RUN_AND_SUBMIT.md)** — Kaggle environment and reproducibility details.

## Dataset and experimental setup

The source is the Kaggle Dogs vs. Cats competition training archive (25,000 labeled images). Images were resized to **150 × 150 RGB**. The six required configurations use a fixed **500-image validation set** and an independent **500-image test set**, with balanced class representation. Each model was trained for 25 epochs, and the best checkpoint was selected using validation loss.

The scratch CNN includes convolution and pooling layers, augmentation, and dropout. The pretrained approach uses ImageNet-pretrained **MobileNetV2** as a frozen feature extractor with a trained classification head.

## Required experiment results

| Model | Training images | Validation accuracy | Test accuracy |
|---|---:|---:|---:|
| Scratch CNN | 1,000 | 76.80% | 74.20% |
| Scratch CNN | 1,500 | 77.00% | 73.20% |
| Scratch CNN | 2,000 | 79.20% | 77.00% |
| Pretrained MobileNetV2 | 1,000 | 97.40% | 96.20% |
| Pretrained MobileNetV2 | 1,500 | 97.40% | 96.60% |
| Pretrained MobileNetV2 | 2,000 | 97.60% | 96.40% |

All test accuracies in this table use the same original 500-image test cohort. The reported results come from the final executed Kaggle run rather than earlier exploratory runs.

## Additional experiments

- **Larger training sets:** Scratch CNNs trained with 12,000 and 18,000 images achieved **92.00%** and **93.20%** accuracy on the original test set. These runs are supplementary and exceed the required 2,000-image training pool.
- **Holdout evaluation:** Scratch 2K, 12K, 18K, and pretrained 2K models were evaluated on an additional 2,000-image holdout.
- **Dropout study:** Rates of 0, 0.25, and 0.5 were compared on validation data and a *different* 2,000-image holdout.
- **Random-seed stability:** Nine runs across three dropout rates and three predetermined random seeds.
- **Error analysis:** Confusion matrices, overlapping errors, representative image galleries, and inspection of suspicious source labels.
- **Data-quality audit:** Five groups of exact duplicate image contents were identified. All **28 model-specific training–evaluation comparisons** had zero shared SHA-256 image hashes. Exact hashing does not exclude visual near-duplicates; repeated observations within some extended datasets are documented.

## Repository structure

```text
Assignment_2_CNN/
├── README.md
├── notebooks/
│   ├── Assignment_2_CNN_Final_Executed.ipynb
│   └── Assignment_2_CNN_Results_and_Code.html
├── report/
│   └── Assignment_2_CNN_Aamir_Sohaib.pdf
├── src/
│   ├── assignment2_cnn.py
│   └── model_specific_duplicate_audit.py
├── results/
├── figures/
│   └── error_examples/
├── audit/
├── docs/
├── archive/
│   └── Original_Exploratory_Notebook.ipynb
└── requirements.txt
```

## Reproducibility and limitations

The experiments were run in Kaggle with **Python 3.13.15**, **TensorFlow 2.20.0**, and **Pandas 2.3.3**. To reproduce training, attach the Dogs vs. Cats competition input containing `train.zip`; the original dataset and large `.keras` model checkpoints are not committed to GitHub. The executed notebook and saved figures document the reported results.

MobileNetV2 loads ImageNet weights associated with its standard 224-pixel configuration while accepting 150 × 150 inputs because `include_top=False`. The scratch and pretrained architectures differ, so their performance gap cannot be attributed solely to pretraining. All supplementary holdouts are identified separately to avoid conflating evaluations of different image cohorts. Exact-content hashing rules out only byte-identical overlap, not perceptually similar images with different encodings.

For detailed methodology and conclusions, see the **[academic report](report/Assignment_2_CNN_Aamir_Sohaib.pdf)**.

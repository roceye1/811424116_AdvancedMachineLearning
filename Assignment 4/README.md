# Assignment 4 — IMDB Sentiment Classification

This folder contains the Kaggle notebook and results of an experiment comparing learned word embeddings with frozen, pretrained GloVe embeddings in the same bidirectional LSTM model.

## Results
| Training reviews | Learned embedding | Frozen GloVe |
|---:|---:|---:|
| 100 | 52.38% | 51.19% |
| 500 | 61.91% | 56.71% |
| 2,000 | 79.31% | 67.34% |
| 5,000 | 82.13% | 76.34% |
| 10,000 | 83.35% | 79.54% |

The learned embedding outperformed frozen GloVe at every tested sample size. These are **validation** accuracies on a fixed set of 10,000 reviews, not independent test-set results. The largest advantage was 11.97 percentage points at 2,000 training reviews.

## Experiment
- IMDB movie-review sentiment prediction (positive / negative).
- Up to 150 tokens per review and a 10,000-token vocabulary cap.
- Training subsets: 100, 500, 2,000, 5,000, 10,000 reviews.
- Two otherwise equivalent bidirectional LSTM classifiers, one with a trainable 100-dimensional embedding and one with frozen GloVe 100-dimensional embeddings.
- Early stopping on validation loss.

## Included files
- `Assignment_4_Kaggle_IMDB_Embeddings.ipynb` — Kaggle-ready code notebook **(source notebook; does not contain saved Kaggle execution outputs)**.
- `assignment4_experiment_results.csv` — all 10 experiment records, accuracies, losses, and training information.
- `assignment4_comparison_table.csv` — side-by-side accuracy comparison.

## Additional submission files
A PDF write-up (`Assignment_4_Final_Report.pdf`) and a performance graph (`assignment4_accuracy_graph.png`) were prepared separately. They must be uploaded to this folder for a complete GitHub submission. The **executed Kaggle notebook exported with outputs saved** should replace the source-only notebook before submission.

The PDF and plot were not yet uploaded by the automated repository update.

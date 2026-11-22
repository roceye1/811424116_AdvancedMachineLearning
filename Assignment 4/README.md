# Assignment 4 — IMDB Sentiment Classification

## Objective

This experiment compares two approaches to representing words for positive/negative IMDB movie-review classification: (1) embeddings learned from the labeled reviews and (2) frozen pretrained GloVe embeddings. Both approaches use the same bidirectional LSTM model. Validation accuracy is compared as the amount of training data increases.

## Methodology

- **Input:** reviews standardized to lowercase with punctuation removed, truncated or padded to 150 tokens.
- **Vocabulary:** up to 10,000 tokens, adapted separately on each training subset without using validation text.
- **Training sizes:** 100, 500, 2,000, 5,000, and 10,000 balanced labeled reviews.
- **Validation:** the same stratified, held-out sample of 10,000 reviews in every run.
- **Shared model:** 100-dimensional word embeddings, a bidirectional LSTM with 16 units in each direction, 30% dropout, and a sigmoid output.
- **Approaches:** trainable embeddings learned from scratch versus frozen pretrained GloVe 100-dimensional vectors; downstream LSTM and classifier are trained for both.
- **Training:** up to six epochs with early stopping based on validation loss.

## Results

| Training reviews | Learned embedding | Frozen GloVe | Learned advantage (pp) |
| ---: | ---: | ---: | ---: |
| 100 | 52.38% | 51.19% | +1.19 |
| 500 | 61.91% | 56.71% | +5.20 |
| 2,000 | 79.26% | 67.34% | +11.92 |
| 5,000 | 81.90% | 76.34% | +5.56 |
| 10,000 | **83.36%** | **79.54%** | **+3.82** |

Learned embeddings performed better at all five tested training sizes. The largest difference was **11.92 percentage points at 2,000 reviews**. The two methods were very close to the 50% chance reference at only 100 reviews.

![Validation accuracy by training size](assignment4_accuracy_graph.png)

## Interpretation

The learned word vectors were adjustable for the movie-review sentiment task, whereas GloVe vectors were pretrained on general text and kept fixed. This offers a plausible explanation for the observed difference but does not establish the cause. Both models improved as training examples increased.

The experiment used an **RNN-based bidirectional LSTM**. Transformers and attention are discussed in the written report for context; no Transformer was trained. Scores are **validation accuracies**, not independent test-set accuracies. The results reflect one model design and one random sampling configuration.

## Files

| File | Contents |
| --- | --- |
| `Assignment_4_Kaggle_IMDB_Embeddings.ipynb` | Executed Kaggle notebook containing the Python code, training logs, validation results, comparison table, and saved chart |
| `Assignment_4_Report.pdf` | Six-page final written analysis with experimental design, charts, findings and limitations |
| `assignment4_report.html` | Kaggle-generated standalone HTML record of the experimental setup, results and accuracy chart |
| `assignment4_experiment_results.csv` | Ten experiment results including accuracy, loss, epoch counts and embedding coverage |
| `assignment4_comparison_table.csv` | Side-by-side validation accuracy at each training size |
| `assignment4_accuracy_graph.png` | Validation accuracy chart exported from the Kaggle experiment |
| `assignment4_summary.md` | Text summary of methodology, results and interpretation |

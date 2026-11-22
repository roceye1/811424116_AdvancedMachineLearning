# Assignment 4 — Experiment Summary

## Setup
- Task: IMDB movie-review sentiment classification (positive vs. negative).
- Maximum review length: 150 tokens.
- Vocabulary cap: 10,000 tokens (learned from each training subset only).
- Validation set: 10,000 held-out reviews; fixed across experiments.
- Training sizes: 100, 500, 2,000, 5,000, 10,000.
- Model: bidirectional LSTM; both approaches share the same architecture.
- Comparison: learned embedding versus 100d GloVe, frozen during training.

## Actual validation results

| Training reviews | Learned embedding | Frozen GloVe | Winner |
|---:|---:|---:|---|
| 100 | 52.4% | 51.2% | Learned embedding |
| 500 | 61.9% | 56.7% | Learned embedding |
| 2,000 | 79.3% | 67.3% | Learned embedding |
| 5,000 | 81.9% | 76.3% | Learned embedding |
| 10,000 | 83.4% | 79.5% | Learned embedding |

## Main findings
- Winner at 100 reviews: Learned embedding.
- First TESTED training size where learned embedding beat frozen GloVe: 100 reviews. This is not necessarily the exact crossover point between tested sizes.

## Interpretation
- Across these experiments, observed validation differences reflect the joint effect of the embeddings and the trained bidirectional LSTM.
- A task-trained embedding can adapt word representations to movie-review sentiment, whereas the frozen GloVe vectors are fixed.
- The comparison does not establish why one model wins in a given run; the effect of sampling, model settings, and training randomness remains relevant.

## Model architecture and study limitations
- The experiment trains bidirectional LSTMs, not Transformers. A Transformer uses attention and positional information rather than recurrence to combine token information.
- The reported scores use a fixed validation set, rather than an independent final test set.
- Generalization beyond this dataset and these training settings was not measured.
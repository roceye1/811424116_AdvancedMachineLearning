# Assignment 3: Time-Series Data

**Course:** Advanced Machine Learning  
**Task:** Predict Jena air temperature 24 hours ahead using the previous five days of hourly weather observations.

## Files

| File | Purpose |
| --- | --- |
| `assignment-3-time-series-data.ipynb` | Kaggle notebook with the Python implementation, executed results and figures. |
| `assignment3_experiments.py` | Standalone Python version of the forecasting experiments. |
| `assignment3_notebook_executed.html` | HTML export of the executed notebook, including code and outputs. |
| `Assignment3_Time_Series_Experiment_Summary.pdf` | Separate, code-free report with experimental design, figures, results and conclusions. |
| `figures/` | Validation comparison, validation learning curves and a sample held-out forecast. |
| `results/` | CSV summaries of validation/test metrics and the selected LSTM's test predictions. |

## Dataset and method

- **Data:** [Jena Climate 2009-2016](https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016), with 420,551 ten-minute records and 14 weather measurements.
- **Preparation:** Hourly aggregation; missing-value handling using past observations; standardization based on the training period only.
- **Splits:** Chronological 70% training, 15% validation and 15% testing; no forecasting window crosses a split boundary.
- **Forecasting setup:** 120 hourly observations (five days) as input; target temperature 24 hours later; six-hour stride between forecast windows.
- **Experiments:** Seasonal-persistence baseline, GRU with 32 or 64 units, LSTM with 32 or 64 units, and a Conv1D + GRU (32-unit) model.
- **Evaluation:** Mean absolute error (MAE) in degrees Celsius. The best model was chosen using validation data before evaluation on the held-out test period.

## Results

| Approach | Validation MAE (°C) | Held-out test MAE (°C) |
| --- | ---: | ---: |
| **LSTM 64** | **2.416** | **2.317** |
| GRU 32 | 2.423 | 2.339 |
| Conv1D + GRU 32 | 2.463 | — |
| GRU 64 | 2.469 | — |
| LSTM 32 | 2.492 | — |
| Seasonal persistence | 2.586 | — |

The LSTM with 64 units reduced validation MAE by **6.6%** relative to the seasonal-persistence baseline. It also had a slightly lower test MAE than GRU 32, although the 0.022°C test gap is too small to establish a general architectural advantage from one run. Increasing recurrent units and adding a convolutional layer did not consistently improve accuracy.

## Reproduction

Open the notebook in Kaggle, attach the Jena Climate dataset linked above, and run the cells in order. The Python script is an equivalent code-only alternative for a Kaggle environment. Both require Python with TensorFlow, NumPy, pandas and Matplotlib. The script writes figures and metrics into `/kaggle/working/`.

Results are from one fixed-seed experiment with a single time-based split and eight maximum training epochs. Re-running neural-network training can produce small numerical differences. The CSV `results/test_predictions.csv` contains **LSTM 64** forecasts only; the GRU 32 test score is recorded in `results/test_model_comparison.csv` and the executed notebook.

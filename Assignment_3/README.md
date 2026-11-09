# Assignment 3 - Time-Series Data

**Student:** Aamir Sohaib  
**Course:** Advanced Machine Learning  
**Dataset:** Jena Climate 2009-2016  
**Objective:** Forecast air temperature 24 hours ahead using five days of hourly observations.


Additional files:

- `Assignment3_Expanded_Report.html` - supplementary model-summary HTML from the Kaggle results export.
- `figures/` - validation-vs-test graph, learning curves, and test-forecast graph.
- `results/` - complete per-model validation/test MAE, all test predictions, validation table, and per-epoch learning-curve values.
- `models/validation_selected_best_model.keras` - trained GRU 32 checkpoint from the original expanded-results ZIP.
- `MANIFEST.txt` - hashes and file sizes of all included files (except itself).

## Experiment design

- Input: 120 hourly observations of 14 weather measurements (five days).
- Target: air temperature 24 hours after the final input observation.
- Window spacing: six hours; 1,730 final test examples.
- Chronological split: 70% training (49,090 hourly rows), 15% validation (10,519), 15% testing (10,520).
- Preprocessing: hourly averaging; forward-fill missing observations; training-only standardization.
- Models: seasonal-persistence baseline; GRU 32/64; LSTM 32/64; Conv1D + GRU 32; stacked GRU 32-64 and 64-32; stacked LSTM 32-64 and 64-32.
- Training: Adam learning rate 0.001, batch size 256, up to 15 epochs with early stopping and best-weight restoration.
- Selection: model with lowest **validation** MAE. All ten test MAEs were computed afterward for descriptive comparison.

## Results (MAE in °C; lower is better)

| Validation rank | Model | Validation MAE | Test MAE |
| ---: | --- | ---: | ---: |
| 1 | GRU 32 | 2.380 | 2.296 |
| 2 | Stacked GRU 64-32 | 2.401 | 2.296 |
| 3 | LSTM 64 | 2.409 | 2.302 |
| 4 | Stacked LSTM 64-32 | 2.411 | 2.361 |
| 5 | Stacked GRU 32-64 | 2.417 | 2.302 |
| 6 | Conv1D + GRU 32 | 2.431 | 2.309 |
| 7 | Stacked LSTM 32-64 | 2.450 | 2.413 |
| 8 | LSTM 32 | 2.467 | 2.347 |
| 9 | GRU 64 | 2.469 | 2.387 |
| 10 | Seasonal persistence | 2.586 | 2.500 |

**Validation-selected winner:** GRU 32, 2.380°C validation MAE and 2.296°C test MAE. Baseline errors were 2.586°C validation and 2.500°C test. This is an 8.0% reduction in validation MAE.

**Test-score nuance:** Stacked GRU 64-32 has a test MAE of 2.295876°C versus 2.295959°C for GRU 32. The difference is only 0.000083°C. The selected winner remains GRU 32 because validation, not test, was used to select the model.

## Reproduce the experiment

1. Open `assignment-3-revised.ipynb` in Kaggle and attach the **Jena Climate 2009-2016** dataset containing `jena_climate_2009_2016.csv`.
2. Run the notebook's cells in order, with Python, TensorFlow/Keras, NumPy, pandas and Matplotlib installed. A GPU is recommended but not essential for correctness.
3. The included `.py` script contains the same model and evaluation code as the revised notebook. The notebook currently stops after the plots and does not automatically regenerate this curated PDF; the PDF is a separate summary based on the saved real-data outputs.
4. Neural-network retraining may produce slightly different scores due to environment, seeds and numerical implementation. The included CSV and executed outputs document the **recorded** experiment.

## Assignment compliance

The original instructions explicitly allow any or all suggested improvement methods. This experiment includes **all three** suggestions: unit-count changes within stacked recurrent layers, LSTM-versus-GRU comparisons, and Conv1D combined with an RNN. It evaluates the best-performing validation models on the test split and reports test MAE for every configuration.

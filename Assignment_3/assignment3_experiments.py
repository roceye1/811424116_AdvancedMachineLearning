"""Assignment 3 - Time-Series Data (Jena climate forecasting).

Run in Kaggle with the Jena Climate 2009-2016 dataset attached as input.
Forecast air temperature 24 hours after five days of hourly observations.
The validation split determines model selection; the test split is held out.
"""

# Outputs are saved in /kaggle/working.
import os, glob, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
paths = glob.glob('/kaggle/input/**/jena_climate_2009_2016.csv', recursive=True)
assert len(paths) == 1, f'Expected one Jena CSV, found: {paths}'
df = pd.read_csv(paths[0])
df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
print('Dataset:', paths[0], '| shape:', df.shape)
print('Dates:', df['Date Time'].min(), 'to', df['Date Time'].max())
print(df.head().to_string(index=False))

# ---

target_col = 'T (degC)'
features = [c for c in df.columns if c != 'Date Time']
sentinel_count = int((df[features] == -9999).sum().sum())
raw = df.sort_values('Date Time').set_index('Date Time')[features].replace(-9999, np.nan)
hourly_raw = raw.resample('1h').mean()
missing_bins = int(hourly_raw[target_col].isna().sum())
hourly = hourly_raw.ffill().reset_index()
assert not hourly[features].isna().any().any()
assert hourly['Date Time'].diff().dropna().eq(pd.Timedelta(hours=1)).all()
print('Original rows:', len(df), '| hourly rows:', len(hourly), '| variables:', len(features))
print('Missing-value markers:', sentinel_count, '| empty hourly temperature bins:', missing_bins)
print('Missing bins/values filled using earlier observations only.')
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(hourly['Date Time'], hourly[target_col], linewidth=.45)
ax.set(title='Jena air temperature, hourly mean', xlabel='Date', ylabel='Temperature (°C)')
fig.tight_layout(); plt.show()

n = len(hourly)
cut1, cut2 = int(n * .70), int(n * .85)
segments = {'train': hourly.iloc[:cut1], 'validation': hourly.iloc[cut1:cut2], 'test': hourly.iloc[cut2:]}
summary = pd.DataFrame([{'split': k, 'rows': len(v), 'from': v['Date Time'].iloc[0], 'through': v['Date Time'].iloc[-1]} for k, v in segments.items()])
print(summary.to_string(index=False))
mu = segments['train'][features].mean().to_numpy(dtype='float32')
sigma = segments['train'][features].std().replace(0, 1).to_numpy(dtype='float32')
print('Training-only temperature mean/std:', round(float(mu[features.index(target_col)]), 3), round(float(sigma[features.index(target_col)]), 3))

# ---

HISTORY, HORIZON, STRIDE = 120, 24, 6
temp_idx = features.index(target_col)

def make_windows(part):
    values = part[features].to_numpy(dtype='float32')
    scaled = (values - mu) / sigma
    starts = np.arange(0, len(part) - HISTORY - HORIZON + 1, STRIDE)
    target_idx = starts + HISTORY - 1 + HORIZON
    X = np.stack([scaled[s:s + HISTORY] for s in starts]).astype('float32')
    y = scaled[target_idx, temp_idx].astype('float32')
    actual = values[target_idx, temp_idx]
    last_temp = values[starts + HISTORY - 1, temp_idx]
    dates = part['Date Time'].iloc[target_idx].reset_index(drop=True)
    assert ((dates.to_numpy() - part['Date Time'].iloc[starts + HISTORY - 1].to_numpy()) == np.timedelta64(24, 'h')).all()
    return X, y, actual, last_temp, dates

X_train, y_train, _, _, train_dates = make_windows(segments['train'])
X_val, y_val, val_actual, val_last, val_dates = make_windows(segments['validation'])
baseline_val_mae = float(np.mean(np.abs(val_actual - val_last)))
print('Train windows:', X_train.shape, '| validation windows:', X_val.shape)
print('First/last training target:', train_dates.iloc[0], train_dates.iloc[-1])
print('First/last validation target:', val_dates.iloc[0], val_dates.iloc[-1])
print(f'Seasonal persistence validation MAE: {baseline_val_mae:.3f} °C')

# ---

tf.keras.backend.clear_session()

def build_model(kind, units):
    inputs = tf.keras.Input(shape=(HISTORY, len(features)))
    if kind == 'Conv1D+GRU':
        x = tf.keras.layers.Conv1D(32, 5, padding='causal', activation='relu')(inputs)
        x = tf.keras.layers.GRU(units)(x)
    elif kind == 'GRU':
        x = tf.keras.layers.GRU(units)(inputs)
    else:
        x = tf.keras.layers.LSTM(units)(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs, name='forecast_' + kind.replace('+', '_') + '_' + str(units))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mae')
    return model

specs = [('GRU', 32), ('GRU', 64), ('LSTM', 32), ('LSTM', 64), ('Conv1D+GRU', 32)]
models, histories = {}, {}
training_rows = [{'model': 'Seasonal persistence', 'architecture': 'Baseline', 'units': 0, 'epochs': 0, 'validation_MAE_C': baseline_val_mae}]
for i, (kind, units) in enumerate(specs):
    tf.keras.utils.set_random_seed(SEED + i)
    name = f'{kind} {units}'
    model = build_model(kind, units)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, min_delta=1e-4, restore_best_weights=True), tf.keras.callbacks.TerminateOnNaN()]
    print('Training', name, flush=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=8, batch_size=256, callbacks=callbacks, verbose=0)
    pred_c = model.predict(X_val, batch_size=512, verbose=0).ravel() * sigma[temp_idx] + mu[temp_idx]
    val_mae_c = float(np.mean(np.abs(val_actual - pred_c)))
    models[name], histories[name] = model, history.history
    training_rows.append({'model': name, 'architecture': kind, 'units': units, 'epochs': len(history.history['loss']), 'validation_MAE_C': val_mae_c})
    print(f'{name}: {val_mae_c:.3f} °C validation MAE after {len(history.history["loss"])} epochs', flush=True)
comparison = pd.DataFrame(training_rows).sort_values('validation_MAE_C').reset_index(drop=True)
print(comparison.round(3).to_string(index=False))

# ---

best_name = comparison.iloc[0]['model']
best_val_mae = float(comparison.iloc[0]['validation_MAE_C'])
print(f'Selected on validation only: {best_name}, MAE {best_val_mae:.3f} °C')
print(f'Change from seasonal persistence: {(baseline_val_mae - best_val_mae) / baseline_val_mae * 100:.1f}% lower MAE')

fig, ax = plt.subplots(figsize=(9, 4))
ax.barh(comparison['model'], comparison['validation_MAE_C'], color=['#1b7895'] + ['#91a8b2'] * (len(comparison)-1))
ax.invert_yaxis()
ax.set(xlabel='Validation MAE (°C)', title='24-hour temperature forecast: validation comparison')
for i, v in enumerate(comparison['validation_MAE_C']): ax.text(v + .015, i, f'{v:.3f}', va='center')
ax.set_xlim(0, comparison['validation_MAE_C'].max() * 1.18)
fig.tight_layout(); fig.savefig('/kaggle/working/validation_comparison.png', dpi=160); plt.show()

fig, ax = plt.subplots(figsize=(9, 4))
for name, h in histories.items():
    ax.plot(np.arange(1, len(h['val_loss']) + 1), np.array(h['val_loss']) * sigma[temp_idx], marker='o', label=name)
ax.set(xlabel='Epoch', ylabel='Validation MAE (°C)', title='Validation learning curves')
ax.legend(ncol=2, fontsize=8); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig('/kaggle/working/learning_curves.png', dpi=160); plt.show()
comparison.to_csv('/kaggle/working/validation_results.csv', index=False)
if best_name in models: models[best_name].save('/kaggle/working/best_temperature_model.keras')

# ---

X_test, y_test, test_actual, test_last, test_dates = make_windows(segments['test'])
if best_name == 'Seasonal persistence':
    test_pred = test_last.copy()
else:
    test_pred = models[best_name].predict(X_test, batch_size=512, verbose=0).ravel() * sigma[temp_idx] + mu[temp_idx]
test_mae = float(np.mean(np.abs(test_actual - test_pred)))
test_summary = pd.DataFrame([{'selected_model': best_name, 'validation_MAE_C': best_val_mae, 'test_MAE_C': test_mae, 'test_windows': len(test_actual)}])
print(test_summary.round(3).to_string(index=False))
print('First/last test target:', test_dates.iloc[0], test_dates.iloc[-1])

predictions = pd.DataFrame({'target_time': test_dates, 'actual_temperature_C': test_actual, 'predicted_temperature_C': test_pred, 'absolute_error_C': np.abs(test_actual - test_pred)})
predictions.to_csv('/kaggle/working/test_predictions.csv', index=False)
test_summary.to_csv('/kaggle/working/final_test_metrics.csv', index=False)
fig, ax = plt.subplots(figsize=(12, 4))
show = slice(0, 120)
ax.plot(test_dates.iloc[show], test_actual[show], label='Actual', linewidth=1.7)
ax.plot(test_dates.iloc[show], test_pred[show], label=best_name, linewidth=1.3)
ax.set(title='Held-out test: first 30 days of 24-hour-ahead forecasts', xlabel='Target date', ylabel='Temperature (°C)')
ax.legend(); ax.grid(alpha=.2)
fig.autofmt_xdate(); fig.tight_layout()
fig.savefig('/kaggle/working/test_forecast_example.png', dpi=160); plt.show()

# ---

# Test GRU 32 on the held-out test dataset

gru32_model = models['GRU 32']

# Predict temperatures and convert back to Celsius
gru32_pred = gru32_model.predict(
    X_test, batch_size=512, verbose=0
).ravel() * sigma[temp_idx] + mu[temp_idx]

# Calculate test MAE
gru32_test_mae = float(
    np.mean(np.abs(test_actual - gru32_pred))
)

print(f"GRU 32 Test MAE: {gru32_test_mae:.3f} °C")
print(f"LSTM 64 Test MAE: {test_mae:.3f} °C")

# Compare both models
test_comparison = pd.DataFrame({
    'Model': ['LSTM 64', 'GRU 32'],
    'Validation MAE (°C)': [
        best_val_mae,
        float(comparison.loc[
            comparison['model'] == 'GRU 32',
            'validation_MAE_C'
        ].iloc[0])
    ],
    'Test MAE (°C)': [
        test_mae,
        gru32_test_mae
    ]
})

print(test_comparison.round(3).to_string(index=False))

# Save comparison results
test_comparison.to_csv(
    '/kaggle/working/test_model_comparison.csv',
    index=False
)

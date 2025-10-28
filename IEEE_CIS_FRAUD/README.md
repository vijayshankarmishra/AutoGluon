# AutoGluon Fraud Detection Pipeline

This README explains, step by step, what each part of your **Google Colab / Jupyter** notebook does.  
Shell commands (lines starting with `!`) are meant for Colab notebooks.

---

## 📌 TL;DR (Pipeline Summary)

1. **Setup**: install tools and download the dataset from Google Drive to `/content/data`.
2. **Load & Merge**: join identity + transaction CSVs by `TransactionID`.
3. **Sample**: take a **20% stratified** sample by `isFraud` to speed up experiments.
4. **Train**: fit an **AutoGluon Tabular** model (`medium_quality`) with **ROC AUC**.
5. **Align**: make test columns match the training feature schema.
6. **Predict & Submit**: generate fraud probabilities and save **`my_submission.csv`**.

---

## 1) Environment Setup & Data Download

```bash
!pip install -U pip setuptools wheel
#!pip install -U autogluon autogluon.timeseries

# Install gdown and create the target directory
!pip -q install gdown
!mkdir -p /content/data

# Download the entire shared folder to /content/data
!gdown --folder --fuzzy "https://drive.google.com/drive/folders/1K6Zdl_rt8AH0XRE4ww_jbvUtP3Zm91Xa?usp=sharing" -O /content/data

# Quick check
!ls -lah /content/data
```

**What this does**  
- Updates packaging tools.  
- Installs **gdown** to fetch a shared Google Drive folder.  
- Ensures `/content/data` exists, downloads the folder there, and lists contents to confirm files.

> **Note:** Outside Colab, remove `!` and run in your shell (or use Python subprocess).

---

## 2) Imports & Basic Settings

```python
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor

directory = '/content/data/'        # where the CSVs live
label = 'isFraud'                   # target column to predict
eval_metric = 'roc_auc'             # competition metric: AUC
save_path = directory + 'AutoGluonModels/'  # model output dir
```

- `TabularPredictor` is AutoGluon’s high-level interface for tabular ML.
- `roc_auc` is well-suited for imbalanced problems like fraud detection.

---

## 3) Read & Merge Training Tables

```python
train_identity = pd.read_csv(directory+'train_identity.csv')
train_transaction = pd.read_csv(directory+'train_transaction.csv')

train_data = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
```

- Loads both training CSVs and **left-joins** on `TransactionID`.  
- Using `how='left'` preserves all transactions even if identity info is missing.

---

## 4) Faster Iteration with a 20% Stratified Sample

```python
# --- SAMPLE 20% OF TRAINING DATA (stratified by label) ---
train_data = (
    train_data.groupby(label, group_keys=False)
              .apply(lambda x: x.sample(frac=0.2, random_state=42))
              .reset_index(drop=True)
)
print(f"Train rows after 20% sample: {len(train_data):,}")
# -----------------------------------------------------------
```

- Downsamples **within each class** to preserve the fraud/non-fraud ratio.  
- Use a small fraction for quick experiments; remove this for final/full training.

---

## 5) Train the AutoGluon Model

```python
predictor = TabularPredictor(label=label, eval_metric=eval_metric, path=save_path, verbosity=3).fit(
    train_data,
    presets='medium_quality',
    time_limit=3600
)

results = predictor.fit_summary()
```

- AutoGluon handles preprocessing, model selection, and ensembling.  
- `medium_quality` balances speed and performance; adjust `time_limit` as needed.  
- `fit_summary()` prints a concise training report.

---

## 6) Prepare Test Data & Align Schema

```python
test_identity = pd.read_csv(directory+'test_identity.csv')
test_transaction = pd.read_csv(directory+'test_transaction.csv')
test_data = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')
```

**Align to training features**:

```python
# --- ALIGN TEST COLUMNS TO TRAIN FEATURES ---
train_features = predictor.features()

# Add missing training features to test as NaN
missing_cols = [c for c in train_features if c not in test_data.columns]
for c in missing_cols:
    test_data[c] = np.nan

# Drop any extras not used during training (including label if present)
extra_cols = [c for c in test_data.columns if c not in train_features]
if extra_cols:
    test_data = test_data.drop(columns=extra_cols)

# Reorder columns to exactly match training
test_data = test_data[train_features]

print(f"Aligned test columns. Added {len(missing_cols)} missing, dropped {len(extra_cols)} extras.")
# -----------------------------------------------------------
```

- Ensures the test set has the **same columns and order** the model expects.

---

## 7) Predict Probabilities

```python
y_predproba = predictor.predict_proba(test_data)
y_predproba.head(5)

predictor.positive_class

y_predproba = predictor.predict_proba(test_data, as_multiclass=False)
```

- Gets **probabilities** for the positive class (fraud = 1).  
- `as_multiclass=False` returns a 1D vector of positive-class probabilities.

---

## 8) Build the Submission File

```python
submission = pd.read_csv(directory+'sample_submission.csv')
submission['isFraud'] = y_predproba
submission.head()
submission.to_csv(directory+'my_submission.csv', index=False)
```

- Uses the official sample to keep required ordering & column names.  
- Writes **`my_submission.csv`**, ready to upload to Kaggle.

---

## ✅ Tips & Pitfalls

- **Sampling**: Use for speed while prototyping; remove for final training.  
- **Imbalance**: ROC AUC is appropriate; for better results increase `time_limit`, try `best_quality`, or engineer features.  
- **Memory**: Consider downcasting (`float32`, `category`) or chunked reads if needed.  
- **Reproducibility**: Fix seeds and pin versions when possible.  
- **Schema**: Always align test features with `predictor.features()`.

---

## 🚀 Extensions

- **Feature importance**: `predictor.feature_importance(train_data)`  
- **Hyperparameter tuning**: pass a `hyperparameters` dict in `.fit(...)`  
- **Deeper ensembling**: larger presets + longer `time_limit`  
- **Deployment**: persist the best model and serve predictions

---

## 🔁 Full-Data Training

For the final leaderboard run, **remove the sampling block** in Step 4 and retrain.

---

**Happy modeling!** 🧠⚡

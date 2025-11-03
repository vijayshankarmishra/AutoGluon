# California Housing Prices — AutoGluon Tabular (End-to-End)

This repository/notebook predicts California housing prices using **AutoGluon Tabular**.  
It provides a practical, fast baseline you can run in **Google Colab** or **local Jupyter**.

## What this project does

1. **Environment setup** (AutoGluon + optional Kaggle CLI)
2. **Data access** (Kaggle download or local CSVs)
3. **Preprocessing** (ID cleanup, light numeric transforms, optional `log1p`)
4. **Model training** (AutoGluon `TabularPredictor` with a small time budget)
5. **Evaluation** (RMSE on an unseen holdout; optional dollar-scale RMSE)
6. **Prediction + export** (creates `submission.csv`)

### Here is the video for explanation : https://drive.google.com/drive/folders/1KPqvk5hlPohl5WU5OqTWx6mAh-xWdMx8

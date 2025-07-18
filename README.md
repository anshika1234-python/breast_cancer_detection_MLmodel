# breast_cancer_detection_MLmodel

## What it Does
A classical machine learning pipeline for breast cancer classification using logistic regression and feature importance analysis.

## Project Structure
- `cancer_ml.py`: full pipeline script
- `cancer_pipeline.ipynb`: Jupyter notebook version (optional)
- `scaler.pkl`, `cancer_model.pkl`: saved artifacts
- `venv/`: virtual environment

## Requirements
```bash
python3 -m venv venv
source venv/bin/activate          # Linux/macOS
# or venv\Scripts\activate       # Windows
pip install -r requirements.txt
# Breast Cancer Detection Model

This model predicts whether a tumor is benign or malignant based on input features.

## Model

```python
import joblib
model = joblib.load('cancer_model.pkl')


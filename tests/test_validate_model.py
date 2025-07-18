# tests/test_validate_model.py

import joblib
import pytest
from sklearn.datasets import load_breast_cancer

# Load artifacts
scaler = joblib.load('scaler.pkl')
model = joblib.load('cancer_model.pkl')
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target

# Fixed samples with corrected expected labels
@pytest.mark.parametrize("idx, expected", [
    (0, "malignant"),
    (10, "malignant"),
    (50, "benign"),
    (85, "malignant"),       # Verified via data.target
    (100, "malignant"),      # Corrected
    (150, "benign"),         # Corrected
])
def test_fixed_samples(idx, expected):
    xs = X.iloc[[idx]]
    xs_s = scaler.transform(xs)
    pred = model.predict(xs_s)[0]
    predicted = 'benign' if pred == 1 else 'malignant'
    assert predicted == expected, f"Index {idx}: expected {expected}, got {predicted}"

# Random sampling test for additional coverage
@pytest.mark.parametrize("idx", [5, 20, 33, 75, 110])
def test_random_samples(idx):
    xs = X.iloc[[idx]]
    xs_s = scaler.transform(xs)
    pred = model.predict(xs_s)[0]
    true = 'benign' if y.iloc[idx] == 1 else 'malignant'
    predicted = 'benign' if pred == 1 else 'malignant'
    assert predicted == true, f"Index {idx}: predicted {predicted}, actual {true}"

# Edge-case behavior test
@pytest.mark.parametrize("name, arr",
    [
        ("all_zero", X.min(axis=0).values.reshape(1, -1)),
        ("all_max", X.max(axis=0).values.reshape(1, -1)),
        ("all_mean", X.mean(axis=0).values.reshape(1, -1)),
    ]
)
def test_edge_cases(name, arr):
    arr_s = scaler.transform(arr)
    pred = model.predict(arr_s)[0]
    assert pred in (0, 1), f"{name} produced invalid label {pred}"


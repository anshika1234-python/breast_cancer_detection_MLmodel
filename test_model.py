import pytest
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# Load model and scaler
model = joblib.load('cancer_model.pkl')
scaler = joblib.load('scaler.pkl')
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target

@pytest.mark.parametrize("idx, expected", [(0, "malignant"), (50, "benign")])
def test_model_predictions(idx, expected):
    xs = X.iloc[[idx]]
    xs_scaled = scaler.transform(xs)
    pred = model.predict(xs_scaled)[0]
    predicted = 'benign' if pred == 1 else 'malignant'
    assert predicted == expected

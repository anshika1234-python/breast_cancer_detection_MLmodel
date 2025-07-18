import joblib, random
import numpy as np
from sklearn.datasets import load_breast_cancer

# Load artifacts
scaler = joblib.load('scaler.pkl')
model = joblib.load('cancer_model.pkl')

# Load dataset
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target

# 1. Fixed indices test
test_indices = [0, 10, 50, 85, 100, 150]
expected_labels = {0:"malignant",10:"malignant",50:"benign",85:"malignant",100:"benign",150:"malignant"}

print("== Fixed index testing ==")
for idx in test_indices:
    xs = X.iloc[[idx]]
    xs_s = scaler.transform(xs)
    pred = model.predict(xs_s)[0]
    prob = model.predict_proba(xs_s)[0,1]
    predicted = 'benign' if pred==1 else 'malignant'
    actual = 'benign' if y[idx]==1 else 'malignant'
    correct = '✅' if predicted==expected_labels[idx] else '❌'
    print(f"Index {idx}: pred={predicted} ({prob:.2f}), actual={actual}, expected={expected_labels[idx]} {correct}")

# 2. Random sampling test
print("\n== Random sampling testing ==")
random_indices = random.sample(range(len(X)), 10)
for idx in random_indices:
    xs = X.iloc[[idx]]
    xs_s = scaler.transform(xs)
    pred = model.predict(xs_s)[0]
    actual = 'benign' if y[idx]==1 else 'malignant'
    print(f"Index {idx}: pred={'benign' if pred==1 else 'malignant'} | actual={actual} | {'OK' if (pred==y[idx]) else 'WRONG'}")

# 3. Edge-case testing
print("\n== Edge-case testing ==")
edge_cases = {
    "all_zero": np.zeros((1, X.shape[1])),
    "all_max": np.full((1, X.shape[1]), X.max().max()),
    "all_min": np.full((1, X.shape[1]), X.min().min())
}
for name, arr in edge_cases.items():
    arr_s = scaler.transform(arr)
    pred = model.predict(arr_s)[0]
    prob = model.predict_proba(arr_s)[0,1]
    print(f"{name}: pred={'benign' if pred==1 else 'malignant'} (prob benign={prob:.2f})")

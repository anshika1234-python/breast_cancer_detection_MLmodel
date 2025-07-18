# cancer_ml.py

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    data = load_breast_cancer(as_frame=True)
    return data.data, data.target, data.feature_names, data.target_names

def preprocess(X_train, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, 'scaler.pkl')
    return X_train_s, X_test_s

def train_model(X_train, y_train):
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'cancer_model.pkl')
    return model

def evaluate(model, X_test, y_test, target_names):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=target_names))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def feature_importance(model, X, y, feature_names):
    r = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    imp = pd.Series(r.importances_mean, index=feature_names)
    imp.nlargest(10).plot(kind='barh')
    plt.title("Top 10 Feature Importances (Permutation)")
    plt.show()

def main():
    X, y, names, targets = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train_s, X_test_s = preprocess(X_train, X_test)
    model = train_model(X_train_s, y_train)
    evaluate(model, X_test_s, y_test, targets)
    feature_importance(model, X_test_s, y_test, names)

if __name__ == '__main__':
    main()

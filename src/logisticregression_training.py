import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, f1_score

# Load the processed data
data = joblib.load(
    r"E:\maktab_sharif\mini_project1_week9\mini-project-01\data\processed_data.pkl"
)

X = data["X_train"]
y = data["y_train"]

# 5-Fold Stratified Cross Validation
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

recalls = []
precisions = []
f1_scores = []

print(type(X))
print(type(y))

for train_idx, test_idx in skf.split(X, y):

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = LogisticRegression(
        random_state=42,
        max_iter=1000
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    recalls.append(recall_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))

results = pd.DataFrame({
    "Mean Recall": [np.mean(recalls)],
    "Std Recall": [np.std(recalls)],
    "Mean Precision": [np.mean(precisions)],
    "Std Precision": [np.std(precisions)],
    "Mean F1-Score": [np.mean(f1_scores)],
    "Std F1-Score": [np.std(f1_scores)]
})

print(results)
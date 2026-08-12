import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, f1_score

# Load the processed data
data = joblib.load(r"E:\maktab_sharif\mini_project1_week9\mini-project-01\data\processed_data.pkl")

X = data["X_train"]
y = data["y_train"]
scaler = data["scaler"]

skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

recalls = []
precisions = []
f_1_scores = []

for train_idx, test_idx in skf.split(X, y):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    LR = LogisticRegression(random_state=42, max_iter=1000)
    LR.fit(X_train, y_train)

    y_hat = LR.predict(X_test)

    recall = recall_score(y_test, y_hat)
    precision = precision_score(y_test, y_hat)
    f_1_score = f1_score(y_test, y_hat)

    recalls.append(recall)
    precisions.append(precision)
    f_1_scores.append(f_1_score)


results = pd.DataFrame({
    "Mean_of_Recalls": [np.mean(recalls)],
    "Std_of_Recalls": [np.std(recalls)],
    "Mean_of_Precisions": [np.mean(precisions)],
    "Std_of_Precisions": [np.std(precisions)],
    "Mean_of_F1-Score": [np.mean(f_1_scores)],
    "Std_of_F1-Score": [np.std(f_1_scores)]
})


results
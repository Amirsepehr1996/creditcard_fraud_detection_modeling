import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the processed data
data = joblib.load(
    r"E:\maktab_sharif\mini_project1_week9\mini-project-01\data\processed_data.pkl"
)

X = data["X_train"]
y = data["y_train"]

print(type(X))
print(type(y))

5-Fold Stratified Cross Validation
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

Depths = [2, 5, 10, 15, None]

results = []

for depth in Depths:

    recalls = []
    precisions = []
    f1_scores = []

    for train_idx, test_idx in skf.split(X, y):

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model = DecisionTreeClassifier(
            max_depth=depth, random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        recalls.append(recall_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))

    results.append({
        "Max Depth": depth,
        "Mean Recall": np.mean(recalls),
        "Std Recall": np.std(recalls),
        "Mean Precision": np.mean(precisions),
        "Std Precision": np.std(precisions),
        "Mean F1-Score": np.mean(f1_scores),
        "Std F1-Score": np.std(f1_scores)
    })

results = pd.DataFrame(results)

print(results)


# Training using all train data
Model = DecisionTreeClassifier(max_depth= 5, random_state=42)
Model.fit(X, y)
y_hat = Model.predict(X)
cm = confusion_matrix(y, y_hat)
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal (0)", "Fraud (1)"]
)

disp.plot(cmap="Blues", values_format="d")

plt.title("Confusion Matrix - DT (Optimum Max_Depth = 5)")
plt.show()

# Manual calculation
TN, FP, FN, TP = 226583, 19, 69, 309

Recall_M = TP / (TP + FN)
Precision_M = TP / (TP + FP)
F1_M = (2 * Recall_M * Precision_M) / (Recall_M + Precision_M)

# Scikit-learn calculation
Recall_A = recall_score(y, y_hat)
Precision_A = precision_score(y, y_hat)
F1_A = f1_score(y, y_hat)

# Compare results
final_training_metrics = pd.DataFrame({
    "Metric": ["Recall", "Precision", "F1-Score"],
    "Manual": [Recall_M, Precision_M, F1_M],
    "Scikit-learn": [Recall_A, Precision_A, F1_A]
})

print(final_training_metrics)
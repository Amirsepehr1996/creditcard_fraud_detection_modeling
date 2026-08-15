import joblib
import numpy as np
import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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

print(type(X))
print(type(y))

K = [3, 5, 7, 10, 15, 20]

results = []

for k in K:

    recalls = []
    precisions = []
    f1_scores = []

    for train_idx, test_idx in skf.split(X, y):

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model = KNeighborsClassifier(
            n_neighbors=k
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        recalls.append(recall_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))

    results.append({
        "Number_of_Neighbors": k,
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
Model = KNeighborsClassifier(n_neighbors= 3)
Model.fit(X, y)

save_path = r"E:\maktab_sharif\mini_project1_week9\mini-project-01\models\knn_model.pkl"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
joblib.dump(Model, save_path)
print(f"Model saved to {save_path}")

y_hat = Model.predict(X)
cm = confusion_matrix(y, y_hat)
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal (0)", "Fraud (1)"]
)

disp.plot(cmap="Blues", values_format="d")

plt.title("Confusion Matrix - KNN (Optimum number of Neighbors = 3)")
plt.show()

# Manual calculation
TN, FP, FN, TP = 226593, 9, 63, 315

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
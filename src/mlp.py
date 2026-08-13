import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)

# Load the processed data
data = joblib.load(
    r"E:\maktab_sharif\mini_project1_week9\mini-project-01\data\processed_data.pkl"
)

X = np.asarray(data["X_train"])
y = data["y_train"]

print(type(X))
print(type(y))

skf = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

learning_rates = [0.005]

results = []


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(30, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)


for alpha in learning_rates:

    recalls = []
    precisions = []
    f1_scores = []

    for train_idx, test_idx in skf.split(X, y):
        X_train_tensor = torch.tensor(X[train_idx], dtype=torch.float32)
        X_test_tensor = torch.tensor(X[test_idx], dtype=torch.float32)
        y_train_tensor = torch.tensor(y.iloc[train_idx].values, dtype=torch.float32).reshape(-1, 1)
        y_test_tensor = torch.tensor(y.iloc[test_idx].values, dtype=torch.float32).reshape(-1, 1)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

        model = MLP()

        n_pos = y_train_tensor.sum()
        n_neg = len(y_train_tensor) - n_pos
        pos_weight = (n_neg / n_pos).clone().detach()

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=alpha)

        epochs = 50

        loss_history = []

        for epoch in range(epochs):

            total_loss = 0
            model.train()

            for X_batch, y_batch in train_loader:

                # Forward pass
                predictions = model(X_batch)

                # Calculate loss
                loss = criterion(predictions, y_batch)

                # Clear previous gradients
                optimizer.zero_grad()

                # Backpropagation
                loss.backward()

                # Update parameters
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            loss_history.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(
                    f"LR={alpha} | Epoch {epoch+1}/{epochs}, "
                    f"Loss: {avg_loss:.4f}"
                )

        model.eval()

        with torch.no_grad():
            logits = model(X_test_tensor)
            outputs = torch.sigmoid(logits)
            predictions = (outputs >= 0.7).float()

        y_test = y_test_tensor.numpy().ravel()
        y_pred = predictions.numpy().ravel()

        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))

    results.append({
        "Learning_Rate": alpha,
        "Mean Recall": np.mean(recalls),
        "Std Recall": np.std(recalls),
        "Mean Precision": np.mean(precisions),
        "Std Precision": np.std(precisions),
        "Mean F1-Score": np.mean(f1_scores),
        "Std F1-Score": np.std(f1_scores)
    })

results = pd.DataFrame(results)

print(results)
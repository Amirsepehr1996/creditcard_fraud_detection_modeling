import joblib
from sklearn.linear_model import LogisticRegression
# Load the processed data
data = joblib.load(r"E:\maktab_sharif\mini_project1_week9\mini-project-01\data\processed_data.pkl")

X_train = data["X_train"]
y_train = data["y_train"]
scaler = data["scaler"]

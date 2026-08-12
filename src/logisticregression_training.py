import joblib

# Load the processed data
data = joblib.load(r"E:\maktab_sharif\mini_project1_week9\mini-project-01\data\processed_data.pkl")

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]
scaler = data["scaler"]
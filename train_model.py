import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

PROC_DIR = "data/processed"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(f"{PROC_DIR}/titanic_clean.csv")
X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Brief validation
acc = model.score(X_test, y_test)
print(f"Test Accuracy: {acc:.3f}")

# Saving model
joblib.dump(model, f"{MODEL_DIR}/model.pkl")

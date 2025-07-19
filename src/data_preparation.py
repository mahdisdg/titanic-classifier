import pandas as pd
import os

RAW_DIR = "data/raw"
PROC_DIR = "data/processed"
os.makedirs(PROC_DIR, exist_ok=True)

df = pd.read_csv(f"{RAW_DIR}/train.csv")

# Only necessary columns
df = df[["Survived", "Pclass", "Sex", "Age", "Fare"]].dropna()
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df.to_csv(f"{PROC_DIR}/titanic_clean.csv", index=False)

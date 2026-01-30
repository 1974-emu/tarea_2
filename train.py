import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Paths
DATA_PATH = "data/heart_failure_clinical_records_dataset.csv"
MODEL_PATH = "model/model.pkl"
SCALER_PATH = "model/scaler.pkl"

# Load data
data = pd.read_csv(DATA_PATH)

# Features and target
X = data.drop("DEATH_EVENT", axis=1)
y = data["DEATH_EVENT"]

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
model = LogisticRegression(
    class_weight="balanced",
    random_state=42,
    max_iter=1000
)
model.fit(X_train_scaled, y_train)

# Evaluation
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# Save artifacts
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("✅ Model and scaler saved in /model")
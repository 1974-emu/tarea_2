from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load artifacts
MODEL_PATH = "model/model.pkl"
SCALER_PATH = "model/scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

app = FastAPI(title="Heart Failure Prediction API")

# Input schema
class PatientData(BaseModel):
    age: float
    anaemia: int
    creatinine_phosphokinase: float
    diabetes: int
    ejection_fraction: float
    high_blood_pressure: int
    platelets: float
    serum_creatinine: float
    serum_sodium: float
    sex: int
    smoking: int
    time: int


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: PatientData):
    # Convert input to array (order MUST match training data)
    features = np.array([[
        data.age,
        data.anaemia,
        data.creatinine_phosphokinase,
        data.diabetes,
        data.ejection_fraction,
        data.high_blood_pressure,
        data.platelets,
        data.serum_creatinine,
        data.serum_sodium,
        data.sex,
        data.smoking,
        data.time
    ]])

    # Scale and predict
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    return {
        "prediction": int(prediction),
        "death_probability": round(float(probability), 4)
    }
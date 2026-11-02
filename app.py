from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib

# Load the saved machine learning model
saved_data = joblib.load("model.pkl")
model = saved_data["model"]
target_names = saved_data["target_names"]

# Create FastAPI application
app = FastAPI(title="ML Prediction API")

# Define input data format
class PredictionInput(BaseModel):
    features: list[float] = Field(
        min_length=4,
        max_length=4
    )

# Home endpoint
@app.get("/")
def home():
    return {"message": "ML API is running successfully!"}

# Prediction endpoint
@app.post("/predict")
def predict(data: PredictionInput):

    # Predict using trained model
    prediction = int(
        model.predict([data.features])[0]
    )

    return {
        "predicted_class": prediction,
        "predicted_flower": target_names[prediction]
    }
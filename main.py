import tensorflow as tf
import numpy as np
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load your ML model and preprocessing tools
model = tf.keras.models.load_model("crop_recommendation_model.h5")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# Initialize FastAPI app
app = FastAPI(title="Crop Recommendation API")

# Enable CORS so that web apps can access this API
origins = [
    "http://localhost:8081",   # React Native Web
    "http://localhost:19006",  # Expo Web
    "*",                        # Allow all origins (for testing, remove * in production)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],            # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],            # Allow all headers
)

# Input schema
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

# Prediction endpoint
@app.post("/predict")
def predict_crop(data: CropInput):
    # Prepare input features
    features = [data.N, data.P, data.K, data.temperature, data.humidity, data.ph, data.rainfall]
    sample = np.array([features])

    # Scale features
    sample_scaled = scaler.transform(sample)

    # Predict probabilities
    pred = model.predict(sample_scaled)[0]

    # Get top 5 recommendations
    top5_idx = pred.argsort()[-5:][::-1]
    results = [{"crop": le.inverse_transform([i])[0], "confidence": float(pred[i])*100} for i in top5_idx]

    # Return JSON response
    return {"recommendations": results}


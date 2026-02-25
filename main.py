from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pycaret.regression import load_model, predict_model
import pandas as pd
from typing import Optional

# initialize app  
app = FastAPI(
    title="Disaster Response Predictor",
    description="Predicts response time & recovery days for disaster events",
    version="1.0.0"
)

# CORS setup for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite default dev port
        "http://localhost:3000",   # just in case
        "https://your-react-app.vercel.app"  # add prod URL later
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model("disaster_response_model")

# input schema
class DisasterInput(BaseModel):
    country: str
    disaster_type: str
    severity_index: float
    casualties: int
    economic_loss_usd: float
    aid_amount_usd: float        
    response_efficiency_score: float 
    recovery_days: int         
    latitude: float
    longitude: float
    month: int    # 1–12
    year: int     # e.g. 2024

# Health check endpoint
@app.get("/")
def root():
    return {"status": "online", "model": "disaster_response_model"}

# Main prediction endpoint 
@app.post("/predict")
def predict(data: DisasterInput):
    try:
        # Convert input to DataFrame (PyCaret expects this)
        df = pd.DataFrame([data.dict()])

        # Run prediction
        result = predict_model(model, data=df)
        prediction = round(float(result["prediction_label"].iloc[0]), 2)

        # Derive severity tier from prediction + input
        if prediction <= 6:
            tier = "CRITICAL"
        elif prediction <= 15:
            tier = "HIGH"
        elif prediction <= 25:
            tier = "MODERATE"
        else:
            tier = "LOW"

        return {
            "success": True,
            "predicted_response_time_hours": prediction,
            "severity_tier": tier,
            "input_received": data.dict()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Batch prediction endpoint 
@app.post("/predict/batch")
def predict_batch(items: list[DisasterInput]):
    try:
        df = pd.DataFrame([i.dict() for i in items])
        result = predict_model(model, data=df)
        predictions = result["prediction_label"].round(2).tolist()
        return {"success": True, "predictions": predictions, "count": len(predictions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
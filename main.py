from fastapi import FastAPI
import pickle
import pandas as pd
from data_model import Water


app = FastAPI(
    title="Water Potability Prediction",
    description="API to predict water potability",
    version="0.1",
)

with open("rf_model.pkl", "rb") as file:
    model = pickle.load(file) 

@app.get("/")
def index():
    return "Welcome to the API"

@app.post("/predict")
def model_predict(water: Water):
    sample = pd.DataFrame({
        "ph": [water.ph],
        "Hardness": [water.Hardness],
        "Solids": [water.Solids],
        "Chloramines": [water.Chloramines],
        "Sulfate": [water.Sulfate],
        "Conductivity": [water.Conductivity],
        "Organic_carbon": [water.Organic_carbon],
        "Trihalomethanes": [water.Trihalomethanes],
        "Turbidity": [water.Turbidity]
    })

    pred = model.predict(sample)

    if pred == 1:
        return "Potable"
    else:
        return "Not Potable"

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

#load model and feutures list
model = joblib.load("/app/models/wine_model.pkl")
feature_names = joblib.load("/app/models/feature_names.pkl")

# define structure of input 
class WineInput(BaseModel):
    values: list[float]


app = FastAPI()


@app.get("/")
def root():
    return {"status": "API is running"}

@app.post("/predict")
def predict(input_data:WineInput):
    if len(input_data.values) != len(feature_names) :
        return {"error": "Wrong number of features"}
    
    X = np.array(input_data.values).reshape(1, -1) # (1, 13)
    prediction = model.predict(X)
    return {"prediction": int(prediction)}


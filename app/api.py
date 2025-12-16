from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

# Load model
with open("models/random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load feature order
with open("models/feature_columns.txt") as f:
    FEATURE_COLUMNS = [line.strip() for line in f]

class StudentInput(BaseModel):
    studytime: int
    absences: int
    failures: int

@app.post("/predict")
def predict(input_data: StudentInput):
    data = input_data.dict()

    # Create empty feature row
    row = {col: 0 for col in FEATURE_COLUMNS}

    # Fill known inputs
    for key in data:
        if key in row:
            row[key] = data[key]

    df = pd.DataFrame([row])
    prediction = model.predict(df)[0]

    return {"at_risk": int(prediction)}

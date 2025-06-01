# app/api/predict.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model

router = APIRouter()

# Model path
model_path = os.path.join("app", "models", "cbc_disease_model.h5")
model = load_model(model_path)

# ✅ Expected feature order
EXPECTED_COLUMNS = [
    "WBC", "Lymphocytes", "Neutrophils", "RBC", "Hemoglobin", "Hematocrit", "Platelets",
    "NLR", "MCV", "MCH", "MCHC", "RDW", "MPV", "PDW", "PCT", "Basophils", "Eosinophils",
    "Monocytes", "LymphsAbs", "NeutroAbs", "BasoAbs", "EosAbs"
]

class CBCInput(BaseModel):
    WBC: float
    Lymphocytes: float
    Neutrophils: float
    RBC: float
    Hemoglobin: float
    Hematocrit: float
    Platelets: float
    NLR: float
    MCV: float
    MCH: float
    MCHC: float
    RDW: float
    MPV: float
    PDW: float
    PCT: float
    Basophils: float
    Eosinophils: float
    Monocytes: float
    LymphsAbs: float
    NeutroAbs: float
    BasoAbs: float
    EosAbs: float

@router.post("/predict")
def predict_cbc(input: CBCInput):
    try:
        # Print input for debugging
        print("\n✅ Received Input:\n", input)

        # Ensure correct feature ordering
        input_data = input.dict()
        print("\n📌 Parsed dict:", input_data)

        df_input = pd.DataFrame([[input_data[col] for col in EXPECTED_COLUMNS]], columns=EXPECTED_COLUMNS)
        print("\n🧪 Final DataFrame for prediction:\n", df_input)

        # Prediction
        probs = model.predict(df_input)[0] * 100
        print("✅ Model prediction:", probs)

        # Risk calculations
        anemia_risk = max(0, min(100, (15 - input.Hemoglobin) * 10))
        infection_risk = max(0, min(100, (input.WBC - 11) * 10))
        cardiovascular_risk = max(0, min(100, input.NLR * 10))
        leukemia_risk = max(0, min(100, (input.WBC - 18) * 20))

        return {
    "anemia_risk": float(round(anemia_risk, 2)),
    "infection_risk": float(round(infection_risk, 2)),
    "cardiovascular_risk": float(round(cardiovascular_risk, 2)),
    "leukemia_risk": float(round(leukemia_risk, 2)),
    "probabilities": [float(round(p, 2)) for p in probs]
}


    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"❌ Prediction failed: {str(e)}")

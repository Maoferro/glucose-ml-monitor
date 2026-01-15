"""
Sistema de Predicción de Glucosa con Machine Learning
API FastAPI con FHIR Compliance R4
Autenticación Bearer Token
7 Modelos ML activos + Base de datos de 61 pacientes
"""

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import datetime
import uvicorn

# ==================== CONFIGURACIÓN ====================
MODELS_DIR = Path(".")
DATA_DIR = Path("data")
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "test-token-2026")

# ==================== MODELOS ML ====================
MODELS = {}
MODEL_FILES = {
    "xgboost": "models/MEJOR_MODELO_XGBoost.joblib",
    "random_forest": "models/Random_Forest.joblib",
    "lightgbm": "models/LightGBM.joblib",
    "gradient_boosting": "models/Gradient_Boosting.joblib",
    "ridge": "models/Ridge.joblib",
    "lasso": "models/Lasso.joblib",
    "elasticnet": "models/ElasticNet.joblib"
}

# ==================== BASE DE DATOS ====================
PATIENTS_DB = None
PREDICTIONS_HISTORY = []

# ==================== INICIALIZACIÓN ====================
app = FastAPI(
    title="Glucose Prediction API with FHIR",
    description="Sistema de predicción de glucosa con 7 modelos ML y compatibilidad FHIR R4",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== MODELOS DE DATOS ====================
class PredictionInput(BaseModel):
    edad: float
    sexo: str
    peso: float
    talla: float
    imc: float
    perimetro_cintura: float
    spo2: float
    frecuencia_cardiaca: float
    actividad_fisica: str
    consumo_frutas: str
    tiene_hipertension: str
    tiene_diabetes: str
    puntaje_findrisc: float

class PatientFHIR(BaseModel):
    resourceType: str = "Patient"
    id: str
    identifier: List[Dict]
    name: List[Dict]
    gender: str
    birthDate: Optional[str]

# ==================== AUTENTICACIÓN ====================
def verify_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token inválido")
    token = authorization.replace("Bearer ", "")
    if token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Token no autorizado")
    return token

# ==================== CARGA DE MODELOS ====================
@app.on_event("startup")
async def load_models():
    global MODELS, PATIENTS_DB
    
    print("\n" + "="*60)
    print("🚀 INICIANDO API DE PREDICCIÓN DE GLUCOSA")
    print("="*60 + "\n")
    
    # Cargar modelos ML
    print("📦 Cargando modelos de Machine Learning...\n")
    for name, filepath in MODEL_FILES.items():
        try:
            if os.path.exists(filepath):
                MODELS[name] = joblib.load(filepath)
                print(f"✅ Modelo {name} cargado desde {filepath}")
            else:
                print(f"❌ Archivo no encontrado: {filepath}")
        except Exception as e:
            print(f"❌ Error al cargar {name}: {str(e)}")
    
    print(f"\n✅ Total de modelos cargados: {len(MODELS)}/7\n")
    
    # Cargar base de datos de pacientes
    csv_path = "data/base_unificada.csv"
    try:
        if os.path.exists(csv_path):
            PATIENTS_DB = pd.read_csv(csv_path)
            print(f"✅ Base de datos cargada: {len(PATIENTS_DB)} pacientes desde {csv_path}\n")
        else:
            print(f"⚠️  Archivo CSV no encontrado: {csv_path}")
            PATIENTS_DB = pd.DataFrame()
    except Exception as e:
        print(f"❌ Error al cargar CSV: {str(e)}")
        PATIENTS_DB = pd.DataFrame()
    
    print("="*60)
    print(f"🎉 API lista con {len(MODELS)} modelos ML activos")
    print("="*60 + "\n")

# ==================== ENDPOINTS ====================

@app.get("/")
def root():
    """Endpoint raíz - Información de la API"""
    return {
        "message": "Glucose ML Prediction API with FHIR Compliance",
        "version": "2.0.0",
        "models_active": len(MODELS),
        "fhir_version": "R4",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "fhir_patient": "/api/v1/patient/{patient_id}",
            "fhir_observations": "/api/v1/patient/{patient_id}/observations",
            "fhir_predictions": "/api/v1/predictions"
        }
    }

@app.get("/health")
def health_check():
    """Health check para Render"""
    return {
        "status": "healthy",
        "models_loaded": len(MODELS),
        "patients_loaded": len(PATIENTS_DB) if PATIENTS_DB is not None else 0,
        "timestamp": datetime.now().isoformat()
    }

# ==================== ENDPOINT ORIGINAL /predict ====================
@app.post("/predict")
def predict_glucose(data: PredictionInput):
    """Endpoint original de predicción (compatibilidad)"""
    if len(MODELS) == 0:
        raise HTTPException(status_code=503, detail="Modelos no cargados")
    
    # Preparar datos
    input_data = pd.DataFrame([{
        "Edad": data.edad,
        "Sexo_M": 1 if data.sexo.upper() == "M" else 0,
        "Peso": data.peso,
        "Talla": data.talla,
        "IMC": data.imc,
        "Perimetro_Cintura": data.perimetro_cintura,
        "SpO2": data.spo2,
        "Frecuencia_Cardiaca": data.frecuencia_cardiaca,
        "Actividad_Fisica_Si": 1 if data.actividad_fisica.lower() == "si" else 0,
        "Consumo_Frutas_Si": 1 if data.consumo_frutas.lower() == "si" else 0,
        "Tiene_Hipertension_Si": 1 if data.tiene_hipertension.lower() == "si" else 0,
        "Tiene_Diabetes_Si": 1 if data.tiene_diabetes.lower() == "si" else 0,
        "Puntaje_FINDRISC": data.puntaje_findrisc
    }])
    
    # Predecir con todos los modelos
    predictions = {}
    for name, model in MODELS.items():
        try:
            pred = model.predict(input_data)[0]
            predictions[name] = float(pred)
        except Exception as e:
            predictions[name] = None
    
    # Calcular promedio
    valid_preds = [p for p in predictions.values() if p is not None]
    avg_prediction = np.mean(valid_preds) if valid_preds else 0.0
    
    # Clasificar
    if avg_prediction < 100:
        categoria = "Normal"
        riesgo = "Bajo"
    elif avg_prediction < 126:
        categoria = "Prediabetes"
        riesgo = "Moderado"
    else:
        categoria = "Diabetes"
        riesgo = "Alto"
    
    return {
        "prediccion_promedio_mg_dl": round(avg_prediction, 2),
        "categoria": categoria,
        "nivel_riesgo": riesgo,
        "predicciones_individuales": predictions,
        "modelos_activos": len(valid_preds)
    }

# ==================== ENDPOINTS FHIR ====================

@app.get("/api/v1/patient/{patient_id}")
def get_patient_fhir(patient_id: str, token: str = Depends(verify_token)):
    """Obtener paciente en formato FHIR R4"""
    if PATIENTS_DB is None or len(PATIENTS_DB) == 0:
        raise HTTPException(status_code=503, detail="Base de datos no disponible")
    
    patient = PATIENTS_DB[PATIENTS_DB['ID_Unico'] == patient_id]
    
    if patient.empty:
        raise HTTPException(status_code=404, detail=f"Paciente {patient_id} no encontrado")
    
    patient_data = patient.iloc[0]
    
    return {
        "resourceType": "Patient",
        "id": patient_id,
        "identifier": [
            {
                "system": "urn:oid:2.16.840.1.113883.4.642.1.1",
                "value": str(patient_data.get('Identificacion', 'N/A'))
            }
        ],
        "name": [
            {
                "use": "official",
                "text": str(patient_data.get('Nombre_Completo', 'N/A'))
            }
        ],
        "gender": "male" if patient_data.get('Sexo', 'M') == 'M' else "female",
        "birthDate": str(2024 - int(patient_data.get('Edad', 0))) + "-01-01",
        "extension": [
            {
                "url": "http://glucose-ml-api.org/fhir/StructureDefinition/diabetes-risk",
                "valueString": str(patient_data.get('Clasificacion_Riesgo', 'N/A'))
            },
            {
                "url": "http://glucose-ml-api.org/fhir/StructureDefinition/findrisc-score",
                "valueDecimal": float(patient_data.get('Puntaje_FINDRISC', 0))
            }
        ]
    }

@app.get("/api/v1/patient/{patient_id}/observations")
def get_patient_observations(patient_id: str, token: str = Depends(verify_token)):
    """Obtener observaciones de un paciente en formato FHIR R4"""
    if PATIENTS_DB is None or len(PATIENTS_DB) == 0:
        raise HTTPException(status_code=503, detail="Base de datos no disponible")
    
    patient = PATIENTS_DB[PATIENTS_DB['ID_Unico'] == patient_id]
    
    if patient.empty:
        raise HTTPException(status_code=404, detail=f"Paciente {patient_id} no encontrado")
    
    patient_data = patient.iloc[0]
    
    observations = [
        {
            "resourceType": "Observation",
            "id": f"{patient_id}-glucose",
            "status": "final",
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "2339-0",
                        "display": "Glucose [Mass/volume] in Blood"
                    }
                ]
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "valueQuantity": {
                "value": float(patient_data.get('Glucosa_Estimada_mgdL', 0)),
                "unit": "mg/dL",
                "system": "http://unitsofmeasure.org",
                "code": "mg/dL"
            }
        },
        {
            "resourceType": "Observation",
            "id": f"{patient_id}-bmi",
            "status": "final",
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "39156-5",
                        "display": "Body mass index (BMI) [Ratio]"
                    }
                ]
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "valueQuantity": {
                "value": float(patient_data.get('IMC', 0)),
                "unit": "kg/m2",
                "system": "http://unitsofmeasure.org",
                "code": "kg/m2"
            }
        }
    ]
    
    return {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": len(observations),
        "entry": [{"resource": obs} for obs in observations]
    }

@app.post("/api/v1/predictions")
def create_prediction_fhir(data: PredictionInput, token: str = Depends(verify_token)):
    """Crear predicción en formato FHIR R4"""
    # Reutilizar lógica de /predict
    prediction_result = predict_glucose(data)
    
    observation = {
        "resourceType": "Observation",
        "id": f"prediction-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "laboratory",
                        "display": "Laboratory"
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "2339-0",
                    "display": "Glucose [Mass/volume] in Blood"
                }
            ],
            "text": "Predicción de Glucosa (ML)"
        },
        "effectiveDateTime": datetime.now().isoformat(),
        "valueQuantity": {
            "value": prediction_result["prediccion_promedio_mg_dl"],
            "unit": "mg/dL",
            "system": "http://unitsofmeasure.org",
            "code": "mg/dL"
        },
        "interpretation": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                        "code": "H" if prediction_result["nivel_riesgo"] == "Alto" else "N",
                        "display": prediction_result["categoria"]
                    }
                ]
            }
        ],
        "note": [
            {
                "text": f"Nivel de riesgo: {prediction_result['nivel_riesgo']}. Modelos activos: {prediction_result['modelos_activos']}"
            }
        ]
    }
    
    # Guardar en historial
    PREDICTIONS_HISTORY.append(observation)
    
    return observation

@app.get("/api/v1/predictions/{patient_id}")
def get_predictions_history(patient_id: str, token: str = Depends(verify_token)):
    """Obtener historial de predicciones de un paciente"""
    patient_predictions = [p for p in PREDICTIONS_HISTORY if patient_id in p.get("id", "")]
    
    return {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": len(patient_predictions),
        "entry": [{"resource": pred} for pred in patient_predictions]
    }

# ==================== EJECUCIÓN ====================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

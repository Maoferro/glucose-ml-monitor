from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import pandas as pd
import numpy as np
from pathlib import Path
import os
import uvicorn
from datetime import datetime
import joblib
import pickle

# ============================================
# CONFIGURACIÓN
# ============================================
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "test-token-2026")
MODELS_DIR = Path("models")
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Variables globales
MODELS = {}
PREPROCESSING = None
PATIENTS_DB = None

# ============================================
# MODELO PYDANTIC
# ============================================
class PredictionRequest(BaseModel):
    edad: int
    sexo: str
    peso: float
    talla: float
    perimetro_cintura: float
    spo2: Optional[float] = 98.0
    frecuencia_cardiaca: Optional[float] = 75.0
    actividad_fisica: str
    consumo_frutas: str
    tiene_hipertension: str
    tiene_diabetes: str
    puntaje_findrisc: int

# ============================================
# FUNCIÓN PARA CARGAR PREPROCESADORES
# ============================================
def load_preprocessing():
    global PREPROCESSING
    preprocessing_path = MODELS_DIR / "preprocessing_objects.pkl"
    try:
        with open(preprocessing_path, 'rb') as f:
            PREPROCESSING = pickle.load(f)
        print(f"✅ Preprocesadores cargados desde {preprocessing_path}")
        print(f"   • Scaler: {type(PREPROCESSING['scaler']).__name__}")
        print(f"   • Label Encoders: {len(PREPROCESSING['label_encoders'])} columnas")
        print(f"   • Features: {len(PREPROCESSING['feature_names'])}")
    except Exception as e:
        print(f"❌ Error cargando preprocesadores: {e}")
        PREPROCESSING = None

# ============================================
# FUNCIÓN PARA CARGAR MODELOS ML
# ============================================
def load_models():
    global MODELS
    model_files = {
        "xgboost": "XGBoost.joblib",
        "random_forest": "Random_Forest.joblib",
        "lightgbm": "LightGBM.joblib",
        "gradient_boosting": "Gradient_Boosting.joblib",
        "ridge": "Ridge.joblib",
        "lasso": "Lasso.joblib",
        "elasticnet": "ElasticNet.joblib"
    }
    
    for name, filename in model_files.items():
        model_path = MODELS_DIR / filename
        try:
            model = joblib.load(model_path)
            MODELS[name] = model
            print(f"✅ Modelo {name} cargado desde {model_path}")
        except Exception as e:
            print(f"❌ Error cargando modelo {name}: {e}")
    
    print(f"✅ Total de modelos cargados: {len(MODELS)}")

# ============================================
# FUNCIÓN PARA CARGAR BASE DE DATOS DE PACIENTES
# ============================================
def load_patients_db():
    global PATIENTS_DB
    csv_path = DATA_DIR / "base_unificada.csv"
    try:
        # ✅ CORRECCIÓN 1: separador correcto
        PATIENTS_DB = pd.read_csv(csv_path, sep=";")

        # Limpieza mínima (la puedes dejar)
        PATIENTS_DB.columns = PATIENTS_DB.columns.str.strip()
        print("Columnas CSV:", PATIENTS_DB.columns.tolist())

        # ✅ CORRECCIÓN 2: lista bien escrita
        PATIENTS_DB = PATIENTS_DB.dropna(
            subset=['ID_Unico', 'Glucosa_Estimada_mgdL']
        )

        print(f"✅ Base de datos cargada: {len(PATIENTS_DB)} pacientes desde {csv_path}")

    except Exception as e:
        print(f"⚠️ Error cargando pacientes: {e}")
        PATIENTS_DB = pd.DataFrame()

# ============================================
# FUNCIÓN DE PREPROCESAMIENTO
# ============================================
def preprocess_input(data: PredictionRequest) -> np.ndarray:
    if PREPROCESSING is None:
        raise HTTPException(status_code=500, detail="Preprocessing objects not loaded")
    
    input_dict = {
        'Edad': data.edad,
        'Sexo': data.sexo.lower(),
        'Peso': data.peso,
        'Talla': data.talla,
        'IMC': data.peso / (data.talla ** 2),
        'Perimetro_Cintura': data.perimetro_cintura,
        'SpO2': data.spo2,
        'Frecuencia_Cardiaca': data.frecuencia_cardiaca,
        'Actividad_Fisica': data.actividad_fisica.lower(),
        'Consumo_Frutas': data.consumo_frutas.lower(),
        'Tiene_Hipertension': data.tiene_hipertension.lower(),
        'Tiene_Diabetes': data.tiene_diabetes.lower(),
        'Puntaje_FINDRISC': data.puntaje_findrisc
    }
    
    df = pd.DataFrame([input_dict])
    
    for col, le in PREPROCESSING['label_encoders'].items():
        if col in df.columns:
            try:
                df[col] = le.transform(df[col].astype(str))
            except:
                df[col] = 0
    
    X_scaled = PREPROCESSING['scaler'].transform(df)
    return X_scaled

# ============================================
# FASTAPI APP
# ============================================
app = FastAPI(
    title="Glucose Prediction API - FHIR Compliance",
    version="2.0.0",
    description="API de predicción de glucosa con 7 modelos ML y compatibilidad FHIR R4"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# DEPENDENCY - AUTENTICACIÓN
# ============================================
def verify_token(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.replace("Bearer ", "")
    if token != AUTH_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")
    return token

# ============================================
# STARTUP EVENT
# ============================================
@app.on_event("startup")
async def startup_event():
    print("="*80)
    print("INFO: Iniciando API de Predicción de Glucosa con FHIR Compliance...")
    print("="*80)
    load_preprocessing()
    load_models()
    load_patients_db()
    print("="*80)
    print(f"✅ API lista - Modelos cargados: {len(MODELS)}")
    print(f"✅ Pacientes cargados: {len(PATIENTS_DB) if PATIENTS_DB is not None else 0}")
    print(f"✅ Preprocesamiento: {'Activo' if PREPROCESSING is not None else 'Error'}")
    print("="*80)

# ============================================
# ENDPOINT ROOT
# ============================================
@app.get("/")
async def root():
    return {
        "message": "Glucose ML Prediction API with FHIR Compliance",
        "version": "2.0.0",
        "models_loaded": len(MODELS),
        "models_active": list(MODELS.keys()),
        "patients_loaded": len(PATIENTS_DB) if PATIENTS_DB is not None else 0,
        "fhir_version": "R4",
        "preprocessing_loaded": PREPROCESSING is not None,
        "endpoints": [
            "GET /",
            "GET /health",
            "GET /docs",
            "POST /predict",
            "GET /api/v1/patient/{patient_id}",
            "GET /api/v1/patient/{patient_id}/observations",
            "POST /api/v1/predictions",
            "GET /api/v1/predictions/{patient_id}"
        ]
    }

# ============================================
# ENDPOINT HEALTH
# ============================================
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": len(MODELS),
        "preprocessing_loaded": PREPROCESSING is not None,
        "patients_loaded": len(PATIENTS_DB) if PATIENTS_DB is not None else 0,
        "timestamp": datetime.now().isoformat()
    }

# ============================================
# ENDPOINT PREDICT (ACTUALIZADO CON PREPROCESAMIENTO)
# ============================================
@app.post("/predict")
async def predict(data: PredictionRequest):
    if not MODELS:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Preprocesar input (LabelEncoder + StandardScaler)
        X = preprocess_input(data)
        
        # Predecir con todos los modelos
        predictions = {}
        for name, model in MODELS.items():
            pred = model.predict(X)[0]
            predictions[name] = float(pred)
        
        # Ensemble (promedio ponderado basado en MAE de validación)
        weights = {
            'xgboost': 0.20,
            'random_forest': 0.18,
            'lightgbm': 0.18,
            'gradient_boosting': 0.16,
            'ridge': 0.10,
            'lasso': 0.10,
            'elasticnet': 0.08
        }
        
        ensemble_pred = sum(predictions[name] * weights.get(name, 1/len(predictions)) 
                           for name in predictions) / sum(weights.values())
        
        # Categoría y riesgo
        if ensemble_pred < 100:
            categoria = "Normal"
            nivel_riesgo = "Bajo"
        elif ensemble_pred < 126:
            categoria = "Prediabetes"
            nivel_riesgo = "Moderado"
        else:
            categoria = "Diabetes"
            nivel_riesgo = "Alto"
        
        # MAE estimado y rango de confianza (95%)
        mae_estimado = 8.2
        rango_inferior = ensemble_pred - (1.96 * mae_estimado)
        rango_superior = ensemble_pred + (1.96 * mae_estimado)
        
        return {
            "glucosa_predicha": round(ensemble_pred, 1),
            "categoria": categoria,
            "nivel_riesgo": nivel_riesgo,
            "confianza": "Alta",
            "mae_estimado": mae_estimado,
            "rango_confianza": f"{round(rango_inferior, 1)} - {round(rango_superior, 1)} mg/dL",
            "predicciones_individuales": {k: round(v, 1) for k, v in predictions.items()},
            "metodo": "Ensemble de 7 modelos ML re-entrenados (XGBoost, RandomForest, LightGBM, GradientBoosting, Ridge, Lasso, ElasticNet)",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ============================================
# ENDPOINTS FHIR
# ============================================

@app.get("/api/v1/patient/{patient_id}")
async def get_patient_fhir(patient_id: str, token: str = Depends(verify_token)):
    """Obtener información del paciente en formato FHIR R4"""
    if PATIENTS_DB is None or PATIENTS_DB.empty:
        raise HTTPException(status_code=404, detail="Patients database not loaded")
    
    patient_row = PATIENTS_DB[PATIENTS_DB['ID_Unico'] == patient_id]
    if patient_row.empty:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    
    p = patient_row.iloc[0]
    
    patient_fhir = {
        "resourceType": "Patient",
        "id": patient_id,
        "identifier": [{
            "system": "http://hospital.example.org/patients",
            "value": str(p.get('Identificacion', ''))
        }],
        "name": [{
            "text": str(p.get('Nombre_Completo', ''))
        }],
        "gender": "male" if str(p.get('Sexo', '')).lower() == "masculino" else "female",
        "birthDate": str(2026 - int(p.get('Edad', 0)))
    }
    
    return patient_fhir

@app.get("/api/v1/patient/{patient_id}/observations")
async def get_patient_observations_fhir(patient_id: str, token: str = Depends(verify_token)):
    """Obtener observaciones del paciente en formato FHIR R4"""
    if PATIENTS_DB is None or PATIENTS_DB.empty:
        raise HTTPException(status_code=404, detail="Patients database not loaded")
    
    patient_row = PATIENTS_DB[PATIENTS_DB['ID_Unico'] == patient_id]
    if patient_row.empty:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    
    p = patient_row.iloc[0]
    
    observations = {
        "resourceType": "Bundle",
        "type": "searchset",
        "entry": [
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": f"{patient_id}-glucose",
                    "status": "final",
                    "code": {
                        "coding": [{
                            "system": "http://loinc.org",
                            "code": "15074-8",
                            "display": "Glucose [Moles/volume] in Blood"
                        }]
                    },
                    "subject": {"reference": f"Patient/{patient_id}"},
                    "valueQuantity": {
                        "value": float(p.get('Glucosa_Estimada_mgdL', 0)),
                        "unit": "mg/dL",
                        "system": "http://unitsofmeasure.org",
                        "code": "mg/dL"
                    }
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": f"{patient_id}-bmi",
                    "status": "final",
                    "code": {
                        "coding": [{
                            "system": "http://loinc.org",
                            "code": "39156-5",
                            "display": "Body mass index (BMI) [Ratio]"
                        }]
                    },
                    "subject": {"reference": f"Patient/{patient_id}"},
                    "valueQuantity": {
                        "value": float(p.get('IMC', 0)),
                        "unit": "kg/m2",
                        "system": "http://unitsofmeasure.org",
                        "code": "kg/m2"
                    }
                }
            }
        ]
    }
    
    return observations

@app.post("/api/v1/predictions")
async def create_prediction_fhir(data: PredictionRequest, token: str = Depends(verify_token)):
    """Crear predicción en formato FHIR R4"""
    result = await predict(data)
    
    observation_fhir = {
        "resourceType": "Observation",
        "id": f"prediction-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "status": "final",
        "code": {
            "coding": [{
                "system": "http://loinc.org",
                "code": "15074-8",
                "display": "Glucose [Moles/volume] in Blood - Predicted"
            }]
        },
        "valueQuantity": {
            "value": result['glucosa_predicha'],
            "unit": "mg/dL",
            "system": "http://unitsofmeasure.org",
            "code": "mg/dL"
        },
        "interpretation": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                "code": "N" if result['categoria'] == "Normal" else ("L" if result['categoria'] == "Prediabetes" else "H"),
                "display": result['categoria']
            }]
        }],
        "note": [{
            "text": f"Predicción ensemble de 7 modelos ML. Nivel de riesgo: {result['nivel_riesgo']}. Rango: {result['rango_confianza']}"
        }]
    }
    
    return observation_fhir

@app.get("/api/v1/predictions/{patient_id}")
async def get_predictions_fhir(patient_id: str, token: str = Depends(verify_token)):
    """Obtener predicciones históricas del paciente en formato FHIR R4"""
    return {
        "resourceType": "Bundle",
        "type": "searchset",
        "entry": []
    }

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

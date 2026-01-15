"""
Sistema de Predicción de Glucosa con Machine Learning + FHIR Compliance
FastAPI Backend con 7 modelos ML y endpoints FHIR R4

Autor: Sistema de IA
Fecha: Enero 2026
Versión: 2.0 (FHIR-compliant)
"""

from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import os
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="Glucose ML Prediction API with FHIR",
    description="API para predicción de glucosa con 7 modelos ML y compliance FHIR R4",
    version="2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== CONFIGURACIÓN ====================

MODELS_DIR = Path("models")
DATA_DIR = Path("data")
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "test-token-2026")

# Cargar modelos ML (7 algoritmos)
MODELS = {}
MODEL_FILES = {
    "xgboost": "MEJOR_MODELO_XGBoost.joblib",
    "random_forest": "Random_Forest.joblib",
    "lightgbm": "LightGBM.joblib",
    "gradient_boosting": "Gradient_Boosting.joblib",
    "ridge": "Ridge.joblib",
    "lasso": "Lasso.joblib",
    "elasticnet": "ElasticNet.joblib"
}

# Cargar base de datos de pacientes (61 registros)
PATIENTS_DB = None

# ==================== MODELOS PYDANTIC ====================

class PredictionRequest(BaseModel):
    edad: int = Field(..., ge=0, le=120, description="Edad del paciente")
    sexo: str = Field(..., description="Sexo (M/F)")
    peso: float = Field(..., gt=0, description="Peso en kg")
    talla: float = Field(..., gt=0, description="Talla en metros")
    perimetro_cintura: float = Field(..., gt=0, description="Perímetro de cintura en cm")
    spo2: float = Field(..., ge=0, le=100, description="SpO2 en %")
    frecuencia_cardiaca: int = Field(..., gt=0, description="FC en bpm")
    actividad_fisica: bool = Field(..., description="Realiza actividad física")
    consumo_frutas: bool = Field(..., description="Consume frutas regularmente")
    tiene_hipertension: bool = Field(..., description="Diagnosticado con HTA")
    tiene_diabetes: bool = Field(..., description="Diagnosticado con DM")
    puntaje_findrisc: int = Field(..., ge=0, le=26, description="Puntaje FINDRISC")

class PredictionResponse(BaseModel):
    glucosa_predicha: float
    categoria: str
    nivel_riesgo: str
    confianza: str
    mae: float
    rango_confianza: str
    modelo_usado: str
    recomendacion: str

# ==================== MODELOS FHIR R4 ====================

class FHIRPatient(BaseModel):
    resourceType: str = "Patient"
    id: str
    identifier: List[Dict[str, Any]]
    name: List[Dict[str, Any]]
    gender: str
    birthDate: Optional[str] = None

class FHIRObservation(BaseModel):
    resourceType: str = "Observation"
    id: str
    status: str = "final"
    code: Dict[str, Any]
    subject: Dict[str, Any]
    effectiveDateTime: str
    valueQuantity: Dict[str, Any]

# ==================== AUTENTICACIÓN ====================

async def verify_token(authorization: Optional[str] = Header(None)):
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header"
        )
    
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication scheme"
            )
        
        if token != AUTH_TOKEN:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format"
        )

# ==================== FUNCIONES DE CARGA ====================

def load_ml_models():
    global MODELS
    
    for name, filename in MODEL_FILES.items():
        model_path = MODELS_DIR / filename
        try:
            MODELS[name] = joblib.load(model_path)
            logger.info(f"✅ Modelo {name} cargado desde {model_path}")
        except FileNotFoundError:
            logger.warning(f"⚠️ Modelo {name} no encontrado en {model_path}")
        except Exception as e:
            logger.error(f"❌ Error cargando {name}: {str(e)}")

def load_patients_database():
    global PATIENTS_DB
    
    csv_path = DATA_DIR / "base_unificada.csv"
    try:
        PATIENTS_DB = pd.read_csv(csv_path)
        logger.info(f"✅ Base de datos cargada: {len(PATIENTS_DB)} pacientes")
    except FileNotFoundError:
        logger.warning(f"⚠️ Base de datos no encontrada en {csv_path}")
        PATIENTS_DB = pd.DataFrame(columns=[
            "ID_Unico", "Tipo_Identificacion", "Identificacion", "Nombre_Completo",
            "Edad", "Sexo", "Peso", "Talla", "IMC", "Perimetro_Cintura",
            "Actividad_Fisica", "Consumo_Frutas", "Tiene_Hipertension", "Tiene_Diabetes",
            "Puntaje_FINDRISC", "Clasificacion_Riesgo", "Glucosa_Estimada_mgdL"
        ])

# ==================== FUNCIONES DE PREDICCIÓN ====================

def predict_glucose(data: PredictionRequest) -> Dict[str, Any]:
    imc = data.peso / (data.talla ** 2)
    
    features = np.array([[
        data.edad,
        1 if data.sexo.upper() == 'M' else 0,
        data.peso,
        data.talla,
        imc,
        data.perimetro_cintura,
        data.spo2,
        data.frecuencia_cardiaca,
        1 if data.actividad_fisica else 0,
        1 if data.consumo_frutas else 0,
        1 if data.tiene_hipertension else 0,
        1 if data.tiene_diabetes else 0,
        data.puntaje_findrisc
    ]])
    
    predictions = []
    models_used = []
    
    for name, model in MODELS.items():
        try:
            pred = model.predict(features)[0]
            predictions.append(pred)
            models_used.append(name)
        except Exception as e:
            logger.error(f"Error en modelo {name}: {str(e)}")
    
    if not predictions:
        raise HTTPException(status_code=500, detail="No hay modelos disponibles")
    
    glucosa_predicha = float(np.mean(predictions))
    std_dev = float(np.std(predictions))
    
    if glucosa_predicha < 100:
        categoria = "Normal"
        nivel_riesgo = "Bajo"
    elif glucosa_predicha < 126:
        categoria = "Prediabetes"
        nivel_riesgo = "Moderado"
    else:
        categoria = "Diabetes"
        nivel_riesgo = "Alto"
    
    if std_dev < 5:
        confianza = "Alta"
    elif std_dev < 10:
        confianza = "Media"
    else:
        confianza = "Baja"
    
    mae = 8.2
    ic_lower = glucosa_predicha - 1.96 * std_dev
    ic_upper = glucosa_predicha + 1.96 * std_dev
    rango_confianza = f"{ic_lower:.1f} - {ic_upper:.1f} mg/dL"
    
    recomendacion = generar_recomendacion(categoria, data.puntaje_findrisc)
    
    return {
        "glucosa_predicha": round(glucosa_predicha, 1),
        "categoria": categoria,
        "nivel_riesgo": nivel_riesgo,
        "confianza": confianza,
        "mae": mae,
        "rango_confianza": rango_confianza,
        "modelo_usado": f"Ensemble de {len(models_used)} modelos",
        "modelos": models_used,
        "recomendacion": recomendacion
    }

def generar_recomendacion(categoria: str, findrisc: int) -> str:
    base = f"Categoría de glucosa: {categoria}. "
    
    if categoria == "Normal":
        return base + "Mantener hábitos saludables. Controles anuales recomendados."
    elif categoria == "Prediabetes":
        return base + "Implementar cambios en estilo de vida. Consultar con médico. Controles cada 6 meses."
    else:
        return base + "Consultar urgentemente con endocrinólogo. Iniciar tratamiento médico."

# ==================== ENDPOINTS LEGACY ====================

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(MODELS),
        "patients_in_db": len(PATIENTS_DB) if PATIENTS_DB is not None else 0
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_legacy(data: PredictionRequest):
    result = predict_glucose(data)
    return PredictionResponse(**result)

# ==================== ENDPOINTS FHIR R4 ====================

@app.get("/api/v1/patient/{patient_id}", dependencies=[Depends(verify_token)])
async def get_patient_fhir(patient_id: str):
    if PATIENTS_DB is None or PATIENTS_DB.empty:
        raise HTTPException(status_code=404, detail="Patient database not loaded")
    
    patient = PATIENTS_DB[PATIENTS_DB["ID_Unico"] == patient_id]
    
    if patient.empty:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    
    patient = patient.iloc[0]
    
    fhir_patient = {
        "resourceType": "Patient",
        "id": patient_id,
        "identifier": [{
            "system": "https://example.org/patient-id",
            "value": str(patient.get("Identificacion", ""))
        }],
        "name": [{
            "use": "official",
            "text": str(patient.get("Nombre_Completo", ""))
        }],
        "gender": "male" if patient.get("Sexo", "").upper() == "M" else "female",
        "birthDate": str(datetime.now().year - int(patient.get("Edad", 0)))[:4] if pd.notna(patient.get("Edad")) else None
    }
    
    return fhir_patient

@app.get("/api/v1/patient/{patient_id}/observations", dependencies=[Depends(verify_token)])
async def get_patient_observations_fhir(patient_id: str):
    if PATIENTS_DB is None or PATIENTS_DB.empty:
        raise HTTPException(status_code=404, detail="Patient database not loaded")
    
    patient = PATIENTS_DB[PATIENTS_DB["ID_Unico"] == patient_id]
    
    if patient.empty:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    
    patient = patient.iloc[0]
    timestamp = datetime.now().isoformat()
    
    observations = []
    
    if pd.notna(patient.get("Glucosa_Estimada_mgdL")):
        observations.append({
            "resourceType": "Observation",
            "id": f"{patient_id}-glucose-001",
            "status": "final",
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": "15074-8",
                    "display": "Glucose [Mass/volume] in Blood"
                }]
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "effectiveDateTime": timestamp,
            "valueQuantity": {
                "value": float(patient["Glucosa_Estimada_mgdL"]),
                "unit": "mg/dL",
                "system": "http://unitsofmeasure.org",
                "code": "mg/dL"
            }
        })
    
    if pd.notna(patient.get("IMC")):
        observations.append({
            "resourceType": "Observation",
            "id": f"{patient_id}-bmi-001",
            "status": "final",
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": "39156-5",
                    "display": "Body mass index (BMI)"
                }]
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "effectiveDateTime": timestamp,
            "valueQuantity": {
                "value": float(patient["IMC"]),
                "unit": "kg/m2",
                "system": "http://unitsofmeasure.org",
                "code": "kg/m2"
            }
        })
    
    if pd.notna(patient.get("Puntaje_FINDRISC")):
        observations.append({
            "resourceType": "Observation",
            "id": f"{patient_id}-findrisc-001",
            "status": "final",
            "code": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": "225338004",
                    "display": "Risk assessment"
                }],
                "text": "FINDRISC Score"
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "effectiveDateTime": timestamp,
            "valueQuantity": {
                "value": int(patient["Puntaje_FINDRISC"]),
                "unit": "score",
                "system": "http://unitsofmeasure.org",
                "code": "{score}"
            }
        })
    
    return {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": len(observations),
        "entry": [{"resource": obs} for obs in observations]
    }

@app.post("/api/v1/predictions", dependencies=[Depends(verify_token)])
async def create_prediction_fhir(data: PredictionRequest):
    result = predict_glucose(data)
    
    timestamp = datetime.now().isoformat()
    prediction_id = f"prediction-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    fhir_observation = {
        "resourceType": "Observation",
        "id": prediction_id,
        "status": "final",
        "code": {
            "coding": [{
                "system": "http://loinc.org",
                "code": "15074-8",
                "display": "Glucose [Mass/volume] in Blood - Predicted"
            }],
            "text": "Predicted Glucose Level"
        },
        "subject": {
            "reference": "Patient/anonymous",
            "display": "Anonymous Patient"
        },
        "effectiveDateTime": timestamp,
        "issued": timestamp,
        "valueQuantity": {
            "value": result["glucosa_predicha"],
            "unit": "mg/dL",
            "system": "http://unitsofmeasure.org",
            "code": "mg/dL"
        },
        "interpretation": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                "code": "N" if result["categoria"] == "Normal" else "H",
                "display": result["categoria"]
            }],
            "text": f"Risk Level: {result['nivel_riesgo']}"
        }],
        "note": [{
            "text": result["recomendacion"]
        }],
        "component": [
            {
                "code": {"text": "Confidence"},
                "valueString": result["confianza"]
            },
            {
                "code": {"text": "MAE"},
                "valueQuantity": {
                    "value": result["mae"],
                    "unit": "mg/dL"
                }
            },
            {
                "code": {"text": "Confidence Range"},
                "valueString": result["rango_confianza"]
            },
            {
                "code": {"text": "Model Used"},
                "valueString": result["modelo_usado"]
            }
        ]
    }
    
    return fhir_observation

@app.get("/api/v1/predictions/{patient_id}", dependencies=[Depends(verify_token)])
async def get_predictions_history_fhir(patient_id: str):
    return {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": 0,
        "entry": [],
        "note": "Prediction history not yet implemented in MVP. Connect to database for full functionality."
    }

# ==================== STARTUP EVENT ====================

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Iniciando API de Predicción de Glucosa...")
    
    MODELS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    
    load_ml_models()
    load_patients_database()
    
    logger.info(f"✅ API lista. Modelos cargados: {len(MODELS)}")
    logger.info(f"✅ Pacientes en DB: {len(PATIENTS_DB) if PATIENTS_DB is not None else 0}")

# ==================== ROOT ====================

@app.get("/")
async def root():
    return {
        "message": "Glucose ML Prediction API with FHIR Compliance",
        "version": "2.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "legacy_predict": "/predict",
            "fhir_patient": "/api/v1/patient/{id}",
            "fhir_observations": "/api/v1/patient/{id}/observations",
            "fhir_predictions": "/api/v1/predictions"
        },
        "authentication": "Bearer token required for FHIR endpoints",
        "models_active": len(MODELS),
        "fhir_version": "R4"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

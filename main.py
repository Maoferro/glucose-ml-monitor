"""
API de Predicción de Glucosa con Machine Learning y FHIR Compliance
Versión: 2.0 - Rutas Corregidas
Fecha: Enero 2026
"""

from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import os
import uvicorn
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== CONFIGURACIÓN ====================
MODELS_DIR = Path("models")
DATA_DIR = Path("data")
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "test-token-2026")

# Nombres de archivos de modelos (sin prefijo de carpeta)
MODEL_FILES = {
    "xgboost": "MEJOR_MODELO_XGBoost.joblib",
    "random_forest": "Random_Forest.joblib",
    "lightgbm": "LightGBM.joblib",
    "gradient_boosting": "Gradient_Boosting.joblib",
    "ridge": "Ridge.joblib",
    "lasso": "Lasso.joblib",
    "elasticnet": "ElasticNet.joblib"
}

# Variables globales
MODELS = {}
PATIENTS_DB = None

# ==================== INICIALIZAR FASTAPI ====================
app = FastAPI(
    title="Glucose ML Prediction API with FHIR",
    description="API de predicción de glucosa con Machine Learning y compliance FHIR R4",
    version="2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ==================== CORS ====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== MODELOS PYDANTIC ====================
class PredictionRequest(BaseModel):
    edad: int = Field(..., ge=0, le=120)
    sexo: str = Field(..., pattern="^(M|F|Masculino|Femenino)$")
    peso: float = Field(..., gt=0)
    talla: float = Field(..., gt=0)
    imc: Optional[float] = None
    perimetro_cintura: Optional[float] = None
    spo2: Optional[float] = Field(None, ge=0, le=100)
    frecuencia_cardiaca: Optional[int] = Field(None, ge=0)
    actividad_fisica: Optional[str] = None
    consumo_frutas: Optional[str] = None
    tiene_hipertension: Optional[str] = None
    tiene_diabetes: Optional[str] = None

class FHIRPatient(BaseModel):
    resourceType: str = "Patient"
    id: str
    identifier: List[Dict[str, Any]]
    name: List[Dict[str, Any]]
    gender: str
    birthDate: Optional[str]

class FHIRObservation(BaseModel):
    resourceType: str = "Observation"
    id: str
    status: str
    code: Dict[str, Any]
    subject: Dict[str, str]
    effectiveDateTime: str
    valueQuantity: Dict[str, Any]

# ==================== FUNCIONES DE CARGA ====================
def load_models():
    """Carga los 7 modelos ML desde la carpeta models/"""
    global MODELS
    logger.info("🔄 Iniciando carga de modelos ML...")
    
    for model_name, filename in MODEL_FILES.items():
        try:
            model_path = MODELS_DIR / filename
            if model_path.exists():
                MODELS[model_name] = joblib.load(model_path)
                logger.info(f"✅ Modelo {model_name} cargado desde {model_path}")
            else:
                logger.warning(f"⚠️ Modelo {model_name} no encontrado en {model_path}")
        except Exception as e:
            logger.error(f"❌ Error cargando modelo {model_name}: {str(e)}")
    
    logger.info(f"✅ Total de modelos cargados: {len(MODELS)}/7")
    return len(MODELS)

def load_patients_db():
    """Carga la base de datos de pacientes desde CSV"""
    global PATIENTS_DB
    try:
        csv_path = DATA_DIR / "base_unificada.csv"
        if csv_path.exists():
            PATIENTS_DB = pd.read_csv(csv_path)
            logger.info(f"✅ Base de datos cargada: {len(PATIENTS_DB)} pacientes desde {csv_path}")
            return len(PATIENTS_DB)
        else:
            logger.warning(f"⚠️ Archivo CSV no encontrado en {csv_path}")
            return 0
    except Exception as e:
        logger.error(f"❌ Error cargando base de datos: {str(e)}")
        return 0

# ==================== AUTENTICACIÓN ====================
async def verify_token(authorization: str = Header(None)):
    """Verifica el token Bearer en el header"""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing"
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

# ==================== ENDPOINTS ====================

@app.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "message": "Glucose ML Prediction API with FHIR Compliance",
        "version": "2.0",
        "models_active": len(MODELS),
        "patients_loaded": len(PATIENTS_DB) if PATIENTS_DB is not None else 0,
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
async def health_check():
    """Endpoint de salud de la API"""
    return {
        "status": "healthy" if len(MODELS) > 0 else "degraded",
        "models_loaded": len(MODELS),
        "models_expected": 7,
        "patients_loaded": len(PATIENTS_DB) if PATIENTS_DB is not None else 0,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Endpoint de predicción original (compatibilidad)"""
    if len(MODELS) == 0:
        raise HTTPException(
            status_code=503,
            detail="No hay modelos cargados"
        )
    
    # Calcular IMC si no está proporcionado
    imc = request.imc if request.imc else request.peso / ((request.talla / 100) ** 2)
    
    # Preparar features para predicción
    features = {
        "edad": request.edad,
        "peso": request.peso,
        "talla": request.talla,
        "imc": imc,
        "perimetro_cintura": request.perimetro_cintura or 0,
        "spo2": request.spo2 or 98.0,
        "frecuencia_cardiaca": request.frecuencia_cardiaca or 75
    }
    
    # Realizar predicción con XGBoost (modelo principal)
    try:
        X = pd.DataFrame([features])
        if "xgboost" in MODELS:
            prediction = MODELS["xgboost"].predict(X)[0]
        else:
            # Usar el primer modelo disponible
            model_name = list(MODELS.keys())[0]
            prediction = MODELS[model_name].predict(X)[0]
        
        # Clasificar resultado
        if prediction < 100:
            categoria = "Normal"
            riesgo = "Bajo"
        elif prediction < 126:
            categoria = "Prediabetes"
            riesgo = "Moderado"
        else:
            categoria = "Diabetes"
            riesgo = "Alto"
        
        return {
            "glucosa_predicha": round(float(prediction), 2),
            "categoria": categoria,
            "nivel_riesgo": riesgo,
            "imc_calculado": round(imc, 2),
            "modelo_usado": "xgboost" if "xgboost" in MODELS else list(MODELS.keys())[0],
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

# ==================== ENDPOINTS FHIR ====================

@app.get("/api/v1/patient/{patient_id}", dependencies=[Depends(verify_token)])
async def get_patient_fhir(patient_id: str):
    """Obtiene un paciente en formato FHIR R4"""
    if PATIENTS_DB is None or len(PATIENTS_DB) == 0:
        raise HTTPException(status_code=404, detail="Base de datos no disponible")
    
    # Buscar paciente
    patient = PATIENTS_DB[PATIENTS_DB["ID_Unico"] == patient_id]
    if patient.empty:
        raise HTTPException(status_code=404, detail=f"Paciente {patient_id} no encontrado")
    
    patient_data = patient.iloc[0]
    
    # Construir respuesta FHIR
    return {
        "resourceType": "Patient",
        "id": patient_id,
        "identifier": [{
            "system": "http://hospital.local/patient-id",
            "value": str(patient_data.get("Identificacion", ""))
        }],
        "name": [{
            "use": "official",
            "text": str(patient_data.get("Nombre_Completo", ""))
        }],
        "gender": "male" if patient_data.get("Sexo") == "M" else "female",
        "birthDate": str(2024 - int(patient_data.get("Edad", 0))) if pd.notna(patient_data.get("Edad")) else None,
        "extension": [{
            "url": "http://hospital.local/fhir/StructureDefinition/imc",
            "valueDecimal": float(patient_data.get("IMC", 0))
        }]
    }

@app.get("/api/v1/patient/{patient_id}/observations", dependencies=[Depends(verify_token)])
async def get_patient_observations_fhir(patient_id: str):
    """Obtiene las observaciones de un paciente en formato FHIR"""
    if PATIENTS_DB is None or len(PATIENTS_DB) == 0:
        raise HTTPException(status_code=404, detail="Base de datos no disponible")
    
    patient = PATIENTS_DB[PATIENTS_DB["ID_Unico"] == patient_id]
    if patient.empty:
        raise HTTPException(status_code=404, detail=f"Paciente {patient_id} no encontrado")
    
    patient_data = patient.iloc[0]
    observations = []
    
    # Observación de glucosa
    if pd.notna(patient_data.get("Glucosa_Estimada_mgdL")):
        observations.append({
            "resourceType": "Observation",
            "id": f"{patient_id}-glucose-1",
            "status": "final",
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": "15074-8",
                    "display": "Glucose [Mass/volume] in Blood"
                }]
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "effectiveDateTime": datetime.now().isoformat(),
            "valueQuantity": {
                "value": float(patient_data["Glucosa_Estimada_mgdL"]),
                "unit": "mg/dL",
                "system": "http://unitsofmeasure.org",
                "code": "mg/dL"
            }
        })
    
    # Observación de IMC
    if pd.notna(patient_data.get("IMC")):
        observations.append({
            "resourceType": "Observation",
            "id": f"{patient_id}-bmi-1",
            "status": "final",
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": "39156-5",
                    "display": "Body mass index (BMI)"
                }]
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "effectiveDateTime": datetime.now().isoformat(),
            "valueQuantity": {
                "value": float(patient_data["IMC"]),
                "unit": "kg/m2",
                "system": "http://unitsofmeasure.org",
                "code": "kg/m2"
            }
        })
    
    return {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": len(observations),
        "entry": [{"resource": obs} for obs in observations]
    }

@app.post("/api/v1/predictions", dependencies=[Depends(verify_token)])
async def create_prediction_fhir(request: PredictionRequest):
    """Crea una predicción y la retorna en formato FHIR Observation"""
    if len(MODELS) == 0:
        raise HTTPException(status_code=503, detail="No hay modelos cargados")
    
    # Realizar predicción
    imc = request.imc if request.imc else request.peso / ((request.talla / 100) ** 2)
    features = {
        "edad": request.edad,
        "peso": request.peso,
        "talla": request.talla,
        "imc": imc,
        "perimetro_cintura": request.perimetro_cintura or 0,
        "spo2": request.spo2 or 98.0,
        "frecuencia_cardiaca": request.frecuencia_cardiaca or 75
    }
    
    X = pd.DataFrame([features])
    model_name = "xgboost" if "xgboost" in MODELS else list(MODELS.keys())[0]
    prediction = MODELS[model_name].predict(X)[0]
    
    # Retornar en formato FHIR Observation
    return {
        "resourceType": "Observation",
        "id": f"prediction-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "status": "final",
        "code": {
            "coding": [{
                "system": "http://loinc.org",
                "code": "15074-8",
                "display": "Glucose [Mass/volume] in Blood - Predicted"
            }],
            "text": "Predicción de Glucosa con ML"
        },
        "effectiveDateTime": datetime.now().isoformat(),
        "valueQuantity": {
            "value": round(float(prediction), 2),
            "unit": "mg/dL",
            "system": "http://unitsofmeasure.org",
            "code": "mg/dL"
        },
        "interpretation": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                "code": "H" if prediction >= 126 else "N" if prediction < 100 else "I",
                "display": "High" if prediction >= 126 else "Normal" if prediction < 100 else "Intermediate"
            }]
        }],
        "note": [{
            "text": f"Predicción realizada con modelo {model_name} basado en IMC {round(imc, 2)}"
        }]
    }

@app.get("/api/v1/predictions/{patient_id}", dependencies=[Depends(verify_token)])
async def get_predictions_history_fhir(patient_id: str):
    """Obtiene el historial de predicciones de un paciente en formato FHIR"""
    # En una implementación real, esto consultaría una base de datos de predicciones
    # Por ahora, retornamos un bundle vacío o de ejemplo
    return {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": 0,
        "entry": [],
        "note": "Historial de predicciones no disponible en esta versión MVP"
    }

# ==================== STARTUP EVENT ====================
@app.on_event("startup")
async def startup_event():
    """Se ejecuta al iniciar la aplicación"""
    logger.info("🚀 Iniciando API de Predicción de Glucosa...")
    
    models_loaded = load_models()
    patients_loaded = load_patients_db()
    
    logger.info(f"✅ API lista - Modelos cargados: {models_loaded}/7")
    logger.info(f"✅ Pacientes cargados: {patients_loaded}")
    
    if models_loaded == 0:
        logger.warning("⚠️ ADVERTENCIA: No se cargó ningún modelo ML")
    
    if patients_loaded == 0:
        logger.warning("⚠️ ADVERTENCIA: No se cargó la base de datos de pacientes")

# ==================== MAIN ====================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )

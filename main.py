from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime

# ===== CONFIGURACIÓN DE LOGGING =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===== CONFIGURACIÓN DE FASTAPI =====
app = FastAPI(
    title="Glucose ML Monitor API",
    description="API para predicción de glucosa usando 7 modelos ML ensemble",
    version="2.1.0"
)

# ===== CONFIGURACIÓN DE CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== RUTAS DE MODELOS =====
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

MODEL_FILES = {
    "xgboost": MODELS_DIR / "XGBoost.joblib",
    "random_forest": MODELS_DIR / "Random_Forest.joblib",
    "lightgbm": MODELS_DIR / "LightGBM.joblib",
    "gradient_boosting": MODELS_DIR / "Gradient_Boosting.joblib",
    "ridge": MODELS_DIR / "Ridge.joblib",
    "lasso": MODELS_DIR / "Lasso.joblib",
    "elasticnet": MODELS_DIR / "ElasticNet.joblib"
}

PREPROCESSING_FILE = MODELS_DIR / "preprocessing_objects.pkl"

# ===== VARIABLES GLOBALES =====
models = {}
label_encoders = {}
scaler = None
FEATURE_NAMES = None

# ===== MAPEO DE FEATURES (CRÍTICO) =====
# El frontend envía lowercase, pero los modelos fueron entrenados con PascalCase
FEATURE_MAP = {
    "edad": "Edad",
    "sexo": "Sexo",
    "peso": "Peso",
    "talla": "Talla",
    "imc": "IMC",
    "perimetro_cintura": "Perimetro_Cintura",
    "spo2": "SpO2",
    "frecuencia_cardiaca": "Frecuencia_Cardiaca",
    "actividad_fisica": "Actividad_Fisica",
    "consumo_frutas": "Consumo_Frutas",
    "tiene_hipertension": "Tiene_Hipertension",
    "tiene_diabetes": "Tiene_Diabetes",
    "puntaje_findrisc": "Puntaje_FINDRISC"
}

# ===== MODELO DE DATOS DE ENTRADA =====
class PredictionInput(BaseModel):
    edad: int = Field(..., ge=18, le=120, description="Edad del paciente")
    sexo: str = Field(..., description="Sexo: Masculino o Femenino")
    peso: float = Field(..., gt=0, description="Peso en kilogramos")
    talla: float = Field(..., gt=0, description="Talla en metros")
    imc: Optional[float] = Field(None, description="Índice de Masa Corporal")
    perimetro_cintura: float = Field(..., gt=0, description="Perímetro de cintura en cm")
    spo2: int = Field(..., ge=70, le=100, description="Saturación de oxígeno")
    frecuencia_cardiaca: int = Field(..., ge=40, le=200, description="Frecuencia cardíaca")
    actividad_fisica: str = Field(..., description="si o no")
    consumo_frutas: str = Field(..., description="si o no")
    tiene_hipertension: str = Field(..., description="Si o No")
    tiene_diabetes: str = Field(..., description="Si o No")
    puntaje_findrisc: int = Field(..., ge=0, le=26, description="Puntaje FINDRISC")

# ===== CARGAR MODELOS Y PREPROCESSING =====
@app.on_event("startup")
async def load_models():
    global models, label_encoders, scaler, FEATURE_NAMES
    
    try:
        logger.info("🚀 Iniciando carga de modelos ML...")
        
        if not PREPROCESSING_FILE.exists():
            raise FileNotFoundError(f"Archivo de preprocesamiento no encontrado: {PREPROCESSING_FILE}")
        
        preprocessing = joblib.load(PREPROCESSING_FILE)
        label_encoders = preprocessing.get('label_encoders', {})
        scaler = preprocessing.get('scaler', None)
        FEATURE_NAMES = preprocessing.get('feature_names', None)
        
        logger.info(f"✅ Preprocesamiento cargado")
        logger.info(f"📋 Feature Names: {FEATURE_NAMES}")
        
        if FEATURE_NAMES is None:
            logger.warning("⚠️ feature_names no encontrado, usando FEATURE_MAP")
            FEATURE_NAMES = list(FEATURE_MAP.values())
        
        models_loaded = 0
        for model_name, model_path in MODEL_FILES.items():
            if not model_path.exists():
                logger.warning(f"⚠️ Modelo no encontrado: {model_path}")
                continue
            
            try:
                models[model_name] = joblib.load(model_path)
                models_loaded += 1
                logger.info(f"✅ Modelo cargado: {model_name}")
            except Exception as e:
                logger.error(f"❌ Error al cargar {model_name}: {str(e)}")
        
        if models_loaded == 0:
            raise RuntimeError("No se pudo cargar ningún modelo ML")
        
        logger.info(f"🎉 Total de modelos cargados: {models_loaded}/7")
        
    except Exception as e:
        logger.error(f"❌ Error crítico al cargar modelos: {str(e)}")
        raise

# ===== FUNCIÓN DE PREPROCESAMIENTO (CORREGIDA) =====
def preprocess_input(data: PredictionInput) -> pd.DataFrame:
    """
    Preprocesa los datos de entrada con FEATURE NAMES en PascalCase.
    """
    try:
        # 1. Calcular IMC si no viene
        imc_value = data.imc if data.imc else (data.peso / (data.talla ** 2))
        
        # 2. Crear diccionario con nombres PascalCase (como espera el modelo)
        input_dict = {
            'Edad': data.edad,
            'Sexo': data.sexo.lower(),
            'Peso': data.peso,
            'Talla': data.talla,
            'IMC': imc_value,
            'Perimetro_Cintura': data.perimetro_cintura,
            'SpO2': data.spo2,
            'Frecuencia_Cardiaca': data.frecuencia_cardiaca,
            'Actividad_Fisica': data.actividad_fisica.lower(),
            'Consumo_Frutas': data.consumo_frutas.lower(),
            'Tiene_Hipertension': data.tiene_hipertension.lower(),
            'Tiene_Diabetes': data.tiene_diabetes.lower(),
            'Puntaje_FINDRISC': data.puntaje_findrisc
        }
        
        # 3. Aplicar LabelEncoders a variables categóricas
        for col, encoder in label_encoders.items():
            if col in input_dict:
                try:
                    input_dict[col] = encoder.transform([input_dict[col]])[0]
                except ValueError as e:
                    logger.warning(f"⚠️ Valor no visto en {col}: {input_dict[col]}, usando 0")
                    input_dict[col] = 0
        
        # 4. Crear DataFrame con orden correcto de columnas
        df = pd.DataFrame([input_dict])
        
        # 5. Reordenar columnas según FEATURE_NAMES si existe
        if FEATURE_NAMES is not None:
            # Asegurar que todas las columnas existan
            for col in FEATURE_NAMES:
                if col not in df.columns:
                    df[col] = 0
            df = df[FEATURE_NAMES]
        
        # 6. Aplicar StandardScaler
        if scaler is not None:
            X_scaled = scaler.transform(df)
        else:
            X_scaled = df.values
        
        # 7. Retornar como DataFrame con feature names
        X_scaled_df = pd.DataFrame(X_scaled, columns=df.columns)
        logger.info(f"✅ DataFrame creado: {list(X_scaled_df.columns)}")
        return X_scaled_df
        
    except Exception as e:
        logger.error(f"❌ Error en preprocesamiento: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en preprocesamiento: {str(e)}")

# ===== ENDPOINT: HEALTH CHECK =====
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(models),
        "models_available": list(models.keys()),
        "feature_names_loaded": FEATURE_NAMES is not None
    }

# ===== ENDPOINT: PREDICCIÓN =====
@app.post("/predict")
async def predict_glucose(data: PredictionInput):
    try:
        logger.info(f"📥 Predicción: edad={data.edad}, sexo={data.sexo}")
        
        X_preprocessed = preprocess_input(data)
        logger.info(f"✅ Datos preprocesados: shape={X_preprocessed.shape}")
        
        predicciones_individuales = {}
        predicciones_validas = []
        
        for model_name, model in models.items():
            try:
                pred = model.predict(X_preprocessed)[0]
                predicciones_individuales[model_name] = float(pred)
                predicciones_validas.append(pred)
                logger.info(f"✅ {model_name}: {pred:.2f} mg/dL")
            except Exception as e:
                logger.error(f"❌ Error en {model_name}: {str(e)}")
                predicciones_individuales[model_name] = None
        
        if len(predicciones_validas) == 0:
            raise HTTPException(status_code=500, detail="Ningún modelo pudo generar predicción")
        
        prediccion_final = float(np.mean(predicciones_validas))
        
        if prediccion_final < 100:
            categoria = "Normal"
        elif prediccion_final < 126:
            categoria = "Prediabetes"
        else:
            categoria = "Diabetes"
        
        std_predicciones = float(np.std(predicciones_validas))
        mae = std_predicciones
        confidence = 1.0 - (std_predicciones / prediccion_final) if prediccion_final > 0 else 0.5
        confidence = max(0.0, min(1.0, confidence))
        
        intervalo_min = prediccion_final - 1.96 * std_predicciones
        intervalo_max = prediccion_final + 1.96 * std_predicciones
        
        mejor_modelo = min(predicciones_individuales.items(), 
                          key=lambda x: abs(x[1] - prediccion_final) if x[1] is not None else float('inf'))
        
        response = {
            "prediccion_final": round(prediccion_final, 2),
            "categoria": categoria,
            "predicciones_individuales": {
                k: round(v, 2) if v is not None else None 
                for k, v in predicciones_individuales.items()
            },
            "confidence": round(confidence, 3),
            "mae": round(mae, 2),
            "intervalo_confianza": [round(intervalo_min, 2), round(intervalo_max, 2)],
            "mejor_modelo": mejor_modelo[0],
            "timestamp": datetime.now().isoformat(),
            "input_data": data.dict()
        }
        
        logger.info(f"✅ Predicción: {prediccion_final:.2f} mg/dL ({categoria})")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

# ===== ENDPOINT: SERVIR INDEX.HTML =====
@app.get("/")
async def serve_index():
    index_path = BASE_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html no encontrado")
    return FileResponse(index_path)

# ===== ENDPOINT: LISTAR MODELOS =====
@app.get("/models")
async def list_models():
    return {
        "models": list(models.keys()),
        "total": len(models),
        "feature_names": FEATURE_NAMES
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


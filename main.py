import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Diabetes Glucose Prediction API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models_path = './models/'
modelos = {}
archivos_modelos = {
    'XGBoost': 'MEJOR_MODELO_XGBoost.joblib',
    'RandomForest': 'Random_Forest.joblib',
    'LightGBM': 'LightGBM.joblib',
    'GradientBoosting': 'Gradient_Boosting.joblib',
    'Ridge': 'Ridge.joblib',
    'Lasso': 'Lasso.joblib',
    'ElasticNet': 'ElasticNet.joblib'
}

print("🔄 Cargando modelos ML...")
for nombre, archivo in archivos_modelos.items():
    ruta = os.path.join(models_path, archivo)
    if os.path.exists(ruta):
        try:
            modelos[nombre] = joblib.load(ruta)
            print(f"  ✅ {nombre} cargado")
        except Exception as e:
            print(f"  ❌ Error en {nombre}: {e}")
    else:
        print(f"  ⚠️ No encontrado: {archivo}")

print(f"\n✅ Total: {len(modelos)}/7 modelos cargados\n")

class DatosPaciente(BaseModel):
    Edad: int
    Sexo: str
    Peso: float
    Talla: float
    IMC: float
    Perimetro_Cintura: int
    SpO2: int
    Frecuencia_Cardiaca: int
    Actividad_Fisica: str
    Consumo_Frutas: str
    Tiene_Hipertension: str
    Tiene_Diabetes: str
    Puntaje_FINDRISC: int

@app.get("/")
def root():
    return FileResponse('index.html')

@app.get("/health")
def health():
    return {
        "estado": "OK",
        "modelos_cargados": len(modelos),
        "lista_modelos": list(modelos.keys())
    }

@app.post("/predict")
def predict(datos: DatosPaciente):
    try:
        df = pd.DataFrame([datos.dict()])
        predicciones = {}
        
        for nombre, modelo in modelos.items():
            try:
                pred = modelo.predict(df)[0]
                predicciones[nombre] = round(float(pred), 2)
            except Exception as e:
                print(f"Error en {nombre}: {e}")
                predicciones[nombre] = None
        
        valores_validos = [v for v in predicciones.values() if v is not None]
        glucosa_final = round(np.mean(valores_validos), 2) if valores_validos else None
        
        if glucosa_final:
            if glucosa_final < 100:
                categoria, riesgo, color = "Normal", "Bajo", "#4CAF50"
            elif glucosa_final < 126:
                categoria, riesgo, color = "Prediabetes", "Moderado", "#FFC107"
            else:
                categoria, riesgo, color = "Diabetes", "Alto", "#F44336"
        else:
            categoria, riesgo, color = "Error", "Desconocido", "#9E9E9E"
        
        modelo_cercano = min(
            predicciones.items(),
            key=lambda x: abs(x[1] - glucosa_final) if x[1] else float('inf')
        )[0]
        
        return {
            "glucosa_predicha": glucosa_final,
            "predicciones_por_modelo": predicciones,
            "modelo_mas_acertado": modelo_cercano,
            "categoria": categoria,
            "riesgo": riesgo,
            "color_categoria": color,
            "confianza": 0.92,
            "error_esperado_mae": 8.2,
            "rango_confianza": {
                "min": round(glucosa_final - 8.2, 2) if glucosa_final else None,
                "max": round(glucosa_final + 8.2, 2) if glucosa_final else None
            },
            "recomendacion": f"Glucosa predicha: {glucosa_final} mg/dL ({categoria})."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

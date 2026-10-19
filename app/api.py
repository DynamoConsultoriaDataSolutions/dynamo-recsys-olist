"""API REST desarrollada con FastAPI para exponer el motor de recomendación híbrido de Olist.

Este componente actúa como la capa de servicios (Backend), aislando la lógica matemática del modelo
entrenado de la interfaz de usuario. Carga en memoria el bundle del modelo de manera eficiente al 
iniciar el servidor y expone endpoints clave para el consumo en tiempo real.

Endpoints disponibles:
    - GET /user/{user_id}/status  -> Devuelve si el cliente es conocido (Warm) o nuevo (Cold Start).
    - GET /user/{user_id}/history -> Recupera el historial de compra real del cliente en Train.
    - GET /recommendations        -> Genera el listado Top-K de productos con sus respectivos motivos.

Antes de ejecutar la API, verificar que exista el archivo en la raíz: 
    models/item2item_hibrido.joblib

Ejemplo de ejecución local desde la raíz del proyecto:
    uvicorn app.api:app --reload

Ejemplo de consumo vía cURL u otra aplicación:
    curl "http://127.0.0"
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# Importamos las funciones que dejaron tus compañeros
from src.models.predict import load_bundle, recommend, user_history, is_known

app = FastAPI(title="Dynamo Consultoría - API Olist", version="1.1")

# Habilitamos CORS por si ejecutas Streamlit y FastAPI en puertos cruzados
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargamos el bundle una sola vez en memoria al levantar la API
try:
    print("⏳ Cargando item2item_hibrido.joblib en memoria...")
    bundle = load_bundle()
    print("🚀 Bundle cargado exitosamente. API lista para operar.")
except Exception as e:
    print(f"❌ Error crítico al cargar el modelo: {e}")
    bundle = None

@app.get("/user/{user_id}/status")
def verificar_usuario(user_id: str):
    """Devuelve si el usuario es conocido (Warm) o nuevo (Cold Start)"""
    if bundle is None:
        raise HTTPException(status_code=500, detail="Modelo no disponible.")
    known = is_known(bundle, user_id)
    return {"user_id": user_id, "is_known": known}

@app.get("/user/{user_id}/history")
def obtener_historial(user_id: str):
    """Devuelve el DataFrame del historial convertido a diccionario de registros"""
    if bundle is None:
        raise HTTPException(status_code=500, detail="Modelo no disponible.")
    df_hist = user_history(bundle, user_id)
    return {"user_id": user_id, "history": df_hist.to_dict(orient="records")}

@app.get("/recommendations")
def obtener_recomendaciones(user_id: str, k: int = 10):
    """Devuelve las Top-K recomendaciones procesadas por el modelo híbrido"""
    if bundle is None:
        raise HTTPException(status_code=500, detail="Modelo no disponible.")
    
    # La función 'recommend' ya maneja el fallback interno si el usuario es nuevo
    df_recs = recommend(bundle, user_id, k=k)
    return {
        "user_id": user_id,
        "recommendations": df_recs.to_dict(orient="records")
    }

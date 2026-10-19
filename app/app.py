"""Interfaz gráfica interactiva desarrollada con Streamlit para la Demo Nro. 2 del proyecto.

Este componente actúa como la fachada (Frontend) orientada al usuario/stakeholder. No realiza
cómputos pesados ni accede de forma directa a la base de datos completa. Su función es consumir
los endpoints de la API en el puerto 8000 y renderizar los resultados de forma estética y comercial.

Características implementadas:
    - Cacheo de recursos (@st.cache_resource) para precargar usuarios de prueba desde el bundle.
    - Manejo de flujo segmentado: Pestaña para Clientes Registrados y Pestaña para Tráfico Anónimo.
    - Resiliencia visual: Alertas dinámicas que exponen si el motor aplicó filtrado híbrido o 
      activó la capa de contingencia por Cold Start.

Antes de ejecutar la interfaz, asegurarse de tener la API web corriendo en paralelo:
    uvicorn app.api:app --reload

Ejemplo de ejecución local desde la raíz del proyecto:
    streamlit run app/app.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Dynamo Engine - Demo Nro. 1", layout="wide")

st.title("📦 Centro de Recomendaciones - Dynamo Consultoría")
st.caption("Fase de Despliegue | Modelo Ítem-to-Ítem Híbrido de Olist")

API_URL = "http://127.0.0.1:8000"

# Usamos el cliente simulado para traer usuarios de prueba dinámicamente desde el bundle
# Importamos la función de ejemplos solo para la UI de Streamlit
from src.models.predict import load_bundle, example_users

@st.cache_resource
def obtener_usuarios_demo():
    try:
        b = load_bundle()
        # Traemos 10 usuarios que tengan al menos 2 compras para la demo
        return example_users(b, n=10, min_products=2)
    except:
        # Fallback en caso de que la UI corra en un entorno sin acceso directo al archivo
        return ["c25a509b05b3d014d58a5c95204c3572", "f29ecf4c6f2ee4e0a4f5f5f5f5f5f5f5"]

usuarios_ejemplo = obtener_usuarios_demo()

# Estructura de interacción en la barra lateral
st.sidebar.header("🕹️ Panel de Control")
user_sel = st.sidebar.selectbox("Seleccione un usuario con historial (Train):", usuarios_ejemplo)
user_manual = st.sidebar.text_input("O ingrese un customer_unique_id manualmente (Prueba Cold Start):")

user_final = user_manual.strip() if user_manual else user_sel
k_recomendaciones = st.sidebar.slider("Cantidad de recomendaciones (K)", min_value=1, max_value=20, value=10)

if st.sidebar.button("⚡ Ejecutar Motor de Recomendación"):
    if not user_final:
        st.error("Por favor ingrese o seleccione un ID de usuario.")
    else:
        try:
            # 1. Consultar estado del usuario
            res_status = requests.get(f"{API_URL}/user/{user_final}/status").json()
            is_known_user = res_status["is_known"]
            
            # Encabezado dinámico según el tipo de cliente
            if is_known_user:
                st.success(f"👤 **Cliente Identificado (Segmento Warm)**: `{user_final}`")
            else:
                st.warning(f"❄️ **Cliente Nuevo Detectado (Cold Start)**: `{user_final}`. Aplicando estrategia de tendencias del último mes.")
            
            # Creamos dos columnas en la interfaz para comparar comportamiento
            col_izq, col_der = st.columns(2)
            
            with col_izq:
                st.subheader("🛒 Historial de Compra (Pasado)")
                res_hist = requests.get(f"{API_URL}/user/{user_final}/history").json()
                df_hist = pd.DataFrame(res_hist["history"])
                
                if df_hist.empty:
                    st.info("Este cliente no registra compras previas en el set de entrenamiento.")
                else:
                    st.dataframe(df_hist[["product_id", "category", "price"]], use_container_width=True, hide_index=True)
            
            with col_der:
                st.subheader("🎯 Recomendaciones sugeridas (Futuro)")
                res_recs = requests.get(f"{API_URL}/recommendations", params={"user_id": user_final, "k": k_recomendaciones}).json()
                df_recs = pd.DataFrame(res_recs["recommendations"])
                
                if df_recs.empty:
                    st.error("No se pudieron generar recomendaciones para este usuario.")
                else:
                    # Mostramos el DataFrame lindo con los motivos y explicaciones que tus compañeros programaron
                    st.dataframe(
                        df_recs[["posicion", "product_id", "category", "price", "explicacion"]],
                        use_container_width=True,
                        hide_index=True
                    )
                    
        except requests.exceptions.ConnectionError:
            st.error("❌ Error de conexión: Asegurate de tener la API de FastAPI corriendo en el puerto 8000 (`uvicorn api:app --reload`).")

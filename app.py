"""
App Avanzada de Streamlit — Nivel de ríos/quebradas (CORNARE / MARCO)
--------------------------------------------------------------------
Modificada para incluir alertas tempranas, análisis exploratorio (EDA),
y personalización de la estación.
"""

import requests
import pandas as pd
import numpy as np
import streamlit as st
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ------------------------------------------------------------------
# Coordenadas por defecto (Ej: Estación Río Nare - Antioquia)
# ------------------------------------------------------------------
LAT_DEFECTO = 6.2891
LON_DEFECTO = -75.1233

API_BASE_URL = "https://marco.cornare.gov.co/api/v1/estaciones"

LLAVE_FECHA = "level_date"
LLAVE_VALOR = "level"
CANDIDATOS_LAT = ["lat", "latitude", "latitud"]
CANDIDATOS_LON = ["lng", "lon", "longitude", "longitud"]

st.set_page_config(page_title="Monitor Hidrológico — MARCO", page_icon="🌊", layout="wide")

# ------------------------------------------------------------------
# Funciones de consulta (Se mantienen estables)
# ------------------------------------------------------------------
def obtener_serie_nivel(codigo_estacion, desde, hasta, calidad=1, timeout=30):
    url = f"{API_BASE_URL}/{codigo_estacion}/nivel"
    params = {"desde": desde, "hasta": hasta, "calidad": calidad}
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout, verify=False)
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"HTTP {resp.status_code}"
    except requests.exceptions.RequestException as e:
        return None, f"Error de red: {e}"

def obtener_todas_las_paginas(datos_json, timeout=30):
    registros = list(datos_json.get("values", []))
    siguiente_url = datos_json.get("next")
    while siguiente_url:
        try:
            resp = requests.get(siguiente_url, timeout=timeout, verify=False)
        except requests.exceptions.RequestException:
            break
        if resp.status_code != 200:
            break
        pagina = resp.json()
        registros.extend(pagina.get("values", []))
        siguiente_url = pagina.get("next")
    return registros

def detectar_coordenadas(datos_json):
    if not isinstance(datos_json, dict):
        return LAT_DEFECTO, LON_DEFECTO, False

    lat = next((datos_json[k] for k in CANDIDATOS_LAT if k in datos_json), None)
    lon = next((datos_json[k] for k in CANDIDATOS_LON if k in datos_json), None)

    if lat is not None and lon is not None:
        try:
            return float(lat), float(lon), True
        except (TypeError, ValueError):
            pass
    return LAT_DEFECTO, LON_DEFECTO, False

def calcular_indice_calidad(df):
    if df.empty or len(df) < 2:
        return 0.0, 0, 0

    df_idx = df.set_index("fecha")
    frecuencia_tipica = df["fecha"].diff().dropna().mode()
    if len(frecuencia_tipica) == 0:
        return 0.0, 0, 0
    frecuencia_tipica = frecuencia_tipica[0]

    rango_completo = pd.date_range(start=df_idx.index.min(), end=df_idx.index.max(), freq=frecuencia_tipica)
    esperados = len(rango_completo)
    huecos = esperados - len(df_idx)
    completitud = max(0.0, 1 - (huecos / esperados)) if esperados > 0 else 0.0

    Q1, Q3 = df["nivel"].quantile(0.25), df["nivel"].quantile(0.75)
    IQR = Q3 - Q1
    lim_inf, lim_sup = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    es_outlier = (df["nivel"] < lim_inf) | (df["nivel"] > lim_sup) | (df["nivel"] < 0)
    proporcion_outliers = es_outlier.mean()

    indice = (completitud * 0.7 + (1 - proporcion_outliers) * 0.3) * 100
    return round(indice, 1), int(huecos), int(es_outlier.sum())

# ------------------------------------------------------------------
# Sidebar — Configuración de la Estación y Alertas
# ------------------------------------------------------------------
st.sidebar.header("⚙️ Configuración de Estación")
nombre_estudiante = st.sidebar.text_input("Analista de Datos", "Tu Nombre Aquí")
nombre_estacion = st.sidebar.text_input("Nombre de la estación", "Río Nare - Principal")
codigo_estacion = st.sidebar.text_input("Código de estación (API)", "42")

st.sidebar.header("📅 Parámetros Temporales")
fecha_desde = st.sidebar.date_input("Desde", pd.to_datetime("2026-08-23")).strftime("%Y-%m-%d")
fecha_hasta = st.sidebar.date_input("Hasta", pd.to_datetime("2026-08-30")).strftime("%Y-%m-%d")
calidad = st.sidebar.selectbox("Filtro de Calidad", [1, 0], index=0, help="1 = Solo datos validados (Recomendado para Modelado)")

st.sidebar.header("🚨 Sistema de Alerta Temprana")
umbral_alerta = st.sidebar.slider("Umbral de desbordamiento (m)", min_value=1.0, max_value=10.0, value=3.5, step=0.1)

consultar = st.sidebar.button("🔍 Extraer Datos", type="primary")

# ------------------------------------------------------------------
# Dashboard Principal
# ------------------------------------------------------------------
st.title(f"🌊 Monitor Hidrológico: {nombre_estacion}")
st.caption(f"Operador: **{nombre_estudiante}** | Código API: **{codigo_estacion}**")

if consultar:
    with st.spinner("Conectando con el puente digital de MARCO..."):
        datos_crudos, error = obtener_serie_nivel(codigo_estacion, fecha_desde, fecha_hasta, calidad)

    if error:
        st.error(f"❌ Error de extracción: {error}")
    else:
        registros = obtener_todas_las_paginas(datos_crudos)

        if not registros:
            st.warning("No hay registros estructurados para esta ventana de tiempo.")
        else:
            # Preparación de datos (Fase 3 CRISP-DM)
            df = pd.DataFrame(registros)
            df = df.rename(columns={LLAVE_FECHA: "fecha", LLAVE_VALOR: "nivel"})
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
            df["nivel"] = pd.to_numeric(df["nivel"], errors="coerce")
            df = df.dropna(subset=["fecha", "nivel"]).sort_values("fecha").reset_index(drop=True)
            
            # Feature Engineering básica: Media Móvil
            df['media_movil_5'] = df['nivel'].rolling(window=5, min_periods=1).mean()

            lat, lon, coords_reales = detectar_coordenadas(datos_crudos)
            indice_calidad, huecos, n_outliers = calcular_indice_calidad(df)

            # --- Alerta Temprana ---
            ultimo_nivel = df['nivel'].iloc[-1]
            nivel_anterior = df['nivel'].iloc[-2] if len(df) > 1 else ultimo_nivel
            variacion = ultimo_nivel - nivel_anterior

            if ultimo_nivel >= umbral_alerta:
                st.error(f"⚠️ **ALERTA DE RIESGO HIDROLÓGICO:** El nivel actual ({ultimo_nivel:.2f}m) supera el umbral de seguridad ({umbral_alerta}m).")
            else:
                st.success("✅ Estación dentro de los parámetros normales de operación.")

            # --- Métricas principales dinámicas ---
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Última Lectura", f"{ultimo_nivel:.2f} m", f"{variacion:.2f} m (vs ant.)", delta_color="inverse")
            col2.metric("Nivel Máximo (Periodo)", f"{df['nivel'].max():.2f} m")
            col3.metric("Índice de Calidad (ICD)", f"{indice_calidad}%", help="Completitud y ausencia de outliers")
            col4.metric("Registros Válidos", len(df))

            # --- Gráficos (Visualización de Tendencias) ---
            st.subheader("📊 Comportamiento del Nivel y Tendencia")
            chart_data = df.set_index("fecha")[["nivel", "media_movil_5"]]
            chart_data.columns = ["Nivel Real", "Tendencia (Media Móvil)"]
            st.line_chart(chart_data)

            # --- Layout de Análisis Exploratorio y Mapa ---
            col_mapa, col_eda = st.columns([1, 1])
            
            with col_mapa:
                st.subheader("📍 Ubicación Espacial")
                if not coords_reales:
                    st.caption("Ubicación por defecto utilizada. La API no reportó coordenadas exactas para este nodo.")
                st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=12)

            with col_eda:
                st.subheader("📈 Resumen Estadístico (EDA)")
                st.dataframe(df['nivel'].describe().to_frame().T, use_container_width=True)
                
            # --- Auditoría de Calidad y Exportación ---
            with st.expander("🛠️ Auditoría de Calidad de Datos (ICD)"):
                st.write(f"- **Valores Faltantes (Huecos):** {huecos}")
                st.write(f"- **Valores Atípicos (Outliers):** {n_outliers} detectados mediante Rango Intercuartílico (IQR).")
                st.info("Un índice de calidad alto es el puente definitivo entre los datos y los modelos predictivos.")

            csv = df[["fecha", "nivel"]].to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Descargar Set de Datos Limpio (CSV)", csv, file_name=f"{nombre_estacion.replace(' ', '_')}_datos.csv", mime="text/csv")
else:
    st.info("Configura la estación en el panel lateral izquierdo y presiona **Extraer Datos** para iniciar el flujo.")

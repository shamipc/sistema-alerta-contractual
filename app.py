from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "datos"

st.set_page_config(
    page_title="Sistema de alerta contractual",
    page_icon="🏗️",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root { --rosa:#D989A5; --azul:#8FAADC; --fondo:#FFF8FB; }
    .stApp { background: linear-gradient(145deg, #fff 0%, var(--fondo) 100%); }
    h1, h2, h3 { color:#3E3540; }
    .hero { padding:1.35rem 1.5rem; border-radius:22px; background:linear-gradient(120deg,#F8DDE7,#E8EEFC); border:1px solid #E8CBD6; margin-bottom:1rem; }
    .hero h1 { margin:0; font-size:2rem; }
    .hero p { margin:.45rem 0 0; color:#5F5661; }
    .resultado { padding:1.15rem; border-radius:18px; background:white; border:1px solid #E9DDE2; box-shadow:0 6px 18px rgba(91,68,79,.06); }
    .puntaje { font-size:2.15rem; font-weight:750; margin:.2rem 0; }
    .aviso { padding:.85rem 1rem; border-radius:14px; background:#FFF3F7; border-left:5px solid var(--rosa); }
    div[data-testid="stForm"] { background:rgba(255,255,255,.78); padding:1.2rem; border-radius:20px; border:1px solid #E9DDE2; }
    .stButton button, div[data-testid="stFormSubmitButton"] button { background:#B96583; color:white; border:0; border-radius:12px; font-weight:700; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def cargar_modelos():
    costo = joblib.load(DATA_DIR / "modelo_final_costo_random_forest.joblib")
    plazo = joblib.load(DATA_DIR / "modelo_final_plazo_logistico.joblib")
    return costo, plazo


@st.cache_data
def cargar_recursos():
    base = pd.read_csv(DATA_DIR / "base_final_modelamiento.csv")
    with open(DATA_DIR / "configuracion_sistema.json", encoding="utf-8") as archivo:
        configuracion = json.load(archivo)
    return base, configuracion


base, configuracion = cargar_recursos()
modelo_costo, modelo_plazo = cargar_modelos()
variables = configuracion["variables_predictoras"]

st.markdown(
    """
    <section class="hero">
      <h1>Sistema de alerta contractual</h1>
      <p>Evaluación preliminar del riesgo de desviaciones de costo y plazo al inicio de la ejecución contractual.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Acerca del sistema")
    st.write("Prototipo académico aplicado a infraestructura educativa pública del departamento de Lima (2019–2024).")
    st.metric("Base histórica", "92 contratos")
    st.caption("Los resultados son puntajes de apoyo a decisiones, no probabilidades exactas ni reemplazan la evaluación técnica.")

tab_evaluacion, tab_resultados = st.tabs(["Evaluar contrato", "Resultados preliminares"])

with tab_evaluacion:
    st.subheader("Datos disponibles al inicio del contrato")
    with st.form("formulario_contrato"):
        c1, c2, c3 = st.columns(3)
        with c1:
            monto = st.number_input("Monto adjudicado del ítem (S/)", min_value=0.0, value=float(base[variables[0]].median()), step=10000.0)
            reduccion = st.number_input("Reducción en la adjudicación (%)", min_value=0.0, max_value=100.0, value=float(base[variables[1]].median()), step=0.1)
            ofertantes = st.number_input("Número de ofertantes", min_value=1, value=int(base[variables[2]].median()), step=1)
            dias_buena_pro = st.number_input("Días entre convocatoria y buena pro", min_value=0, value=int(base[variables[3]].median()), step=1)
        with c2:
            dias_consentimiento = st.number_input("Días entre buena pro y consentimiento", min_value=0, value=int(base[variables[4]].median()), step=1)
            plazo_original = st.number_input("Plazo contractual original (días)", min_value=1, value=int(base[variables[5]].median()), step=1)
            anio = st.selectbox("Año de inicio contractual", sorted(base[variables[6]].unique(), reverse=True))
            nivel = st.selectbox("Nivel de gobierno", sorted(base[variables[7]].dropna().unique()))
        with c3:
            sistema = st.selectbox("Sistema de contratación", sorted(base[variables[8]].dropna().unique()))
            proceso = st.selectbox("Tipo de proceso de selección", sorted(base[variables[9]].dropna().unique()))
            proveedor = st.selectbox("Tipo de proveedor", sorted(base[variables[10]].dropna().unique()))
            provincia = st.selectbox("Provincia", sorted(base[variables[11]].dropna().unique()))
        evaluar = st.form_submit_button("Evaluar contrato", use_container_width=True)

    if evaluar:
        registro = pd.DataFrame([{
            "monto_adjudicado_item_soles": monto,
            "reduccion_adjudicacion_pct": reduccion,
            "numero_ofertantes": ofertantes,
            "dias_convocatoria_buenapro_limpio": dias_buena_pro,
            "dias_buenapro_consentimiento_limpio": dias_consentimiento,
            "plazo_contractual_original_dias": plazo_original,
            "anio_inicio_contractual": anio,
            "nivel_gobierno_modelo": nivel,
            "sistema_contratacion": sistema,
            "tipo_proceso_seleccion": proceso,
            "tipo_proveedor": proveedor,
            "provincia": provincia,
        }])[variables]

        puntaje_costo = float(modelo_costo.predict_proba(registro)[0, 1])
        puntaje_plazo = float(modelo_plazo.predict_proba(registro)[0, 1])
        nivel_costo = "Vigilancia prioritaria" if puntaje_costo >= 0.55 else "Vigilancia estándar"
        if puntaje_plazo < 0.35:
            nivel_plazo, color_plazo = "Riesgo bajo", "#91C99D"
        elif puntaje_plazo < 0.65:
            nivel_plazo, color_plazo = "Riesgo medio", "#F2C66D"
        else:
            nivel_plazo, color_plazo = "Riesgo alto", "#D97B86"

        st.subheader("Resultado de la evaluación")
        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f'<div class="resultado"><h3>Componente de costo</h3><div class="puntaje" style="color:#D989A5">{puntaje_costo:.3f}</div><b>{nivel_costo}</b><p>Puntaje exploratorio para orientar la vigilancia contractual.</p></div>', unsafe_allow_html=True)
        with r2:
            st.markdown(f'<div class="resultado"><h3>Componente de plazo</h3><div class="puntaje" style="color:{color_plazo}">{puntaje_plazo:.3f}</div><b>{nivel_plazo}</b><p>Nivel preliminar para priorizar la revisión técnica.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="aviso"><b>Interpretación:</b> estos puntajes permiten ordenar y priorizar contratos. No deben interpretarse como probabilidades exactas ni como una decisión automática.</div>', unsafe_allow_html=True)

        reporte = pd.DataFrame([{
            "Fecha de evaluación": pd.Timestamp.now(tz="America/Lima").strftime("%d/%m/%Y %H:%M"),
            "Monto adjudicado del ítem (S/)": round(monto, 2),
            "Reducción en la adjudicación (%)": round(reduccion, 2),
            "Número de ofertantes": ofertantes,
            "Días entre convocatoria y buena pro": dias_buena_pro,
            "Días entre buena pro y consentimiento": dias_consentimiento,
            "Plazo contractual original (días)": plazo_original,
            "Año de inicio contractual": anio,
            "Nivel de gobierno": nivel,
            "Sistema de contratación": sistema,
            "Tipo de proceso de selección": proceso,
            "Tipo de proveedor": proveedor,
            "Provincia": provincia,
            "Puntaje del componente de costo": round(puntaje_costo, 3),
            "Clasificación del componente de costo": nivel_costo,
            "Puntaje del componente de plazo": round(puntaje_plazo, 3),
            "Clasificación del componente de plazo": nivel_plazo,
            "Nota de interpretación": (
                "Los resultados son puntajes preliminares de apoyo para ordenar y priorizar "
                "contratos; no son probabilidades exactas ni decisiones automáticas."
            ),
        }])
        archivo_csv = reporte.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Descargar resultado en CSV",
            data=archivo_csv,
            file_name="resultado_evaluacion_contractual.csv",
            mime="text/csv",
            use_container_width=True,
        )

with tab_resultados:
    st.subheader("Desempeño externo de los modelos seleccionados")
    tabla = pd.DataFrame({
        "Componente": ["Costo — Random Forest", "Plazo — Regresión logística"],
        "Exactitud balanceada": [0.615, 0.741],
        "Sensibilidad": [0.917, 0.703],
        "Especificidad": [0.313, 0.779],
        "ROC-AUC": [0.607, 0.850],
    })
    st.dataframe(tabla, hide_index=True, use_container_width=True)
    st.info("El componente de plazo mostró el desempeño más sólido. El componente de costo se conserva como herramienta exploratoria de vigilancia prioritaria.")
    imagen = DATA_DIR / "figura_02_metricas_modelos.png"
    if imagen.exists():
        st.image(str(imagen), caption="Resultados medios de la validación anidada", use_container_width=True)

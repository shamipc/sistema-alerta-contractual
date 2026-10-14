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
    :root {
        --azul-principal: #183153;
        --azul-secundario: #52789C;
        --texto: #243247;
        --fondo: #F5F7FB;
        --borde: #D8E0EA;
    }

    .stApp,
    [data-testid="stAppViewContainer"] {
        background: #F5F7FB !important;
        color: #243247 !important;
    }

    /* Encabezados y texto principal */
    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3,
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] label p {
        color: #243247 !important;
    }

    /* Encabezado principal */
    .hero {
        padding: 1.5rem 1.7rem;
        border-radius: 16px;
        background: linear-gradient(
            120deg, #183153, #355B82
        );
        border: 1px solid #29496D;
        margin-bottom: 1rem;
    }

    .hero h1,
    .hero p {
        color: #FFFFFF !important;
    }

    .hero h1 {
        margin: 0;
        font-size: 2rem;
    }

    .hero p {
        margin-top: 0.5rem;
        color: #E5EDF6 !important;
    }

    /* Formulario */
    div[data-testid="stForm"] {
        background: #FFFFFF;
        padding: 1.3rem;
        border-radius: 16px;
        border: 1px solid #D8E0EA;
    }

    [data-testid="stWidgetLabel"] p {
        color: #334155 !important;
        font-weight: 500;
    }

    /* Tarjetas de resultados */
    .resultado {
        padding: 1.3rem;
        border-radius: 16px;
        background: #FFFFFF;
        border: 1px solid #D8E0EA;
        box-shadow: 0 4px 14px rgba(24,49,83,.06);
        color: #243247 !important;
    }

    .resultado h3,
    .resultado b,
    .resultado p {
        color: #243247 !important;
    }

    .puntaje {
        font-size: 2.2rem;
        font-weight: 750;
        margin: 0.3rem 0;
    }

    /* Aviso técnico */
    .aviso {
        padding: 1rem;
        border-radius: 12px;
        background: #E8F0F8;
        border-left: 5px solid #52789C;
        color: #183153 !important;
    }

    .aviso b {
        color: #183153 !important;
    }

    /* Botón principal */
    .stButton button,
    div[data-testid="stFormSubmitButton"] button {
        background: #183153 !important;
        color: #FFFFFF !important;
        border: none;
        border-radius: 9px;
        font-weight: 650;
    }

    .stButton button:hover,
    div[data-testid="stFormSubmitButton"] button:hover {
        background: #2C527B !important;
        color: #FFFFFF !important;
    }

    /* Pestañas */
    button[data-baseweb="tab"] p {
        color: #475569 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] p {
        color: #183153 !important;
        font-weight: 700;
    }

    /* Barra lateral */
    [data-testid="stSidebar"] {
        background: #202A3B;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p {
        color: #F8FAFC !important;
    }
    </style>

    """,
    unsafe_allow_html=True,
)


# P2_TEMA_SOBRIO
st.markdown('<style>\n.stApp,[data-testid="stAppViewContainer"]{background:#F7F9FC!important;color:#1E2A38!important;}\n[data-testid="stFormSubmitButton"] button{background:#1F4E79!important;border:1px solid #1F4E79!important;min-height:48px!important;box-shadow:0 3px 12px #1F4E7933!important;}\n[data-testid="stFormSubmitButton"] button *{color:#FFFFFF!important;font-weight:700!important;font-size:1.02rem!important;}\n[data-testid="stFormSubmitButton"] button:hover{background:#355C8A!important;}\ndiv[data-baseweb="input"],div[data-baseweb="input"]>div,div[data-baseweb="select"]>div{background:#FFFFFF!important;color:#1E2A38!important;border-color:#CBD5E1!important;}\ndiv[data-baseweb="input"] input,div[data-baseweb="select"] input,div[data-baseweb="select"] span{color:#1E2A38!important;background:transparent!important;}\n[data-testid="stExpander"] summary,[data-testid="stExpander"] summary *{background:#E8EEF5!important;color:#183153!important;}\n[data-testid="stSidebar"] [data-testid="stMetricLabel"] *{color:#FFFFFF!important;}\n[data-baseweb="tab-highlight"]{background:#1F4E79!important;}\n</style>', unsafe_allow_html=True)

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

# P2_NOMBRES_CLAROS
NOMBRES_FACTORES = {'monto_adjudicado_item_soles': 'Monto adjudicado (S/)', 'reduccion_adjudicacion_pct': 'Reducción de adjudicación (%)', 'numero_ofertantes': 'Número de ofertantes', 'dias_convocatoria_buenapro_limpio': 'Días: convocatoria a buena pro', 'dias_buenapro_consentimiento_limpio': 'Días: buena pro a consentimiento', 'plazo_contractual_original_dias': 'Plazo contractual inicial (días)', 'anio_inicio_contractual': 'Año de inicio contractual', 'nivel_gobierno_modelo': 'Nivel de gobierno', 'sistema_contratacion': 'Sistema de contratación', 'tipo_proceso_seleccion': 'Procedimiento de selección', 'tipo_proveedor': 'Tipo de proveedor', 'provincia': 'Provincia'}

# P2_BOTON_DESCARGA_VISIBLE
st.markdown('<style>\n[data-testid="stDownloadButton"] button {\n  background:#1F4E79 !important;\n  border:1px solid #1F4E79 !important;\n  border-radius:10px !important;\n  min-height:48px !important;\n  box-shadow:0 3px 10px #1F4E7926 !important;\n}\n[data-testid="stDownloadButton"] button,\n[data-testid="stDownloadButton"] button * {\n  color:#FFFFFF !important;\n  font-weight:700 !important;\n}\n[data-testid="stDownloadButton"] button:hover {\n  background:#355C8A !important;\n  border-color:#355C8A !important;\n}\n</style>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Acerca del sistema")
    st.write("Apoyo exploratorio a la vigilancia de contratos de infraestructura educativa.")
    st.caption("Cobertura histórica: Lima, 2019–2024")
    st.metric("Base histórica", "92 contratos")
    st.caption("Puntajes de priorización, no probabilidades calibradas. No reemplazan la evaluación técnica.")

tab_evaluacion, tab_resultados = st.tabs(["Evaluar contrato", "Resultados preliminares"])

with tab_evaluacion:
    st.subheader("Datos disponibles al inicio del contrato")
    st.caption(
        "Puede escribir directamente los valores numéricos: "
        "seleccione el número y reemplácelo. Para el monto, "
        "use un punto decimal y no coloque separadores de miles. "
        "Los botones + y − son opcionales."
    )

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
            anios_historicos = sorted(
                base[variables[6]].dropna().astype(int).unique(),
                reverse=True
            )
            anios_disponibles = sorted(
                set(anios_historicos) | {2025, 2026},
                reverse=True
            )
            anio = st.selectbox(
                "Año de inicio contractual",
                anios_disponibles,
                index=anios_disponibles.index(max(anios_historicos)),
                help=(
                    "2019–2024 corresponde al periodo de entrenamiento. "
                    "2025 y 2026 permiten una evaluación exploratoria "
                    "fuera de ese periodo."
                )
            )
            nivel = st.selectbox("Nivel de gobierno", sorted(base[variables[7]].dropna().unique()))
        with c3:
            sistema = st.selectbox("Sistema de contratación", sorted(base[variables[8]].dropna().unique()))
            proceso = st.selectbox("Tipo de proceso de selección", sorted(base[variables[9]].dropna().unique()))
            proveedor = st.selectbox("Tipo de proveedor", sorted(base[variables[10]].dropna().unique()))
            provincia = st.selectbox("Provincia", sorted(base[variables[11]].dropna().unique()))
        evaluar = st.form_submit_button("Evaluar contrato", use_container_width=True)

    if evaluar:
        # Comprobar cobertura temporal sin sustituir el año ingresado.
        anios_entrenamiento = set(
            base[variables[6]].dropna().astype(int)
        )

        if int(anio) not in anios_entrenamiento:
            st.warning(
                f"Evaluación fuera del periodo de entrenamiento: "
                f"el contrato corresponde a {int(anio)}, mientras "
                "que la base histórica utilizada abarca 2019–2024. "
                "Se conserva el año ingresado. El desempeño para "
                "este año no está validado y el resultado debe "
                "interpretarse como exploratorio."
            )
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
            nivel_plazo, color_plazo = "Riesgo bajo", "#247A4C"
        elif puntaje_plazo < 0.65:
            nivel_plazo, color_plazo = "Riesgo medio", "#A76A0C"
        else:
            nivel_plazo, color_plazo = "Riesgo alto", "#D97B86"

        st.subheader("Resultado de la evaluación")
        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f'<div class="resultado"><h3>Componente de costo</h3><div class="puntaje" style="color:#183153">{puntaje_costo:.3f}</div><b>{nivel_costo}</b><p>Puntaje exploratorio para orientar la vigilancia contractual.</p></div>', unsafe_allow_html=True)
        with r2:
            st.markdown(f'<div class="resultado"><h3>Componente de plazo</h3><div class="puntaje" style="color:{color_plazo}">{puntaje_plazo:.3f}</div><b>{nivel_plazo}</b><p>Nivel preliminar para priorizar la revisión técnica.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="aviso"><b>Interpretación:</b> estos puntajes permiten ordenar y priorizar contratos. No deben interpretarse como probabilidades exactas ni como una decisión automática.</div>', unsafe_allow_html=True)



        # P2_BARRAS_ALERTA
        def mostrar_banda_alerta(puntaje, componente):
            posicion = min(100, max(0, puntaje * 100))

            if componente == "Costo":
                fondo = (
                    "linear-gradient(to right, "
                    "#F3D480 0%, #F3D480 55%, "
                    "#D98A8A 55%, #D98A8A 100%)"
                )
                cortes = (
                    '<span style="position:absolute;left:55%;'
                    'transform:translateX(-50%)">0,55</span>'
                )
                leyenda = (
                    "Amarillo: vigilancia estándar · "
                    "Rojo: vigilancia prioritaria"
                )
            else:
                fondo = (
                    "linear-gradient(to right, "
                    "#88BDA3 0%, #88BDA3 35%, "
                    "#F3D480 35%, #F3D480 65%, "
                    "#D98A8A 65%, #D98A8A 100%)"
                )
                cortes = (
                    '<span style="position:absolute;left:35%;'
                    'transform:translateX(-50%)">0,35</span>'
                    '<span style="position:absolute;left:65%;'
                    'transform:translateX(-50%)">0,65</span>'
                )
                leyenda = (
                    "Verde: bajo · Amarillo: medio · Rojo: alto"
                )

            st.markdown(
                f"""
                <div style="background:white;padding:16px;
                            border:1px solid #D9E3EE;
                            border-radius:12px;margin:8px 0">
                    <div style="color:#173554;font-weight:600">
                        Ubicación del puntaje de {componente.lower()}
                    </div>
                    <div style="position:relative;margin-top:22px;
                                height:20px;border-radius:6px;
                                background:{fondo}">
                        <div style="position:absolute;left:{posicion}%;
                                    top:-10px;transform:translateX(-50%);
                                    color:#173554;font-size:18px;
                                    line-height:18px">▼</div>
                        <div style="position:absolute;left:{posicion}%;
                                    top:0;height:20px;width:2px;
                                    background:#173554"></div>
                    </div>
                    <div style="position:relative;height:24px;
                                margin-top:5px;font-size:12px;
                                color:#42566C">
                        <span style="position:absolute;left:0">0</span>
                        {cortes}
                        <span style="position:absolute;right:0">1</span>
                    </div>
                    <div style="color:#42566C;font-size:13px">
                        {leyenda}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        barra_costo, barra_plazo = st.columns(2)

        with barra_costo:
            mostrar_banda_alerta(puntaje_costo, "Costo")

        with barra_plazo:
            mostrar_banda_alerta(puntaje_plazo, "Plazo")

        st.caption(
            "El marcador muestra el puntaje del contrato; los colores "
            "representan las reglas operativas del prototipo. "
            "Los cortes son exploratorios y no convierten el puntaje "
            "en una probabilidad calibrada."
        )

        # P2: sensibilidad local de los 12 predictores
        filas = []

        for variable in variables:
            serie = base[variable].dropna()

            if serie.empty:
                continue

            if pd.api.types.is_numeric_dtype(serie):
                referencia = float(serie.median())
            else:
                referencia = serie.mode().iloc[0]

            alternativo = registro.copy()
            alternativo.loc[
                alternativo.index[0], variable
            ] = referencia

            costo_ref = float(
                modelo_costo.predict_proba(alternativo)[0, 1]
            )
            plazo_ref = float(
                modelo_plazo.predict_proba(alternativo)[0, 1]
            )

            filas.append({
                "Factor": variable,
                "Valor ingresado": str(registro.iloc[0][variable]),
                "Referencia": str(referencia),
                "Cambio costo": puntaje_costo - costo_ref,
                "Cambio plazo": puntaje_plazo - plazo_ref
            })

        with st.expander(
            "Cómo cambia el puntaje al comparar cada dato con una referencia",
            expanded=True
        ):
            st.caption(
                "Comparamos un dato a la vez con la mediana numérica o la "
                "categoría más frecuente de la base histórica, "
                "manteniendo los demás datos. La diferencia es el "
                "puntaje original menos el obtenido con la referencia. "
                "Un signo + indica un puntaje original mayor y un signo "
                "− uno menor. Son diferencias del modelo, no causas "
                "del cambio contractual ni porcentajes de probabilidad."
            )

            sensibilidad = pd.DataFrame(filas)

            col_c, col_p = st.columns(2)

            for columna, nombre, metrica in [
                (col_c, "Costo", "Cambio costo"),
                (col_p, "Plazo", "Cambio plazo")
            ]:
                with columna:
                    st.markdown(
                        f"**Comparaciones que más cambian el puntaje de {nombre.lower()}**"
                    )

                    tabla = (
                        sensibilidad
                        .assign(
                            magnitud=lambda x: x[metrica].abs()
                        )
                        .sort_values(
                            "magnitud",
                            ascending=False
                        )
                        .head(6)
                    )

                    tabla = tabla[
                        ["Factor", "Valor ingresado", "Referencia", metrica]
                    ].rename(
                        columns={
                            metrica: "Diferencia del puntaje"
                        }
                    )

                    # Formatear valores únicamente para su presentación.
                    def mostrar_valor(valor):
                        try:
                            numero = float(valor)
                            if numero.is_integer():
                                return str(int(numero))
                            return f"{numero:,.2f}".rstrip("0").rstrip(".")
                        except (ValueError, TypeError):
                            return str(valor)

                    tabla_visible = tabla.copy()
                    tabla_visible["Factor"] = (
                        tabla_visible["Factor"].replace(NOMBRES_FACTORES)
                    )

                    for campo in ["Valor ingresado", "Referencia"]:
                        tabla_visible[campo] = (
                            tabla_visible[campo].map(mostrar_valor)
                        )

                    tabla_visible["Lectura"] = (
                        tabla_visible["Diferencia del puntaje"].map(
                            lambda diferencia:
                                "Cambio menor de 0,001"
                                if abs(diferencia) < 0.001
                                else (
                                    "Mayor que con la referencia"
                                    if diferencia > 0
                                    else "Menor que con la referencia"
                                )
                        )
                    )

                    tabla_visible["Diferencia del puntaje"] = (
                        tabla_visible["Diferencia del puntaje"].map(
                            lambda valor: f"{valor:+.3f}"
                        )
                    )

                    # HTML permite controlar el fondo de estas tablas.
                    st.markdown(
                        """
                        <style>
                        .tabla-factores-p2 {
                            overflow-x: auto;
                            margin-bottom: 14px;
                        }
                        .tabla-factores-p2 table {
                            width: 100%;
                            border-collapse: collapse;
                            background: #FFFFFF;
                            color: #24364B;
                            font-size: 0.85rem;
                        }
                        .tabla-factores-p2 th {
                            background: #E8EFF7;
                            color: #173554;
                            text-align: left;
                            padding: 10px;
                            border-bottom: 2px solid #CCD9E7;
                        }
                        .tabla-factores-p2 td {
                            padding: 10px;
                            border-bottom: 1px solid #E2E8F0;
                        }
                        .tabla-factores-p2 tbody tr:nth-child(even) {
                            background: #F5F8FC;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )

                    html_tabla = tabla_visible.to_html(
                        index=False,
                        escape=True,
                        border=0
                    )
                    st.markdown(
                        '<div class="tabla-factores-p2">'
                        + html_tabla
                        + '</div>',
                        unsafe_allow_html=True
                    )

                    # Explicar la comparación de mayor magnitud.
                    if not tabla.empty:
                        fila = tabla.iloc[0]
                        diferencia = float(fila["Diferencia del puntaje"])
                        puntaje_original = (
                            puntaje_costo
                            if nombre == "Costo"
                            else puntaje_plazo
                        )
                        puntaje_referencia = puntaje_original - diferencia
                        factor = NOMBRES_FACTORES.get(
                            fila["Factor"], fila["Factor"]
                        )

                        st.info(
                            f"Ejemplo — {factor}: el dato ingresado es "
                            f"{mostrar_valor(fila['Valor ingresado'])} "
                            f"y la referencia histórica es "
                            f"{mostrar_valor(fila['Referencia'])}. "
                            f"Al sustituir únicamente ese dato, el puntaje "
                            f"de {nombre.lower()} pasa de "
                            f"{puntaje_original:.3f} a "
                            f"{puntaje_referencia:.3f}. "
                            "Los demás datos permanecen iguales."
                        )

                    st.caption(
                        "Se muestran las seis comparaciones de mayor "
                        "magnitud. La referencia representa un valor "
                        "habitual de la base, no una condición ideal. "
                        "Las diferencias no se suman ni demuestran "
                        "las causas de una desviación. Una mediana "
                        "puede ser decimal aunque la variable sea un conteo."
                    )



        # P2_COBERTURA_DEL_CASO
        with st.expander(
            "Cobertura de categorías en los 92 contratos de entrenamiento",
            expanded=True
        ):
            st.info(
                "Los modelos finales se entrenaron con la base "
                "completa de 92 contratos y 12 predictores. "
                "Los conteos siguientes describen cada categoría "
                "por separado: no indican cuántos contratos se "
                "utilizan para calcular esta predicción, ni son "
                "una búsqueda de casos similares o una medida "
                "de confianza."
            )

            campos_cobertura = {
                "provincia": "Provincia",
                "sistema_contratacion": "Sistema de contratación",
                "tipo_proceso_seleccion": "Proceso de selección",
                "tipo_proveedor": "Tipo de proveedor"
            }

            filas_cobertura = []

            for campo, nombre_campo in campos_cobertura.items():
                valor_ingresado = registro.iloc[0][campo]

                cantidad = int(
                    base[campo].eq(valor_ingresado).sum()
                )

                filas_cobertura.append({
                    "Dato": nombre_campo,
                    "Categoría ingresada": str(valor_ingresado),
                    "Registros de esta categoría": cantidad
                })

            tabla_cobertura = pd.DataFrame(filas_cobertura)

            st.markdown(
                '<div class="tabla-factores-p2">'
                + tabla_cobertura.to_html(
                    index=False,
                    escape=True,
                    border=0
                )
                + '</div>',
                unsafe_allow_html=True
            )

            categorias_escasas = [
                fila["Dato"]
                for fila in filas_cobertura
                if fila["Registros de esta categoría"] < 5
            ]

            if categorias_escasas:
                st.warning(
                    "Hay menos de cinco contratos históricos para "
                    "estas categorías: "
                    + ", ".join(categorias_escasas)
                    + ". El desempeño específico para ellas "
                    "no se ha establecido."
                )

            st.caption(
                "Cada conteo corresponde a una categoría por separado, "
                "no a la combinación completa del contrato. "
                "La marca de menos de cinco registros es descriptiva: "
                "no establece un mínimo estadístico suficiente. "
                "Tener más registros tampoco garantiza una "
                "predicción correcta."
            )

            st.caption(
                "Cobertura territorial: la muestra representa nueve "
                "provincias del departamento de Lima. Canta no tiene "
                "registros en la base final de entrenamiento "
                "de esta versión."
            )

        # P2_ORIENTACION_REVISION
        st.subheader("Interpretación y siguiente revisión")

        st.caption(
            "Orientaciones generales para apoyar la revisión técnica "
            "al inicio contractual, cuando los datos solicitados están "
            "disponibles. No son decisiones automáticas ni diagnósticos."
        )

        orientacion_costo, orientacion_plazo = st.columns(2)

        with orientacion_costo:
            st.markdown("**Costo — ¿qué significa y qué revisar?**")

            if puntaje_costo >= 0.55:
                st.warning(
                    f"Vigilancia prioritaria: el puntaje "
                    f"{puntaje_costo:.3f} alcanza o supera "
                    "el corte exploratorio de 0,55."
                )
                st.markdown(
                    "- Priorizar la revisión del presupuesto, "
                    "metrados y alcance contractual.\n"
                    "- Contrastar la consistencia entre planos, "
                    "especificaciones y presupuesto.\n"
                    "- Revisar el sustento y procedimiento aplicable "
                    "a posibles modificaciones contractuales."
                )
            else:
                st.info(
                    f"Vigilancia estándar: el puntaje "
                    f"{puntaje_costo:.3f} es inferior a 0,55. "
                    "Esto no garantiza ausencia de incrementos."
                )
                st.markdown(
                    "- Mantener la revisión habitual del presupuesto "
                    "y del alcance contractual.\n"
                    "- Documentar posibles cambios y su sustento.\n"
                    "- Complementar el resultado con la evaluación "
                    "del responsable técnico."
                )

            st.caption(
                "Evidencia histórica: exactitud balanceada media "
                "de 0,615 en validación agrupada anidada. "
                "El componente de costo presenta discriminación "
                "limitada y debe utilizarse con cautela."
            )

        with orientacion_plazo:
            st.markdown("**Plazo — ¿qué significa y qué revisar?**")

            if puntaje_plazo < 0.35:
                st.info(
                    f"Nivel bajo: el puntaje {puntaje_plazo:.3f} "
                    "es inferior a 0,35. No garantiza que el "
                    "contrato termine sin ampliaciones."
                )
                st.markdown(
                    "- Mantener el seguimiento del cronograma "
                    "y de los hitos contractuales.\n"
                    "- Verificar las condiciones previstas "
                    "para iniciar y ejecutar la obra."
                )

            elif puntaje_plazo < 0.65:
                st.warning(
                    f"Nivel medio: el puntaje {puntaje_plazo:.3f} "
                    "se encuentra entre 0,35 y menos de 0,65."
                )
                st.markdown(
                    "- Revisar la coherencia del cronograma "
                    "con el alcance de la obra.\n"
                    "- Identificar actividades críticas, "
                    "dependencias y suministros relevantes.\n"
                    "- Verificar disponibilidad del terreno, "
                    "permisos y otras condiciones de ejecución."
                )

            else:
                st.error(
                    f"Nivel alto: el puntaje {puntaje_plazo:.3f} "
                    "alcanza o supera 0,65."
                )
                st.markdown(
                    "- Priorizar la revisión del cronograma "
                    "y de las actividades críticas.\n"
                    "- Revisar restricciones de ejecución, "
                    "suministros y disponibilidad de recursos.\n"
                    "- Definir responsables y seguimiento "
                    "de los hitos que requieran mayor atención."
                )

            st.caption(
                "Evidencia histórica: exactitud balanceada media "
                "de 0,741 en validación agrupada anidada. "
                "El puntaje no estima la cantidad de días "
                "de una posible ampliación."
            )

        with st.expander("Alcance de estas orientaciones"):
            st.markdown(
                "- Las recomendaciones son pautas generales "
                "de revisión; no se deducen como causas "
                "del resultado del modelo.\n"
                "- La evaluación utiliza una base histórica "
                "de 92 contratos de infraestructura educativa "
                "del departamento de Lima.\n"
                "- Los puntajes no son probabilidades calibradas.\n"
                "- Los cortes se examinaron sobre la muestra "
                "histórica y requieren validación independiente.\n"
                "- El sistema no verifica por sí mismo documentos "
                "ni condiciones físicas de la obra.\n"
                "- Su uso operativo requiere una prueba piloto "
                "con nuevos contratos y usuarios de una entidad."
            )

        # P2_DESCARGA_CSV
        reporte = registro.iloc[0].to_dict()
        reporte.update({
            "puntaje_costo": round(puntaje_costo, 6),
            "nivel_costo": nivel_costo,
            "puntaje_plazo": round(puntaje_plazo, 6),
            "nivel_plazo": nivel_plazo,
            "nota": "Puntajes exploratorios; no son probabilidades calibradas.",
        })
        csv_reporte = pd.DataFrame([reporte]).to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Descargar evaluación contractual (CSV)",
            data=csv_reporte,
            file_name="evaluacion_contractual.csv",
            mime="text/csv",
            use_container_width=True,
            on_click="ignore",
        )

# P2_RESULTADOS_PRESENTACION
st.markdown(
    """
    <style>
    header[data-testid="stHeader"] {
        background: #F7F9FC !important;
    }
    header[data-testid="stHeader"] button {
        color: #173554 !important;
    }
    .hero {
        padding: 20px 24px !important;
    }
    .hero h1 {
        font-size: 1.65rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with tab_resultados:
    st.subheader("Resultados medios de validación agrupada anidada")
    tabla = pd.DataFrame({
        "Componente": ["Costo — Random Forest", "Plazo — Regresión logística"],
        "Exactitud balanceada": [0.615, 0.741],
        "Sensibilidad": [0.917, 0.703],
        "Especificidad": [0.313, 0.779],
        "ROC-AUC": [0.607, 0.850],
    })
    # Tabla clara con las cifras originales.
    tabla_visible = tabla.copy()

    for campo in tabla_visible.columns:
        if campo != "Componente":
            tabla_visible[campo] = tabla_visible[campo].map(
                lambda valor: f"{valor:.3f}"
            )

    st.markdown(
        """
        <style>
        .tabla-metricas-p2 {
            overflow-x: auto;
            margin-bottom: 16px;
        }
        .tabla-metricas-p2 table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            color: #24364B;
            font-size: 0.95rem;
        }
        .tabla-metricas-p2 th {
            background: #E8EFF7;
            color: #173554;
            padding: 12px;
            text-align: left;
        }
        .tabla-metricas-p2 td {
            padding: 12px;
            border-bottom: 1px solid #DFE7F0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="tabla-metricas-p2">'
        + tabla_visible.to_html(
            index=False, escape=True, border=0
        )
        + '</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Medias de las cinco particiones externas de validación "
        "agrupada anidada, con corte de clasificación 0,50. "
        "No son una evaluación independiente de los cortes "
        "operativos exploratorios del formulario."
    )

    with st.expander("Cómo interpretar estas métricas", expanded=True):
        st.markdown(
            "- **Sensibilidad:** proporción de casos positivos "
            "que el modelo identifica.\n"
            "- **Especificidad:** proporción de casos negativos "
            "que el modelo reconoce.\n"
            "- **Exactitud balanceada:** promedio de sensibilidad "
            "y especificidad; permite considerar ambas clases.\n"
            "- **ROC-AUC:** capacidad de distinguir las clases "
            "a través de distintos cortes. No es un porcentaje "
            "de predicciones correctas."
        )

        st.markdown(
            "**Costo:** presenta sensibilidad elevada y "
            "especificidad baja. Detecta muchos positivos, "
            "pero genera numerosas falsas alertas.\n\n"
            "**Plazo:** presenta un mejor equilibrio entre "
            "sensibilidad y especificidad. Estas medias no "
            "garantizan el desempeño en contratos nuevos."
        )
    st.info("El componente de plazo mostró el desempeño más sólido. El componente de costo se conserva como herramienta exploratoria de vigilancia prioritaria.")
    imagen = DATA_DIR / "figura_02_metricas_modelos.png"
    if imagen.exists():
        margen_izquierdo, columna_figura, margen_derecho = (
            st.columns([1, 4, 1])
        )
        with columna_figura:
            st.image(
                str(imagen),
                caption="Resultados medios de la validación anidada",
                use_container_width=True
            )


# P2_REFERENCIAS_FACTORES

# P2_FACTORES_LECTURA_CLARA

# P2_ANIOS_Y_AYUDAS

# P2_ACLARACION_COBERTURA

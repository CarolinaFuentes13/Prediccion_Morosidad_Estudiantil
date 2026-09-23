"""Constantes, rutas y parámetros de negocio del Sistema de Alertas Tempranas (SAT).

Todo lo que viene del "Reporte de implementación y experimentos" (umbrales,
requerimientos, resultados de calidad, métricas reportadas en MLflow) vive aquí,
para que las páginas no repitan cifras sueltas.
"""
from __future__ import annotations

from pathlib import Path

# ================= Rutas =================
APP_DIR = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = APP_DIR.parent
REPO_DIR = DASHBOARD_DIR.parent
ASSETS_DIR = DASHBOARD_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "logo_uni.png"
APP_DATA_DIR = APP_DIR / "data"
MODEL_ARTIFACTS_PATH = APP_DATA_DIR / "model_artifacts.json"

PRED_DIR = REPO_DIR / "Database" / "Data_model_predictions"
BASE_DATA_PATH = PRED_DIR / "df_dash_with_preds.csv"
TEST_PRED_FILES = {
    "rf": PRED_DIR / "df_con_predicciones_TEST_modelo1_RF.csv",
    "xgb": PRED_DIR / "df_con_predicciones_TEST_modelo2_XGB.csv",
    "lgbm": PRED_DIR / "df_con_predicciones_TEST_modelo3_LGBM.csv",
    "psa_xgb": PRED_DIR / "df_con_predicciones_TEST_modelo4_PSA_XGB.csv",
    "boruta_xgb": PRED_DIR / "df_con_predicciones_TEST_modelo5_BORUTA_XGB.csv",
}

# ================= Identidad =================
APP_NAME = "SAT · Morosidad Estudiantil"
APP_TITLE = "Señales tempranas de riesgo de incumplimiento en créditos estudiantiles"
APP_TAGLINE = (
    "Sistema de alertas tempranas que convierte el modelo predictivo en decisiones "
    "de gestión preventiva de cartera."
)
UNIVERSIDAD = "Universidad de los Andes"
PROGRAMA_ACADEMICO = "Maestría en Inteligencia Analítica de Datos"
EQUIPO = [
    "María Fernanda González",
    "Carolina Fuentes Pulido",
    "Juan Camilo Romero",
    "Renato Patiño Fierro",
    "Rafael Negrete",
]
REPO_URL = "https://github.com/Juan-crss/Prediccion_Morosidad_Estudiantil"

# ================= Riesgo =================
RISK_ORDER = ["Alto", "Medio", "Bajo"]
RISK_COLORS = {"Alto": "#E5484D", "Medio": "#F5A524", "Bajo": "#30A46C"}
RISK_ICONS = {"Alto": "🔴", "Medio": "🟠", "Bajo": "🟢"}
# Bandas de la variable continua Y acordadas con Cartera (Tabla 7 del reporte).
RISK_BANDS_TEXT = {
    "Bajo": "Y < 0,20 — comportamiento de pago sano",
    "Medio": "0,20 ≤ Y < 0,25 — señales de deterioro",
    "Alto": "Y ≥ 0,25 — incumplimiento relevante de cuotas vencidas",
}

# ================= Umbrales de negocio (reporte) =================
AUC_RETRAIN_THRESHOLD = 0.80   # ME-01: disparador de reentrenamiento
RECALL_ALTO_TARGET = 0.75      # DE-03: recall comprometido para la clase Alto
RECAUDO_BASELINE = 0.85        # Indicador institucional de Recaudo de Cartera
SATISFACCION_TARGET = 0.80     # EV-03
MAX_FINANCIACION_PCT = 0.85    # DA-02: financiación ≤ 85 % de Vr_neto_matricula

# ================= Modelos =================
MODEL_LABELS = {
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
    "psa_xgb": "PSA + XGBoost",
    "boruta_xgb": "Boruta + XGBoost",
}
SELECTED_MODEL = "rf"
# Métricas de la versión optimizada del RF en Databricks + MLflow (conclusiones del reporte).
MLFLOW_REPORTED = {
    "rf": {"auc_range": (0.82, 0.83), "accuracy": 0.873, "auc_macro": 0.822, "precision_macro": 0.478,
           "recall_macro_max": 0.57},
    "xgb": {"auc_range": (0.76, 0.81), "accuracy": 0.89, "precision_range": (0.66, 0.85), "recall_macro": 0.355},
    "lgbm": {"auc_range": (0.77, 0.79), "accuracy": 0.89, "precision_range": (0.54, 0.64), "recall_macro": 0.35},
}
# Hiperparámetros del modelo empaquetado (model_morosidad-0.0.1, config.yml).
PACKAGED_MODEL = {
    "version": "0.0.1",
    "algoritmo": "RandomForestClassifier",
    "hiperparametros": {
        "n_estimators": 457, "max_depth": 22, "max_features": 0.5, "min_samples_leaf": 5,
        "bootstrap": True, "class_weight": "balanced", "random_state": 42,
    },
    "preprocesamiento": "SimpleImputer (mediana / moda) + OrdinalEncoder (desconocidos = -1)",
    "particion": "Estratificada 80/20, random_state = 42",
}
BORUTA_FEATURES = [
    "Tipo_interes", "Antiguedad_meses", "Cuotas", "TIPOESTUDIANTE", "Fecha_aprobacion", "Mes", "AñoB",
    "Cohorte", "CARGA", "Valor_financiacion", "Valor_primera_cuota", "Valor_cuota_inicial",
    "Validación valor financiado", "PROGRAMA", "FACULTAD", "Vr_neto_matricula",
]
# Variables con comportamiento de identificador: su importancia alta es una alerta de gobierno.
ID_LIKE_FEATURES = {"llave2", "fecha_nacimiento", "fecha_aprobacion", "id_estudiante"}

# Variables que consume el modelo empaquetado (orden de config.yml).
MODEL_FEATURES = [
    "llave2", "nombre_linea", "fecha_aprobacion", "antiguedad_meses", "nombre_fondo", "valor_financiacion",
    "cuotas", "tipo_interes", "vr_neto_matricula", "fecha_nacimiento", "estado_civil", "genero", "facultad",
    "programa", "nivel", "estado", "tipoestudiante", "sede", "sello", "carga", "grupo_etnico",
    "tipo_discapacidad", "nacionalidad", "mora", "valor_cuota_inicial", "valor_primera_cuota",
    "fecha_de_pago", "validacion_valor_financiado", "detalle_estado_final", "tipo_estudiante", "operacion",
    "cate", "subcate", "cohorte", "mes", "cliente", "media_score", "anob", "valor_maximo", "valor_medio",
    "valor_bajo", "plataforma",
]
NUMERIC_FEATURES = [
    "antiguedad_meses", "valor_financiacion", "cuotas", "vr_neto_matricula", "valor_cuota_inicial",
    "valor_primera_cuota", "fecha_de_pago", "validacion_valor_financiado", "anob", "valor_maximo",
    "valor_medio", "valor_bajo",
]

# ================= Calidad de datos (reporte, Tablas 3–6) =================
QUALITY_SOURCES = [
    {"fuente": "CRq · Créditos Financieros", "area": "Crédito y Cartera", "campos": 13, "reglas": 23,
     "completitud": 1.0, "exactitud": 1.0, "consistencia": 1.0, "validez": 0.998, "global": 0.999,
     "hallazgo": "812 de 71.493 registros incumplen financiación ≤ 85 % de Vr_neto_matricula (DA-02)."},
    {"fuente": "CCq · Cartera edades cliente", "area": "Crédito y Cartera", "campos": 13, "reglas": 26,
     "completitud": 1.0, "exactitud": None, "consistencia": 1.0, "validez": 1.0, "global": 1.0,
     "hallazgo": "Cumple el 100 % de las reglas definidas."},
    {"fuente": "Matriz total · Scoring de riesgo", "area": "Crédito y Cartera", "campos": 26, "reglas": 44,
     "completitud": 0.964, "exactitud": 1.0, "consistencia": None, "validez": 0.957, "global": 0.962,
     "hallazgo": "Incidencias en cuota inicial, primera cuota, fecha de pago, ciudad y Datacrédito."},
    {"fuente": "MAFI · Registro y Control", "area": "Registro y Control", "campos": 19, "reglas": 27,
     "completitud": 0.968, "exactitud": 0.999, "consistencia": None, "validez": 0.896, "global": 0.914,
     "hallazgo": "Ninguna regla al 100 %: requiere limpieza adicional antes de modelar."},
]
QUALITY_GLOBAL = 0.968
QUALITY_TOTALS = {"reglas_validacion": 147, "transformaciones": 75, "variables_origen": 212,
                  "variables_modelo": 41, "variables_numericas": 7, "variables_categoricas": 34}

# ================= Requerimientos (trazabilidad) =================
# estado: "tablero" = resuelto en esta versión del tablero; "parcial"; "pendiente" (fuera del tablero).
REQUIREMENTS = [
    {"codigo": "DB-01", "criticidad": "Alta", "descripcion": "Filtros operativos sobre todo el tablero",
     "criterio": "100 % de casos de prueba de filtros superados", "estado": "tablero",
     "donde": "Barra lateral · filtros globales persistentes y compartibles por URL"},
    {"codigo": "DB-02", "criticidad": "Media", "descripcion": "Exportación de resultados",
     "criterio": "100 % de éxito en pruebas de exportación", "estado": "tablero",
     "donde": "Resumen, Cola de gestión, Segmentos y Carga y predicción (CSV / Excel / HTML)"},
    {"codigo": "DB-03", "criticidad": "Alta", "descripcion": "KPI exigidos en la vista de resumen",
     "criterio": "100 % de los KPI presentes", "estado": "tablero", "donde": "Resumen ejecutivo"},
    {"codigo": "DB-04", "criticidad": "Media", "descripcion": "Integración con el sistema de gestión de cartera",
     "criterio": "≥ 90 % de casos de prueba de integración", "estado": "parcial",
     "donde": "Cola de gestión · archivo de intercambio (CSV/JSON) con esquema fijo"},
    {"codigo": "DE-01", "criticidad": "Alta", "descripcion": "Automatización ETL + predicción",
     "criterio": "Al menos un ciclo probado", "estado": "parcial",
     "donde": "Carga y predicción: CSV → validación → API/modelo → tablero"},
    {"codigo": "DE-03", "criticidad": "Alta", "descripcion": "Ajuste de umbral para recall de Alto ≥ 0,75",
     "criterio": "Recall clase Alto ≥ 0,75", "estado": "tablero",
     "donde": "Desempeño del modelo · umbral de alerta"},
    {"codigo": "MO-02", "criticidad": "Alta", "descripcion": "F1-Score macro del modelo final",
     "criterio": "Métrica registrada", "estado": "tablero", "donde": "Desempeño del modelo"},
    {"codigo": "EV-02", "criticidad": "Alta", "descripcion": "Reporte de desempeño por clase",
     "criterio": "Reporte por clase disponible", "estado": "tablero", "donde": "Desempeño del modelo"},
    {"codigo": "MO-03", "criticidad": "Media", "descripcion": "Top 10 de variables validado con asesor",
     "criterio": "Acta de validación", "estado": "parcial",
     "donde": "Desempeño del modelo · importancia (falta el acta)"},
    {"codigo": "ME-01", "criticidad": "Alta", "descripcion": "Disparador de reentrenamiento si AUC < 0,80",
     "criterio": "Alerta automática", "estado": "parcial",
     "donde": "AUC por semestre precalculado (data/model_artifacts.json); vista de monitoreo fuera de esta versión"},
    {"codigo": "ME-02", "criticidad": "Media", "descripcion": "Controles de calidad automáticos a la entrada",
     "criterio": "Reglas ejecutadas en cada carga", "estado": "tablero",
     "donde": "Carga y predicción · reglas de calidad en cada archivo"},
    {"codigo": "ME-03", "criticidad": "Baja", "descripcion": "Reportes periódicos automáticos",
     "criterio": "Reporte generado", "estado": "parcial", "donde": "Resumen ejecutivo · reporte descargable"},
    {"codigo": "DA-02", "criticidad": "Media", "descripcion": "Remediación registros con financiación > 85 %",
     "criterio": "Registros remediados", "estado": "parcial",
     "donde": "Carga y predicción · regla VA-02 detecta los casos"},
    {"codigo": "EV-03", "criticidad": "Alta", "descripcion": "Validación con experto (satisfacción ≥ 80 %)",
     "criterio": "≥ 1 experto con satisfacción ≥ 80 %", "estado": "pendiente", "donde": "Sesión con usuarios"},
    {"codigo": "SEC", "criticidad": "Alta", "descripcion": "Control de acceso y autenticación del tablero",
     "criterio": "Acceso restringido por usuario", "estado": "tablero",
     "donde": "Inicio de sesión con roles (st.secrets)"},
]

# ================= Rutas de gestión (cola de cobro preventivo) =================
ACTION_ROUTES = {
    "R1": {"nombre": "Contacto inmediato", "color": "#E5484D", "sla": "48 h",
           "accion": "Llamada del gestor + acuerdo de pago preventivo antes del próximo vencimiento."},
    "R2": {"nombre": "Acompañamiento preventivo", "color": "#F5A524", "sla": "5 días",
           "accion": "WhatsApp/SMS y correo previos a la fecha de pago + consejería financiera."},
    "R3": {"nombre": "Recordatorio automático", "color": "#3E63DD", "sla": "Automático",
           "accion": "Recordatorio automatizado 3 días antes de la fecha de pago."},
    "R4": {"nombre": "Monitoreo estándar", "color": "#30A46C", "sla": "Mensual",
           "accion": "Seguimiento regular; sin intervención adicional."},
}

# ================= Roles y páginas =================
ROLES = {
    "admin": "Administrador",
    "cartera": "Gestión de Cartera",
    "direccion": "Dirección Financiera",
    "analitica": "Equipo de Analítica",
}
# Claves de página → roles con acceso. Las claves las usa app.py para construir la navegación.
PAGE_ACCESS = {
    "resumen": {"admin", "cartera", "direccion", "analitica"},
    "cola": {"admin", "cartera"},
    "segmentos": {"admin", "cartera", "direccion", "analitica"},
    "modelo": {"admin", "direccion", "analitica"},
    "carga": {"admin", "cartera", "analitica"},
}

# ================= Etiquetas legibles =================
FIELD_LABELS = {
    "nombre": "Estudiante", "id_estudiante": "ID estudiante", "llave2": "Llave crédito",
    "programa": "Programa", "facultad": "Facultad", "programa_cluster": "Segmento de programa",
    "nivel": "Nivel", "sede": "Sede", "fecha_aprobacion": "Fecha de aprobación",
    "valor_financiacion": "Valor financiado", "vr_neto_matricula": "Matrícula neta", "cuotas": "Cuotas",
    "tipo_interes": "Tipo de interés", "y_pred": "Riesgo predicho", "y_true": "Riesgo observado",
    "proba_pred": "Confianza del modelo", "mora": "Mora Datacrédito", "media_score": "Scoring externo",
    "cohorte": "Cohorte", "genero": "Género", "estado_civil": "Estado civil", "tipo_estudiante": "Tipo de estudiante",
    "cliente": "Tipo de cliente", "departamento": "Departamento", "ciudad_norm": "Ciudad",
    "prioridad": "Índice de prioridad", "ruta": "Ruta de gestión", "edad": "Edad",
    "antiguedad_meses": "Antigüedad (meses)", "plataforma": "Plataforma scoring", "operacion": "Operación",
}

# Orden lógico del scoring externo (media_score) de peor a mejor.
SCORE_ORDER = [
    "Scoring 0", "Scoring < o = 400", "Scoring = o > 401  y < o = 480", "Scoring = o > 481  y < o = 560",
    "Scoring = o > 561  y < o = 639", "Scoring = o > 640  y < o = 720", "Scoring = o > 721",
]
SCORE_SHORT = {
    "Scoring 0": "Sin score", "Scoring < o = 400": "≤ 400", "Scoring = o > 401  y < o = 480": "401–480",
    "Scoring = o > 481  y < o = 560": "481–560", "Scoring = o > 561  y < o = 639": "561–639",
    "Scoring = o > 640  y < o = 720": "640–720", "Scoring = o > 721": "≥ 721",
}
MESES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre",
         "Noviembre", "Diciembre"]

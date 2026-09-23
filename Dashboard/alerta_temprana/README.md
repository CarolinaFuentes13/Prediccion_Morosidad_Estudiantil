# SAT · Sistema de Alertas Tempranas de morosidad estudiantil

Tablero Streamlit multipágina que convierte el modelo predictivo de morosidad (Random Forest) en decisiones de
**gestión preventiva de cartera**: priorización, rutas de cobro, seguimiento del desempeño del modelo, calidad de
datos y predicción en línea de nuevos créditos.

> Proyecto final · Maestría en Inteligencia Analítica de Datos · Universidad de los Andes.
> Los nombres de estudiantes son **anonimizados** (determinísticos por ID).

## Ejecutar

```bash
pip install -r requirements.txt                 # desde la raíz del repositorio
streamlit run Dashboard/alerta_temprana/app.py
```

En Streamlit Community Cloud: *Main file path* = `Dashboard/alerta_temprana/app.py`.
Las apps anteriores (`Dashboard/streamlit_app.py`, `Dashboard/streamlit_app_dashlike.py`) siguen funcionando igual.

### Acceso por roles

Sin configuración, el tablero arranca en **modo demostración** (se elige un rol sin contraseña). Para exigir
contraseña, copie `.streamlit/secrets.toml.example` a `.streamlit/secrets.toml` (o péguelo en *Secrets* de Cloud)
y defina los usuarios con su contraseña en SHA-256:

```bash
python -c "import hashlib; print(hashlib.sha256(b'mi-clave').hexdigest())"
```

| Rol | Páginas |
|---|---|
| `admin` | Todas |
| `cartera` | Resumen, Cola de gestión, Segmentos y territorio, Cargar y predecir |
| `direccion` | Resumen, Segmentos y territorio, Desempeño del modelo |
| `analitica` | Resumen, Segmentos y territorio, Desempeño del modelo, Cargar y predecir |

### Motor de predicción (carga de CSV)

El tablero elige automáticamente, en este orden:

1. **API FastAPI** del modelo (`Despliegue/API.zip`): defina `[api] url = "https://…"` en secrets o la variable de
   entorno `SAT_API_URL`. Usa `GET /api/v1/health` y `POST /api/v1/predict` y devuelve la clase predicha.
2. **Modelo local** (paquete `Despliegue/model_morosidad-0.0.1-py3-none-any.whl`): además de la clase entrega las
   probabilidades por clase. Requiere Python 3.10/3.11 y ~550 MB de RAM:
   `pip install -r Dashboard/alerta_temprana/requirements-model.txt`.
3. **Sin motor**: se aceptan CSV que ya traigan `y_pred`.

## Páginas

| Página | Qué resuelve | Requerimientos |
|---|---|---|
| Resumen ejecutivo | KPI exigidos, evolución del riesgo, segmentos críticos, hallazgos automáticos y reporte descargable | DB-03, ME-03 |
| Cola de gestión | Índice de prioridad con pesos editables, rutas R1–R4, curva de cobertura por capacidad, archivo de integración | DB-02, DB-04 |
| Segmentos y territorio | Explorador de segmentos con IC de Wilson, perfil de riesgo por scoring y tipo de interés, mapa por ciudad | Enfoque descriptivo |
| Desempeño del modelo | Comparación de 5 modelos, reporte por clase, umbral de alerta para Alto, importancia de variables | MO-02, EV-02, DE-03, MO-03 |
| Cargar y predecir | Plantilla, validación de esquema y calidad, predicción con API o modelo local, uso en todo el tablero | DE-01, ME-02 |

Los **filtros globales** (DB-01) de la barra lateral persisten entre páginas y pueden fijarse en la URL para
compartir una vista.

## Arquitectura

```
Dashboard/alerta_temprana/
├── app.py                 # entrada: login, navegación por rol, barra lateral y filtros
├── core/
│   ├── config.py          # rutas, umbrales de negocio, requerimientos, rutas de gestión, roles
│   ├── theme.py           # paleta, CSS, plantilla Plotly y formatos es-CO
│   ├── data.py            # carga, limpieza, variables derivadas, prioridad, rutas, dataset activo
│   ├── filters.py         # filtros globales persistentes y compartibles
│   ├── components.py      # hero, KPI, secciones, hallazgos, exportes
│   ├── metrics.py         # métricas en numpy + artefactos precalculados del modelo
│   ├── quality.py         # reglas de calidad (completitud, validez, exactitud, consistencia, unicidad)
│   ├── engine.py          # motor de predicción: API → modelo local
│   ├── auth.py            # login por roles con st.secrets
│   └── nav.py             # registro de páginas y navegación
├── views/                 # una página por archivo (5 páginas)
├── data/                  # model_artifacts.json, feature_importance.json
├── tools/                 # scripts para regenerar artefactos
└── tests/                 # pruebas unitarias y de humo (AppTest)
```

### Datos

* Cartera: `Database/Data_model_predictions/df_dash_with_preds.csv` (14.660 créditos con `y_pred`, `proba_pred`,
  `y_true`).
* Evaluación del modelo: predicciones de prueba (partición estratificada 80/20) de los cinco notebooks
  (`df_con_predicciones_TEST_modelo*.csv`), resumidas en `data/model_artifacts.json`.

Para regenerar los artefactos después de reentrenar:

```bash
# importancia de variables (en un entorno con el modelo instalado)
python Dashboard/alerta_temprana/tools/extract_feature_importance.py
# métricas, curvas, umbrales y desempeño temporal (valida contra scikit-learn si está instalado)
python Dashboard/alerta_temprana/tools/build_model_artifacts.py
```

## Pruebas

```bash
python -m pytest Dashboard/alerta_temprana/tests -q
```

`test_core.py` valida la capa de datos, métricas (contra ejemplos conocidos), reglas de calidad, filtros y el
motor; `test_smoke.py` ejecuta cada página con `streamlit.testing.AppTest` para cada rol.

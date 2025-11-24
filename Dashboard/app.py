from dash import Dash, html, dcc, dash_table 
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, timedelta  # <--- AHORA TAMBIÉN timedelta
import re, unicodedata, hashlib
import base64
import io
import requests  


# ================= Utilidades =================

def nrm(s):
    s = str(s).lower().strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9\s]", " ", s).strip()

def nombre_fake(seed, genero=None):
    m = ["juan", "carlos", "andres", "diego", "luis",
         "mateo", "jorge", "felipe", "daniel", "santiago"]
    f = ["maria", "laura", "ana", "camila", "valentina",
         "carolina", "paula", "daniela", "sara", "gabriela"]
    ap = ["garcia", "rodriguez", "lopez", "martinez", "hernandez",
          "gomez", "diaz", "ramirez", "torres", "vargas"]
    h = int(hashlib.sha256(str(seed).encode()).hexdigest(), 16)
    base = f if str(genero).lower() in {"f", "femenino"} else m
    return f"{base[h % 10]} {ap[(h // 97) % 10]}".title()

# ================= Datos =================

# Ruta: ../Database/Data_model_predictions/df_dash_with_preds.csv
DATA = Path(__file__).resolve().parents[1] / "Database" / "Data_model_predictions" / "df_dash_with_preds.csv"
df = pd.read_csv(DATA)

# Fecha de aprobación
if "fecha_aprobacion" in df.columns:
    df["fecha_aprobacion"] = pd.to_datetime(df["fecha_aprobacion"], errors="coerce")
else:
    df["fecha_aprobacion"] = pd.NaT

# Fecha de actualización (siempre hoy)
fecha_hoy = date.today().strftime("%Y-%m-%d")

# Riesgo predicho 
RIESGO = "y_pred"
ORDEN = ["Alto", "Medio", "Bajo"]
if RIESGO in df.columns:
    df[RIESGO] = df[RIESGO].astype(str).str.strip().str.capitalize()
    df[RIESGO] = pd.Categorical(df[RIESGO], categories=ORDEN, ordered=True)

# Probabilidad predicha (si existe)
if "proba_pred" not in df.columns:
    df["proba_pred"] = np.nan

# Año y periodo año-mes
df["anio"] = df["fecha_aprobacion"].dt.year
df["periodo"] = df["fecha_aprobacion"].dt.to_period("M").astype(str)

# Nombre anonimizado si no existe
col_id = next((c for c in df.columns if c.lower() == "idbanner"), None)
col_gen = next((c for c in df.columns if c.lower() in {"genero", "sexo"}), None)

if "nombre" not in df.columns:
    if col_id is None:
        df["nombre"] = [nombre_fake(i) for i in range(len(df))]
    else:
        df["nombre"] = df.apply(
            lambda r: nombre_fake(f"{r[col_id]}-{r.get(col_gen, '')}", r.get(col_gen, "")),
            axis=1
        )

# Programa y Facultad en bruto 
for c in ["programa", "facultad"]:
    if c not in df.columns:
        df[c] = "No definido"

# Clusters simples de programa y facultad
def rule_cluster(txt):
    t = nrm(txt)
    if any(x in t for x in ["ingenier", "sistemas", "software", "datos"]):
        return "Software y TI"
    if any(x in t for x in ["medic", "salud", "enfermer", "odont"]):
        return "Medicina y Salud"
    if any(x in t for x in ["admin", "negoc", "finan", "conta", "mercad"]):
        return "Negocios y Administración"
    if any(x in t for x in ["derech", "jur"]):
        return "Derecho"
    return "Otros"

df["programa_cluster"] = df["programa"].astype(str).map(rule_cluster)
df["facultad_cluster"] = df["facultad"].astype(str).map(rule_cluster)

# Flag de mora si existe
pos_mora = [c for c in df.columns if c.lower() in {"en_mora_datacredito", "flag_mora_bureau", "mora"}]
if pos_mora:
    c = pos_mora[0]
    df["mora_flag"] = (
        df[c].astype(str).str.lower().str.strip()
          .isin({"1", "si", "true", "yes", "y", "en mora", "mora"})
          .astype(int)
    )
else:
    df["mora_flag"] = 0

# Créditos activos por estudiante
if col_id:
    df["_credits_by_id"] = df.groupby(col_id)[col_id].transform("size")
else:
    df["_credits_by_id"] = 1

# Tipo de cliente limpio (Estudiante / No estudiante / Otro)
if "cliente" in df.columns:
    cli = df["cliente"].astype(str).str.strip().str.lower()
    cli = cli.replace({
        "estudiante": "estudiante",
        "no estudiante": "no estudiante"
    })
    df["cliente_limpio"] = cli.map({
        "estudiante": "Estudiante",
        "no estudiante": "No estudiante"
    }).fillna("Otro")
else:
    df["cliente_limpio"] = "Otro"

# Coordenadas (mapa)
lat_col = next((c for c in df.columns if c.lower() == "latitud"), None)
lon_col = next((c for c in df.columns if c.lower() == "longitud"), None)

# Valor de exposición (monto financiado)
VAL_COL = next(
    (c for c in df.columns if c.lower() in {"valor_financiacion", "vr_neto_matricula"}),
    None
)

# ORDENAR POR FECHA (más reciente primero) ANTES DE CREAR df_base
df = df.sort_values("fecha_aprobacion", ascending=False).reset_index(drop=True)

# Rango de fechas por defecto: último mes (últimos 30 días desde la fecha máxima disponible)
if df["fecha_aprobacion"].notna().any():
    fecha_max = df["fecha_aprobacion"].max().date()
    fecha_min = df["fecha_aprobacion"].min().date()
    fecha_ini_default = max(fecha_min, fecha_max - timedelta(days=30))
    fecha_fin_default = fecha_max
else:
    fecha_ini_default = None
    fecha_fin_default = None

df_base = df.copy()

# ================= Figuras =================

# Paleta tipo semáforo: Alto rojo, Medio naranja, Bajo verde
pal_riesgo = {
    "Alto": "#d9534f",   
    "Medio": "#f0ad4e",  
    "Bajo": "#5cb85c"    
}

def fig_riesgo_resumen(dff):
    if RIESGO not in dff.columns or dff[RIESGO].dropna().empty:
        return px.bar(title="Créditos por nivel de riesgo")
    g = dff[RIESGO].value_counts(dropna=True).rename_axis("riesgo").reset_index(name="n")
    g["pct"] = g["n"] / g["n"].sum() * 100
    g["etq"] = g["n"].map("{:,.0f}".format) + " | " + g["pct"].map("{:,.1f}%".format)
    order_map = {k: i for i, k in enumerate(ORDEN)}
    g = g.sort_values("riesgo", key=lambda s: s.map(order_map))
    fig = px.bar(
        g, x="riesgo", y="n", text="etq", color="riesgo",
        color_discrete_map=pal_riesgo,
        title="Créditos por nivel de riesgo"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title="Nivel de riesgo",
        yaxis_title="Número de créditos",
        showlegend=False,
        title_font={"size": 16, "color": "#003366"}
    )
    return fig

def fig_cuotas_tiempo(dff):
    if "cuotas" not in dff.columns or dff["fecha_aprobacion"].isna().all():
        return px.line(title="Cuotas promedio por riesgo y mes de aprobación")
    tmp = dff.dropna(subset=["fecha_aprobacion"])
    if tmp.empty:
        return px.line(title="Cuotas promedio por riesgo y mes de aprobación")
    g = (
        tmp.groupby(["periodo", RIESGO])["cuotas"]
           .mean()
           .reset_index(name="cuotas_prom")
    )
    order_map = {k: i for i, k in enumerate(ORDEN)}
    g = g.sort_values(
        ["periodo", RIESGO],
        key=lambda col: col.map(order_map) if col.name == RIESGO else col
    )
    fig = px.line(
        g, x="periodo", y="cuotas_prom", color=RIESGO,
        color_discrete_map=pal_riesgo,
        markers=True,
        title="Cuotas promedio por riesgo y mes de aprobación"
    )
    fig.update_layout(
        xaxis_title="Año-mes de aprobación",
        yaxis_title="Cuotas promedio",
        legend_title_text="Nivel de riesgo",
        title_font={"size": 16, "color": "#003366"}
    )
    return fig

def fig_heat_cluster_anio(dff):
    tmp = dff.copy()
    if tmp["anio"].isna().all():
        return px.density_heatmap(title="Porcentaje de mora por segmento de programa y año")
    tmp = (
        tmp.groupby(["programa_cluster", "anio"])["mora_flag"]
           .mean()
           .reset_index(name="mora_pct")
    )
    tmp["mora_pct"] = tmp["mora_pct"] * 100
    fig = px.density_heatmap(
        tmp, x="anio", y="programa_cluster", z="mora_pct",
        color_continuous_scale="Blues",
        title="Porcentaje de mora por segmento de programa y año"
    )
    fig.update_layout(
        xaxis_title="Año",
        yaxis_title="Segmento de programa",
        coloraxis_colorbar_title="% en mora",
        title_font={"size": 16, "color": "#003366"}
    )
    return fig

def fig_riesgo_tiempo(dff):
    if RIESGO not in dff.columns or dff["fecha_aprobacion"].isna().all():
        return px.area(title="Distribución de niveles de riesgo en el tiempo")
    tmp = dff.dropna(subset=["fecha_aprobacion"])
    if tmp.empty:
        return px.area(title="Distribución de niveles de riesgo en el tiempo")
    g = (
        tmp.groupby(["periodo", RIESGO]).size()
           .rename("n")
           .reset_index()
    )
    tot = g.groupby("periodo")["n"].transform("sum")
    g["pct"] = g["n"] / tot * 100
    order_map = {k: i for i, k in enumerate(ORDEN)}
    g = g.sort_values(
        ["periodo", RIESGO],
        key=lambda col: col.map(order_map) if col.name == RIESGO else col
    )
    fig = px.area(
        g, x="periodo", y="pct", color=RIESGO,
        color_discrete_map=pal_riesgo,
        title="Distribución de niveles de riesgo en el tiempo"
    )
    fig.update_layout(
        xaxis_title="Año-mes de aprobación",
        yaxis_title="% de créditos",
        legend_title_text="Nivel de riesgo",
        title_font={"size": 16, "color": "#003366"}
    )
    return fig

def fig_mapa(dff):
    # Verificar que existan las columnas requeridas
    if dff is None or dff.empty or lat_col not in dff.columns or lon_col not in dff.columns:
        return px.scatter_mapbox(
            lat=[],
            lon=[],
            zoom=4,
            height=420,
            title="Mapa de créditos (sin datos geográficos)"
        ).update_layout(mapbox_style="open-street-map")

    # Filtrar filas válidas
    dd = dff.dropna(subset=[lat_col, lon_col])
    if dd.empty:
        return px.scatter_mapbox(
            lat=[],
            lon=[],
            zoom=4,
            height=420,
            title="Mapa de créditos (sin datos geográficos válidos)"
        ).update_layout(mapbox_style="open-street-map")

    # Mapa real
    fig = px.scatter_mapbox(
        dd,
        lat=lat_col,
        lon=lon_col,
        color=RIESGO,
        hover_name="nombre",
        color_discrete_map=pal_riesgo,
        zoom=4,
        height=420,
        title="Mapa de créditos por categoría de riesgo predicha",
        labels={RIESGO: "Categoría predicha"}
    )

    fig.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=0, r=0, t=60, b=0),
        title_font={"size": 16, "color": "#003366"},
        legend_title_text="Categoría predicha"
    )

    return fig

# ================= App =================

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
brand, bg = "#003366", "#f7f9fb"

def kpi_card(title, value, subtitle=None):
    body = [
        html.Div(title, style={"color": "#667", "fontWeight": "600"}),
        html.H3(value, style={"color": brand, "margin": 0})
    ]
    if subtitle:
        body.append(html.Div(subtitle, style={"color": "#999", "fontSize": "11px", "marginTop": "4px"}))
    return dbc.Card(dbc.CardBody(body), style={"textAlign": "center"})

# Header
header = html.Div([
    html.Img(
        src="/assets/logo_uni.png",
        style={"height": "68px", "marginRight": "18px", "objectFit": "contain"}
    ),
    html.Div(style={"borderLeft": "2px solid #ccc", "height": "58px", "marginRight": "18px"}),
    html.H1(
        "Riesgo de morosidad en créditos estudiantiles",
        style={
            "textAlign": "left", "fontSize": "26px", "fontWeight": "bold",
            "color": brand, "margin": "0", "display": "flex", "alignItems": "center"
        }
    ),
    html.Div(style={"borderLeft": "2px solid #ccc", "height": "58px", "marginLeft": "auto"}),
    html.Div([
        html.Span(
            f"Actualizado al {fecha_hoy}",
            style={"color": "white", "fontWeight": "500", "fontSize": "16px"}
        )
    ], style={
        "backgroundColor": brand, "padding": "10px 20px", "borderRadius": "8px",
        "margin": "10px 25px 20px 25px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
        "display": "inline-block"
    })
], style={
    "display": "flex", "alignItems": "center", "justifyContent": "flex-start",
    "padding": "15px 25px", "backgroundColor": bg,
    "borderBottom": "2px solid #ccc", "boxShadow": "0 2px 5px rgba(0,0,0,0.1)"
})

intro = html.Div([
    html.P(
        "Este tablero muestra las predicciones de nivel de riesgo de morosidad "
        "(Alto, Medio y Bajo) para los créditos estudiantiles internos. "
        "Use los filtros para explorar por fecha de aprobación, programa, facultad y tipo de cliente.",
        style={"marginBottom": "4px", "color": "#333"}
    ),
    html.P(
        "Los montos corresponden al valor financiado de los créditos y se utilizan para dimensionar "
        "la exposición de la cartera por nivel de riesgo.",
        style={"fontSize": "12px", "color": "#666"}
    )
], style={"padding": "10px 25px 0 25px"})

upload = dbc.Card([
    html.Div("Cargar nuevas observaciones",
             style={"fontWeight": "600", "marginBottom": "6px", "color": "#003366"}),
    html.P(
        "Sube un archivo CSV con la misma estructura de variables de entrada del modelo. "
        "El sistema enviará estos datos a la API, obtendrá el nivel de riesgo y actualizará el tablero.",
        style={"fontSize": "12px", "color": "#555", "marginBottom": "8px"}
    ),
    dcc.Upload(
        id="upload-csv",
        children=html.Div([
            "Arrastra y suelta o haz clic para seleccionar un archivo CSV"
        ]),
        style={
            "width": "100%", "height": "60px", "lineHeight": "60px",
            "borderWidth": "1px", "borderStyle": "dashed",
            "borderRadius": "4px", "textAlign": "center", "backgroundColor": "#fdfdfd"
        },
        multiple=False
    ),
    html.Div(id="upload-msg", style={"fontSize": "12px", "color": "#666", "marginTop": "6px"})
], body=True, style={"backgroundColor": "#f7f9fb"})


# Filtros
filtros = dbc.Card([
    dbc.Row([
        dbc.Col([
            html.Label("Buscar nombre"),
            dcc.Dropdown(
                id="f-nombre",
                options=[{"label": n, "value": n} for n in sorted(df["nombre"].unique())],
                multi=True,
                placeholder="Selecciona o busca un nombre",
                searchable=True,
                clearable=True
            )
        ], md=3),
        dbc.Col([
            html.Label("Nivel de riesgo predicho"),
            dcc.Dropdown(
                id="f-riesgo",
                options=[{"label": c, "value": c} for c in ORDEN if c in df.get(RIESGO, pd.Series()).unique()],
                multi=True, placeholder="Selecciona nivel de riesgo"
            )
        ], md=3),
        dbc.Col([
            html.Label("Fecha de aprobación"),
            dcc.DatePickerRange(
                id="f-fecha",
                start_date=fecha_ini_default,   # <--- ÚLTIMO MES POR DEFECTO
                end_date=fecha_fin_default,     # <--- HASTA LA FECHA MÁXIMA
                display_format="YYYY-MM-DD"
            )
        ], md=3),
        dbc.Col([
            html.Label("Tipo de cliente"),
            dcc.Dropdown(
                id="f-cli",
                options=[{"label": c, "value": c} for c in sorted(df["cliente_limpio"].unique())],
                multi=True, placeholder="Selecciona tipo de cliente"
            )
        ], md=3),
    ], className="g-3"),
    dbc.Row([
        dbc.Col([
            html.Label("Facultad"),
            dcc.Dropdown(
                id="f-fac",
                options=[{"label": c, "value": c} for c in sorted(df["facultad"].astype(str).unique())],
                multi=True, placeholder="Selecciona facultad"
            )
        ], md=6),
        dbc.Col([
            html.Label("Programa"),
            dcc.Dropdown(
                id="f-prog",
                options=[{"label": c, "value": c} for c in sorted(df["programa"].astype(str).unique())],
                multi=True, placeholder="Selecciona programa"
            )
        ], md=6),
    ], className="g-3"),
    dbc.Row([
        dbc.Col([
            html.Label("Facultad (segmento)"),
            dcc.Dropdown(
                id="f-fac-clu",
                options=[{"label": c, "value": c} for c in sorted(df["facultad_cluster"].unique())],
                multi=True, placeholder="Selecciona segmento de facultad"
            )
        ], md=6),
        dbc.Col([
            html.Label("Programa (segmento)"),
            dcc.Dropdown(
                id="f-prog-clu",
                options=[{"label": c, "value": c} for c in sorted(df["programa_cluster"].unique())],
                multi=True, placeholder="Selecciona segmento de programa"
            )
        ], md=6),
    ], className="g-3"),
], body=True, style={"backgroundColor": bg})

# KPIs
kpis_row1 = dbc.Row([
    dbc.Col(kpi_card("Créditos filtrados", "", "Número total de créditos según los filtros aplicados"), md=3, id="kpi-n"),
    dbc.Col(kpi_card("Créditos en riesgo alto", "", "Cantidad de créditos clasificados en nivel Alto"), md=3, id="kpi-alto"),
    dbc.Col(kpi_card("% en riesgo alto", "", "Proporción de créditos en nivel Alto sobre el total filtrado"), md=3, id="kpi-alto-pct"),
    dbc.Col(kpi_card("% en mora (Datacrédito)", "", "Porcentaje de créditos reportados en mora"), md=3, id="kpi-mora"),
], className="g-3")

kpis_row2 = dbc.Row([
    dbc.Col(kpi_card("Monto total financiado (COP)", "", "Suma del valor financiado de los créditos filtrados"), md=3, id="kpi-exp-tot"),
    dbc.Col(kpi_card("Monto en riesgo alto (COP)", "", "Valor financiado asociado a créditos en nivel Alto"), md=3, id="kpi-exp-alto"),
    dbc.Col(kpi_card("Monto en riesgo medio (COP)", "", "Valor financiado asociado a créditos en nivel Medio"), md=3, id="kpi-exp-med"),
    dbc.Col(kpi_card("Monto en riesgo bajo (COP)", "", "Valor financiado asociado a créditos en nivel Bajo"), md=3, id="kpi-exp-baj"),
], className="g-3")

# Layout
app.layout = dbc.Container([
    header,
    dcc.Store(id="store-datos", data=df_base.to_dict("records")),  
    intro,
    html.Br(),
    upload,
    html.Br(),
    filtros,
    html.Br(),
    kpis_row1,
    html.Br(),
    kpis_row2,
    html.Br(),
    dbc.Row([
        dbc.Col(dcc.Graph(id="g-resumen"), md=4),
        dbc.Col(dcc.Graph(id="g-cuotas-tiempo"), md=4),
        dbc.Col(dcc.Graph(id="g-heat"), md=4),
    ], className="g-3"),
    html.Br(),
    dbc.Row([
        dbc.Col(dcc.Graph(id="g-riesgo-tiempo"), md=6),
        dbc.Col(dcc.Graph(id="g-mapa"), md=6),
    ], className="g-3"),
    html.Br(),
    dbc.Card([
        html.Div(
            "Casos en mora (Datacrédito)",
            style={"color": brand, "fontWeight": "600", "padding": "8px 12px"}
        ),
        html.Div(
            "Se listan los créditos con reporte de mora según Datacrédito, "
            "junto con su nivel de riesgo predicho y la probabilidad estimada.",
            style={"padding": "0 12px 8px 12px", "fontSize": "12px", "color": "#555"}
        ),
        dash_table.DataTable(
            id="tbl-mora",
            columns=[
                {"name": "Nombre", "id": "nombre"},
                {"name": "Programa", "id": "programa"},
                {"name": "Fecha de aprobación", "id": "fecha_aprobacion"},
                {"name": "Nivel de riesgo predicho", "id": RIESGO},
                {"name": "Probabilidad de morosidad", "id": "proba_str"},
                {"name": "Créditos activos", "id": "_credits_by_id"},
            ],
            data=[], page_size=10,
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": brand, "color": "white"},
            style_cell={"padding": "8px", "border": "none"}
        )
    ]),
    html.Br()
], fluid=True)

# ================= Callbacks =================

def filtrar(_df, nombres, riesgo, f_ini, f_fin, fac, prog, fac_clu, prog_clu, cli):
    dff = _df.copy()
    # Filtro por nombre (Dropdown multi)
    if nombres:
        dff = dff[dff["nombre"].isin(nombres)]
    # Filtro por riesgo
    if riesgo:
        dff = dff[dff[RIESGO].isin(riesgo)]
    # Fechas
    if f_ini:
        dff = dff[dff["fecha_aprobacion"] >= f_ini]
    if f_fin:
        dff = dff[dff["fecha_aprobacion"] <= f_fin]
    # Facultad / Programa brutos
    if fac:
        dff = dff[dff["facultad"].astype(str).isin(fac)]
    if prog:
        dff = dff[dff["programa"].astype(str).isin(prog)]
    # Clusters
    if fac_clu:
        dff = dff[dff["facultad_cluster"].isin(fac_clu)]
    if prog_clu:
        dff = dff[dff["programa_cluster"].isin(prog_clu)]
    # Tipo de cliente
    if cli:
        dff = dff[dff["cliente_limpio"].isin(cli)]
    return dff


@app.callback(
    Output("store-datos", "data"),
    Output("upload-msg", "children"),
    Input("upload-csv", "contents"),
    Input("upload-csv", "filename"),
    prevent_initial_call=True
)
def procesar_csv_con_api(contents, filename):
    # Si no han cargado archivo aún, devolvemos la base original
    if contents is None:
        return df_base.to_dict("records"), ""

    try:
        # 1. Decodificar el CSV que viene desde el navegador
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        # Intentar leer primero como UTF-8, si falla usar latin1
        try:
            df_new = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        except UnicodeDecodeError:
            df_new = pd.read_csv(io.BytesIO(decoded), encoding="latin1")

        # Normalizar nombres de columnas (quita BOM, espacios, etc.)
        df_new.columns = (
            df_new.columns.astype(str)
                  .str.replace('\ufeff', '', regex=False)  # quita BOM si viene
                  .str.strip()
        )

        # 2. Verificar que traiga Llave2 (o alguna variante)
        #    Construimos un diccionario "columna normalizada" -> "nombre original"
        clean_cols = {re.sub(r"[^A-Za-z0-9]", "", c).lower(): c for c in df_new.columns}
        if "llave2" not in clean_cols:
            msg = f"El archivo debe incluir la columna 'llave2'. Columnas detectadas: {list(df_new.columns)}"
            return df_base.to_dict("records"), msg

        # Normalizar nombre de la columna llave a exactamente 'llave2'
        df_new = df_new.rename(columns={clean_cols["llave2"]: "llave2"})

        cols_permitidas = [
            "llave2", "nombre_linea", "fecha_aprobacion", "antiguedad_meses",
            "nombre_fondo", "valor_financiacion", "cuotas", "tipo_interes",
            "vr_neto_matricula", "fecha_nacimiento", "estado_civil", "genero",
            "facultad", "programa", "nivel", "estado", "tipoestudiante", "sede",
            "sello", "carga", "grupo_etnico", "tipo_discapacidad", "nacionalidad",
            "mora", "valor_cuota_inicial", "valor_primera_cuota", "fecha_de_pago",
            "validacion_valor_financiado", "detalle_estado_final", "tipo_estudiante",
            "operacion", "cate", "subcate", "cohorte", "mes", "cliente",
            "media_score", "anob", "valor_maximo", "valor_medio", "valor_bajo",
            "plataforma"
        ]

        df_new = df_new[cols_permitidas]

        # 3. Llamar a la API del modelo con las filas nuevas
        payload = {"inputs": df_new.to_dict(orient="records")}

        resp = requests.post(
            "http://13.220.177.100:8001/api/v1/predict",
            json=payload
        )
        resp.raise_for_status()
        data_api = resp.json()

        # 4. Extraer predicciones (lista de clases)
        preds = data_api.get("predictions", [])
        if len(preds) != len(df_new):
            msg = "Error: el número de predicciones no coincide con el número de filas."
            return df_base.to_dict("records"), msg

        # 5. Añadir columnas de resultado al df_new
        df_new["y_pred"] = preds
        df_new["proba_pred"] = np.nan  # la API todavía no devuelve probabilidades

        # Normalizar riesgo
        df_new["y_pred"] = (
            df_new["y_pred"]
              .astype(str)
              .str.strip()
              .str.capitalize()
        )

        # 6. Fechas, año, periodo
        if "fecha_aprobacion" in df_new.columns:
            df_new["fecha_aprobacion"] = pd.to_datetime(df_new["fecha_aprobacion"], errors="coerce")
            df_new["anio"] = df_new["fecha_aprobacion"].dt.year
            df_new["periodo"] = df_new["fecha_aprobacion"].dt.to_period("M").astype(str)
        else:
            df_new["fecha_aprobacion"] = pd.NaT
            df_new["anio"] = np.nan
            df_new["periodo"] = ""

        # 7. Nombre si hace falta
        if "nombre" not in df_new.columns:
            if col_id and col_id in df_new.columns:
                df_new["nombre"] = df_new.apply(
                    lambda r: nombre_fake(f"{r.get(col_id,'')}-{r.get(col_gen,'')}", r.get(col_gen, "")),
                    axis=1
                )
            else:
                df_new["nombre"] = [nombre_fake(f"nuevo-{i}") for i in range(len(df_new))]

        # 8. Programa / Facultad si faltan
        for c in ["programa", "facultad"]:
            if c not in df_new.columns:
                df_new[c] = "No definido"

        # 9. Tipo de cliente limpio
        if "cliente" in df_new.columns:
            cli = df_new["cliente"].astype(str).str.strip().str.lower()
            cli = cli.replace({"estudiante": "estudiante", "no estudiante": "no estudiante"})
            df_new["cliente_limpio"] = cli.map({
                "estudiante": "Estudiante",
                "no estudiante": "No estudiante"
            }).fillna("Otro")
        else:
            df_new["cliente_limpio"] = "Otro"

        # 10. Créditos activos por estudiante
        if col_id and col_id in df_new.columns:
            df_new["_credits_by_id"] = df_new.groupby(col_id)[col_id].transform("size")
        else:
            df_new["_credits_by_id"] = 1

        # 11. Cluster de programa y facultad
        df_new["programa_cluster"] = df_new["programa"].astype(str).map(rule_cluster)
        df_new["facultad_cluster"] = df_new["facultad"].astype(str).map(rule_cluster)

        # 12. Mora flag (si viene; si no, 0)
        if pos_mora:
            c = pos_mora[0]
            if c in df_new.columns:
                df_new["mora_flag"] = (
                    df_new[c].astype(str).str.lower().str.strip()
                    .isin({"1", "si", "true", "yes", "y", "en mora", "mora"})
                    .astype(int)
                )
            else:
                df_new["mora_flag"] = 0
        else:
            df_new["mora_flag"] = 0

        # 13. Concatenar base + nuevos y ordenar por fecha (más reciente primero)
        df_total = pd.concat([df_base, df_new], ignore_index=True)
        df_total = df_total.sort_values("fecha_aprobacion", ascending=False).reset_index(drop=True)

        msg = f"Archivo '{filename}' cargado correctamente. Se añadieron {len(df_new)} observaciones con clasificación de riesgo."
        return df_total.to_dict("records"), msg

    except Exception as e:
        return df_base.to_dict("records"), f"Ocurrió un error al procesar el archivo: {e}"


@app.callback(
    [
        Output("kpi-n", "children"),
        Output("kpi-alto", "children"),
        Output("kpi-alto-pct", "children"),
        Output("kpi-mora", "children"),
        Output("kpi-exp-tot", "children"),
        Output("kpi-exp-alto", "children"),
        Output("kpi-exp-med", "children"),
        Output("kpi-exp-baj", "children"),
        Output("g-resumen", "figure"),
        Output("g-cuotas-tiempo", "figure"),
        Output("g-heat", "figure"),
        Output("g-riesgo-tiempo", "figure"),
        Output("g-mapa", "figure"),
        Output("tbl-mora", "data"),
    ],
    [
        Input("store-datos", "data"),   
        Input("f-nombre", "value"),
        Input("f-riesgo", "value"),
        Input("f-fecha", "start_date"),
        Input("f-fecha", "end_date"),
        Input("f-fac", "value"),
        Input("f-prog", "value"),
        Input("f-fac-clu", "value"),
        Input("f-prog-clu", "value"),
        Input("f-cli", "value"),
    ]
)
def update(data_store, nombres, riesgo, f_ini, f_fin, fac, prog, fac_clu, prog_clu, cli):
    # Reconstruir df actual
    if data_store is None:
        df_actual = df_base.copy()
    else:
        df_actual = pd.DataFrame(data_store)

    dff = filtrar(df_actual, nombres, riesgo, f_ini, f_fin, fac, prog, fac_clu, prog_clu, cli)
    n = len(dff)

    # KPIs de riesgo
    n_alto = int(dff[dff[RIESGO].eq("Alto")].shape[0]) if n > 0 else 0
    pct_alto = (n_alto / n * 100) if n > 0 else 0.0
    mora_pct = float(dff["mora_flag"].mean() * 100) if n > 0 else 0.0

    # KPIs de monto
    if VAL_COL and VAL_COL in dff.columns and n > 0:
        val = dff[VAL_COL].fillna(0).clip(lower=0)
        exp_tot = float(val.sum())
        exp_alto = float(val[dff[RIESGO].eq("Alto")].sum())
        exp_med = float(val[dff[RIESGO].eq("Medio")].sum())
        exp_baj = float(val[dff[RIESGO].eq("Bajo")].sum())
    else:
        exp_tot = exp_alto = exp_med = exp_baj = 0.0

    # Figuras
    f1 = fig_riesgo_resumen(dff)
    f2 = fig_cuotas_tiempo(dff)
    f3 = fig_heat_cluster_anio(dff)
    f4 = fig_riesgo_tiempo(dff)
    f5 = fig_mapa(dff)

    # Tabla mora
    top = dff[dff["mora_flag"] == 1].copy()
    if not top.empty:
        top["proba_str"] = top["proba_pred"].mul(100).map(
            lambda v: f"{v:.1f}%" if pd.notna(v) else ""
        )
        cols = ["nombre", "programa", "fecha_aprobacion", RIESGO, "proba_str", "_credits_by_id"]
        data = top.sort_values("fecha_aprobacion", ascending=False)[cols].head(10).to_dict("records")
    else:
        data = []

    # Armar KPIs como tarjetas
    k1 = kpi_card("Créditos filtrados", f"{n:,}", "Número total de créditos según los filtros aplicados")
    k2 = kpi_card("Créditos en riesgo alto", f"{n_alto:,}", "Cantidad de créditos clasificados en nivel Alto")
    k3 = kpi_card("% en riesgo alto", f"{pct_alto:.1f}%", "Proporción de créditos en nivel Alto sobre el total filtrado")
    k4 = kpi_card("% en mora (Datacrédito)", f"{mora_pct:.1f}%", "Porcentaje de créditos reportados en mora")

    k5 = kpi_card("Monto total financiado (COP)", f"{exp_tot:,.0f}", "Suma del valor financiado de los créditos filtrados")
    k6 = kpi_card("Monto en riesgo alto (COP)", f"{exp_alto:,.0f}", "Valor financiado asociado a créditos en nivel Alto")
    k7 = kpi_card("Monto en riesgo medio (COP)", f"{exp_med:,.0f}", "Valor financiado asociado a créditos en nivel Medio")
    k8 = kpi_card("Monto en riesgo bajo (COP)", f"{exp_baj:,.0f}", "Valor financiado asociado a créditos en nivel Bajo")

    return k1, k2, k3, k4, k5, k6, k7, k8, f1, f2, f3, f4, f5, data


if __name__ == "__main__":
    app.run(debug=True)

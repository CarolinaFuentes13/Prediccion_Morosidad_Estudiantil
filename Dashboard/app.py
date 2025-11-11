from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
import re, unicodedata, hashlib

def nrm(s):
    s = str(s).lower().strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    return re.sub(r"[^a-z0-9\s]", " ", s).strip()

def _cap1_token(tok: str) -> str:
    return tok[:1].upper() + tok[1:].lower() if tok else tok

def _proper_case(name: str) -> str:
    """
    Convierte 'sebaSTian caSTillo' -> 'Sebastian Castillo'
    Respeta separadores: espacio, guion y apóstrofo.
    """
    parts = re.split(r"([-\s'])", str(name).strip())
    out = []
    for p in parts:
        if p and not re.fullmatch(r"[-\s']", p) and any(ch.isalpha() for ch in p):
            out.append(_cap1_token(p))
        else:
            out.append(p)
    return "".join(out)

def nombre_fake(seed, genero=None):
    m = [
        "juan","carlos","andres","diego","luis","mateo","jorge","felipe","daniel","santiago",
        "sebastian","nicolas","alejandro","miguel","ricardo","tomas","bruno","rafael"
    ]
    f = [
        "maria","laura","ana","camila","valentina","carolina","paula","daniela","sara","gabriela",
        "andrea","sofia","juliana","natalia","isabel","manuela","fernanda","lucia"
    ]
    ap = [
        "garcia","rodriguez","lopez","martinez","hernandez","gomez","diaz","ramirez","torres","vargas",
        "perez","sanchez","castillo","rojas","moreno","ortiz","alvarez","jimenez","flores","cruz"
    ]

    h = int(hashlib.sha256(str(seed).encode()).hexdigest(), 16)
    base = f if str(genero).lower() in {"f","femenino"} else m

    nombre = base[h % len(base)]
    apellido = ap[(h // 97) % len(ap)]

    return f"{_proper_case(nombre)} {_proper_case(apellido)}"



# data 
DATA = Path(__file__).resolve().parents[1] / "DB_Model" / "df_dash_with_preds.csv"
df = pd.read_csv(DATA)

if "fecha_aprobacion" in df.columns:
    df["fecha_aprobacion"] = pd.to_datetime(df["fecha_aprobacion"], errors="coerce")
    ultima_fecha = df["fecha_aprobacion"].max()
else:
    df["fecha_aprobacion"] = pd.NaT
    ultima_fecha = None

# riesgo predicho
RIESGO = "y_pred"
ORDEN = ["Alto","Medio","Bajo"]
if RIESGO in df.columns:
    df[RIESGO] = df[RIESGO].astype(str).str.strip().str.capitalize()
    df[RIESGO] = pd.Categorical(df[RIESGO], categories=ORDEN, ordered=True)

# proba, año
if "proba_pred" not in df.columns: df["proba_pred"] = np.nan
df["anio"] = df["fecha_aprobacion"].dt.year

# nombre
col_id = next((c for c in df.columns if c.lower()=="idbanner"), None)
col_gen = next((c for c in df.columns if c.lower() in {"genero","sexo"}), None)
if "nombre" not in df.columns:
    if col_id is None:
        df["nombre"] = [nombre_fake(i) for i in range(len(df))]
    else:
        df["nombre"] = df.apply(lambda r: nombre_fake(f"{r[col_id]}-{r.get(col_gen,'')}", r.get(col_gen,"")), axis=1)

# clusters (para heatmap y filtros extra)
for c in ["programa","facultad"]:
    if c not in df.columns: df[c] = "no definido"
def rule_cluster(txt):
    t = nrm(txt)
    if any(x in t for x in ["ingenier","sistemas","software","datos"]): return "software y ti"
    if any(x in t for x in ["medic","salud","enfermer","odont"]):       return "medicina y salud"
    if any(x in t for x in ["admin","negoc","finan","conta","mercad"]): return "negocios y adm"
    if any(x in t for x in ["derech","jur"]):                           return "derecho"
    return "otros"
df["programa_cluster"] = df["programa"].astype(str).map(rule_cluster)
df["facultad_cluster"] = df["facultad"].astype(str).map(rule_cluster)

# mora flag si existe
pos_mora = [c for c in df.columns if c.lower() in {"en_mora_datacredito","flag_mora_bureau","mora"}]
if pos_mora:
    c = pos_mora[0]
    df["mora_flag"] = df[c].astype(str).str.lower().str.strip().isin({"1","si","true","yes","y","en mora","mora"}).astype(int)
else:
    df["mora_flag"] = 0

# créditos activos por estudiante
if col_id:
    df["_credits_by_id"] = df.groupby(col_id)[col_id].transform("size")
else:
    df["_credits_by_id"] = 1

# coordenadas
lat_col = next((c for c in df.columns if c.lower()=="latitud"), None)
lon_col = next((c for c in df.columns if c.lower()=="longitud"), None)

# valor de exposición (para métricas financieras)
VAL_COL = next((c for c in df.columns if c.lower() in {"valor_financiacion","vr_neto_matricula"}), None)

# estilo 
brand, bg, bg_soft = "#003366", "#f7f9fb", "#f0f4f8"

def apply_fig_min(fig, title_text, x_title=None, y_title=None):
    fig.update_layout(
        title=dict(text=title_text, x=0.02, xanchor="left", font=dict(size=18, family="Arial", color=brand)),
        font=dict(size=12, family="Arial"),
        legend_title_text="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0, bgcolor="rgba(0,0,0,0)")
    )
    if x_title: fig.update_xaxes(title_text=x_title)
    if y_title: fig.update_yaxes(title_text=y_title)
    return fig

# figs 
def fig_riesgo_resumen(dff):
    if RIESGO not in dff.columns:
        return apply_fig_min(px.bar(), "Créditos por riesgo", "Riesgo", "Créditos")
    g = dff[RIESGO].value_counts(dropna=True).rename_axis("riesgo").reset_index(name="n")
    g["pct"] = g["n"]/g["n"].sum()*100
    g["etq"] = g["n"].map("{:,.0f}".format) + " | " + g["pct"].map("{:,.1f}%".format)
    fig = px.bar(
        g.sort_values("riesgo", key=lambda s: s.map({k:i for i,k in enumerate(ORDEN)})),
        x="riesgo", y="n", text="etq", color="riesgo",
        color_discrete_map={"Alto":"#1f77b4","Medio":"#4ba3d3","Bajo":"#9ecae1"}
    )
    fig.update_traces(textposition="outside", showlegend=False)
    fig = apply_fig_min(fig, "Créditos por riesgo", "Riesgo", "Créditos")
    return fig

def fig_cuotas_mes(dff):
    if "cuotas" not in dff.columns or dff["fecha_aprobacion"].isna().all():
        return apply_fig_min(px.line(), "Cuotas por mes y riesgo", "Año-Mes", "Cuotas promedio")
    tmp = dff.dropna(subset=["fecha_aprobacion"]).copy()
    tmp["ym"] = tmp["fecha_aprobacion"].dt.to_period("M").astype(str)
    g = (tmp.groupby(["ym", RIESGO])["cuotas"]
             .mean()
             .reset_index(name="cuotas_prom"))
    g = g[g[RIESGO].notna()]
    fig = px.line(
        g.sort_values("ym"),
        x="ym", y="cuotas_prom", color=RIESGO, markers=True,
        color_discrete_map={"Alto":"#1f77b4","Medio":"#4ba3d3","Bajo":"#9ecae1"}
    )
    fig = apply_fig_min(fig, "Cuotas por mes y riesgo", "Año-Mes", "Cuotas promedio")
    return fig

def fig_heat_cluster_anio(dff):
    tmp = dff.groupby(["programa_cluster","anio"])["mora_flag"].mean().reset_index(name="mora_pct")
    tmp["mora_pct"] = tmp["mora_pct"]*100
    tmp["programa_cluster_t"] = tmp["programa_cluster"].astype(str).str.title()
    fig = px.density_heatmap(
        tmp, x="anio", y="programa_cluster_t", z="mora_pct",
        color_continuous_scale="Blues"
    )
    fig = apply_fig_min(fig, "Mora por programa y año", "Año", "Programa (cluster)")
    fig.update_layout(coloraxis_colorbar_title="%")
    return fig

def fig_mapa(dff):
    if not lat_col or not lon_col:
        return apply_fig_min(px.scatter_mapbox(), "Mapa por riesgo")
    dd = dff.dropna(subset=[lat_col, lon_col])
    if dd.empty:
        return apply_fig_min(px.scatter_mapbox(), "Mapa por riesgo")
    fig = px.scatter_mapbox(
        dd, lat=lat_col, lon=lon_col, color=RIESGO,
        hover_name="nombre",
        color_discrete_map={"Alto":"#1f77b4","Medio":"#4ba3d3","Bajo":"#9ecae1"},
        zoom=4, height=420
    )
    fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,t=60,b=0))
    fig = apply_fig_min(fig, "Mapa por riesgo")
    return fig

def fig_real_vs_pred(dff):
    # Mostrar etiquetas en % sobre cada barra 
    if "y_true" not in dff.columns:
        return apply_fig_min(px.bar(), "Real vs predicho", "Clase", "Créditos")
    g1 = dff["y_true"].astype(str).str.capitalize().value_counts().rename_axis("clase").reset_index(name="n_real")
    g2 = dff[RIESGO].astype(str).value_counts().rename_axis("clase").reset_index(name="n_pred")
    g = pd.merge(g1, g2, on="clase", how="outer").fillna(0)
    g["n_real"] = g["n_real"].astype(int); g["n_pred"] = g["n_pred"].astype(int)
    g = g.sort_values("clase", key=lambda s: s.map({k:i for i,k in enumerate(ORDEN)}))

    total_real = g["n_real"].sum() if g["n_real"].sum()>0 else 1
    total_pred = g["n_pred"].sum() if g["n_pred"].sum()>0 else 1
    pct_real = (g["n_real"] / total_real * 100).round(1).astype(str) + "%"
    pct_pred = (g["n_pred"] / total_pred * 100).round(1).astype(str) + "%"

    fig = px.bar(g, x="clase", y=["n_real","n_pred"], barmode="group")
    if len(fig.data) >= 2:
        fig.data[0].text = pct_real
        fig.data[1].text = pct_pred
        fig.data[0].textposition = "outside"
        fig.data[1].textposition = "outside"

    fig = apply_fig_min(fig, "Real vs predicho", "Clase", "Créditos")
    return fig

# app 
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
brand, bg = "#003366", "#f7f9fb"

def kpi_card(title, value):
    return dbc.Card(
        dbc.CardBody([
            html.Div(title, style={"color":"#667", "fontWeight":"600"}),
            html.H3(value, style={"color":brand, "margin":0})
        ]),
        style={"textAlign":"center"}
    )

# HEADER 
header = html.Div([
    html.Img(src="/assets/logo_uni.png",
             style={"height":"68px","marginRight":"18px","objectFit":"contain"}),
    html.Div(style={"borderLeft":"2px solid #ccc","height":"58px","marginRight":"18px"}),
    html.H1("Riesgo de morosidad en créditos estudiantiles",
            style={"textAlign":"left","fontSize":"26px","fontWeight":"bold",
                   "color":brand,"margin":"0","display":"flex","alignItems":"center"}),
    html.Div(style={"borderLeft":"2px solid #ccc","height":"58px","marginLeft":"auto"}),
    html.Div([
        html.Span(
            f"Data last updated on | {ultima_fecha.strftime('%Y-%m-%d') if pd.notna(ultima_fecha) else '-'}",
            style={"color":"white","fontWeight":"500","fontSize":"16px"}
        )
    ], style={"backgroundColor":brand,"padding":"10px 20px","borderRadius":"8px",
              "margin":"10px 25px 20px 25px","boxShadow":"0 2px 4px rgba(0,0,0,0.1)","display":"inline-block"})
], style={"display":"flex","alignItems":"center","justifyContent":"flex-start",
          "padding":"15px 25px","backgroundColor":bg_soft,
          "borderBottom":"2px solid #ccc","boxShadow":"0 2px 5px rgba(0,0,0,0.1)"})

# OPCIONES de filtros 
opts_riesgo = [{"label": c, "value": c} for c in ORDEN if RIESGO in df.columns]
opts_cli = [{"label": c, "value": c} for c in sorted(df.get("cliente", pd.Series(dtype=str)).dropna().unique())]
opts_fac = [{"label": c, "value": c} for c in sorted(df["facultad"].astype(str).dropna().unique())]
opts_prog = [{"label": c, "value": c} for c in sorted(df["programa"].astype(str).dropna().unique())]
opts_fac_clu = [{"label": c.title(), "value": c} for c in sorted(df["facultad_cluster"].astype(str).dropna().unique())]
opts_prog_clu = [{"label": c.title(), "value": c} for c in sorted(df["programa_cluster"].astype(str).dropna().unique())]

# FILTROS 
filtros = dbc.Card([
    dbc.Row([
        dbc.Col(dcc.Input(id="f-nombre", type="text", placeholder="Buscar nombre...", debounce=True), md=3),
        dbc.Col(dcc.Dropdown(id="f-riesgo", options=opts_riesgo, multi=True, placeholder="Riesgo predicho"), md=3),
        dbc.Col(dcc.DatePickerRange(
            id="f-fecha",
            start_date=df["fecha_aprobacion"].min().date() if df["fecha_aprobacion"].notna().any() else None,
            end_date=df["fecha_aprobacion"].max().date() if df["fecha_aprobacion"].notna().any() else None,
            display_format="YYYY-MM-DD"
        ), md=3),
        dbc.Col(dcc.Dropdown(id="f-cli", options=opts_cli, multi=True, placeholder="Cliente"), md=3),
    ], className="g-3"),
    dbc.Row([
        dbc.Col(dcc.Dropdown(id="f-fac", options=opts_fac, multi=True, placeholder="Facultad (original)"), md=3),
        dbc.Col(dcc.Dropdown(id="f-prog", options=opts_prog, multi=True, placeholder="Programa (original)"), md=3),
        dbc.Col(dcc.Dropdown(id="f-fac-clu", options=opts_fac_clu, multi=True, placeholder="Facultad (cluster)"), md=3),
        dbc.Col(dcc.Dropdown(id="f-prog-clu", options=opts_prog_clu, multi=True, placeholder="Programa (cluster)"), md=3),
    ], className="g-3"),
], body=True, style={"backgroundColor": bg})

# KPIs
kpis_row1 = dbc.Row([
    dbc.Col(kpi_card("Créditos (filtro)", ""), md=3, id="kpi-n"),
    dbc.Col(kpi_card("Mora Datacrédito", ""), md=3, id="kpi-mora"),
    dbc.Col(kpi_card("Probabilidad promedio", ""), md=3, id="kpi-prob"),
    dbc.Col(dbc.Card(dbc.CardBody([
        html.Div("Niveles de riesgo", style={"color":"#667","fontWeight":"600"}),
        html.Div(id="kpi-riesgos", style={"display":"flex","gap":"8px","justifyContent":"center","flexWrap":"wrap"})
    ]), style={"textAlign":"center"}), md=3),
], className="g-3")

kpis_row2 = dbc.Row([
    dbc.Col(kpi_card("Accuracy del modelo", ""), md=2, id="kpi-acc"),
    dbc.Col(kpi_card("Recall en alto", ""), md=2, id="kpi-rec-alto"),
    dbc.Col(kpi_card("Gap alto (real - pred.)", ""), md=2, id="kpi-gap-alto"),
    dbc.Col(kpi_card("Lift alto vs base", ""), md=2, id="kpi-lift"),
    dbc.Col(kpi_card("Exposición esperada", ""), md=2, id="kpi-exp-esp"),
    dbc.Col(kpi_card("Exposición en alto", ""), md=2, id="kpi-exp-alto"),
], className="g-3")

# LAYOUT
app.layout = html.Div([
    header,
    dbc.Container([
        html.Br(),
        html.Div(f"Actualización: {ultima_fecha.strftime('%Y-%m-%d') if pd.notna(ultima_fecha) else '-'}",
                 style={"color":"#666"}),
        html.Br(),
        filtros, html.Br(),
        kpis_row1, html.Br(),
        kpis_row2, html.Br(),
        dbc.Row([
            dbc.Col(dcc.Graph(id="g-resumen"), md=4),
            dbc.Col(dcc.Graph(id="g-cuotas-mes"), md=4),
            dbc.Col(dcc.Graph(id="g-heat"), md=4),
        ], className="g-3"),
        html.Br(),
        dbc.Row([
            dbc.Col(dcc.Graph(id="g-comp"), md=6),
            dbc.Col(dcc.Graph(id="g-mapa"), md=6),
        ], className="g-3"),
        html.Br(),
        dbc.Card([
            html.Div("Casos en Mora Datacrédito", style={"color":brand, "fontWeight":"700", "padding":"8px 12px"}),
            dash_table.DataTable(
                id="tbl-mora",
                columns=[
                    {"name":"Nombre","id":"nombre"},
                    {"name":"Programa","id":"programa"},
                    {"name":"Fecha de Aprobación","id":"fecha_aprobacion"},
                    {"name":"Riesgo Predicho","id":RIESGO},
                    {"name":"Probabilidad (%)","id":"proba_pred"},
                    {"name":"Créditos Activos","id":"_credits_by_id"}
                ],
                data=[],
                page_size=20,
                sort_action="native",
                style_table={"overflowX":"auto"},
                style_header={"backgroundColor":brand,"color":"white","fontWeight":"700"},
                style_cell={"padding":"8px","border":"none"}
            )
        ]),
        html.Br()
    ], fluid=True)
])

# callbacks
def filtrar(_df, q, riesgo, f_ini, f_fin, fac, prog, cli, fac_clu, prog_clu):
    dff = _df.copy()
    if q: dff = dff[dff["nombre"].str.lower().str.contains(q.lower(), na=False)]
    if riesgo: dff = dff[dff[RIESGO].isin(riesgo)]
    if f_ini: dff = dff[dff["fecha_aprobacion"] >= f_ini]
    if f_fin: dff = dff[dff["fecha_aprobacion"] <= f_fin]
    if fac: dff = dff[dff["facultad"].astype(str).isin(fac)]
    if prog: dff = dff[dff["programa"].astype(str).isin(prog)]
    if fac_clu: dff = dff[dff["facultad_cluster"].astype(str).isin(fac_clu)]
    if prog_clu: dff = dff[dff["programa_cluster"].astype(str).isin(prog_clu)]
    if cli and "cliente" in dff.columns: dff = dff[dff["cliente"].isin(cli)]
    return dff

@app.callback(
    [Output("kpi-n","children"), Output("kpi-mora","children"),
     Output("kpi-prob","children"), Output("kpi-riesgos","children"),
     Output("kpi-acc","children"), Output("kpi-rec-alto","children"),
     Output("kpi-gap-alto","children"), Output("kpi-lift","children"),
     Output("kpi-exp-esp","children"), Output("kpi-exp-alto","children"),
     Output("g-resumen","figure"), Output("g-cuotas-mes","figure"),
     Output("g-heat","figure"), Output("g-comp","figure"),
     Output("g-mapa","figure"), Output("tbl-mora","data")],
    [Input("f-nombre","value"), Input("f-riesgo","value"),
     Input("f-fecha","start_date"), Input("f-fecha","end_date"),
     Input("f-fac","value"), Input("f-prog","value"), Input("f-cli","value"),
     Input("f-fac-clu","value"), Input("f-prog-clu","value")]
)
def update(q, riesgo, f_ini, f_fin, fac, prog, cli, fac_clu, prog_clu):
    dff = filtrar(df, q, riesgo, f_ini, f_fin, fac, prog, cli, fac_clu, prog_clu)
    n = len(dff)
    mora = (dff["mora_flag"].mean()*100 if n>0 else 0.0)
    prob = (dff["proba_pred"].mean()*100 if n>0 else 0.0)

    # chips riesgo
    r_cnt = dff[RIESGO].value_counts(dropna=True)
    chips = [html.Span(f"{r}: {int(r_cnt.get(r,0))}", style={
        "backgroundColor":"#eef3fb","padding":"4px 8px","borderRadius":"999px","fontWeight":"700"}) for r in ORDEN]

    # desempeño (si hay y_true)
    if "y_true" in dff.columns:
        y_true = dff["y_true"].astype(str).str.capitalize()
        y_pred = dff[RIESGO].astype(str)
        acc = (y_true == y_pred).mean()*100 if n>0 else 0.0
        mask_alto = y_true.eq("Alto")
        recall_alto = ((y_pred[mask_alto].eq("Alto")).mean()*100) if mask_alto.any() else 0.0
        p_real_alto = y_true.eq("Alto").mean()*100
        p_pred_alto = y_pred.eq("Alto").mean()*100
        gap_alto = p_real_alto - p_pred_alto
    else:
        acc = recall_alto = gap_alto = 0.0

    # lift vs base
    base = dff["mora_flag"].mean() if n>0 else 0.0
    palto = dff.loc[dff[RIESGO].eq("Alto"), "mora_flag"].mean() if (RIESGO in dff and dff[RIESGO].eq("Alto").any()) else np.nan
    lift = (palto/base) if base>0 and pd.notna(palto) else 0.0

    # métricas financieras
    if VAL_COL and VAL_COL in dff.columns and n>0:
        val = dff[VAL_COL].fillna(0).clip(lower=0)
        exp_esp = ((dff["proba_pred"].fillna(0)*val).sum() / val.replace(0,np.nan).sum())*100 if val.sum()>0 else 0.0
        exp_alto = val[dff[RIESGO].eq("Alto")].sum()
    else:
        exp_esp = 0.0; exp_alto = 0.0

    # figuras
    f1 = fig_riesgo_resumen(dff)
    f2 = fig_cuotas_mes(dff)
    f3 = fig_heat_cluster_anio(dff)
    f4 = fig_real_vs_pred(dff)
    f5 = fig_mapa(dff)

    # tabla: todos los casos en mora, orden fecha desc y riesgo 
    cols = ["nombre","programa","fecha_aprobacion", RIESGO, "proba_pred","_credits_by_id"]
    top = dff[dff["mora_flag"]==1].copy()
    top[RIESGO] = pd.Categorical(top[RIESGO], categories=ORDEN, ordered=True)
    top = top.sort_values(["fecha_aprobacion", RIESGO], ascending=[False, True])
    if "proba_pred" in top.columns:
        top["proba_pred"] = (top["proba_pred"].astype(float)*100).round(1)
    data = top[cols].to_dict("records")

    # KPIs
    k1 = kpi_card("Créditos (filtro)", f"{n:,}")
    k2 = kpi_card("Mora Datacrédito", f"{mora:.1f}%")
    k3 = kpi_card("Probabilidad promedio", f"{prob:.1f}%")
    k4 = chips
    k5 = kpi_card("Accuracy del modelo", f"{acc:.1f}%")
    k6 = kpi_card("Recall en alto", f"{recall_alto:.1f}%")
    k7 = kpi_card("Gap alto (real - pred.)", f"{gap_alto:.1f} pp")
    k8 = kpi_card("Lift alto vs base", f"{lift:.2f}x")
    k9 = kpi_card("Exposición esperada", f"{exp_esp:.1f}%")
    k10 = kpi_card("Exposición en alto", f"{exp_alto:,.0f}")

    return k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, f1, f2, f3, f4, f5, data

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)



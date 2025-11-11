# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import re, unicodedata, hashlib

# ================== Config & Theme ==================
st.set_page_config(page_title="Riesgo de morosidad en créditos estudiantiles", layout="wide")
brand, bg, bg_soft = "#003366", "#f7f9fb", "#f0f4f8"

# Inject minimal Bootstrap-like CSS (cards, spacing, shadows, rounded, fonts)
st.markdown(f"""
<style>
:root {{
  --brand: {brand};
  --bg: {bg};
  --bg-soft: {bg_soft};
  --text: #222;
  --muted: #667;
}}
html, body, [class*="css"]  {{
  font-family: Arial, sans-serif;
}}
.container-soft {{
  background: var(--bg);
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 16px;
}}
.card {{
  background: white;
  border: 1px solid #e6e6e6;
  border-radius: 16px;
  box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}}
.card > .card-body {{
  padding: 16px 18px;
}}
.kpi-title {{
  color: var(--muted);
  font-weight: 600;
  margin-bottom: 4px;
}}
.kpi-value {{
  color: var(--brand);
  margin: 0;
}}
.chip {{
  display: inline-block;
  background: #eef3fb;
  padding: 4px 8px;
  border-radius: 999px;
  font-weight: 700;
  margin: 2px 4px;
}}
.header-wrap {{
  display: flex;
  align-items: center;
  justify-content: flex-start;
  padding: 15px 25px;
  background: var(--bg-soft);
  border-bottom: 2px solid #ccc;
  box-shadow: 0 2px 5px rgba(0,0,0,0.1);
  border-radius: 8px;
  margin-bottom: 12px;
}}
.header-divider {{
  border-left: 2px solid #ccc;
  height: 58px;
  margin: 0 18px;
}}
.badge-updated {{
  background: var(--brand);
  color: white;
  padding: 10px 16px;
  border-radius: 8px;
  margin: 10px 0 20px 25px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  display: inline-block;
  font-weight: 500;
  font-size: 16px;
}}
.h2-title {{
  color: var(--brand);
}}
.small-muted {{
  color: #666;
}}
</style>
""", unsafe_allow_html=True)

# ================== Utils ==================
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

# ================== Data ==================
@st.cache_data
def load_data():
    DATA = Path("DB_Model") / "df_dash_with_preds.csv"
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
    df["nombre"] = df["nombre"].astype(str).map(_proper_case)

    # clusters mínimos
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
    col_id2 = next((c for c in df.columns if c.lower()=="idbanner"), None)
    if col_id2:
        df["_credits_by_id"] = df.groupby(col_id2)[col_id2].transform("size")
    else:
        df["_credits_by_id"] = 1

    # coordenadas
    lat_col = next((c for c in df.columns if c.lower()=="latitud"), None)
    lon_col = next((c for c in df.columns if c.lower()=="longitud"), None)

    # valor de exposición (para métricas financieras)
    VAL_COL = next((c for c in df.columns if c.lower() in {"valor_financiacion","vr_neto_matricula"}), None)

    return df, ultima_fecha, RIESGO, ORDEN, lat_col, lon_col, VAL_COL

df, ultima_fecha, RIESGO, ORDEN, lat_col, lon_col, VAL_COL = load_data()

# ================== Header ==================
st.markdown('<div class="header-wrap">', unsafe_allow_html=True)
col_a, col_b, col_c, col_d = st.columns([1,0.1,6,3], gap="small")
with col_a:
    try:
        st.image("assets/logo_uni.png", use_container_width=False)
    except Exception:
        st.write("")
with col_b:
    st.markdown('<div class="header-divider"></div>', unsafe_allow_html=True)
with col_c:
    st.markdown('<h3 class="h2-title" style="margin:0;">Riesgo de morosidad en créditos estudiantiles</h3>', unsafe_allow_html=True)
with col_d:
    st.markdown(
        f'<div class="badge-updated">Data last updated on | {ultima_fecha.strftime("%Y-%m-%d") if pd.notna(ultima_fecha) else "-"}</div>',
        unsafe_allow_html=True
    )
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f'<div class="small-muted">Actualización: {ultima_fecha.strftime("%Y-%m-%d") if pd.notna(ultima_fecha) else "-"}</div>', unsafe_allow_html=True)
st.write("")

# ================== Filtros (sin labels, sobrios) ==================
with st.container():
    st.markdown('<div class="container-soft card"><div class="card-body">', unsafe_allow_html=True)
    r1c1, r1c2, r1c3, r1c4 = st.columns(4, gap="medium")
    with r1c1:
        q = st.text_input(" ", placeholder="Buscar nombre...", label_visibility="collapsed")
    with r1c2:
        riesgos = st.multiselect(" ", ORDEN, placeholder="Riesgo predicho", label_visibility="collapsed")
    with r1c3:
        if df["fecha_aprobacion"].notna().any():
            f_ini, f_fin = st.date_input(" ", value=(df["fecha_aprobacion"].min().date(),
                                                     df["fecha_aprobacion"].max().date()),
                                         format="YYYY-MM-DD", label_visibility="collapsed")
        else:
            f_ini = f_fin = None
    with r1c4:
        clientes = sorted(df.get("cliente", pd.Series(dtype=str)).dropna().unique()) if "cliente" in df.columns else []
        cli = st.multiselect(" ", options=clientes, placeholder="Cliente", label_visibility="collapsed")

    r2c1, r2c2, r2c3, r2c4 = st.columns(4, gap="medium")
    with r2c1:
        fac = st.multiselect(" ", sorted(df["facultad"].astype(str).dropna().unique()), placeholder="Facultad (original)", label_visibility="collapsed")
    with r2c2:
        prog = st.multiselect("  ", sorted(df["programa"].astype(str).dropna().unique()), placeholder="Programa (original)", label_visibility="collapsed")
    with r2c3:
        fac_clu = st.multiselect("   ", sorted(df["facultad_cluster"].astype(str).dropna().unique()), placeholder="Facultad (cluster)", label_visibility="collapsed")
    with r2c4:
        prog_clu = st.multiselect("    ", sorted(df["programa_cluster"].astype(str).dropna().unique()), placeholder="Programa (cluster)", label_visibility="collapsed")
    st.markdown('</div></div>', unsafe_allow_html=True)

# ================== Filtrado ==================
dff = df.copy()
if q: dff = dff[dff["nombre"].str.lower().str.contains(q.lower(), na=False)]
if riesgos: dff = dff[dff[RIESGO].isin(riesgos)]
if f_ini: dff = dff[dff["fecha_aprobacion"] >= pd.to_datetime(f_ini)]
if f_fin: dff = dff[dff["fecha_aprobacion"] <= pd.to_datetime(f_fin)]
if fac: dff = dff[dff["facultad"].astype(str).isin(fac)]
if prog: dff = dff[dff["programa"].astype(str).isin(prog)]
if fac_clu: dff = dff[dff["facultad_cluster"].astype(str).isin(fac_clu)]
if prog_clu: dff = dff[dff["programa_cluster"].astype(str).isin(prog_clu)]
if "cliente" in dff.columns and cli:
    dff = dff[dff["cliente"].isin(cli)]

# ================== KPIs ==================
n = len(dff)
mora = (dff["mora_flag"].mean()*100 if n>0 else 0.0)
prob = (dff["proba_pred"].mean()*100 if n>0 else 0.0)

st.write("")
k1,k2,k3,k4 = st.columns([1,1,1,2], gap="medium")
with k1:
    st.markdown('<div class="card"><div class="card-body"><div class="kpi-title">Créditos (filtro)</div><h3 class="kpi-value">{:,}</h3></div></div>'.format(n), unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="card"><div class="card-body"><div class="kpi-title">Mora Datacrédito</div><h3 class="kpi-value">{mora:.1f}%</h3></div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="card"><div class="card-body"><div class="kpi-title">Probabilidad promedio</div><h3 class="kpi-value">{prob:.1f}%</h3></div></div>', unsafe_allow_html=True)
with k4:
    r_cnt = dff[RIESGO].value_counts(dropna=True) if RIESGO in dff.columns else pd.Series(dtype=int)
    chips_html = "".join([f'<span class="chip">{r}: {int(r_cnt.get(r,0))}</span>' for r in ["Alto","Medio","Bajo"]])
    st.markdown(f'<div class="card"><div class="card-body"><div class="kpi-title">Niveles de riesgo</div><div>{chips_html}</div></div></div>', unsafe_allow_html=True)

# Segunda fila de KPIs
if "y_true" in dff.columns:
    y_true = dff["y_true"].astype(str).str.capitalize()
    y_pred = dff[RIESGO].astype(str) if RIESGO in dff.columns else pd.Series([""], index=dff.index)
    acc = (y_true == y_pred).mean()*100 if n>0 else 0.0
    mask_alto = y_true.eq("Alto")
    recall_alto = ((y_pred[mask_alto].eq("Alto")).mean()*100) if mask_alto.any() else 0.0
    p_real_alto = y_true.eq("Alto").mean()*100
    p_pred_alto = y_pred.eq("Alto").mean()*100
    gap_alto = p_real_alto - p_pred_alto
else:
    acc = recall_alto = gap_alto = 0.0

base = dff["mora_flag"].mean() if n>0 else 0.0
palto = dff.loc[dff[RIESGO].eq("Alto"), "mora_flag"].mean() if (RIESGO in dff.columns and dff[RIESGO].eq("Alto").any()) else np.nan
lift = (palto/base) if base>0 and pd.notna(palto) else 0.0

if VAL_COL and VAL_COL in dff.columns and n>0:
    val = dff[VAL_COL].fillna(0).clip(lower=0)
    exp_esp = ((dff["proba_pred"].fillna(0)*val).sum() / val.replace(0,np.nan).sum())*100 if val.sum()>0 else 0.0
    exp_alto = val[dff[RIESGO].eq("Alto")].sum() if RIESGO in dff.columns else 0.0
else:
    exp_esp = 0.0; exp_alto = 0.0

st.write("")
k21,k22,k23,k24,k25,k26 = st.columns(6, gap="medium")
with k21: st.markdown(f'<div class="card"><div class="card-body"><div class="kpi-title">Accuracy del modelo</div><h3 class="kpi-value">{acc:.1f}%</h3></div></div>', unsafe_allow_html=True)
with k22: st.markdown(f'<div class="card"><div class="card-body"><div class="kpi-title">Recall en alto</div><h3 class="kpi-value">{recall_alto:.1f}%</h3></div></div>', unsafe_allow_html=True)
with k23: st.markdown(f'<div class="card"><div class="card-body"><div class="kpi-title">Gap alto (real - pred.)</div><h3 class="kpi-value">{gap_alto:.1f} pp</h3></div></div>', unsafe_allow_html=True)
with k24: st.markdown(f'<div class="card"><div class="card-body"><div class="kpi-title">Lift alto vs base</div><h3 class="kpi-value">{lift:.2f}x</h3></div></div>', unsafe_allow_html=True)
with k25: st.markdown(f'<div class="card"><div class="card-body"><div class="kpi-title">Exposición esperada</div><h3 class="kpi-value">{exp_esp:.1f}%</h3></div></div>', unsafe_allow_html=True)
with k26: st.markdown(f'<div class="card"><div class="card-body"><div class="kpi-title">Exposición en alto</div><h3 class="kpi-value">{exp_alto:,.0f}</h3></div></div>', unsafe_allow_html=True)

# ================== Gráficos (3 + 2) ==================
st.write("")
g_row1_c1, g_row1_c2, g_row1_c3 = st.columns(3, gap="large")

# 1) Resumen riesgo
with g_row1_c1:
    if RIESGO in dff.columns and len(dff)>0:
        g = dff[RIESGO].value_counts(dropna=True).rename_axis("riesgo").reset_index(name="n")
        g["pct"] = g["n"]/g["n"].sum()*100
        g["etq"] = g["n"].map("{:,.0f}".format) + " | " + g["pct"].map("{:,.1f}%".format)
        fig1 = px.bar(
            g.sort_values("riesgo", key=lambda s: s.map({k:i for i,k in enumerate(ORDEN)})),
            x="riesgo", y="n", text="etq", color="riesgo",
            color_discrete_map={"Alto":"#1f77b4","Medio":"#4ba3d3","Bajo":"#9ecae1"},
            title="Créditos por riesgo"
        )
        fig1.update_traces(textposition="outside", showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

# 2) Cuotas por mes y riesgo
with g_row1_c2:
    if all(c in dff.columns for c in ["fecha_aprobacion", "cuotas", RIESGO]) and dff["fecha_aprobacion"].notna().any():
        tmp = dff.dropna(subset=["fecha_aprobacion"]).copy()
        tmp["ym"] = tmp["fecha_aprobacion"].dt.to_period("M").astype(str)
        g2 = (tmp.groupby(["ym", RIESGO])["cuotas"].mean().reset_index(name="cuotas_prom"))
        g2 = g2[g2[RIESGO].notna()]
        fig2 = px.line(
            g2.sort_values("ym"),
            x="ym", y="cuotas_prom", color=RIESGO, markers=True,
            color_discrete_map={"Alto":"#1f77b4","Medio":"#4ba3d3","Bajo":"#9ecae1"},
            title="Cuotas por mes y riesgo"
        )
        st.plotly_chart(fig2, use_container_width=True)

# 3) Heatmap mora por programa (cluster) con Title Case
with g_row1_c3:
    tmp = dff.groupby(["programa_cluster","anio"], observed=False)["mora_flag"].mean().reset_index(name="mora_pct")
    tmp["mora_pct"] = tmp["mora_pct"]*100
    tmp["programa_cluster_t"] = tmp["programa_cluster"].astype(str).str.title()
    fig3 = px.density_heatmap(
        tmp, x="anio", y="programa_cluster_t", z="mora_pct",
        color_continuous_scale="Blues", title="Mora por programa y año"
    )
    fig3.update_layout(coloraxis_colorbar_title="%")
    st.plotly_chart(fig3, use_container_width=True)

# Fila 2 de gráficos
g_row2_c1, g_row2_c2 = st.columns(2, gap="large")

# 4) Real vs predicho (etiquetas en %)
with g_row2_c1:
    if "y_true" in dff.columns and RIESGO in dff.columns:
        g1 = dff["y_true"].astype(str).str.capitalize().value_counts().rename_axis("clase").reset_index(name="n_real")
        g2 = dff[RIESGO].astype(str).value_counts().rename_axis("clase").reset_index(name="n_pred")
        g4 = pd.merge(g1, g2, on="clase", how="outer").fillna(0)
        g4["n_real"] = g4["n_real"].astype(int); g4["n_pred"] = g4["n_pred"].astype(int)
        g4 = g4.sort_values("clase", key=lambda s: s.map({k:i for i,k in enumerate(ORDEN)}))
        total_real = g4["n_real"].sum() if g4["n_real"].sum()>0 else 1
        total_pred = g4["n_pred"].sum() if g4["n_pred"].sum()>0 else 1
        pct_real = (g4["n_real"]/total_real*100).round(1).astype(str) + "%"
        pct_pred = (g4["n_pred"]/total_pred*100).round(1).astype(str) + "%"
        fig4 = px.bar(g4, x="clase", y=["n_real","n_pred"], barmode="group", title="Real vs predicho")
        if len(fig4.data) >= 2:
            fig4.data[0].text = pct_real
            fig4.data[1].text = pct_pred
            fig4.data[0].textposition = "outside"
            fig4.data[1].textposition = "outside"
        st.plotly_chart(fig4, use_container_width=True)

# 5) Mapa por riesgo (si hay lat/lon)
with g_row2_c2:
    if lat_col and lon_col and dff[[lat_col, lon_col]].notna().any().any():
        dd = dff.dropna(subset=[lat_col, lon_col]).copy()
        if not dd.empty:
            fig5 = px.scatter_mapbox(
                dd, lat=lat_col, lon=lon_col, color=RIESGO, hover_name="nombre",
                color_discrete_map={"Alto":"#1f77b4","Medio":"#4ba3d3","Bajo":"#9ecae1"},
                zoom=4, height=420, title="Mapa por riesgo"
            )
            fig5.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,t=60,b=0))
            st.plotly_chart(fig5, use_container_width=True)

# ================== Tabla ==================
st.write("")
st.markdown('<div class="card"><div class="card-body"><div style="color: var(--brand);font-weight:700;padding:0 0 8px 0;">Casos en Mora Datacrédito</div>', unsafe_allow_html=True)
top = dff[dff["mora_flag"]==1].copy()
if RIESGO in top.columns:
    top[RIESGO] = pd.Categorical(top[RIESGO], categories=ORDEN, ordered=True)
top = top.sort_values(["fecha_aprobacion", RIESGO], ascending=[False, True])
if "proba_pred" in top.columns:
    top["Probabilidad (%)"] = (top["proba_pred"].astype(float)*100).round(1)
cols = ["nombre","programa","fecha_aprobacion", RIESGO, "Probabilidad (%)","_credits_by_id"]
cols = [c for c in cols if c in top.columns]
st.dataframe(top[cols], use_container_width=True, height=420)
st.markdown('</div></div>', unsafe_allow_html=True)

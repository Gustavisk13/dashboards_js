import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HOURS = list(range(23))

DAY_LABELS = [
    "Domingo",
    "Segunda",
    "Terça",
    "Quarta",
    "Quinta",
    "Sexta",
    "Sábado",
]

DAY_ORDER_MAP = {
    6: "Domingo",
    0: "Segunda",
    1: "Terça",
    2: "Quarta",
    3: "Quinta",
    4: "Sexta",
    5: "Sábado",
}

DAY_SORT_INDEX = {
    "Domingo": 0,
    "Segunda": 1,
    "Terça": 2,
    "Quarta": 3,
    "Quinta": 4,
    "Sexta": 5,
    "Sábado": 6,
}


@st.cache_data(show_spinner=False)
def load_data(json_source) -> pd.DataFrame:
    if hasattr(json_source, "read"):
        raw = json.load(json_source)
    else:
        with open(json_source, "r", encoding="utf-8") as f:
            raw = json.load(f)

    df = pd.DataFrame(raw)
    df.columns = df.columns.str.lower()

    df["ad_dhalter"] = pd.to_datetime(df["ad_dhalter"], errors="coerce")
    df["dtneg"] = pd.to_datetime(df["dtneg"], errors="coerce")

    df["event_at"] = df["ad_dhalter"].fillna(df["dtneg"])
    df = df.dropna(subset=["event_at"]).copy()

    df["tipo"] = df["tipo"].astype(str).str.upper().str.strip()
    df["origem"] = df["ad_digitalorigemapp"].astype(str).str.upper().str.strip()

    df["year_month"] = df["event_at"].dt.to_period("M").astype(str)
    df["date"] = df["event_at"].dt.date
    df["hour"] = df["event_at"].dt.hour
    df["dayofweek_pd"] = df["event_at"].dt.dayofweek
    df["day_name_pt"] = df["dayofweek_pd"].map(DAY_ORDER_MAP)
    df["day_sort"] = df["day_name_pt"].map(DAY_SORT_INDEX)

    df = df[df["hour"].isin(HOURS)].copy()

    return df


def filter_df(df: pd.DataFrame, month: str, tipo: str, origem: str) -> pd.DataFrame:
    filtered = df[df["year_month"] == month].copy()

    if tipo != "TODOS":
        filtered = filtered[filtered["tipo"] == tipo]

    if origem != "TODOS":
        filtered = filtered[filtered["origem"] == origem]

    return filtered


def previous_month(period_str: str) -> str:
    p = pd.Period(period_str, freq="M")
    return str(p - 1)


def format_delta(current: int, previous: int | None) -> str | None:
    if previous is None:
        return None
    if previous == 0:
        if current == 0:
            return "0.0%"
        return "Novo"
    pct = ((current - previous) / previous) * 100
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.1f}%"


def compute_metrics(current_df: pd.DataFrame, previous_df: pd.DataFrame | None) -> dict:
    total_sales = len(current_df)

    previous_total = None if previous_df is None else len(previous_df)
    delta_total = format_delta(total_sales, previous_total)

    if current_df.empty:
        avg_daily = 0.0
        best_day = "-"
        best_hour_range = "-"
    else:
        daily_counts = current_df.groupby("date").size().sort_values(ascending=False)
        avg_daily = daily_counts.mean()

        best_day_key = (
            current_df.groupby("day_name_pt")
            .size()
            .sort_values(ascending=False)
            .index[0]
        )
        best_day = best_day_key

        hour_counts = (
            current_df.groupby("hour")
            .size()
            .reindex(HOURS, fill_value=0)
            .sort_values(ascending=False)
        )
        best_hour = int(hour_counts.index[0])
        best_hour_range = f"{best_hour:02d}:00 - {best_hour:02d}:59"

    return {
        "total_sales": total_sales,
        "delta_total": delta_total,
        "avg_daily": avg_daily,
        "best_day": best_day,
        "best_hour_range": best_hour_range,
    }


def build_heatmap_matrix(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        base = pd.DataFrame(0, index=DAY_LABELS, columns=HOURS)
        return base

    heatmap = (
        df.groupby(["day_name_pt", "hour"])
        .size()
        .reset_index(name="count")
        .pivot(index="day_name_pt", columns="hour", values="count")
        .fillna(0)
    )

    heatmap = heatmap.reindex(DAY_LABELS, fill_value=0)
    heatmap = heatmap.reindex(columns=HOURS, fill_value=0)

    return heatmap


PURPLE_COLORSCALE = [
    [0.0, "#f5eeff"],
    [0.2, "#d4a8ff"],
    [0.4, "#9933ff"],
    [0.6, "#6600cc"],
    [0.8, "#3d0066"],
    [1.0, "#1a0033"],
]


def make_heatmap_figure(heatmap_df: pd.DataFrame, titulo: str) -> go.Figure:
    z = heatmap_df.values
    x = [f"{h:02d}h" for h in heatmap_df.columns]
    y = list(heatmap_df.index)

    max_val = int(np.nanmax(z)) if z.size else 0
    text = [[("" if int(v) == 0 else str(int(v))) for v in row] for row in z]

    axis_font = dict(size=14, family="Arial Black, Arial, sans-serif")

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            text=text,
            texttemplate="%{text}",
            textfont={
                "size": 14,
                "color": "#ffffff",
                "family": "Arial Black, Arial, sans-serif",
            },
            colorscale=PURPLE_COLORSCALE,
            hovertemplate=("<b>%{y}</b><br>Hora: %{x}<br>Vendas: %{z}<extra></extra>"),
            colorbar={"title": {"text": "Qtd"}},
            zmin=0,
            zmax=max(max_val, 1),
            xgap=2,
            ygap=2,
        )
    )

    fig.update_layout(
        title=dict(text=titulo),
        height=560,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    fig.update_xaxes(side="top", tickfont=axis_font, title_font=axis_font)
    fig.update_yaxes(autorange="reversed", tickfont=axis_font, title_font=axis_font)

    return fig


def build_summary_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "Sem dados para os filtros selecionados."

    total = len(df)
    total_value = df["vlrnota"].fillna(0).sum()

    return (
        (f"{total:,} registros filtrados • R$ {total_value:,.2f} em valor total")
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )


st.set_page_config(
    page_title="Mapa de Calor de Vendas",
    page_icon=":material/local_fire_department:",
    layout="wide",
)

st.title(":material/local_fire_department: Mapa de Calor de Vendas")
st.caption(
    "Alterne entre pedidos, orçamentos e origem digital para visualizar concentração de vendas por dia e hora."
)

with st.sidebar:
    st.header("Filtros")

    uploaded = st.file_uploader(
        "JSON de dados",
        type=["json"],
    )

    if uploaded is None:
        st.info("Envie um arquivo JSON para começar.")
        st.stop()

    try:
        df = load_data(uploaded)
    except Exception as e:
        st.error(f"Não consegui carregar o JSON: {e}")
        st.stop()

    available_months = sorted(df["year_month"].dropna().unique().tolist())
    available_months = list(reversed(available_months))

    if not available_months:
        st.warning("Nenhum mês disponível no arquivo.")
        st.stop()

    selected_month = st.selectbox("Mês", available_months, index=0)

    tipo_options = ["TODOS"] + sorted(df["tipo"].dropna().unique().tolist())
    origem_options = ["TODOS"] + sorted(df["origem"].dropna().unique().tolist())

    selected_tipo = st.segmented_control(
        "Tipo",
        options=tipo_options,
        default="TODOS",
        selection_mode="single",
    )

    if selected_tipo is None:
        selected_tipo = "TODOS"

    selected_origem = st.segmented_control(
        "Origem",
        options=origem_options,
        default="TODOS",
        selection_mode="single",
    )

    if selected_origem is None:
        selected_origem = "TODOS"

current_df = filter_df(df, selected_month, selected_tipo, selected_origem)

prev_month = previous_month(selected_month)
previous_df = None
if prev_month in df["year_month"].unique():
    previous_df = filter_df(df, prev_month, selected_tipo, selected_origem)

metrics = compute_metrics(current_df, previous_df)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Vendas totais",
    f"{metrics['total_sales']:,}".replace(",", "."),
    metrics["delta_total"],
)

c2.metric(
    "Venda média diária",
    f"{metrics['avg_daily']:.1f}".replace(".", ","),
)

c3.metric(
    "Dia com mais vendas",
    metrics["best_day"],
)

c4.metric(
    "Intervalo com mais vendas",
    metrics["best_hour_range"],
)

st.markdown("")

summary_left, summary_right = st.columns([2, 1])
with summary_left:
    st.subheader("Distribuição semanal por hora")
    st.caption(build_summary_text(current_df))

with summary_right:
    show_table = st.toggle("Mostrar tabela-base", value=False)

heatmap_df = build_heatmap_matrix(current_df)

title = f"{selected_month} • {selected_tipo} • {selected_origem}"
fig = make_heatmap_figure(heatmap_df, title)
st.plotly_chart(fig, use_container_width=True)

if show_table:
    st.dataframe(
        heatmap_df.astype(int),
        use_container_width=True,
    )

st.markdown("---")
st.markdown(
    """
**Leitura do gráfico**
- Tons mais quentes indicam maior concentração de vendas.
- Células com `0` ficam mais frias e sem rótulo.
- O delta do card de vendas totais compara o mês selecionado com o mês anterior, usando os mesmos filtros.
"""
)

"""Componentes visuais do MLP Studio, sem cálculos de aprendizado."""

import plotly.graph_objects as go
import streamlit as st

from mlp.ui_theme import PURPLE, CYAN, LABELS, chart_style
from mlp.architecture_view import architecture_summary, architecture_figure, show_architecture


def page_header(title, description, icon):
    """Cabeçalho consistente para cada área da aplicação."""
    st.caption(f":material/{icon}: MLP STUDIO")
    st.header(title, anchor=False)
    st.caption(description)


def section(title, description=None):
    """Título de seção com descrição opcional."""
    st.subheader(title, anchor=False)
    if description:
        st.caption(description)


def message(text, kind="info"):
    """Mensagens nativas com ícones discretos."""
    icons = {"info": ":material/info:", "success": ":material/check_circle:", "error": ":material/error:"}
    getattr(st, kind)(text, icon=icons[kind])


def metric_card(label, value, caption=None, delta=None):
    """Card de métrica real; usa travessão quando o valor não está disponível."""
    with st.container(border=True, key=None):
        st.metric(label, "—" if value is None else value, delta=delta,
                  delta_color="inverse" if delta is not None else "off")
        if caption:
            st.caption(caption)


def training_figure(history, height=350):
    """Curvas com os valores reais do histórico existente, sem interpolação."""
    figure = go.Figure()
    for field, label, color in [("train_mse", "Treinamento", PURPLE), ("validation_mse", "Validação", CYAN)]:
        figure.add_trace(go.Scatter(
            x=[row["epoch"] for row in history], y=[row[field] for row in history],
            mode="lines+markers" if len(history) == 1 else "lines", name=label,
            line=dict(color=color, width=2.5),
            hovertemplate=f"{label}<br>Época %{{x}}<br>MSE %{{y:.8g}}<extra></extra>",
        ))
    chart_style(figure, height)
    figure.update_layout(hovermode="x unified", xaxis_title="Época", yaxis_title="Erro quadrático médio")
    return figure


def split_figure(datasets):
    """Proporções reais após arredondamento, sem usar percentuais pretendidos."""
    names = ["train", "validation", "test"]
    figure = go.Figure(go.Pie(
        labels=["Treinamento", "Validação", "Teste"],
        values=[len(datasets[name][0]) for name in names], hole=0.72,
        marker=dict(colors=[PURPLE, CYAN, "#64748B"], line=dict(color="#172033", width=3)),
        textinfo="percent", hovertemplate="%{label}<br>%{value} exemplos · %{percent}<extra></extra>",
        sort=False,
    ))
    chart_style(figure, 280)
    figure.update_layout(legend=dict(orientation="h", y=-0.1, x=0))
    return figure

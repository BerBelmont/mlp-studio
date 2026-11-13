"""Página inicial com estados vazios e ações rápidas."""

import streamlit as st

from mlp.visual import page_header, section, metric_card, architecture_figure, architecture_summary, training_figure
from mlp.ui.common import navigate, quick_actions


def dashboard_page():
    page_header("Dashboard", "Seu modelo, seus dados e a evolução do aprendizado.", "dashboard")
    network = st.session_state.network
    history = st.session_state.history
    if network is None:
        left, right = st.columns([0.6, 0.4], gap="large")
        with left:
            with st.container(border=True):
                st.subheader("Da primeira camada à primeira previsão")
                st.write("Configure uma arquitetura e acompanhe sua rede aprendendo com os seus dados.")
                st.button("Criar minha primeira rede", type="primary", on_click=navigate, args=("Arquitetura Neural",))
                st.button("Já tenho um modelo JSON", on_click=navigate, args=("Modelos e Previsões",))
        with right:
            section("Seu próximo passo")
            st.write("01 · Desenhe a rede")
            st.write("02 · Prepare o dataset")
            st.write("03 · Treine, avalie e explore")
        return
    quick_actions()
    status = "Treinado nesta sessão" if history else "Modelo disponível" if st.session_state.model_ready else "Pronto para configurar dados"
    for column, label, value in zip(st.columns(4), ["Arquitetura", "Neurônios", "Camadas", "Status"],
                                   [architecture_summary(network), sum(len(layer.neurons) for layer in network.layers), len(network.layers), status]):
        with column:
            metric_card(label, value)
    map_column, learning = st.columns([0.48, 0.52], gap="large")
    with map_column:
        section("Modelo ativo")
        st.plotly_chart(architecture_figure(network), width="stretch", config={"displaylogo": False})
        if st.session_state.prepared is not None:
            st.caption(" · ".join(f"{label}: {len(st.session_state.prepared[name][0])}" for name, label in [("train", "Treino"), ("validation", "Validação"), ("test", "Teste")]))
    with learning:
        section("Aprendizado")
        if history:
            st.plotly_chart(training_figure(history, height=360), width="stretch", config={"displaylogo": False})
            for column, name, label in zip(st.columns(2), ["train_mse", "validation_mse"], ["MSE treino", "MSE validação"]):
                with column:
                    metric_card(label, f"{history[-1][name]:.6g}")
        else:
            with st.container(border=True):
                st.subheader("Um experimento começa com seus dados")
                st.write("As curvas e métricas aparecerão após o treinamento nesta sessão.")
                st.button("Preparar dataset", type="primary", on_click=navigate, args=("Dataset",))

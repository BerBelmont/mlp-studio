"""Bancada de experimentos: configurações à esquerda e resultados à direita."""

import pandas as pd
import streamlit as st

from mlp.interface import train_session
from mlp.ui.common import optional_seed, parse_seed, navigate, training_results
from mlp.visual import page_header, section, message, metric_card


def training_page():
    page_header("Treinamento", "Configure um experimento e acompanhe o aprendizado do modelo.", "monitoring")
    data = st.session_state.prepared
    if data is None:
        message("Prepare os conjuntos antes de iniciar um experimento.")
        st.button("Preparar dataset", type="primary", on_click=navigate, args=("Dataset",))
        return
    settings, results = st.columns([0.30, 0.70], gap="large")
    with settings:
        with st.container(border=True):
            section("Novo experimento")
            st.caption(f"SGD · {len(data['train'][0])} exemplos de treino")
            with st.form("training_form", border=False):
                epochs = st.number_input("Épocas", min_value=1, value=100, step=1)
                rate = st.number_input("Taxa de aprendizado", min_value=0.000001, value=0.05, format="%.6f")
                with st.expander("Reprodutibilidade"):
                    seed = optional_seed("Seed do embaralhamento", "training_seed")
                submitted = st.form_submit_button("Iniciar treinamento", type="primary", width="stretch")
            st.caption("Continua os pesos atuais. Teste não participa do treinamento.")
        run_summary = st.empty()
    with results:
        if submitted:
            seed = parse_seed(seed)
            with st.status("Executando SGD…", expanded=True) as status:
                train_session(st.session_state, epochs, rate, seed)
                status.update(label="Experimento concluído", state="complete", expanded=False)
        history = st.session_state.history
        if history:
            section("Evolução do MSE", f"Concluído · {len(history)} épocas · métricas ao final de cada época")
            training_results()
            with st.expander("Histórico por época"):
                st.dataframe(pd.DataFrame(history).rename(columns={"epoch": "Época", "train_mse": "MSE treino", "validation_mse": "MSE validação"}), hide_index=True, width="stretch")
        else:
            section("Pronto para experimentar")
            st.write("Defina épocas e taxa de aprendizado. As curvas aparecem aqui quando a execução terminar.")
            for column, name, label in zip(st.columns(2), ["train", "validation"], ["Exemplos de treino", "Exemplos de validação"]):
                with column:
                    metric_card(label, len(data[name][0]))
            message("A validação é usada apenas para avaliação. Nenhuma métrica foi calculada ainda.")

    with run_summary.container():
        config = st.session_state.get("training_config")
        if config and st.session_state.history:
            st.caption("ÚLTIMA EXECUÇÃO")
            st.write(f"{config['epochs']} épocas · taxa {config['learning_rate']:g}")

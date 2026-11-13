"""Avaliação, persistência e previsão usando as APIs existentes."""

import pandas as pd
import streamlit as st
from mlp.interface import comparison_table, import_json, export_json, replace_network
from mlp.visual import page_header, section, metric_card, message

def evaluation_page():
    page_header("Avaliação", "Meça a generalização do modelo no conjunto reservado de teste.", "analytics")
    if st.session_state.prepared is None:
        st.info("Prepare os conjuntos para avaliar o teste.")
        return
    st.caption("O conjunto de teste é usado somente nesta avaliação, sem atualizar pesos ou vieses.")
    if st.button("Avaliar conjunto de teste", type="primary"):
        network = st.session_state.network
        dataset = st.session_state.prepared["test"]
        st.session_state.evaluation = (
            network.evaluate_mse(*dataset),
            comparison_table(network, dataset, st.session_state.output_columns),
        )
    if st.session_state.evaluation is not None:
        mse, table = st.session_state.evaluation
        with st.columns(2)[0]:
            metric_card("MSE final · teste", f"{mse:.6g}", "Avaliação sem atualização dos parâmetros")
        section("Esperado × previsto")
        st.dataframe(table, width="stretch")

def persistence_page():
    restore, export = st.columns(2, gap="large")
    with restore:
        with st.container(border=True):
            section("Restaurar modelo", "Continue a partir de um arquivo JSON salvo.")
            upload = st.file_uploader("Arquivo JSON da rede", type=["json"])
            st.caption("Substitui a rede e descarta conjuntos preparados e histórico.")
            if st.button("Carregar rede JSON", disabled=upload is None, width="stretch"):
                network = import_json(upload.getvalue())
                replace_network(st.session_state, network, loaded=True)
                message("Rede restaurada. Use a aba Previsão manual para explorar novas entradas.", "success")
    with export:
        with st.container(border=True):
            section("Exportar modelo", "Pesos, vieses, arquitetura e normalização em um único arquivo.")
            if st.session_state.network is not None:
                content = export_json(st.session_state.network)
                st.download_button("Baixar rede em JSON", data=content, file_name="mlp_studio.json",
                                   mime="application/json", type="primary", width="stretch")
                st.caption("O formato do arquivo permanece compatível com as versões anteriores.")
            else:
                message("Crie ou carregue uma rede para exportar.")

def prediction_page():
    network = st.session_state.network
    if network is None:
        st.info("Crie ou carregue uma rede antes de prever.")
        return
    method = {"none": "Nenhuma", "minmax": "Min-Max", "zscore": "Z-score"}[network.normalizer.method if network.normalizer is not None else "none"]
    st.caption(f"Informe valores originais. Normalização automática: {method}. A ordem é a mesma das colunas escolhidas no dataset.")
    with st.form(f"prediction_form_{st.session_state.revision}"):
        values = []
        input_fields = st.columns(2)
        names = st.session_state.input_columns
        for i in range(network.input_size):
            label = names[i] if len(names) == network.input_size else f"Entrada {i+1}"
            with input_fields[i % 2]:
                values.append(st.number_input(label, value=0.0, key=f"predict_{st.session_state.revision}_{i}"))
        if st.form_submit_button("Realizar previsão", type="primary"):
            result = network.predict(values)
            st.dataframe(pd.DataFrame({"Saída": [i+1 for i in range(len(result))], "Valor previsto": result}), hide_index=True)
    if not names:
        st.caption("O JSON armazena a ordem dos atributos, mas não seus nomes. Use a ordem original de treinamento.")

def models_page():
    page_header("Modelos e Previsões", "Preserve seus parâmetros e explore novas entradas com o modelo atual.", "model_training")
    files, inference = st.tabs(["Modelos · JSON", "Previsão manual"])
    with files:
        persistence_page()
    with inference:
        section("Executar previsão", "Use valores originais na mesma ordem do treinamento.")
        prediction_page()

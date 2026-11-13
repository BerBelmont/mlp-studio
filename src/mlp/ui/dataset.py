"""Preparação do dataset em três etapas, reutilizando o módulo de dados."""

import pandas as pd
import streamlit as st

from mlp.interface import replace_dataset, prepare_data
from mlp.ui.common import optional_seed, parse_seed
from mlp.visual import page_header, section, metric_card, split_figure, message


def import_stage():
    """Upload e exploração lado a lado, sem alterar o CSV durante a edição."""
    upload_column, overview = st.columns([0.35, 0.65], gap="large")
    with upload_column:
        section("Importar CSV")
        upload = st.file_uploader("Arquivo CSV com cabeçalho", type=["csv"])
        if st.button("Carregar CSV", type="primary", disabled=upload is None, width="stretch"):
            replace_dataset(st.session_state, upload.getvalue(), upload.name)
            st.success("CSV carregado. Continue na etapa 2.")
    with overview:
        frame = st.session_state.dataset
        if frame is None:
            section("Seus dados começam aqui")
            st.write("Envie um CSV numérico com cabeçalho. Depois, escolha quais colunas representam entradas e saídas.")
            st.caption("EXEMPLO DE FORMATO · apenas ilustrativo")
            st.code("entrada_1,entrada_2,saida\n0.1,0.2,0.3\n0.4,0.5,0.9", language="csv")
        else:
            for column, label, value in zip(st.columns(2), ["Registros", "Colunas"], [len(frame), len(frame.columns)]):
                with column:
                    metric_card(label, value)
            st.caption(st.session_state.dataset_name)
            st.dataframe(frame.head(10), width="stretch", hide_index=True)


def configure_stage():
    """Seleção de atributos e divisão, mantendo todas as validações existentes."""
    frame = st.session_state.dataset
    network = st.session_state.network
    if frame is None:
        message("Importe um CSV na etapa 1 para selecionar os atributos.")
        return
    if network is None:
        message("Crie ou carregue uma rede para validar a quantidade de entradas e saídas.")
        return
    locked = st.session_state.model_ready
    registered = network.normalizer
    configuration = st.session_state.preparation_config or {"percentages": [70., 15., 15.], "seed": 42, "method": "none"}
    if locked:
        st.caption("Modelo treinado/carregado: a normalização registrada será preservada.")
    with st.form("dataset_form"):
        attributes, splitting = st.columns([0.55, 0.45], gap="large")
        with attributes:
            section("Entradas e saídas")
            st.caption(f"A rede espera {network.input_size} entradas e {len(network.layers[-1].neurons)} saídas.")
            x = st.multiselect("Colunas de entrada (na ordem desejada)", list(frame.columns), default=st.session_state.input_columns)
            y = st.multiselect("Colunas de saída", list(frame.columns), default=st.session_state.output_columns)
            method = st.selectbox("Normalização das entradas", ["none", "minmax", "zscore"],
                                  index=["none", "minmax", "zscore"].index(registered.method if locked and registered is not None else configuration["method"]),
                                  format_func=lambda name: {"none": "Nenhuma", "minmax": "Min-Max", "zscore": "Z-score"}[name], disabled=locked)
        with splitting:
            section("Divisão dos exemplos")
            columns = st.columns(3)
            percentages = [column.number_input(label, min_value=0., max_value=100., value=value)
                           for column, label, value in zip(columns, ["Treino (%)", "Validação (%)", "Teste (%)"], configuration["percentages"])]
            seed = optional_seed("Seed da divisão", "split_seed", value="" if configuration["seed"] is None else str(configuration["seed"]))
            st.caption("A divisão ocorre antes da normalização. Os parâmetros são ajustados somente no treino.")
        if st.form_submit_button("Validar e preparar conjuntos", type="primary", width="stretch"):
            prepare_data(st.session_state, x, y, percentages, parse_seed(seed), method)
            st.success("Conjuntos preparados. Confira a etapa 3.")


def prepared_stage():
    """Mostra divisão efetiva e tabelas das entradas preparadas e saídas intactas."""
    datasets = st.session_state.prepared
    if datasets is None:
        message("Confirme a preparação na etapa 2 para explorar os conjuntos.")
        return
    for column, name, label in zip(st.columns(3), ["train", "validation", "test"], ["Treinamento", "Validação", "Teste"]):
        with column:
            metric_card(label, len(datasets[name][0]), "Exemplos disponíveis")
    proportion, samples = st.columns([0.35, 0.65], gap="large")
    with proportion:
        section("Divisão efetiva")
        st.plotly_chart(split_figure(datasets), width="stretch", config={"displaylogo": False})
        st.caption("Treino e validação arredondam para baixo. Teste recebe o restante.")
    with samples:
        section("Explore os conjuntos")
        tabs = st.tabs(["Treino", "Validação", "Teste"])
        for tab, name in zip(tabs, ["train", "validation", "test"]):
            with tab:
                x, y = datasets[name]
                table = pd.DataFrame(x, columns=[f"Entrada · {label}" for label in st.session_state.input_columns])
                for i, label in enumerate(st.session_state.output_columns):
                    table[f"Saída · {label}"] = y[:, i]
                st.dataframe(table.head(10), hide_index=True, width="stretch")
                st.caption("Até 10 exemplos · entradas na escala preparada · saídas originais")


def dataset_page():
    page_header("Dataset", "Do CSV aos conjuntos de treino, validação e teste.", "table_chart")
    stages = st.tabs(["1 · Importar CSV", "2 · Configurar dados", "3 · Conjuntos preparados"])
    for tab, render in zip(stages, [import_stage, configure_stage, prepared_stage]):
        with tab:
            render()

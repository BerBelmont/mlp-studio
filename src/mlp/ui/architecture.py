"""Editor reativo com rascunho separado da rede ativa."""

import pandas as pd
import streamlit as st

from mlp.architecture_view import draft_figure, architecture_summary
from mlp.interface import replace_network
from mlp.persistencia import ACTIVATIONS
from mlp.rede import NeuralNetwork
from mlp.ui.common import parse_seed
from mlp.visual import page_header, section, metric_card, LABELS


def network_page():
    """Atualiza o desenho com widgets reativos, sem instanciar redes na prévia."""
    page_header("Arquitetura Neural", "Desenhe a próxima rede. O modelo ativo só muda quando você confirmar.", "hub")
    draft = st.session_state.get("architecture_draft", {
        "input_size": 2, "hidden": [(3, "relu")], "output": (1, "linear"), "seed": "42",
    })
    editor, preview = st.columns([0.35, 0.65], gap="large")
    prefix = f"draft_{st.session_state.revision}"
    with editor:
        with st.container(border=True):
            section("Editor da rede")
            hidden = st.number_input("Quantidade de camadas ocultas", min_value=0, value=len(draft["hidden"]), step=1, key=f"{prefix}_hidden")
            inputs = st.number_input("Quantidade de entradas", min_value=1, value=draft["input_size"], step=1, key=f"{prefix}_inputs")
            configurations = []
            for i in range(hidden):
                default_count, default_activation = draft["hidden"][i] if i < len(draft["hidden"]) else (3, "relu")
                left, right = st.columns([0.45, 0.55])
                count = left.number_input(f"Neurônios · oculta {i+1}", min_value=1, value=default_count, step=1, key=f"{prefix}_count_{i}")
                name = right.selectbox(f"Ativação · oculta {i+1}", list(ACTIVATIONS), index=list(ACTIVATIONS).index(default_activation), format_func=LABELS.get, key=f"{prefix}_activation_{i}")
                configurations.append((count, name))
            st.caption("CAMADA DE SAÍDA")
            left, right = st.columns([0.45, 0.55])
            count = left.number_input("Neurônios · saída", min_value=1, value=draft["output"][0], step=1, key=f"{prefix}_output")
            name = right.selectbox("Ativação · saída", list(ACTIVATIONS), index=list(ACTIVATIONS).index(draft["output"][1]), format_func=LABELS.get, key=f"{prefix}_output_activation")
            seed = st.text_input("Seed da inicialização", value=draft["seed"], key=f"{prefix}_seed", help="Opcional: deixe vazio para inicialização aleatória.")
            st.session_state.architecture_draft = {
                "input_size": inputs, "hidden": configurations, "output": (count, name), "seed": seed,
            }
            layers = configurations + [(count, name)]
            if st.button("Criar rede", type="primary", width="stretch"):
                network = NeuralNetwork(inputs, seed=parse_seed(seed))
                for neurons, activation in layers:
                    network.add_layer(neurons, ACTIVATIONS[activation])
                replace_network(st.session_state, network)
                st.success("Rede criada e ativada.")
            st.caption("Criar substitui o modelo ativo e descarta conjuntos e resultados anteriores.")
    with preview:
        section("Prévia da arquitetura", "Rascunho em tempo real · nenhuma alteração nos pesos ativos")
        sizes = [inputs] + [neurons for neurons, _ in layers]
        connections = sum(a * b for a, b in zip(sizes, sizes[1:]))
        for column, label, value in zip(st.columns(3), ["Neurônios", "Conexões", "Camadas"], [sum(sizes[1:]), connections, len(layers)]):
            with column:
                metric_card(label, value)
        st.plotly_chart(draft_figure(inputs, layers), key="architecture_preview", width="stretch", config={"displaylogo": False})
        st.caption("Ciano: entrada · Roxo: ocultas · Verde: saída. Até 8 círculos por camada e 10 camadas no desenho; sem limitar a rede real.")
        active = st.session_state.network
        if active is not None:
            st.caption(f"MODELO ATIVO · {architecture_summary(active)}")
        with st.expander("Detalhes e inicialização"):
            st.dataframe(pd.DataFrame([
                {"Camada": i + 1, "Neurônios": n, "Ativação": LABELS[a], "Inicialização": "He" if a == "relu" else "Xavier/Glorot"}
                for i, (n, a) in enumerate(layers)
            ]), hide_index=True, width="stretch")

"""Ponto de entrada Streamlit: tema, sessão e navegação do MLP Studio."""

from pathlib import Path

import pandas as pd
import streamlit as st

from mlp.interface import initialize_state
from mlp.visual import message, architecture_summary
from mlp.ui.architecture import network_page
from mlp.ui.dashboard import dashboard_page
from mlp.ui.dataset import dataset_page
from mlp.ui.training import training_page
from mlp.ui.models import evaluation_page, models_page


def main():
    """Renderiza o dashboard mantendo a rede ativa entre navegações."""
    st.set_page_config(page_title="MLP Studio", page_icon="🧠", layout="wide")
    st.html(Path(__file__).parent / ".streamlit" / "studio.css")
    initialize_state(st.session_state)
    if "pending_page" in st.session_state:
        st.session_state["navigation"] = st.session_state.pop("pending_page")
    pages = {
        "Dashboard": (dashboard_page, "◈"),
        "Arquitetura Neural": (network_page, "⌘"),
        "Dataset": (dataset_page, "▦"),
        "Treinamento": (training_page, "↗"),
        "Avaliação": (evaluation_page, "◎"),
        "Modelos e Previsões": (models_page, "◇"),
    }
    with st.sidebar:
        st.title("MLP Studio")
        st.caption("ESPAÇO DE TRABALHO · REDES NEURAIS")
        st.divider()
        page = st.radio("Navegação", list(pages), format_func=lambda name: f"{pages[name][1]}  {name}", label_visibility="collapsed", key="navigation")
        st.divider()
        footer = st.empty()
    try:
        pages[page][0]()
    except (ValueError, TypeError, OSError, pd.errors.ParserError) as error:
        message(f"Não foi possível concluir a operação: {error}", "error")

    with footer.container():
        network = st.session_state.network
        if network is not None:
            st.caption("MODELO ATIVO")
            st.write(architecture_summary(network))
            st.caption(f"{len(network.layers)} camadas · {sum(len(layer.neurons) for layer in network.layers)} neurônios")
        else:
            st.caption("Nenhum modelo ativo")
        st.caption("Seu trabalho permanece nesta sessão. Exporte o JSON para preservar os parâmetros.")


if __name__ == "__main__":
    main()

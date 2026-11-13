"""Controles e navegação compartilhados pelas páginas."""

import streamlit as st
from mlp.visual import metric_card, training_figure


def navigate(page):
    """Agenda a troca de página preservando a sessão e os pesos ativos."""
    st.session_state.pending_page = page


def quick_actions():
    """Atalhos para as operações principais, sem executar ações no modelo."""
    for column, label, page in zip(st.columns(3), ["Configurar rede", "Treinar modelo", "Carregar modelo"],
                                   ["Arquitetura Neural", "Treinamento", "Modelos e Previsões"]):
        column.button(label, on_click=navigate, args=(page,), width="stretch")


def optional_seed(label, key, value="42"):
    """Campo de seed opcional: vazio significa None."""
    text = st.text_input(label, value=value, key=key, help="Deixe vazio para uma execução aleatória.")
    return text

def parse_seed(text):
    """Valida a seed do formulário sem interromper sua edição."""
    if not text.strip():
        return None
    try:
        value = int(text)
    except ValueError as error:
        raise ValueError("A seed deve ser um inteiro não negativo ou ficar vazia.") from error
    if value < 0:
        raise ValueError("A seed deve ser um inteiro não negativo ou ficar vazia.")
    return value

def training_results():
    """Curvas e comparação inicial/final da última execução de treinamento."""
    history = st.session_state.history
    initial = st.session_state.initial_mse
    for column, name, label in zip(st.columns(2), ["train", "validation"], ["Treinamento", "Validação"]):
        final = history[-1][name + "_mse"]
        start = initial[name] if initial is not None else None
        with column:
            metric_card(f"MSE · {label}", f"{final:.6g}",
                        f"Inicial: {start:.6g}" if start is not None else None,
                        delta=f"{final - start:+.4g} em relação ao inicial" if start is not None else None)
    st.plotly_chart(training_figure(history, height=460), width="stretch", config={"displaylogo": False})

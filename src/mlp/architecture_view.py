"""Diagrama interativo a partir da rede ativa ou de uma configuração em edição."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from mlp.ui_theme import CYAN, PURPLE, MUTED, LABELS, chart_style


def architecture_summary(network):
    """Resume os tamanhos reais, incluindo as entradas apenas no diagrama."""
    return " → ".join(str(n) for n in [network.input_size] + [len(layer.neurons) for layer in network.layers])


def architecture_figure(network):
    """Adapta a rede ativa ao diagrama, sem criar ou alterar neurônios."""
    return draft_figure(network.input_size, [(len(layer.neurons), layer.activation.__name__) for layer in network.layers], height=360)


def draft_figure(input_size, layers, height=520):
    """Desenha camadas, conexões e neurônios sem modificar a rede.

    Exibe até 8 nós por camada e 10 camadas no diagrama. Para redes maiores,
    preserva entrada, saída e uma seleção de camadas ocultas; saltos ficam
    explícitos e não são conectados. A tabela contém a arquitetura completa.
    """
    sizes = [input_size] + [count for count, activation in layers]
    indices = list(range(len(sizes)))
    if len(indices) > 10:
        indices = [0] + list(range(1, 5)) + list(range(len(sizes) - 5, len(sizes)))
    positions = {index: position for position, index in enumerate(indices)}
    nodes = {i: np.linspace(-1, 1, min(sizes[i], 8)) if sizes[i] > 1 else np.array([0.]) for i in indices}
    edge_x, edge_y = [], []
    for previous, current in zip(indices, indices[1:]):
        if current != previous + 1:
            continue
        for a in nodes[previous]:
            for b in nodes[current]:
                edge_x.extend([positions[previous], positions[current], None])
                edge_y.extend([a, b, None])
    figure = go.Figure(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                  line=dict(color="rgba(148,163,184,0.16)", width=0.8),
                                  hoverinfo="skip", showlegend=False))
    for i in indices:
        is_output = i == len(sizes) - 1
        label = "Entrada" if i == 0 else "Saída" if is_output else f"Oculta {i}"
        activation = "Atributos" if i == 0 else LABELS[layers[i-1][1]]
        color = CYAN if i == 0 else "#34D399" if is_output else PURPLE
        count = sizes[i]
        figure.add_trace(go.Scatter(
            x=[positions[i]] * len(nodes[i]), y=nodes[i], mode="markers",
            marker=dict(size=17, color=color, line=dict(width=2, color="#172033")),
            customdata=[[label, activation, count]] * len(nodes[i]),
            hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br>%{customdata[2]} neurônios<extra></extra>",
            showlegend=False,
        ))
        figure.add_annotation(x=positions[i], y=1.48, text=f"{label}<br>{activation} · {count}",
                              showarrow=False, font=dict(size=11, color=MUTED))
        if count > 8:
            figure.add_annotation(x=positions[i], y=-1.3, text=f"+{count - 8} não exibidos",
                                  showarrow=False, font=dict(size=10, color=MUTED))
    if len(sizes) > 10:
        figure.add_annotation(x=4.5, y=0, text=f"…<br>{len(sizes)-10} camadas omitidas",
                              showarrow=False, font=dict(color=MUTED, size=10))
    chart_style(figure, height)
    figure.update_xaxes(visible=False, range=[-0.6, len(indices) - 0.4])
    figure.update_yaxes(visible=False, range=[-1.6, 1.8], fixedrange=True)
    return figure


def show_architecture(network):
    """Renderiza a arquitetura interativa, com simplificação visual documentada."""
    st.plotly_chart(architecture_figure(network), width="stretch", config={"displaylogo": False})
    st.caption("Diagrama ilustrativo · até 8 nós por camada e 10 camadas · detalhes completos na tabela de arquitetura.")

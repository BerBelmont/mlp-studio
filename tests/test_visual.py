"""Valida gráficos com dados reais e simplificação de arquiteturas grandes."""

import numpy as np

from mlp.ativacoes import relu, sigmoid, tanh
from mlp.rede import NeuralNetwork
from mlp.visual import architecture_figure, architecture_summary, training_figure, split_figure


def test_arquitetura_mostra_conexoes_ativacoes_e_quantidades():
    network = NeuralNetwork(3, seed=42)
    network.add_layer(5, relu)
    network.add_layer(2, sigmoid)
    weights = network.layers[0].neurons[0].weights.copy()
    figure = architecture_figure(network)
    assert architecture_summary(network) == "3 → 5 → 2"
    assert len(figure.data[0].x) == (3 * 5 + 5 * 2) * 3
    assert [len(trace.x) for trace in figure.data[1:]] == [3, 5, 2]
    labels = [annotation.text for annotation in figure.layout.annotations]
    assert "Entrada<br>Atributos · 3" in labels
    assert "Oculta 1<br>ReLU · 5" in labels
    assert "Saída<br>Sigmoid · 2" in labels
    np.testing.assert_array_equal(network.layers[0].neurons[0].weights, weights)


def test_arquitetura_grande_limita_desenho_sem_alterar_rede():
    network = NeuralNetwork(100, seed=42)
    for _ in range(15):
        network.add_layer(20, tanh)
    network.add_layer(2, sigmoid)
    figure = architecture_figure(network)
    assert len(figure.data) == 11  # Um traço de conexões e dez camadas visíveis.
    assert all(len(trace.x) <= 8 for trace in figure.data[1:])
    assert len(network.layers) == 16
    assert any("camadas omitidas" in a.text for a in figure.layout.annotations)
    assert len(figure.data[0].x) <= 9 * 8 * 8 * 3


def test_grafico_preserva_valores_do_historico():
    history = [
        {"epoch": 1, "train_mse": 1.2, "validation_mse": 1.5},
        {"epoch": 2, "train_mse": 0.4, "validation_mse": 0.7},
    ]
    figure = training_figure(history)
    assert list(figure.data[0].y) == [1.2, 0.4]
    assert list(figure.data[1].y) == [1.5, 0.7]
    assert figure.data[0].line.color == "#8B5CF6"
    assert figure.data[1].line.color == "#22D3EE"


def test_grafico_divisao_usa_quantidades_reais():
    datasets = {name: (np.zeros((count, 1)), np.zeros((count, 1)))
                for name, count in [("train", 3), ("validation", 1), ("test", 3)]}
    figure = split_figure(datasets)
    assert list(figure.data[0].values) == [3, 1, 3]

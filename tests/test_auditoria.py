"""Regressões de estabilidade identificadas na auditoria de entrega."""

import numpy as np
import pytest

from mlp.dados import split_dataset
from mlp.interface import train_session
from mlp.metricas import mse
from mlp.neuronio import Neuron
from mlp.rede import NeuralNetwork
from mlp.treinamento import train


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_api_rejeita_valores_nao_finitos(value):
    with pytest.raises(ValueError, match="finitos"):
        Neuron([value])
    with pytest.raises(ValueError, match="finitos"):
        Neuron([1], bias=value)
    neuron = Neuron([1])
    with pytest.raises(ValueError, match="finitos"):
        neuron.forward([value])
    assert neuron.z is None
    with pytest.raises(ValueError, match="finitos"):
        mse([1], [value])
    network = NeuralNetwork(1)
    network.add_layer(1)
    with pytest.raises(ValueError, match="finitos"):
        network.train_step([1], [value], 0.1)
    with pytest.raises(ValueError, match="finitos"):
        split_dataset([[value], [1], [2]], [[1], [2], [3]])


def test_taxa_complexa_rejeitada_com_erro_claro():
    network = NeuralNetwork(1)
    network.add_layer(1)
    with pytest.raises(ValueError, match="learning_rate"):
        network.train_step([1], [0], 1j)
    with pytest.raises(ValueError, match="learning_rate"):
        train(network, ([[1]], [[1]]), ([[1]], [[1]]), 1, 1j)


def test_percentuais_nao_numericos():
    with pytest.raises(ValueError, match="percentuais"):
        split_dataset(np.ones((10, 1)), np.ones((10, 1)), "texto", 15, 15)


def test_falha_numerica_preserva_modelo_historico_e_conjuntos():
    network = NeuralNetwork(1, seed=42)
    network.add_layer(1)
    data = {name: (np.array([[1.], [2.]]), np.array([[2.], [4.]]))
            for name in ("train", "validation", "test")}
    state = {"network": network, "prepared": data}
    train_session(state, 2, 0.1, 42)
    before = [(n.weights.copy(), n.bias, n.z, n.output, n.delta) for layer in network.layers for n in layer.neurons]
    history = state["history"]
    with pytest.raises(ValueError, match="divergiu"):
        train_session(state, 10, 1e308, 42)
    assert state["history"] is history
    assert state["network"] is network
    assert state["prepared"] is data
    for neuron, (weights, bias, z, output, delta) in zip([n for layer in network.layers for n in layer.neurons], before):
        np.testing.assert_array_equal(neuron.weights, weights)
        assert (neuron.bias, neuron.z, neuron.output, neuron.delta) == (bias, z, output, delta)

"""Avaliação do MSE de conjuntos com a rede existente."""

import numpy as np
import pytest

from mlp.ativacoes import relu
from mlp.rede import NeuralNetwork


def criar_rede():
    """Cria saídas lineares conhecidas: x + y + 1 e 2x - y - 1."""
    network = NeuralNetwork(2, seed=42)
    layer = network.add_layer(2)
    layer.neurons[0].weights[:] = [1, 1]
    layer.neurons[0].bias = 1
    layer.neurons[1].weights[:] = [2, -1]
    layer.neurons[1].bias = -1
    return network


def test_mse_rede_com_multiplos_exemplos_e_saidas():
    # Previsões [[4, -1], [4, 5]]; erros quadráticos [[1, 4], [9, 16]].
    assert criar_rede().evaluate_mse([[1, 2], [3, 0]], [[3, 1], [1, 1]]) == 7.5


def test_previsoes_perfeitas():
    assert criar_rede().evaluate_mse([[1, 2], [3, 0]], [[4, -1], [4, 5]]) == 0.0


def test_um_exemplo_com_uma_saida():
    network = NeuralNetwork(2, seed=42)
    neuron = network.add_layer(1).neurons[0]
    neuron.weights[:] = [1, 2]
    neuron.bias = 1
    result = network.evaluate_mse([[1, 2]], [[4]])
    assert isinstance(result, float)
    assert result == 4.0


def test_avaliacao_multicamada_preserva_parametros():
    network = NeuralNetwork(2, seed=42)
    network.add_layer(3, relu)
    network.add_layer(2)
    neurons = [n for layer in network.layers for n in layer.neurons]
    for i, neuron in enumerate(neurons):
        neuron.bias = i + 0.5
    before = [(n.weights.copy(), n.bias) for n in neurons]
    inputs = np.array([[1., 2.], [3., 4.]])
    expected = np.array([[1., 0.], [0., 1.]])
    inputs_copy, expected_copy = inputs.copy(), expected.copy()
    first = network.evaluate_mse(inputs, expected)
    assert network.evaluate_mse(inputs, expected) == first
    for neuron, (weights, bias) in zip(neurons, before):
        np.testing.assert_array_equal(neuron.weights, weights)
        assert neuron.bias == bias
    np.testing.assert_array_equal(inputs, inputs_copy)
    np.testing.assert_array_equal(expected, expected_copy)


@pytest.mark.parametrize("inputs, expected", [
    ([1, 2], [[1, 2]]),
    ([[1]], [[1, 2]]),
    ([[1, 2, 3]], [[1, 2]]),
    ([[1, 2]], [1, 2]),
    ([[1, 2]], [[1]]),
    ([[1, 2]], [[1, 2], [3, 4]]),
    ([[[1, 2]]], [[1, 2]]),
])
def test_dimensoes_invalidas(inputs, expected):
    network = criar_rede()
    with pytest.raises(ValueError):
        network.evaluate_mse(inputs, expected)
    assert all(n.z is None for n in network.layers[0].neurons)


@pytest.mark.parametrize("inputs, expected", [
    ([], []), ([[]], [[]]), ([], [[1, 2]]), ([[1, 2]], []),
])
def test_conjuntos_vazios(inputs, expected):
    with pytest.raises(ValueError, match="vazios"):
        criar_rede().evaluate_mse(inputs, expected)


def test_rede_sem_camadas():
    with pytest.raises(ValueError, match="camada"):
        NeuralNetwork(2).evaluate_mse([[1, 2]], [[1]])

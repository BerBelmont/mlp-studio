"""Testes das escalas de inicialização e da sequência aleatória por rede."""

import numpy as np
import pytest

from mlp.ativacoes import linear, sigmoid, tanh, relu
from mlp.camada import Layer
from mlp.rede import NeuralNetwork


@pytest.mark.parametrize("activation, std", [
    (linear, np.sqrt(2 / 7)),
    (sigmoid, np.sqrt(2 / 7)),
    (tanh, np.sqrt(2 / 7)),
    (relu, np.sqrt(2 / 3)),
])
def test_distribuicao_e_dimensoes(activation, std):
    layer = Layer(3, 4, activation, rng=np.random.default_rng(42))
    # A sequência esperada também detecta uma reinicialização a cada neurônio.
    expected_rng = np.random.default_rng(42)
    for neuron in layer.neurons:
        expected = expected_rng.normal(0.0, std, size=3)
        np.testing.assert_array_equal(neuron.weights, expected)
        assert neuron.weights.shape == (3,)
        assert neuron.bias == 0.0


def criar_rede(seed, input_size, counts, activations):
    """Constrói arquiteturas utilizadas nos testes de reprodutibilidade."""
    network = NeuralNetwork(input_size, seed=seed)
    for count, activation in zip(counts, activations):
        network.add_layer(count, activation)
    return network


@pytest.mark.parametrize("input_size, counts, activations", [
    (3, [5, 4, 2], [relu, tanh, sigmoid]),
    (2, [3, 1], [relu, sigmoid]),
    (1, [1], [linear]),
])
def test_mesma_seed_e_arquiteturas(input_size, counts, activations):
    a = criar_rede(42, input_size, counts, activations)
    b = criar_rede(42, input_size, counts, activations)
    for layer_a, layer_b in zip(a.layers, b.layers):
        for neuron_a, neuron_b in zip(layer_a.neurons, layer_b.neurons):
            np.testing.assert_array_equal(neuron_a.weights, neuron_b.weights)
            assert neuron_a.bias == neuron_b.bias == 0.0
    np.testing.assert_array_equal(a.forward(np.ones(input_size)), b.forward(np.ones(input_size)))


def test_seeds_diferentes():
    a = criar_rede(42, 3, [5, 2], [relu, sigmoid])
    b = criar_rede(43, 3, [5, 2], [relu, sigmoid])
    for layer_a, layer_b in zip(a.layers, b.layers):
        assert not np.array_equal(
            [n.weights for n in layer_a.neurons],
            [n.weights for n in layer_b.neurons],
        )


def test_gerador_continua_entre_camadas():
    network = NeuralNetwork(3, seed=42)
    network.add_layer(4, tanh)
    network.add_layer(2, relu)
    expected_rng = np.random.default_rng(42)
    for layer, std in zip(network.layers, [np.sqrt(2 / 7), np.sqrt(2 / 4)]):
        for neuron in layer.neurons:
            np.testing.assert_array_equal(
                neuron.weights, expected_rng.normal(0.0, std, size=layer.input_size)
            )


def test_parametros_continuam_editaveis_e_independentes():
    layer = NeuralNetwork(2, seed=42).add_layer(2)
    original = layer.neurons[1].weights.copy()
    layer.neurons[0].weights[:] = [1, 2]
    layer.neurons[0].bias = 3
    np.testing.assert_array_equal(layer.neurons[1].weights, original)
    assert layer.neurons[1].bias == 0
    assert layer.forward([1, 1])[0] == 6

"""Testes de deltas, SGD e comparação com gradientes numéricos."""

import numpy as np
import pytest

from mlp.ativacoes import linear, sigmoid, tanh, relu
from mlp.metricas import mse
from mlp.rede import NeuralNetwork


@pytest.mark.parametrize("rate", [0.01, 0.1, 0.25])
def test_deltas_e_atualizacao_com_pesos_antigos(rate):
    network = NeuralNetwork(1, seed=42)
    hidden = network.add_layer(1).neurons[0]
    outputs = network.add_layer(2).neurons
    hidden.weights[:] = [2]
    hidden.bias = 1
    outputs[0].weights[:] = [3]
    outputs[0].bias = 1
    outputs[1].weights[:] = [-2]
    outputs[1].bias = 2
    # x=1 -> h=3 -> saídas [10,-4]; alvos [8,-1].
    # Deltas de saída [2,-3]; delta oculto 3*2 + (-2)*(-3) = 12.
    assert network.train_step([1], [8, -1], rate) == 6.5
    assert [n.delta for n in outputs] == [2, -3]
    assert hidden.delta == 12
    np.testing.assert_allclose(hidden.weights, [2 - rate * 12])
    assert hidden.bias == pytest.approx(1 - rate * 12)
    np.testing.assert_allclose(outputs[0].weights, [3 - rate * 6])
    np.testing.assert_allclose(outputs[1].weights, [-2 + rate * 9])
    assert outputs[0].bias == pytest.approx(1 - rate * 2)
    assert outputs[1].bias == pytest.approx(2 + rate * 3)


@pytest.mark.parametrize("activation", [linear, sigmoid, tanh, relu])
def test_gradient_check_todos_os_pesos_e_vieses(activation):
    network = NeuralNetwork(2, seed=42)
    # Duas camadas ocultas, duas saídas. ReLU longe da origem.
    network.add_layer(2, activation)
    network.add_layer(2, tanh)
    network.add_layer(2, sigmoid)
    neurons = [n for layer in network.layers for n in layer.neurons]
    for i, neuron in enumerate(neurons):
        neuron.weights[:] = [0.1 + i * 0.02, 0.2 + i * 0.01]
        neuron.bias = 0.3
    inputs, expected = [0.4, 0.7], [0.2, 0.8]
    original = [(n.weights.copy(), n.bias) for n in neurons]
    epsilon = 1e-6
    numerical = []
    for neuron in neurons:
        weight_gradients = []
        for j in range(neuron.weights.size):
            old = neuron.weights[j]
            neuron.weights[j] = old + epsilon
            plus = mse(expected, network.forward(inputs))
            neuron.weights[j] = old - epsilon
            minus = mse(expected, network.forward(inputs))
            neuron.weights[j] = old
            weight_gradients.append((plus - minus) / (2 * epsilon))
        old_bias = neuron.bias
        neuron.bias = old_bias + epsilon
        plus = mse(expected, network.forward(inputs))
        neuron.bias = old_bias - epsilon
        minus = mse(expected, network.forward(inputs))
        neuron.bias = old_bias
        numerical.append((weight_gradients, (plus - minus) / (2 * epsilon)))

    rate = 0.1
    network.train_step(inputs, expected, rate)
    for neuron, (weights, bias), (grad_weights, grad_bias) in zip(neurons, original, numerical):
        np.testing.assert_allclose((weights - neuron.weights) / rate, grad_weights,
                                   rtol=1e-5, atol=1e-8)
        assert neuron.delta == pytest.approx(grad_bias, rel=1e-5, abs=1e-8)
        assert (bias - neuron.bias) / rate == pytest.approx(grad_bias, rel=1e-5, abs=1e-8)


@pytest.mark.parametrize("activation", [linear, sigmoid, tanh, relu])
def test_reducao_do_erro(activation):
    network = NeuralNetwork(1, seed=42)
    neuron = network.add_layer(1, activation).neurons[0]
    neuron.weights[:] = [0.4]
    neuron.bias = 0.1
    before = mse([0.2], network.forward([1]))
    for _ in range(10):
        network.train_step([1], [0.2], 0.1)
    assert mse([0.2], network.forward([1])) < before


@pytest.mark.parametrize("z", [-1., 0.])
def test_relu_delta_zero(z):
    network = NeuralNetwork(1)
    neuron = network.add_layer(1, relu).neurons[0]
    neuron.weights[:] = [z]
    network.train_step([1], [2], 0.1)
    assert neuron.delta == 0
    assert neuron.weights[0] == z
    assert neuron.bias == 0


@pytest.mark.parametrize("rate", [0, -0.1, np.nan, np.inf, -np.inf, "0.1", None, [0.1], True])
def test_taxa_invalida_nao_altera_parametros(rate):
    network = NeuralNetwork(1, seed=42)
    neuron = network.add_layer(1).neurons[0]
    weights = neuron.weights.copy()
    with pytest.raises(ValueError, match="learning_rate"):
        network.train_step([1], [0], rate)
    np.testing.assert_array_equal(neuron.weights, weights)
    assert neuron.bias == 0
    assert neuron.z is None


@pytest.mark.parametrize("inputs, expected", [
    ([], [1]), ([1], [1]), ([[1, 2]], [1]), ([1, 2, 3], [1]),
    ([1, 2], []), ([1, 2], [[1]]), ([1, 2], [1, 2]), ([1, 2], 1),
])
def test_dimensoes_invalidas(inputs, expected):
    network = NeuralNetwork(2, seed=42)
    neuron = network.add_layer(1).neurons[0]
    weights = neuron.weights.copy()
    with pytest.raises(ValueError):
        network.train_step(inputs, expected, 0.1)
    np.testing.assert_array_equal(neuron.weights, weights)
    assert neuron.z is None


def test_rede_sem_camadas():
    with pytest.raises(ValueError, match="camada"):
        NeuralNetwork(1).train_step([1], [0], 0.1)


def test_ativacao_sem_derivada_preserva_parametros():
    network = NeuralNetwork(1, seed=42)
    neuron = network.add_layer(1, lambda z: z).neurons[0]
    weights = neuron.weights.copy()
    with pytest.raises(ValueError, match="derivada"):
        network.train_step([1], [0], 0.1)
    np.testing.assert_array_equal(neuron.weights, weights)
    assert neuron.z is None

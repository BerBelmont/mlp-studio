"""Validação integrada da propagação direta, sem treinamento."""

import math

import numpy as np
import pytest

from mlp.ativacoes import linear, relu, sigmoid, tanh
from mlp.rede import NeuralNetwork


@pytest.mark.parametrize("activation, expected", [
    (linear, [-1.0, 2.0]),
    (relu, [0.0, 2.0]),
    (sigmoid, [0.2689414213699951, 0.8807970779778823]),
    (tanh, [-0.7615941559557649, 0.9640275800758169]),
])
def test_camada_unica_com_multiplas_saidas(activation, expected):
    network = NeuralNetwork(2, seed=42)
    layer = network.add_layer(2, activation)
    layer.neurons[0].weights[:] = [1, -1]
    layer.neurons[1].weights[:] = [0.5, 0.5]
    layer.neurons[1].bias = 0.5

    output = network.forward([1, 2])

    assert isinstance(output, np.ndarray)
    assert output.shape == (2,)
    np.testing.assert_allclose(output, expected, rtol=1e-12)
    np.testing.assert_array_equal([n.z for n in layer.neurons], [-1, 2])
    np.testing.assert_allclose([n.output for n in layer.neurons], expected, rtol=1e-12)


def test_duas_camadas_ocultas_com_estado_em_todos_os_neuronios():
    network = NeuralNetwork(2, seed=42)
    first = network.add_layer(2, relu)
    second = network.add_layer(2, tanh)
    last = network.add_layer(2, sigmoid)

    first.neurons[0].weights[:] = [1, -1]
    first.neurons[1].weights[:] = [2, 1]
    first.neurons[1].bias = 1
    second.neurons[0].weights[:] = [2, 1]
    second.neurons[0].bias = -5
    second.neurons[1].weights[:] = [1, 0.2]
    last.neurons[0].weights[:] = [3, 1]
    last.neurons[1].weights[:] = [2, -1]

    # [1, 2] -> ReLU([-1, 5]) -> tanh([0, 1]) -> sigmoid([t, -t]).
    t = math.tanh(1)
    expected = [1 / (1 + math.exp(-t)), 1 / (1 + math.exp(t))]
    output = network.forward([1, 2])

    for layer, expected_z, expected_output in [
        (first, [-1, 5], [0, 5]),
        (second, [0, 1], [0, t]),
        (last, [t, -t], expected),
    ]:
        np.testing.assert_allclose([n.z for n in layer.neurons], expected_z, atol=1e-15)
        np.testing.assert_allclose([n.output for n in layer.neurons], expected_output)
    np.testing.assert_allclose(output, expected)

    # Uma nova entrada deve substituir o estado de todas as camadas.
    network.forward([0, 0])
    for layer in network.layers:
        assert all(n.z is not None and n.output is not None for n in layer.neurons)
    np.testing.assert_array_equal([n.z for n in first.neurons], [0, 1])
    np.testing.assert_allclose([n.z for n in second.neurons], [-4, 0.2])
    np.testing.assert_allclose(
        [n.z for n in last.neurons],
        [3 * math.tanh(-4) + math.tanh(0.2), 2 * math.tanh(-4) - math.tanh(0.2)],
    )


@pytest.mark.parametrize("input_size, architecture", [
    (1, [(1, linear)]),
    (2, [(3, relu), (1, sigmoid)]),
    (3, [(5, relu), (4, tanh), (2, sigmoid)]),
])
def test_repetibilidade_sem_modificar_parametros(input_size, architecture):
    network = NeuralNetwork(input_size, seed=42)
    for count, activation in architecture:
        network.add_layer(count, activation)
    neurons = [n for layer in network.layers for n in layer.neurons]
    # Vieses não nulos permitem detectar também uma eventual reinicialização.
    for i, neuron in enumerate(neurons):
        neuron.bias = (i + 1) / 10
        neuron.delta = (i + 1) / 20
    parameters = [(n.weights.copy(), n.bias, n.delta) for n in neurons]
    inputs = np.arange(input_size, dtype=float)
    original_inputs = inputs.copy()

    first_output = network.forward(inputs)
    first_state = [(n.z, n.output) for n in neurons]
    network.forward(inputs + 1)
    repeated_output = network.forward(inputs)

    np.testing.assert_array_equal(repeated_output, first_output)
    assert [(n.z, n.output) for n in neurons] == first_state
    np.testing.assert_array_equal(inputs, original_inputs)
    for neuron, (weights, bias, delta) in zip(neurons, parameters):
        np.testing.assert_array_equal(neuron.weights, weights)
        assert neuron.bias == bias
        assert neuron.delta == delta


@pytest.mark.parametrize("inputs", [
    1, [], [1], [1, 2, 3], [[1, 2]], [[1], [2]], [[[1, 2]]],
])
def test_entrada_incompativel_nao_processa_neuronios(inputs):
    network = NeuralNetwork(2, seed=42)
    network.add_layer(3, relu)
    network.add_layer(2, sigmoid)
    with pytest.raises(ValueError, match="entradas"):
        network.forward(inputs)
    for layer in network.layers:
        for neuron in layer.neurons:
            assert neuron.z is None
            assert neuron.output is None

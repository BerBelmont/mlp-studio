"""Testes de configuração e processamento de camadas."""

import numpy as np
import pytest

from mlp.ativacoes import linear, relu, sigmoid, tanh
from mlp.camada import Layer
from mlp.neuronio import Neuron


@pytest.mark.parametrize("input_size, count", [(1, 1), (2, 3), (3, 5)])
def test_criacao(input_size, count):
    layer = Layer(input_size, count, relu)
    assert len(layer.neurons) == count
    assert layer.input_size == input_size
    for neuron in layer.neurons:
        assert isinstance(neuron, Neuron)
        assert neuron.weights.shape == (input_size,)
        assert neuron.activation is relu
        assert neuron.bias == 0
    assert layer.forward(np.zeros(input_size)).shape == (count,)


@pytest.mark.parametrize("activation, esperado", [
    (linear, [-1, 2]), (relu, [0, 2]),
    (sigmoid, [0.2689414214, 0.8807970780]),
    (tanh, [-0.7615941560, 0.9640275801]),
])
def test_processamento(activation, esperado):
    layer = Layer(2, 2, activation)
    layer.neurons[0].weights[:] = [1, -1]
    layer.neurons[1].weights[:] = [0.5, 0.5]
    layer.neurons[1].bias = 0.5
    np.testing.assert_allclose(layer.forward([1, 2]), esperado)
    assert [n.z for n in layer.neurons] == [-1, 2]


@pytest.mark.parametrize("inputs", [1, [], [1], [1, 2, 3], [[1, 2]], [[1], [2]]])
def test_dimensoes_invalidas(inputs):
    layer = Layer(2, 3)
    with pytest.raises(ValueError, match="entradas"):
        layer.forward(inputs)
    assert all(n.z is None for n in layer.neurons)


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "2", None])
def test_quantidades_invalidas(value):
    with pytest.raises(ValueError):
        Layer(value, 2)
    with pytest.raises(ValueError):
        Layer(2, value)


def test_ativacao_invalida():
    with pytest.raises(ValueError, match="ativação"):
        Layer(2, 3, activation="relu")

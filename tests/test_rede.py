"""Testes de arquiteturas dinâmicas e propagação entre camadas."""

import numpy as np
import pytest

from mlp.ativacoes import relu, tanh, sigmoid
from mlp.rede import NeuralNetwork


@pytest.mark.parametrize("inputs, counts, activations", [
    (3, [5, 4, 2], [relu, tanh, sigmoid]),
    (2, [3, 1], [relu, sigmoid]),
    (1, [1], [tanh]),
])
def test_arquiteturas(inputs, counts, activations):
    network = NeuralNetwork(inputs)
    previous = inputs
    for count, activation in zip(counts, activations):
        layer = network.add_layer(count, activation)
        assert layer is network.layers[-1]
        assert layer.input_size == previous
        assert layer.activation is activation
        previous = count
    assert len(network.layers) == len(counts)
    result = network.forward(np.ones(inputs))
    assert isinstance(result, np.ndarray)
    assert result.shape == (counts[-1],)


def test_propagacao_numerica_entre_camadas():
    network = NeuralNetwork(2)
    hidden = network.add_layer(2, relu)
    output = network.add_layer(1, sigmoid)
    hidden.neurons[0].weights[:] = [1, -1]
    hidden.neurons[1].weights[:] = [2, 1]
    hidden.neurons[1].bias = 1
    output.neurons[0].weights[:] = [3, 0.5]
    output.neurons[0].bias = -1
    # [1, 2] -> ReLU([-1, 5]) -> [0, 5] -> sigmoid(1.5).
    np.testing.assert_allclose(network.forward([1, 2]), [0.8175744762])
    assert [n.output for n in hidden.neurons] == [0, 5]
    assert output.neurons[0].z == 1.5


@pytest.mark.parametrize("inputs", [1, [], [1], [1, 2, 3], [[1, 2]]])
def test_entradas_invalidas(inputs):
    network = NeuralNetwork(2)
    network.add_layer(1)
    with pytest.raises(ValueError, match="entradas"):
        network.forward(inputs)


@pytest.mark.parametrize("value", [0, -2, 1.5, True, None])
def test_quantidade_de_entradas_invalida(value):
    with pytest.raises(ValueError):
        NeuralNetwork(value)


def test_rede_sem_camadas():
    with pytest.raises(ValueError, match="camada"):
        NeuralNetwork(2).forward([1, 2])


def test_camada_invalida_nao_altera_rede():
    network = NeuralNetwork(2)
    with pytest.raises(ValueError):
        network.add_layer(0)
    assert network.layers == []


def test_incompatibilidade_entre_camadas_e_detectada():
    network = NeuralNetwork(2)
    network.add_layer(2)
    network.add_layer(1)
    network.layers[1].input_size = 3
    with pytest.raises(ValueError, match="camada"):
        network.forward([1, 2])

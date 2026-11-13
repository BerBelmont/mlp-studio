"""Testes da soma ponderada, estado e validação do neurônio."""

import numpy as np
import pytest

from mlp.ativacoes import linear, sigmoid, tanh, relu
from mlp.neuronio import Neuron


def test_soma_ponderada_e_estado():
    neuron = Neuron([0.5, -0.25], bias=0.1)
    assert neuron.z is None
    assert neuron.output is None
    assert neuron.delta == 0.0
    neuron.delta = 0.7
    assert neuron.forward([2, 4]) == pytest.approx(0.1)
    assert neuron.z == pytest.approx(0.1)
    assert neuron.output == pytest.approx(0.1)
    assert neuron.delta == 0.7
    assert neuron.forward([0, 0]) == pytest.approx(0.1)
    np.testing.assert_array_equal(neuron.weights, [0.5, -0.25])
    assert neuron.bias == 0.1


@pytest.mark.parametrize("bias", [-2.0, 0.0, 3.0])
def test_bias(bias):
    assert Neuron([1, 2], bias=bias).forward([0, 0]) == bias


@pytest.mark.parametrize("ativacao, esperado", [
    (linear, -1.0), (sigmoid, 0.2689414214),
    (tanh, -0.7615941560), (relu, 0.0),
])
def test_aplica_ativacao(ativacao, esperado):
    neuron = Neuron([1, -1], activation=ativacao)
    assert neuron.forward([1, 2]) == pytest.approx(esperado)
    assert neuron.z == -1.0
    assert neuron.output == pytest.approx(esperado)


@pytest.mark.parametrize("entradas", [[], [1], [1, 2, 3], [[1, 2]], [[1], [2]], 1])
def test_rejeita_dimensoes_incorretas(entradas):
    neuron = Neuron([1, 2])
    with pytest.raises(ValueError, match="entradas"):
        neuron.forward(entradas)
    assert neuron.z is None
    assert neuron.output is None


@pytest.mark.parametrize("pesos", [[], [[1, 2]], 1])
def test_rejeita_pesos_sem_formato_vetorial(pesos):
    with pytest.raises(ValueError, match="pesos"):
        Neuron(pesos)

"""Verificações numéricas das ativações e de suas derivadas."""

import numpy as np
import pytest

from mlp.ativacoes import (
    linear, linear_derivative, sigmoid, sigmoid_derivative,
    tanh, tanh_derivative, relu, relu_derivative,
)


@pytest.mark.parametrize("funcao, esperado", [
    (linear, [-1, 0, 1]),
    (sigmoid, [0.2689414214, 0.5, 0.7310585786]),
    (tanh, [-0.7615941560, 0, 0.7615941560]),
    (relu, [0, 0, 1]),
    (linear_derivative, [1, 1, 1]),
    (sigmoid_derivative, [0.1966119332, 0.25, 0.1966119332]),
    (tanh_derivative, [0.4199743416, 1, 0.4199743416]),
    (relu_derivative, [0, 0, 1]),
])
def test_valores_escalares_e_vetoriais(funcao, esperado):
    np.testing.assert_allclose(funcao(np.array([-1., 0., 1.])), esperado)
    for z, valor in zip([-1., 0., 1.], esperado):
        assert float(funcao(z)) == pytest.approx(valor)


@pytest.mark.parametrize("funcao, derivada", [
    (linear, linear_derivative), (sigmoid, sigmoid_derivative),
    (tanh, tanh_derivative), (relu, relu_derivative),
])
def test_derivadas_por_diferencas_finitas(funcao, derivada):
    z = np.array([-2.0, -0.3, 0.4, 2.0])
    passo = 1e-6
    aproximacao = (funcao(z + passo) - funcao(z - passo)) / (2 * passo)
    np.testing.assert_allclose(derivada(z), aproximacao, atol=1e-8)


def test_sigmoid_extremos_sem_overflow():
    with np.errstate(over="raise", invalid="raise"):
        np.testing.assert_allclose(sigmoid([-1000, 1000]), [0, 1])

"""Funções de ativação e derivadas em relação à soma ponderada z.

As funções aceitam escalares ou arrays NumPy.
"""

import numpy as np


def linear(z):
    """Retorna a própria entrada."""
    return np.asarray(z, dtype=float)


def linear_derivative(z):
    """Retorna um para cada valor de z."""
    return np.ones_like(z, dtype=float)


def sigmoid(z):
    """Calcula a sigmoide evitando overflow para entradas de grande magnitude."""
    z = np.asarray(z, dtype=float)
    exponencial = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + exponencial), exponencial / (1 + exponencial))


def sigmoid_derivative(z):
    """Calcula f(z) * (1 - f(z))."""
    saida = sigmoid(z)
    return saida * (1 - saida)


def tanh(z):
    """Calcula a tangente hiperbólica de z."""
    return np.tanh(z)


def tanh_derivative(z):
    """Calcula 1 - tanh(z)²."""
    return 1 - np.square(tanh(z))


def relu(z):
    """Retorna zero para valores negativos e z para valores positivos."""
    return np.maximum(0, z)


def relu_derivative(z):
    """Retorna um se z > 0 e zero caso contrário, inclusive em z = 0."""
    return (np.asarray(z) > 0).astype(float)


# Relaciona cada ativação implementada à sua derivada em relação a z.
DERIVATIVES = {
    linear: linear_derivative,
    sigmoid: sigmoid_derivative,
    tanh: tanh_derivative,
    relu: relu_derivative,
}

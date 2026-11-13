"""Entidade Neuron, com estado da última propagação direta."""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from mlp.ativacoes import linear


class Neuron:
    """Armazena parâmetros e aplica uma ativação à soma ponderada das entradas."""

    def __init__(
        self,
        weights: ArrayLike,
        bias: float = 0.0,
        activation: Callable[[float], float] = linear,
    ) -> None:
        """Recebe um vetor não vazio de pesos, um viés e uma função de ativação."""
        self.weights = np.array(weights, dtype=float, copy=True)
        if self.weights.ndim != 1 or self.weights.size == 0:
            raise ValueError("Os pesos devem ser um vetor unidimensional não vazio.")
        self.bias = float(bias)
        if not np.isfinite(self.weights).all() or not np.isfinite(self.bias):
            raise ValueError("Pesos e bias devem ser valores finitos.")
        if not callable(activation):
            raise ValueError("A ativação deve ser uma função.")
        self.activation = activation
        self.z: float | None = None
        self.output: float | None = None
        self.delta: float = 0.0

    def forward(self, inputs: ArrayLike) -> float:
        """Valida as entradas, calcula z = entradas · pesos + bias e guarda a saída.

        O delta é preservado; seu cálculo pertence à retropropagação na rede.
        """
        inputs = np.asarray(inputs, dtype=float)
        if inputs.ndim != 1 or inputs.shape != self.weights.shape:
            raise ValueError(
                "As entradas devem ser um vetor unidimensional "
                "com a mesma quantidade de elementos dos pesos."
            )
        if not np.isfinite(inputs).all():
            raise ValueError("As entradas devem conter somente valores finitos.")
        self.z = float(np.dot(inputs, self.weights) + self.bias)
        self.output = float(self.activation(self.z))
        return self.output

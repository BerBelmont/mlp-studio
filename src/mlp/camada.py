"""Camada de neurônios com uma função de ativação compartilhada."""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from mlp.ativacoes import linear, relu
from mlp.neuronio import Neuron


class Layer:
    """Agrupa neurônios que recebem o mesmo vetor de entradas."""

    def __init__(
        self,
        input_size: int,
        neuron_count: int,
        activation: Callable[[float], float] = linear,
        *,
        rng: np.random.Generator | None = None,
        weights: ArrayLike | None = None,
        biases: ArrayLike | None = None,
    ) -> None:
        """Cria pesos com He para ReLU e Xavier/Glorot para as demais ativações.

        Usa o gerador recebido ou cria um para uma camada independente.
        Cada neurônio mantém seu próprio bias, inicialmente zero.
        """
        if isinstance(input_size, bool) or not isinstance(input_size, int) or input_size <= 0:
            raise ValueError("A quantidade de entradas deve ser um inteiro positivo.")
        if isinstance(neuron_count, bool) or not isinstance(neuron_count, int) or neuron_count <= 0:
            raise ValueError("A quantidade de neurônios deve ser um inteiro positivo.")
        if not callable(activation):
            raise ValueError("A ativação deve ser uma função.")
        self.input_size = input_size
        self.activation = activation
        # O carregamento fornece parâmetros prontos, sem gerar novos pesos.
        if weights is not None or biases is not None:
            weights = np.asarray(weights, dtype=float)
            biases = np.asarray(biases, dtype=float)
            if weights.shape != (neuron_count, input_size) or biases.shape != (neuron_count,):
                raise ValueError("Dimensões incompatíveis nos parâmetros da camada.")
            if not np.isfinite(weights).all() or not np.isfinite(biases).all():
                raise ValueError("Os parâmetros da camada devem ser finitos.")
            self.neurons = [
                Neuron(w, bias=b, activation=activation) for w, b in zip(weights, biases)
            ]
            return
        if rng is None:
            rng = np.random.default_rng()
        # fan_in é o número de entradas; fan_out é o tamanho desta camada.
        if activation is relu:
            std = np.sqrt(2 / input_size)
        else:
            std = np.sqrt(2 / (input_size + neuron_count))
        self.neurons = [
            Neuron(rng.normal(0.0, std, size=input_size), bias=0.0, activation=activation)
            for _ in range(neuron_count)
        ]

    def forward(self, inputs: ArrayLike) -> np.ndarray:
        """Valida um vetor de entradas e retorna uma saída por neurônio."""
        inputs = np.asarray(inputs, dtype=float)
        if inputs.ndim != 1 or inputs.size != self.input_size:
            raise ValueError(
                f"A camada espera um vetor com {self.input_size} entradas."
            )
        return np.array([neuron.forward(inputs) for neuron in self.neurons])

"""Rede MLP construída dinamicamente a partir de camadas sequenciais."""

from collections.abc import Callable
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike

from mlp.ativacoes import DERIVATIVES, linear
from mlp.camada import Layer
from mlp.metricas import mse
from mlp.normalizacao import Normalizer


class NeuralNetwork:
    """Armazena a arquitetura e propaga um vetor pelas suas camadas."""

    def __init__(self, input_size: int, seed: int | None = None) -> None:
        """Inicia a rede e seu único gerador aleatório, com seed opcional."""
        if isinstance(input_size, bool) or not isinstance(input_size, int) or input_size <= 0:
            raise ValueError("A quantidade de entradas deve ser um inteiro positivo.")
        self.input_size = input_size
        self.rng = np.random.default_rng(seed)
        self.layers: list[Layer] = []
        self.normalizer: Normalizer | None = None

    def predict(self, inputs: ArrayLike) -> np.ndarray:
        """Prevê um vetor original, normalizando uma vez antes do forward.

        Não altera pesos ou vieses e não desnormaliza as saídas.
        Associe um Normalizer já ajustado em self.normalizer, se necessário.
        """
        inputs = np.asarray(inputs, dtype=float)
        if inputs.shape != (self.input_size,) or not np.isfinite(inputs).all():
            raise ValueError("A previsão exige um vetor finito compatível com as entradas da rede.")
        if self.normalizer is not None:
            inputs = self.normalizer.transform(inputs.reshape(1, -1))[0]
        return self.forward(inputs)

    def add_layer(
        self,
        neuron_count: int,
        activation: Callable[[float], float] = linear,
    ) -> Layer:
        """Adiciona e retorna uma camada compatível com a saída da anterior."""
        input_size = len(self.layers[-1].neurons) if self.layers else self.input_size
        layer = Layer(input_size, neuron_count, activation, rng=self.rng)
        self.layers.append(layer)
        return layer

    def forward(self, inputs: ArrayLike) -> np.ndarray:
        """Processa um vetor sequencialmente e retorna a saída da última camada."""
        inputs = np.asarray(inputs, dtype=float)
        if inputs.ndim != 1 or inputs.size != self.input_size:
            raise ValueError(
                f"A rede espera um vetor com {self.input_size} entradas."
            )
        if not self.layers:
            raise ValueError("Adicione pelo menos uma camada antes de processar entradas.")
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def evaluate_mse(self, inputs: ArrayLike, expected: ArrayLike) -> float:
        """Avalia um conjunto sem alterar pesos ou vieses.

        Recebe matrizes (exemplos, entradas) e (exemplos, saídas), inclusive
        para um único exemplo ou uma única saída. O forward atualiza apenas
        o estado z e output dos neurônios, como no processamento individual.
        """
        inputs = np.asarray(inputs, dtype=float)
        expected = np.asarray(expected, dtype=float)
        if not self.layers:
            raise ValueError("Adicione pelo menos uma camada antes de avaliar a rede.")
        if inputs.size == 0 or expected.size == 0:
            raise ValueError("A avaliação não aceita conjuntos vazios.")
        if inputs.ndim != 2 or inputs.shape[1] != self.input_size:
            raise ValueError(
                f"As entradas devem ser uma matriz com {self.input_size} colunas."
            )
        output_size = len(self.layers[-1].neurons)
        if expected.shape != (inputs.shape[0], output_size):
            raise ValueError(
                "As saídas esperadas devem ser uma matriz com uma linha por "
                f"exemplo e {output_size} colunas."
            )
        predicted = np.array([self.forward(example) for example in inputs])
        return mse(expected, predicted)

    def train_step(
        self, inputs: ArrayLike, expected: ArrayLike, learning_rate: float
    ) -> float:
        """Executa uma atualização SGD e retorna o MSE anterior à atualização.

        Entradas e saídas esperadas são vetores de um único exemplo.
        Todos os deltas usam os pesos antigos e o z do forward atual.
        O estado z/output/delta permanece referente a esse cálculo, sem um
        novo forward após a atualização. Não executa um loop de épocas.
        """
        if (
            not isinstance(learning_rate, Real)
            or isinstance(learning_rate, (bool, np.bool_))
            or not np.isfinite(learning_rate)
            or learning_rate <= 0
        ):
            raise ValueError("learning_rate deve ser um número finito maior que zero.")
        inputs = np.asarray(inputs, dtype=float)
        expected = np.asarray(expected, dtype=float)
        if not self.layers:
            raise ValueError("Adicione pelo menos uma camada antes da atualização SGD.")
        if inputs.shape != (self.input_size,):
            raise ValueError("As entradas devem ser um vetor compatível com a rede.")
        if expected.shape != (len(self.layers[-1].neurons),):
            raise ValueError("As saídas esperadas devem ser um vetor compatível com a rede.")
        if not np.isfinite(inputs).all() or not np.isfinite(expected).all():
            raise ValueError("Entradas e saídas esperadas devem conter somente valores finitos.")
        # Valida as ativações antes de alterar o estado ou qualquer parâmetro.
        for layer in self.layers:
            for neuron in layer.neurons:
                if neuron.activation not in DERIVATIVES:
                    raise ValueError("Ativação sem derivada implementada para SGD.")

        predicted = self.forward(inputs)
        loss = mse(expected, predicted)
        output_layer = self.layers[-1]
        output_size = len(output_layer.neurons)
        for neuron, target in zip(output_layer.neurons, expected):
            derivative = DERIVATIVES[neuron.activation]
            neuron.delta = float(
                (2 / output_size) * (neuron.output - target) * derivative(neuron.z)
            )

        # Retropropaga com todos os pesos ainda intactos.
        for index in range(len(self.layers) - 2, -1, -1):
            layer = self.layers[index]
            next_layer = self.layers[index + 1]
            for j, neuron in enumerate(layer.neurons):
                propagated = sum(n.weights[j] * n.delta for n in next_layer.neurons)
                derivative = DERIVATIVES[neuron.activation]
                neuron.delta = float(derivative(neuron.z) * propagated)

        # Somente após calcular TODOS os deltas, atualiza pesos e vieses.
        layer_inputs = inputs
        for layer in self.layers:
            for neuron in layer.neurons:
                weight_gradient = neuron.delta * layer_inputs
                bias_gradient = neuron.delta
                neuron.weights -= learning_rate * weight_gradient
                neuron.bias -= learning_rate * bias_gradient
            # Saídas armazenadas pertencem ao forward anterior à atualização.
            layer_inputs = np.array([n.output for n in layer.neurons])
        return loss

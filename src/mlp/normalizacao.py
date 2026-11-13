"""Normalização de entradas com parâmetros ajustados exclusivamente no treino."""

import numpy as np
from numpy.typing import ArrayLike


def _input_matrix(inputs: ArrayLike) -> np.ndarray:
    """Converte entradas em matriz numérica não vazia e finita."""
    inputs = np.asarray(inputs, dtype=float)
    if inputs.ndim != 2 or inputs.size == 0:
        raise ValueError("As entradas devem ser uma matriz não vazia (exemplos, atributos).")
    if not np.isfinite(inputs).all():
        raise ValueError("As entradas devem conter somente valores numéricos finitos.")
    return inputs


class Normalizer:
    """Armazena parâmetros por atributo para none, minmax ou zscore.

    fit deve receber somente entradas de treinamento, após a divisão.
    transform nunca recalcula parâmetros e não modifica os dados recebidos.
    """

    def __init__(self, method: str = "none") -> None:
        """Escolhe a normalização; none mantém os valores originais."""
        if method not in ("none", "minmax", "zscore"):
            raise ValueError("Método inválido. Use none, minmax ou zscore.")
        self.method = method
        self.input_size: int | None = None
        self.minimum: np.ndarray | None = None
        self.maximum: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, train_inputs: ArrayLike) -> None:
        """Calcula parâmetros exclusivamente no treino, usando std com ddof=0."""
        inputs = _input_matrix(train_inputs)
        self.input_size = inputs.shape[1]
        self.minimum = self.maximum = self.mean = self.std = None
        if self.method == "minmax":
            self.minimum = np.min(inputs, axis=0)
            self.maximum = np.max(inputs, axis=0)
        elif self.method == "zscore":
            self.mean = np.mean(inputs, axis=0)
            self.std = np.std(inputs, axis=0, ddof=0)

    def transform(self, inputs: ArrayLike) -> np.ndarray:
        """Aplica parâmetros armazenados, inclusive em novos dados de previsão.

        Para amplitude ou desvio zero, usa divisor 1. Valores constantes do
        treino tornam-se zero; novos valores mantêm a diferença para a origem.
        Min-Max não limita os resultados ao intervalo [0, 1].
        """
        if self.input_size is None:
            raise ValueError("Ajuste o normalizador com fit antes de transformar dados.")
        inputs = _input_matrix(inputs)
        if inputs.shape[1] != self.input_size:
            raise ValueError(f"A normalização espera {self.input_size} atributos de entrada.")
        if self.method == "none":
            return inputs.copy()
        if self.method == "minmax":
            origin = self.minimum
            scale = self.maximum - self.minimum
        else:
            origin = self.mean
            scale = self.std
        return (inputs - origin) / np.where(scale == 0, 1.0, scale)


def normalize_splits(
    datasets: dict[str, tuple[np.ndarray, np.ndarray]],
    method: str = "none",
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], Normalizer]:
    """Normaliza o resultado de split_dataset, preservando as saídas esperadas.

    Retorna novos conjuntos e o normalizador para uso posterior em previsões.
    Somente datasets['train'][0] participa do ajuste dos parâmetros.
    """
    names = ("train", "validation", "test")
    if set(datasets) != set(names):
        raise ValueError("Forneça os conjuntos train, validation e test após a divisão.")
    for name in names:
        inputs, expected = datasets[name]
        inputs = _input_matrix(inputs)
        expected = np.asarray(expected)
        if expected.ndim != 2 or expected.size == 0 or expected.shape[0] != inputs.shape[0]:
            raise ValueError(f"Entradas e saídas incompatíveis no conjunto {name}.")
    normalizer = Normalizer(method)
    normalizer.fit(datasets["train"][0])
    normalized = {
        name: (normalizer.transform(datasets[name][0]), np.array(datasets[name][1], copy=True))
        for name in names
    }
    return normalized, normalizer

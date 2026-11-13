"""Métricas para comparar valores esperados e previsões."""

import numpy as np
from numpy.typing import ArrayLike


def mse(expected: ArrayLike, predicted: ArrayLike) -> float:
    """Calcula a soma dos erros ao quadrado dividida pelo total de valores.

    Aceita vetores ou matrizes não vazios, com formatos exatamente iguais.
    Uma matriz representa exemplos nas linhas e saídas nas colunas.
    Para uma única saída de um único exemplo, use um vetor de tamanho um.
    """
    expected = np.asarray(expected, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if expected.shape != predicted.shape:
        raise ValueError("Valores esperados e previstos devem ter o mesmo formato.")
    if expected.ndim not in (1, 2):
        raise ValueError("Os valores devem ser vetores ou matrizes.")
    if expected.size == 0:
        raise ValueError("O MSE não pode ser calculado para conjuntos vazios.")
    if not np.isfinite(expected).all() or not np.isfinite(predicted).all():
        raise ValueError("O MSE exige valores esperados e previstos finitos.")
    errors = expected - predicted
    return float(np.sum(errors ** 2) / expected.size)

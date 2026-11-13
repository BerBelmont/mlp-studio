"""Verificações numéricas e de formato do erro quadrático médio."""

import numpy as np
import pytest

from mlp.metricas import mse


@pytest.mark.parametrize("expected, predicted, result", [
    ([2], [2], 0.0),
    ([[1, 2], [3, 4]], [[1, 2], [3, 4]], 0.0),
    ([3], [1], 4.0),
    ([1, 2, 3], [2, 4, 6], 14 / 3),
    ([[1], [3]], [[2], [5]], 2.5),
    ([[1, 2], [3, 4]], [[2, 4], [6, 8]], 7.5),
])
def test_mse_conhecido(expected, predicted, result):
    actual = mse(expected, predicted)
    assert isinstance(actual, float)
    assert actual == pytest.approx(result)


@pytest.mark.parametrize("expected, predicted", [
    ([1, 2], [1]),
    ([1, 2], [[1, 2]]),
    ([[1], [2]], [[1, 2]]),
    ([[1, 2], [3, 4]], [[1, 2]]),
    (1, 1),
    ([[[1]]], [[[1]]]),
])
def test_rejeita_dimensoes_incompativeis(expected, predicted):
    with pytest.raises(ValueError):
        mse(expected, predicted)


@pytest.mark.parametrize("empty", [[], [[]], np.empty((0, 2))])
def test_rejeita_conjuntos_vazios(empty):
    with pytest.raises(ValueError, match="vazios"):
        mse(empty, empty)

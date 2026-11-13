"""Testes de normalização sem vazamento de validação ou teste."""

import numpy as np
import pytest

from mlp.dados import split_dataset
from mlp.normalizacao import Normalizer, normalize_splits


@pytest.mark.parametrize("method, expected", [
    ("minmax", [[0, 0], [1, 1]]),
    ("zscore", [[-1, -1], [1, 1]]),
    ("none", [[2, 10], [4, 20]]),
])
def test_formulas(method, expected):
    inputs = np.array([[2., 10.], [4., 20.]])
    original = inputs.copy()
    normalizer = Normalizer(method)
    normalizer.fit(inputs)
    np.testing.assert_allclose(normalizer.transform(inputs), expected)
    np.testing.assert_array_equal(inputs, original)


@pytest.mark.parametrize("method", ["minmax", "zscore", "none"])
def test_parametros_somente_no_treino_e_saidas_preservadas(method):
    datasets = {
        "train": (np.array([[2., 10.], [4., 20.]]), np.array([[100.], [200.]])),
        "validation": (np.array([[6., 30.]]), np.array([[300.]])),
        "test": (np.array([[0., 0.]]), np.array([[400.]])),
    }
    before = {name: (x.copy(), y.copy()) for name, (x, y) in datasets.items()}
    normalized, normalizer = normalize_splits(datasets, method)
    if method == "minmax":
        np.testing.assert_array_equal(normalizer.minimum, [2, 10])
        np.testing.assert_array_equal(normalizer.maximum, [4, 20])
        expected_validation, expected_test = [[2, 2]], [[-1, -1]]
    elif method == "zscore":
        np.testing.assert_array_equal(normalizer.mean, [3, 15])
        np.testing.assert_array_equal(normalizer.std, [1, 5])
        expected_validation, expected_test = [[3, 3]], [[-3, -3]]
    else:
        assert normalizer.minimum is normalizer.maximum is normalizer.mean is normalizer.std is None
        expected_validation, expected_test = [[6, 30]], [[0, 0]]
    np.testing.assert_allclose(normalized["validation"][0], expected_validation)
    np.testing.assert_allclose(normalized["test"][0], expected_test)
    parameters = [None if p is None else p.copy() for p in
                  (normalizer.minimum, normalizer.maximum, normalizer.mean, normalizer.std)]
    normalizer.transform([[1000, -1000]])
    for old, current in zip(parameters, (normalizer.minimum, normalizer.maximum, normalizer.mean, normalizer.std)):
        np.testing.assert_array_equal(old, current)
    for name in datasets:
        np.testing.assert_array_equal(normalized[name][1], before[name][1])
        for original, unchanged in zip(before[name], datasets[name]):
            np.testing.assert_array_equal(original, unchanged)


@pytest.mark.parametrize("method", ["minmax", "zscore"])
def test_constantes_e_novos_dados(method):
    normalizer = Normalizer(method)
    normalizer.fit([[5, 10], [5, 10]])
    with np.errstate(divide="raise", invalid="raise"):
        np.testing.assert_array_equal(normalizer.transform([[5, 10]]), [[0, 0]])
        np.testing.assert_array_equal(normalizer.transform([[7, 9]]), [[2, -1]])
    if method == "minmax":
        np.testing.assert_array_equal(normalizer.maximum - normalizer.minimum, [0, 0])
    else:
        np.testing.assert_array_equal(normalizer.std, [0, 0])


@pytest.mark.parametrize("method", ["none", "minmax", "zscore"])
def test_reprodutibilidade_apos_divisao(method):
    x = np.arange(40).reshape(20, 2)
    y = np.arange(20).reshape(20, 1)
    a, _ = normalize_splits(split_dataset(x, y, seed=42), method)
    b, _ = normalize_splits(split_dataset(x, y, seed=42), method)
    for name in a:
        for left, right in zip(a[name], b[name]):
            np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("invalid", [[], [[]], [1, 2], 1, [[[1, 2]]], [[np.nan, 1]], [[np.inf, 2]]])
def test_dados_invalidos(invalid):
    normalizer = Normalizer("minmax")
    with pytest.raises(ValueError):
        normalizer.fit(invalid)
    normalizer.fit([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        normalizer.transform(invalid)


def test_quantidade_de_atributos_incompativel():
    normalizer = Normalizer()
    normalizer.fit([[1, 2]])
    with pytest.raises(ValueError, match="atributos"):
        normalizer.transform([[1]])


def test_transform_sem_fit():
    with pytest.raises(ValueError, match="fit"):
        Normalizer().transform([[1]])


def test_metodo_invalido():
    with pytest.raises(ValueError, match="Método"):
        Normalizer("outro")


def test_conjuntos_nao_divididos():
    with pytest.raises(ValueError, match="divisão"):
        normalize_splits({"train": (np.ones((2, 1)), np.ones((2, 1)))})


def test_dimensoes_invalidas_nos_conjuntos():
    datasets = {name: (np.ones((2, 1)), np.ones((2, 1))) for name in ("train", "validation", "test")}
    datasets["test"] = (np.ones((2, 1)), np.ones((1, 1)))
    with pytest.raises(ValueError, match="incompatíveis"):
        normalize_splits(datasets)

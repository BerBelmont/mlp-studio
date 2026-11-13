"""Testes de leitura, seleção e separação dos exemplos sem perdas."""

import numpy as np
import pandas as pd
import pytest

from mlp.dados import load_csv, select_columns, split_dataset
from mlp.rede import NeuralNetwork


def rede(inputs=2, outputs=1):
    network = NeuralNetwork(inputs, seed=42)
    network.add_layer(3)
    network.add_layer(outputs)
    return network


def test_csv_valido_e_ordem_das_colunas(tmp_path):
    path = tmp_path / "dados.csv"
    path.write_text("a,b,y,ignorada\n1,2,3,texto\n4,5,6,outro\n", encoding="utf-8")
    frame = load_csv(path)
    assert list(frame.columns) == ["a", "b", "y", "ignorada"]
    inputs, outputs = select_columns(frame, ["b", "a"], ["y"], rede())
    np.testing.assert_array_equal(inputs, [[2, 1], [5, 4]])
    np.testing.assert_array_equal(outputs, [[3], [6]])
    assert inputs.dtype == outputs.dtype == float


@pytest.mark.parametrize("input_size, output_size", [(1, 1), (2, 1), (3, 2)])
def test_arquiteturas_diferentes(input_size, output_size):
    frame = pd.DataFrame({"a": [1], "b": [2], "c": [3], "y": [4], "z": [5]})
    x, y = select_columns(frame, ["a", "b", "c"][:input_size], ["y", "z"][:output_size], rede(input_size, output_size))
    assert x.shape == (1, input_size)
    assert y.shape == (1, output_size)


@pytest.mark.parametrize("inputs, outputs, message", [
    (["ausente", "b"], ["y"], "inexistentes"),
    (["a"], ["y"], "entrada"),
    (["a", "b"], ["y", "z"], "saída"),
    (["a", "b"], ["a"], "simultaneamente"),
    ([], ["y"], "ao menos"),
    (["a", "a"], ["y"], "repetidas"),
])
def test_selecao_invalida(inputs, outputs, message):
    frame = pd.DataFrame({"a": [1], "b": [2], "y": [3], "z": [4]})
    with pytest.raises(ValueError, match=message):
        select_columns(frame, inputs, outputs, rede())


@pytest.mark.parametrize("value", ["texto", np.nan, np.inf])
def test_rejeita_dados_nao_numericos_ou_nao_finitos(value):
    with pytest.raises(ValueError, match="numéricos"):
        select_columns(pd.DataFrame({"x": [value], "y": [1]}), ["x"], ["y"], rede(1))


@pytest.mark.parametrize("content", ["", "x,y\n"])
def test_csv_vazio(tmp_path, content):
    path = tmp_path / "vazio.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_csv(path)


def test_rede_sem_camadas():
    with pytest.raises(ValueError, match="camada"):
        select_columns(pd.DataFrame({"x": [1], "y": [2]}), ["x"], ["y"], NeuralNetwork(1))


def test_divisao_associacao_sem_perda_e_sem_duplicacao():
    inputs = np.arange(20).reshape(-1, 1)
    outputs = np.column_stack((inputs[:, 0] * 10, -inputs[:, 0]))
    result = split_dataset(inputs, outputs, seed=42)
    assert [len(x) for x, _ in result.values()] == [14, 3, 3]
    for x, y in result.values():
        np.testing.assert_array_equal(y[:, 0], x[:, 0] * 10)
        np.testing.assert_array_equal(y[:, 1], -x[:, 0])
    combined = np.concatenate([x[:, 0] for x, _ in result.values()])
    np.testing.assert_array_equal(np.sort(combined), np.arange(20))
    assert not np.array_equal(combined, inputs[:, 0])
    np.testing.assert_array_equal(inputs[:, 0], np.arange(20))


def test_seed_reproduzivel_e_seeds_diferentes():
    inputs = np.arange(100).reshape(-1, 1)
    a = split_dataset(inputs, inputs, seed=42)
    b = split_dataset(inputs, inputs, seed=42)
    c = split_dataset(inputs, inputs, seed=43)
    for name in a:
        for left, right in zip(a[name], b[name]):
            np.testing.assert_array_equal(left, right)
        assert not np.array_equal(a[name][0], c[name][0])


def test_arredondamento_e_percentuais_personalizados():
    inputs = np.arange(7).reshape(-1, 1)
    result = split_dataset(inputs, inputs, 50, 25, 25, seed=1)
    assert [len(x) for x, _ in result.values()] == [3, 1, 3]


@pytest.mark.parametrize("percentages", [
    (70, 15, 10), (0, 50, 50), (-10, 55, 55), (101, 1, -2),
    (np.nan, 15, 15), (np.inf, 15, 15),
])
def test_percentuais_invalidos(percentages):
    inputs = np.ones((20, 1))
    with pytest.raises(ValueError, match="percentuais"):
        split_dataset(inputs, inputs, *percentages)


@pytest.mark.parametrize("count, percentages", [(1, (70, 15, 15)), (3, (70, 15, 15)), (10, (98, 1, 1))])
def test_dataset_pequeno(count, percentages):
    inputs = np.ones((count, 1))
    with pytest.raises(ValueError, match="pequeno demais"):
        split_dataset(inputs, inputs, *percentages)


@pytest.mark.parametrize("inputs, outputs", [([], []), ([[1]], []), ([1, 2], [1, 2]), ([[1], [2]], [[1]])])
def test_dimensoes_invalidas_na_divisao(inputs, outputs):
    with pytest.raises(ValueError):
        split_dataset(inputs, outputs)

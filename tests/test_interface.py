"""Integração da interface sem duplicar testes dos algoritmos matemáticos."""

import numpy as np
import pytest

from mlp.interface import (
    initialize_state, replace_network, replace_dataset, prepare_data,
    export_json, import_json, comparison_table,
)
from mlp.rede import NeuralNetwork


CSV = b"x,y\n0,1\n1,3\n2,5\n3,7\n4,9\n5,11\n6,13\n7,15\n8,17\n9,19\n"


def session():
    state = {}
    initialize_state(state)
    network = NeuralNetwork(1, seed=42)
    network.add_layer(1)
    replace_network(state, network)
    replace_dataset(state, CSV, "dados.csv")
    return state


def test_rerun_preserva_rede_e_historico():
    state = session()
    network = state["network"]
    state["history"] = [{"epoch": 1}]
    initialize_state(state)
    assert state["network"] is network
    assert state["history"] == [{"epoch": 1}]


@pytest.mark.parametrize("method", ["none", "minmax", "zscore"])
def test_preparacao_e_normalizador_associado(method):
    state = session()
    prepare_data(state, ["x"], ["y"], [60, 20, 20], 42, method)
    assert state["normalizer"] is state["network"].normalizer
    assert state["normalizer"].method == method
    assert [len(x) for x, _ in state["prepared"].values()] == [6, 2, 2]
    for x, y in state["prepared"].values():
        assert np.all(y % 2 == 1)  # Saídas originais preservadas.


def test_substituicao_invalida_dados_mas_preserva_modelo():
    state = session()
    prepare_data(state, ["x"], ["y"], [60, 20, 20], 42, "minmax")
    state["history"] = [1]
    network = state["network"]
    weights = network.layers[0].neurons[0].weights.copy()
    replace_dataset(state, CSV, "outro.csv")
    assert state["prepared"] is None
    assert state["history"] == []
    assert state["network"] is network
    np.testing.assert_array_equal(network.layers[0].neurons[0].weights, weights)
    replace_network(state, NeuralNetwork(2))
    assert state["normalizer"] is None
    assert state["dataset"] is not None


def test_erro_de_upload_ou_preparacao_preserva_estado():
    state = session()
    prepare_data(state, ["x"], ["y"], [60, 20, 20], 42, "none")
    previous = state.copy()
    with pytest.raises(ValueError):
        replace_dataset(state, b"", "vazio.csv")
    with pytest.raises(ValueError):
        prepare_data(state, ["ausente"], ["y"], [60, 20, 20], 42, "none")
    assert state["prepared"] is previous["prepared"]
    assert state["dataset"] is previous["dataset"]


def test_download_upload_e_reuso_de_normalizador_treinado():
    state = session()
    prepare_data(state, ["x"], ["y"], [60, 20, 20], 42, "minmax")
    original = state["network"].predict([3])
    loaded = import_json(export_json(state["network"]))
    replace_network(state, loaded, loaded=True)
    normalizer = loaded.normalizer
    minimum = normalizer.minimum.copy()
    maximum = normalizer.maximum.copy()
    replace_dataset(state, CSV.replace(b"0,1", b"100,201"), "novo.csv")
    prepare_data(state, ["x"], ["y"], [60, 20, 20], 10, "zscore")
    assert state["network"].normalizer is normalizer
    assert normalizer.method == "minmax"
    np.testing.assert_array_equal(normalizer.minimum, minimum)
    np.testing.assert_array_equal(normalizer.maximum, maximum)
    np.testing.assert_array_equal(loaded.predict([3]), original)


def test_tabela_usa_forward_sem_segunda_normalizacao():
    state = session()
    prepare_data(state, ["x"], ["y"], [60, 20, 20], 42, "minmax")
    network = state["network"]
    dataset = state["prepared"]["test"]
    table = comparison_table(network, dataset, ["y"])
    expected = np.array([network.forward(row) for row in dataset[0]])[:, 0]
    np.testing.assert_array_equal(table["y · previsto"], expected)
    np.testing.assert_array_equal(table["y · esperado"], dataset[1][:, 0])


def test_json_invalido():
    with pytest.raises(ValueError):
        import_json(b"{nao json}")

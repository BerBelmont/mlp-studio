"""Fluxo de entrega: CSV → divisão → escala → SGD → teste → JSON → previsão."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import runpy

from mlp.ativacoes import relu, tanh
from mlp.dados import load_csv, select_columns, split_dataset
from mlp.normalizacao import normalize_splits
from mlp.persistencia import load_network, save_network
from mlp.rede import NeuralNetwork
from mlp.treinamento import train


@pytest.mark.parametrize("method", ["none", "minmax", "zscore"])
@pytest.mark.parametrize("hidden", [False, True])
@pytest.mark.parametrize("outputs", [1, 2])
def test_ciclo_completo(tmp_path, method, hidden, outputs):
    values = np.linspace(-0.5, 0.5, 20)
    frame = pd.DataFrame({"x": values, "y": 0.3 * values + 0.2, "z": -0.2 * values + 0.1})
    path = tmp_path / "dados.csv"
    frame.to_csv(path, index=False)
    network = NeuralNetwork(1, seed=42)
    if hidden:
        network.add_layer(3, relu)
        network.add_layer(3, tanh)
    network.add_layer(outputs)
    inputs, expected = select_columns(load_csv(path), ["x"], ["y", "z"][:outputs], network)
    raw = split_dataset(inputs, expected, seed=42)
    data, normalizer = normalize_splits(raw, method)
    for name in data:
        np.testing.assert_array_equal(data[name][1], raw[name][1])
    initial = network.evaluate_mse(*data["train"])
    history = train(network, data["train"], data["validation"], 150, 0.03, seed=42)
    assert len(history) == 150
    assert history[-1]["train_mse"] < initial
    assert history[-1]["train_mse"] == network.evaluate_mse(*data["train"])
    assert history[-1]["validation_mse"] == network.evaluate_mse(*data["validation"])
    original = [(n.weights.copy(), n.bias) for layer in network.layers for n in layer.neurons]
    assert np.isfinite(network.evaluate_mse(*data["test"]))
    network.normalizer = normalizer
    predictions = np.array([network.predict(row) for row in inputs])
    # A previsão em escala original equivale a forward com a escala preparada.
    prepared_predictions = np.array([network.forward(row) for row in normalizer.transform(inputs)])
    np.testing.assert_array_equal(predictions, prepared_predictions)
    saved = tmp_path / "rede.json"
    save_network(network, saved)
    loaded = load_network(saved)
    np.testing.assert_array_equal(predictions, [loaded.predict(row) for row in inputs])
    for neuron, (weights, bias) in zip([n for layer in network.layers for n in layer.neurons], original):
        np.testing.assert_array_equal(neuron.weights, weights)
        assert neuron.bias == bias


def test_csv_incompativel_e_json_invalido(tmp_path):
    path = tmp_path / "dados.csv"
    pd.DataFrame({"x": [1, 2, 3], "y": [3, 5, 7]}).to_csv(path, index=False)
    network = NeuralNetwork(2)
    network.add_layer(1)
    with pytest.raises(ValueError, match="entrada"):
        select_columns(load_csv(path), ["x"], ["y"], network)
    path.write_text('{"version": 1, "layers": []}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_network(path)


@pytest.mark.parametrize("name", ["linear", "xor"])
def test_demos_entregues_executam_fora_da_raiz(name, tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "examples" / "demonstracao.py"
    monkeypatch.chdir(tmp_path)
    result = runpy.run_path(str(script))["demonstrate"](name)
    assert result["final"]["train_mse"] < result["initial"]["train"]
    assert result["final"]["validation_mse"] < result["initial"]["validation"]
    if name == "xor":
        np.testing.assert_array_equal((result["predictions"] >= 0.5).astype(int), [[0], [1], [1], [0]])

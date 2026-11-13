"""Persistência e previsão com parâmetros restaurados, sem novo ajuste."""

import json

import numpy as np
import pytest

from mlp.ativacoes import linear, relu, sigmoid, tanh
from mlp.dados import split_dataset
from mlp.normalizacao import Normalizer, normalize_splits
from mlp.persistencia import load_network, save_network
from mlp.rede import NeuralNetwork
from mlp.treinamento import train


def criar_rede(method=None):
    network = NeuralNetwork(2, seed=42)
    for count, activation in [(3, relu), (2, tanh), (2, sigmoid), (1, linear)]:
        network.add_layer(count, activation)
    for i, layer in enumerate(network.layers):
        for neuron in layer.neurons:
            neuron.bias = i + 0.25
    if method is not None:
        network.normalizer = Normalizer(method)
        network.normalizer.fit([[2, 10], [4, 20]])
    return network


@pytest.mark.parametrize("method", [None, "none", "minmax", "zscore"])
def test_roundtrip_preserva_parametros_e_previsoes(tmp_path, method, monkeypatch):
    network = criar_rede(method)
    before = network.predict([3, 15])
    path = tmp_path / "rede.json"
    save_network(network, path)

    # Falha se o carregamento tentar gerar pesos pela distribuição normal.
    class NoRandom:
        def normal(self, *args, **kwargs):
            pytest.fail("Carregamento não deve gerar pesos aleatórios")

    monkeypatch.setattr(np.random, "default_rng", lambda *args: NoRandom())
    restored = load_network(path)
    assert restored.input_size == network.input_size
    assert len(restored.layers) == len(network.layers)
    for a, b in zip(network.layers, restored.layers):
        assert a.activation is b.activation
        assert a.input_size == b.input_size
        assert len(a.neurons) == len(b.neurons)
        for n, m in zip(a.neurons, b.neurons):
            np.testing.assert_array_equal(n.weights, m.weights)
            assert n.bias == m.bias
            assert n.activation is m.activation
    np.testing.assert_array_equal(restored.predict([3, 15]), before)
    if method is None:
        assert restored.normalizer is None
    else:
        assert restored.normalizer.method == method
        for name in ("minimum", "maximum", "mean", "std"):
            np.testing.assert_array_equal(getattr(restored.normalizer, name), getattr(network.normalizer, name))


@pytest.mark.parametrize("method", [None, "none", "minmax", "zscore"])
def test_predict_normaliza_uma_vez_e_preserva_parametros(method, monkeypatch):
    network = NeuralNetwork(1, seed=42)
    neuron = network.add_layer(1).neurons[0]
    neuron.weights[:] = [2]
    neuron.bias = 1
    if method is not None:
        network.normalizer = Normalizer(method)
        network.normalizer.fit([[2], [4]])
        transform = network.normalizer.transform
        calls = []

        def tracked(x):
            calls.append(np.array(x))
            return transform(x)

        monkeypatch.setattr(network.normalizer, "transform", tracked)
    original = neuron.weights.copy()
    expected = {None: 7, "none": 7, "minmax": 2, "zscore": 1}[method]
    np.testing.assert_allclose(network.predict([3]), [expected])
    if method is not None:
        assert len(calls) == 1
        np.testing.assert_array_equal(calls[0], [[3]])
    np.testing.assert_array_equal(neuron.weights, original)
    assert neuron.bias == 1
    # forward continua recebendo entradas já preparadas, sem transformação.
    np.testing.assert_array_equal(network.forward([3]), [7])


@pytest.mark.parametrize("inputs", [[], [1], [1, 2, 3], [[1, 2]], [np.nan, 2]])
def test_predict_dimensoes_invalidas(inputs):
    with pytest.raises(ValueError):
        criar_rede().predict(inputs)


@pytest.mark.parametrize("content", ["{", "[]", "null", "{}", '{"version": 2}'])
def test_json_invalido(tmp_path, content):
    path = tmp_path / "invalido.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="inválid"):
        load_network(path)


@pytest.mark.parametrize("field, value", [
    ("layer_count", 2), ("input_size", 3), ("input_size", True),
    ("weights", [[1]]), ("weights", [["1", "2"]]),
    ("biases", [float("nan")]), ("biases", [True]),
    ("neuron_count", 0), ("activation", "inexistente"),
    ("std", [-1, 1]), ("std", [1]), ("mean", None),
    ("normalizer_input_size", 1), ("method", "desconhecido"),
])
def test_estrutura_inconsistente(tmp_path, field, value):
    network = NeuralNetwork(2)
    network.add_layer(1)
    network.normalizer = Normalizer("zscore")
    network.normalizer.fit([[1, 2], [3, 4]])
    path = tmp_path / "rede.json"
    save_network(network, path)
    data = json.loads(path.read_text())
    if field in ("layer_count", "input_size"):
        data[field] = value
    elif field in ("std", "mean", "method", "normalizer_input_size"):
        data["normalization"]["input_size" if field == "normalizer_input_size" else field] = value
    else:
        data["layers"][0][field] = value
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_network(path)


def test_normalizador_nao_ajustado_e_rede_vazia_rejeitados(tmp_path):
    network = NeuralNetwork(1)
    with pytest.raises(ValueError):
        save_network(network, tmp_path / "vazio.json")
    network.add_layer(1)
    network.normalizer = Normalizer("minmax")
    with pytest.raises(ValueError):
        save_network(network, tmp_path / "nao_ajustado.json")
    with pytest.raises(ValueError, match="fit"):
        network.predict([1])


def test_integrado_treina_salva_carrega(tmp_path):
    x = np.linspace(-1, 1, 20).reshape(-1, 1)
    datasets, normalizer = normalize_splits(split_dataset(x, 2 * x + 1, seed=42), "minmax")
    network = NeuralNetwork(1, seed=42)
    network.add_layer(1)
    train(network, datasets["train"], datasets["validation"], 200, 0.05, seed=42)
    network.normalizer = normalizer
    before = network.predict([0.25])
    np.testing.assert_allclose(before, [1.5], atol=1e-7)
    path = tmp_path / "treinada.json"
    save_network(network, path)
    del network
    restored = load_network(path)
    np.testing.assert_array_equal(restored.predict([0.25]), before)

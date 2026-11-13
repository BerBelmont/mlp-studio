"""Testes do loop de épocas e isolamento dos conjuntos."""

import numpy as np
import pytest

from mlp.dados import split_dataset
from mlp.normalizacao import normalize_splits
from mlp.rede import NeuralNetwork
from mlp.treinamento import train


def rede():
    network = NeuralNetwork(1, seed=42)
    network.add_layer(1)
    return network


def dados():
    return (np.array([[0.], [0.5], [1.]]), np.array([[1.], [2.], [3.]]))


@pytest.mark.parametrize("epochs", [1, 5])
def test_historico_e_mse_ao_final_de_cada_epoca(epochs):
    network, reference = rede(), rede()
    training = dados()
    validation = (np.array([[0.25], [0.75]]), np.array([[1.5], [2.5]]))
    rate, seed = 0.1, 17
    rng = np.random.default_rng(seed)
    expected_history = []
    for epoch in range(1, epochs + 1):
        for index in rng.permutation(3):
            reference.train_step(training[0][index], training[1][index], rate)
        expected_history.append({
            "epoch": epoch,
            "train_mse": reference.evaluate_mse(*training),
            "validation_mse": reference.evaluate_mse(*validation),
        })
    before = network.layers[0].neurons[0].weights.copy()
    history = train(network, training, validation, epochs, rate, seed)
    assert len(history) == epochs
    assert history == expected_history
    assert not np.array_equal(before, network.layers[0].neurons[0].weights)
    assert network.layers[0].neurons[0].bias != 0


def test_seed_igual_produz_historico_e_parametros_iguais():
    a, b = rede(), rede()
    assert train(a, dados(), dados(), 8, 0.1, 10) == train(b, dados(), dados(), 8, 0.1, 10)
    np.testing.assert_array_equal(a.layers[0].neurons[0].weights, b.layers[0].neurons[0].weights)
    assert a.layers[0].neurons[0].bias == b.layers[0].neurons[0].bias


def test_so_treino_atualiza_e_embaralhamento_por_epoca(monkeypatch):
    network = rede()
    training = dados()
    validation = (np.array([[10.], [20.]]), np.array([[21.], [41.]]))
    calls = []
    original_step = network.train_step
    original_evaluate = network.evaluate_mse

    def step(x, y, rate):
        calls.append((float(x[0]), float(y[0])))
        return original_step(x, y, rate)

    def evaluate(x, y):
        neuron = network.layers[0].neurons[0]
        weights, bias = neuron.weights.copy(), neuron.bias
        assert len(calls) % 3 == 0
        result = original_evaluate(x, y)
        np.testing.assert_array_equal(neuron.weights, weights)
        assert neuron.bias == bias
        return result

    monkeypatch.setattr(network, "train_step", step)
    monkeypatch.setattr(network, "evaluate_mse", evaluate)
    train(network, training, validation, 3, 0.1, seed=42)
    rng = np.random.default_rng(42)
    expected = []
    for _ in range(3):
        expected.extend((float(training[0][i, 0]), float(training[1][i, 0])) for i in rng.permutation(3))
    assert calls == expected


def test_integrado_aprende_relacao_normalizada_sem_usar_teste():
    x = np.linspace(-1, 1, 20).reshape(-1, 1)
    y = 2 * x + 1
    prepared, normalizer = normalize_splits(split_dataset(x, y, seed=42), "minmax")
    original = {name: (a.copy(), b.copy()) for name, (a, b) in prepared.items()}
    a, b = rede(), rede()
    before = a.evaluate_mse(*prepared["train"])
    history = train(a, prepared["train"], prepared["validation"], 200, 0.05, seed=42)
    # Alterar o teste não afeta o treinamento, pois ele não é recebido pela API.
    prepared["test"] = (np.full_like(prepared["test"][0], 1e9), np.full_like(prepared["test"][1], -1e9))
    assert history == train(b, prepared["train"], prepared["validation"], 200, 0.05, seed=42)
    assert history[-1]["train_mse"] < min(before, 1e-8)
    assert history[-1]["validation_mse"] < 1e-8
    for name in ("train", "validation"):
        for current, previous in zip(prepared[name], original[name]):
            np.testing.assert_array_equal(current, previous)
    np.testing.assert_array_equal(normalizer.minimum, [x.min()])


@pytest.mark.parametrize("epochs", [0, -1, 1.5, True])
def test_epocas_invalidas(epochs):
    with pytest.raises(ValueError, match="epochs"):
        train(rede(), dados(), dados(), epochs, 0.1)


@pytest.mark.parametrize("rate", [0, -1, np.nan, np.inf, None, True])
def test_taxa_invalida(rate):
    with pytest.raises(ValueError, match="learning_rate"):
        train(rede(), dados(), dados(), 1, rate)


@pytest.mark.parametrize("invalid", [
    ([], []), ([1, 2], [1, 2]), ([[1, 2]], [[1]]),
    ([[1]], [[1, 2]]), ([[1], [2]], [[1]]), ([[np.nan]], [[1]]),
])
@pytest.mark.parametrize("invalid_train", [True, False])
def test_dataset_invalido_rejeitado_antes_de_atualizar(invalid, invalid_train):
    network = rede()
    neuron = network.layers[0].neurons[0]
    weights = neuron.weights.copy()
    with pytest.raises(ValueError):
        train(network, invalid if invalid_train else dados(), dados() if invalid_train else invalid, 1, 0.1)
    np.testing.assert_array_equal(neuron.weights, weights)
    assert neuron.bias == 0


def test_rede_sem_camadas():
    with pytest.raises(ValueError, match="camada"):
        train(NeuralNetwork(1), dados(), dados(), 1, 0.1)

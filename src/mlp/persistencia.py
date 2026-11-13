"""Persistência JSON da arquitetura, parâmetros aprendidos e normalização."""

import json
from pathlib import Path

import numpy as np

from mlp.ativacoes import linear, sigmoid, tanh, relu
from mlp.camada import Layer
from mlp.normalizacao import Normalizer
from mlp.rede import NeuralNetwork

ACTIVATIONS = {"linear": linear, "sigmoid": sigmoid, "tanh": tanh, "relu": relu}


def _positive_integer(value):
    """Valida contagens declaradas no JSON, sem aceitar bool como inteiro."""
    if type(value) is not int or value <= 0:
        raise ValueError("As quantidades da arquitetura devem ser inteiros positivos.")
    return value


def _numeric_array(value, shape):
    """Exige números JSON reais finitos e o formato especificado."""
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError("Parâmetros numéricos inválidos ou dimensões incompatíveis.")
    return array.astype(float)


def _restore(data):
    """Valida o documento e reconstrói a rede sem inicializar pesos aleatórios."""
    if not isinstance(data, dict) or type(data.get("version")) is not int or data["version"] != 1:
        raise ValueError("Formato ou versão JSON inválida.")
    network = NeuralNetwork(_positive_integer(data["input_size"]))
    layers = data["layers"]
    if not isinstance(layers, list) or len(layers) != _positive_integer(data["layer_count"]):
        raise ValueError("Quantidade de camadas inconsistente.")
    input_size = network.input_size
    for item in layers:
        count = _positive_integer(item["neuron_count"])
        activation = ACTIVATIONS[item["activation"]]
        weights = _numeric_array(item["weights"], (count, input_size))
        biases = _numeric_array(item["biases"], (count,))
        network.layers.append(Layer(input_size, count, activation, weights=weights, biases=biases))
        input_size = count
    normalization = data["normalization"]
    if normalization is not None:
        normalizer = Normalizer(normalization["method"])
        if _positive_integer(normalization["input_size"]) != network.input_size:
            raise ValueError("Normalizador incompatível com as entradas da rede.")
        normalizer.input_size = network.input_size
        for name in ("minimum", "maximum", "mean", "std"):
            required = (
                normalizer.method == "minmax" and name in ("minimum", "maximum")
                or normalizer.method == "zscore" and name in ("mean", "std")
            )
            value = normalization[name]
            if required:
                setattr(normalizer, name, _numeric_array(value, (network.input_size,)))
            elif value is not None:
                raise ValueError("Parâmetro incompatível com o tipo de normalização.")
        if normalizer.method == "minmax" and np.any(normalizer.maximum < normalizer.minimum):
            raise ValueError("Máximos não podem ser menores que os mínimos.")
        if normalizer.method == "zscore" and np.any(normalizer.std < 0):
            raise ValueError("Desvios-padrão não podem ser negativos.")
        network.normalizer = normalizer
    return network


def save_network(network: NeuralNetwork, path: str | Path) -> None:
    """Salva somente arquitetura, pesos, vieses e normalizador ajustado.

    z, output, delta, histórico e estado dos geradores não são persistidos.
    Valida os dados antes de abrir o arquivo para escrita.
    """
    names = {function: name for name, function in ACTIVATIONS.items()}
    try:
        layers = []
        input_size = network.input_size
        for layer in network.layers:
            if layer.input_size != input_size:
                raise ValueError("Quantidade de entradas inconsistente entre camadas.")
            if any(n.activation is not layer.activation for n in layer.neurons):
                raise ValueError("A ativação deve ser a mesma em todos os neurônios da camada.")
            layers.append({
                "neuron_count": len(layer.neurons),
                "activation": names[layer.activation],
                "weights": [n.weights.tolist() for n in layer.neurons],
                "biases": [float(n.bias) for n in layer.neurons],
            })
            input_size = len(layer.neurons)
        normalization = None
        if network.normalizer is not None:
            normalizer = network.normalizer
            normalization = {"method": normalizer.method, "input_size": normalizer.input_size}
            for name in ("minimum", "maximum", "mean", "std"):
                value = getattr(normalizer, name)
                normalization[name] = None if value is None else value.tolist()
        data = {
            "version": 1, "input_size": network.input_size,
            "layer_count": len(layers), "layers": layers, "normalization": normalization,
        }
        _restore(data)
        content = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise ValueError(f"Não foi possível salvar a rede: {error}") from error
    Path(path).write_text(content + "\n", encoding="utf-8")


def load_network(path: str | Path) -> NeuralNetwork:
    """Lê e valida JSON, restaurando a rede pronta para predict com dados originais."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return _restore(data)
    except (KeyError, TypeError, AttributeError, ValueError, OverflowError) as error:
        raise ValueError(f"Arquivo de rede inválido: {error}") from error

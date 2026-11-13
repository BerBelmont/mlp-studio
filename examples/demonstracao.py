"""Demonstrações acadêmicas reproduzíveis com CSV, treino, teste e JSON."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from mlp.ativacoes import linear, tanh, sigmoid
from mlp.dados import load_csv, select_columns, split_dataset
from mlp.normalizacao import normalize_splits
from mlp.persistencia import save_network, load_network
from mlp.rede import NeuralNetwork
from mlp.treinamento import train

ROOT = Path(__file__).resolve().parents[1]


def demonstrate(name):
    """Executa um exemplo completo; retorna métricas realmente calculadas."""
    is_xor = name == "xor"
    network = NeuralNetwork(2 if is_xor else 1, seed=42)
    if is_xor:
        network.add_layer(4, tanh)
    network.add_layer(1, sigmoid if is_xor else linear)
    frame = load_csv(ROOT / "data" / f"{name}.csv")
    x, y = select_columns(frame, ["x1", "x2"] if is_xor else ["x"], ["y"], network)
    datasets, normalizer = normalize_splits(split_dataset(x, y, seed=42), "none" if is_xor else "minmax")
    initial = {key: network.evaluate_mse(*datasets[key]) for key in ("train", "validation")}
    history = train(network, datasets["train"], datasets["validation"],
                    epochs=1000 if is_xor else 200, learning_rate=0.3 if is_xor else 0.05, seed=42)
    # O teste aparece pela primeira vez após todas as épocas.
    test_mse = network.evaluate_mse(*datasets["test"])
    network.normalizer = normalizer
    samples = [[0, 0], [0, 1], [1, 0], [1, 1]] if is_xor else [[0.25]]
    before = np.array([network.predict(row) for row in samples])
    with TemporaryDirectory() as directory:
        path = Path(directory) / "modelo.json"
        save_network(network, path)
        restored = load_network(path)
        after = np.array([restored.predict(row) for row in samples])
    np.testing.assert_array_equal(before, after)
    print(f"\n{name.upper()} · seed 42 · divisão 70/15/15")
    print(f"MSE treino: {initial['train']:.9g} -> {history[-1]['train_mse']:.9g}")
    print(f"MSE validação: {initial['validation']:.9g} -> {history[-1]['validation_mse']:.9g}")
    print(f"MSE teste (avaliação final): {test_mse:.9g}")
    for sample, prediction in zip(samples, after):
        print(f"{sample} -> {prediction.tolist()}")
    print("Previsões idênticas após JSON: True")
    if is_xor:
        print("XOR repete os quatro padrões: demonstra aprendizado da tabela, não generalização para padrões inéditos.")
    return {"initial": initial, "final": history[-1], "test_mse": test_mse, "predictions": after}


def main():
    """Seleciona demonstração linear, XOR ou ambas sem editar o código."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exemplo", choices=["linear", "xor", "todos"], default="todos")
    args = parser.parse_args()
    for name in ["linear", "xor"] if args.exemplo == "todos" else [args.exemplo]:
        demonstrate(name)


if __name__ == "__main__":
    main()

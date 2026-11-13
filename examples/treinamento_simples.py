"""Exemplo reproduzível: aprende y = 2x + 1 sem usar teste no treinamento."""

import numpy as np

from mlp.dados import split_dataset
from mlp.normalizacao import normalize_splits
from mlp.rede import NeuralNetwork
from mlp.treinamento import train


def main() -> None:
    """Prepara dados, mostra métricas e treina uma rede linear pequena."""
    inputs = np.linspace(-1, 1, 20).reshape(-1, 1)
    expected = 2 * inputs + 1
    datasets = split_dataset(inputs, expected, seed=42)
    datasets, normalizer = normalize_splits(datasets, method="minmax")

    network = NeuralNetwork(input_size=1, seed=42)
    network.add_layer(1)  # Linear é a ativação padrão.
    for name in ("train", "validation"):
        print(f"MSE inicial {name}: {network.evaluate_mse(*datasets[name]):.6e}")

    history = train(
        network, datasets["train"], datasets["validation"],
        epochs=200, learning_rate=0.05, seed=42,
    )
    print(f"MSE final train: {history[-1]['train_mse']:.6e}")
    print(f"MSE final validation: {history[-1]['validation_mse']:.6e}")
    print("Teste reservado; não foi usado no treinamento.")
    # O normalizador permanece disponível para futuras previsões.
    print(f"Normalização utilizada: {normalizer.method}")


if __name__ == "__main__":
    main()

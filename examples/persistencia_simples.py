"""Treina, salva e carrega uma rede para prever com entradas originais."""

from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np

from mlp.dados import split_dataset
from mlp.normalizacao import normalize_splits
from mlp.persistencia import load_network, save_network
from mlp.rede import NeuralNetwork
from mlp.treinamento import train


def main() -> None:
    """Demonstra o ciclo completo usando um arquivo JSON temporário."""
    x = np.linspace(-1, 1, 20).reshape(-1, 1)
    datasets, normalizer = normalize_splits(
        split_dataset(x, 2 * x + 1, seed=42), method="minmax"
    )
    network = NeuralNetwork(1, seed=42)
    network.add_layer(1)
    train(network, datasets["train"], datasets["validation"], 200, 0.05, seed=42)
    network.normalizer = normalizer
    before = network.predict([0.25])
    with TemporaryDirectory() as directory:
        path = Path(directory) / "rede.json"
        save_network(network, path)
        del network
        restored = load_network(path)
        after = restored.predict([0.25])
    print(f"Antes de salvar: {before[0]:.12f}")
    print(f"Após carregar:  {after[0]:.12f}")
    print(f"Previsões idênticas: {np.array_equal(before, after)}")


if __name__ == "__main__":
    main()

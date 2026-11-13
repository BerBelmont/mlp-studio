"""Leitura de CSV, seleção de colunas e divisão aleatória de exemplos."""

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from mlp.rede import NeuralNetwork


def load_csv(path: str | Path) -> pd.DataFrame:
    """Lê um CSV com cabeçalho; as colunas disponíveis ficam em frame.columns."""
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError as error:
        raise ValueError("O CSV está vazio, sem colunas ou exemplos.") from error
    if frame.empty:
        raise ValueError("O CSV não contém exemplos disponíveis.")
    return frame


def select_columns(
    frame: pd.DataFrame,
    input_columns: list[str],
    output_columns: list[str],
    network: NeuralNetwork,
) -> tuple[np.ndarray, np.ndarray]:
    """Seleciona colunas na ordem informada e valida a compatibilidade com a rede.

    Retorna matrizes float (exemplos, entradas) e (exemplos, saídas).
    Não realiza limpeza, codificação de categorias ou normalização.
    """
    if frame.empty:
        raise ValueError("O dataset não contém exemplos disponíveis.")
    if not input_columns or not output_columns:
        raise ValueError("Selecione ao menos uma coluna de entrada e uma de saída.")
    if len(set(input_columns)) != len(input_columns) or len(set(output_columns)) != len(output_columns):
        raise ValueError("Não selecione colunas repetidas.")
    if set(input_columns) & set(output_columns):
        raise ValueError("Uma coluna não pode ser entrada e saída simultaneamente.")
    selected = list(input_columns) + list(output_columns)
    missing = [column for column in selected if column not in frame.columns]
    if missing:
        raise ValueError(f"Colunas inexistentes: {', '.join(missing)}.")
    if not frame.columns.is_unique:
        raise ValueError("Os nomes das colunas do dataset devem ser únicos.")
    if not network.layers:
        raise ValueError("A rede deve possuir pelo menos uma camada de saída.")
    if len(input_columns) != network.input_size:
        raise ValueError(f"A rede exige {network.input_size} colunas de entrada.")
    output_size = len(network.layers[-1].neurons)
    if len(output_columns) != output_size:
        raise ValueError(f"A rede exige {output_size} colunas de saída.")
    try:
        inputs = frame.loc[:, list(input_columns)].to_numpy(dtype=float, copy=True)
        outputs = frame.loc[:, list(output_columns)].to_numpy(dtype=float, copy=True)
    except (ValueError, TypeError) as error:
        raise ValueError("As colunas selecionadas devem conter somente dados numéricos.") from error
    if not np.isfinite(inputs).all() or not np.isfinite(outputs).all():
        raise ValueError("As colunas selecionadas devem conter valores numéricos finitos.")
    return inputs, outputs


def split_dataset(
    inputs: ArrayLike,
    expected: ArrayLike,
    train_percent: float = 70,
    validation_percent: float = 15,
    test_percent: float = 15,
    seed: int | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Embaralha índices e divide os pares de exemplos em três conjuntos.

    Os percentuais positivos devem totalizar 100 (tolerância absoluta 1e-8).
    Treino e validação usam floor(n * percentual / 100); teste recebe o
    restante. Se algum tamanho resultar em zero, a divisão é rejeitada.
    Retorna train, validation e test, cada um contendo (entradas, saídas).
    """
    inputs = np.asarray(inputs, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if inputs.ndim != 2 or expected.ndim != 2:
        raise ValueError("Entradas e saídas devem ser matrizes com uma linha por exemplo.")
    if inputs.size == 0 or expected.size == 0:
        raise ValueError("O dataset não pode ser vazio.")
    if inputs.shape[0] != expected.shape[0]:
        raise ValueError("Entradas e saídas devem ter a mesma quantidade de exemplos.")
    if not np.isfinite(inputs).all() or not np.isfinite(expected).all():
        raise ValueError("Entradas e saídas devem conter valores numéricos finitos.")
    try:
        percentages = np.asarray([train_percent, validation_percent, test_percent], dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Os percentuais devem ser números finitos maiores que zero.") from error
    if percentages.shape != (3,) or not np.isfinite(percentages).all() or np.any(percentages <= 0):
        raise ValueError("Os percentuais devem ser números finitos maiores que zero.")
    if not np.isclose(percentages.sum(), 100, rtol=0, atol=1e-8):
        raise ValueError("Os percentuais devem totalizar 100%.")
    count = inputs.shape[0]
    train_size = int(np.floor(count * percentages[0] / 100))
    validation_size = int(np.floor(count * percentages[1] / 100))
    test_size = count - train_size - validation_size
    if min(train_size, validation_size, test_size) <= 0:
        raise ValueError("Dataset pequeno demais para a divisão escolhida: todos os conjuntos devem ser não vazios.")
    # O mesmo índice seleciona a entrada e sua saída, preservando os pares.
    indices = np.random.default_rng(seed).permutation(count)
    train, validation, test = np.split(indices, [train_size, train_size + validation_size])
    return {
        "train": (inputs[train], expected[train]),
        "validation": (inputs[validation], expected[validation]),
        "test": (inputs[test], expected[test]),
    }

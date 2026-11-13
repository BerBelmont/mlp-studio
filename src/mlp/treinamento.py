"""Treinamento por épocas com SGD e avaliação após as atualizações."""

import numpy as np
from numbers import Real
from numpy.typing import ArrayLike

from mlp.rede import NeuralNetwork


def _validate_dataset(network, dataset, name):
    """Valida um par de matrizes antes de qualquer atualização da rede."""
    inputs, expected = (np.asarray(values, dtype=float) for values in dataset)
    if inputs.ndim != 2 or expected.ndim != 2 or inputs.size == 0 or expected.size == 0:
        raise ValueError(f"O conjunto de {name} deve conter matrizes não vazias.")
    if inputs.shape[1] != network.input_size:
        raise ValueError(f"Quantidade de entradas incompatível no conjunto de {name}.")
    if expected.shape != (inputs.shape[0], len(network.layers[-1].neurons)):
        raise ValueError(f"Dimensões das saídas incompatíveis no conjunto de {name}.")
    if not np.isfinite(inputs).all() or not np.isfinite(expected).all():
        raise ValueError(f"O conjunto de {name} deve conter valores numéricos finitos.")
    return inputs, expected


def train(
    network: NeuralNetwork,
    train_data: tuple[ArrayLike, ArrayLike],
    validation_data: tuple[ArrayLike, ArrayLike],
    epochs: int,
    learning_rate: float,
    seed: int | None = None,
) -> list[dict[str, int | float]]:
    """Treina com SGD e retorna epoch, train_mse e validation_mse por época.

    Recebe pares (entradas, saídas) já divididos e, opcionalmente, normalizados.
    Não ajusta nem aplica normalização. Somente train_data atualiza parâmetros.
    O teste não é recebido e deve ser reservado para avaliação final separada.
    A seed controla um único gerador de embaralhamento durante esta chamada.
    """
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs deve ser um inteiro maior que zero.")
    if (
        not isinstance(learning_rate, Real)
        or isinstance(learning_rate, (bool, np.bool_))
        or not np.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("learning_rate deve ser um número finito maior que zero.")
    if not network.layers:
        raise ValueError("A rede deve possuir pelo menos uma camada.")
    train_inputs, train_expected = _validate_dataset(network, train_data, "treinamento")
    validation_inputs, validation_expected = _validate_dataset(network, validation_data, "validação")
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(1, epochs + 1):
        # O mesmo índice preserva a associação entre entrada e saída.
        for index in rng.permutation(len(train_inputs)):
            network.train_step(train_inputs[index], train_expected[index], learning_rate)
        # Reavalia os conjuntos inteiros com os parâmetros finais da época.
        history.append({
            "epoch": epoch,
            "train_mse": network.evaluate_mse(train_inputs, train_expected),
            "validation_mse": network.evaluate_mse(validation_inputs, validation_expected),
        })
    return history

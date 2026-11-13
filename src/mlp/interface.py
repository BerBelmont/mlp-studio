"""Integração da interface: estado da sessão e adaptação de arquivos do navegador.

Não implementa algoritmos de rede: delega aos módulos acadêmicos existentes.
"""

from io import BytesIO
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from mlp.dados import load_csv, select_columns, split_dataset
from mlp.normalizacao import normalize_splits
from mlp.persistencia import load_network, save_network
from mlp.treinamento import train


def _update_state(state, **values):
    """Atualiza chaves pelo acesso comum a dict e session_state."""
    for key, value in values.items():
        state[key] = value


def initialize_state(state):
    """Inicializa apenas chaves ausentes, preservando a rede entre reruns."""
    defaults = {
        "network": None, "dataset": None, "dataset_name": None,
        "prepared": None, "history": [], "initial_mse": None,
        "evaluation": None, "input_columns": [], "output_columns": [],
        "model_ready": False, "normalizer": None, "revision": 0,
        "training_config": None,
        "preparation_config": None,
    }
    for key, value in defaults.items():
        if key not in state:
            state[key] = value


def invalidate_data(state):
    """Descarta resultados derivados, sem alterar os parâmetros da rede."""
    _update_state(state, prepared=None, history=[], initial_mse=None, evaluation=None,
                 input_columns=[], output_columns=[], training_config=None, preparation_config=None)
    state["revision"] += 1


def replace_network(state, network, loaded=False):
    """Troca a rede somente após uma criação ou leitura concluída com sucesso."""
    invalidate_data(state)
    _update_state(state, network=network, normalizer=network.normalizer, model_ready=loaded)


def replace_dataset(state, content, name):
    """Lê o CSV antes de invalidar conjuntos e histórico anteriores."""
    frame = load_csv(BytesIO(content))
    invalidate_data(state)
    _update_state(state, dataset=frame, dataset_name=name)


def prepare_data(state, input_columns, output_columns, percentages, seed, method):
    """Prepara conjuntos após divisão, sem reajustar a escala de um modelo treinado.

    Uma rede já treinada ou carregada conserva sua normalização registrada.
    Para mudar a escala, o usuário cria uma nova rede antes de preparar dados.
    """
    network = state["network"]
    inputs, expected = select_columns(state["dataset"], input_columns, output_columns, network)
    datasets = split_dataset(inputs, expected, *percentages, seed=seed)
    if state["model_ready"]:
        normalizer = network.normalizer
        if normalizer is not None:
            datasets = {name: (normalizer.transform(x), y) for name, (x, y) in datasets.items()}
    else:
        datasets, normalizer = normalize_splits(datasets, method)
    # Só altera o estado depois de todas as validações terem passado.
    invalidate_data(state)
    network.normalizer = normalizer
    _update_state(state, prepared=datasets, normalizer=normalizer,
                 input_columns=list(input_columns), output_columns=list(output_columns),
                 preparation_config={"percentages": [float(value) for value in percentages], "seed": seed,
                                     "method": normalizer.method if normalizer is not None else "none"})


def export_json(network):
    """Adapta save_network para download, mantendo a validação existente."""
    with TemporaryDirectory() as directory:
        path = Path(directory) / "rede.json"
        save_network(network, path)
        return path.read_bytes()


def import_json(content):
    """Adapta os bytes de upload para o carregador JSON existente."""
    with TemporaryDirectory() as directory:
        path = Path(directory) / "rede.json"
        path.write_bytes(content)
        return load_network(path)


def comparison_table(network, dataset, names):
    """Compara saídas com forward: as entradas do conjunto já estão preparadas."""
    inputs, expected = dataset
    predicted = np.array([network.forward(row) for row in inputs])
    columns = {}
    for i in range(expected.shape[1]):
        label = names[i] if len(names) == expected.shape[1] else f"Saída {i + 1}"
        columns[f"{label} · esperado"] = expected[:, i]
        columns[f"{label} · previsto"] = predicted[:, i]
    return pd.DataFrame(columns)


def train_session(state, epochs, learning_rate, seed=None):
    """Executa o treinamento em cópia e confirma o resultado somente se válido.

    Falhas numéricas não destroem a rede ativa nem o histórico anterior.
    A função delega integralmente SGD, épocas e MSE às APIs existentes.
    """
    network = state["network"]
    datasets = state["prepared"]
    candidate = deepcopy(network)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            initial = {name: candidate.evaluate_mse(*datasets[name]) for name in ("train", "validation")}
            history = train(candidate, datasets["train"], datasets["validation"], epochs, learning_rate, seed)
        if not all(np.isfinite(row[name]) for row in history for name in ("train_mse", "validation_mse")):
            raise ValueError("O treinamento produziu MSE não finito.")
    except (FloatingPointError, OverflowError) as error:
        raise ValueError("O treinamento divergiu numericamente. Reduza a taxa de aprendizado; o modelo anterior foi preservado.") from error
    for original_layer, trained_layer in zip(network.layers, candidate.layers):
        for original, trained in zip(original_layer.neurons, trained_layer.neurons):
            original.weights[:] = trained.weights
            original.bias = trained.bias
            original.z, original.output, original.delta = trained.z, trained.output, trained.delta
    _update_state(state, history=history, initial_mse=initial, model_ready=True, evaluation=None,
                  training_config={"epochs": epochs, "learning_rate": learning_rate,
                                   "examples": len(datasets["train"][0])})
    return history

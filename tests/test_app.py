"""Verifica renderização e ações reais com o AppTest oficial do Streamlit."""

from pathlib import Path
import json
import pytest

from streamlit.testing.v1 import AppTest

from mlp.interface import prepare_data, replace_dataset, export_json


APP = Path(__file__).resolve().parents[1] / "app.py"


def test_inicio_e_navegacao_sem_rede():
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert app.header[0].value == "Dashboard"
    for page in ["Dataset", "Treinamento", "Avaliação", "Modelos e Previsões", "Arquitetura Neural", "Dashboard"]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception


def test_fluxo_criacao_treino_avaliacao_previsao_e_estado():
    app = AppTest.from_file(str(APP), default_timeout=15).run()
    app.sidebar.radio[0].set_value("Arquitetura Neural").run()
    # Cria uma rede com uma entrada e apenas a camada de saída linear.
    app.number_input[0].set_value(0).run()
    next(widget for widget in app.number_input if widget.label == "Quantidade de entradas").set_value(1)
    app.button[0].click().run()
    assert not app.exception
    network = app.session_state["network"]
    csv = b"x,y\n0,1\n1,3\n2,5\n3,7\n4,9\n5,11\n6,13\n7,15\n8,17\n9,19\n"
    replace_dataset(app.session_state, csv, "dados.csv")
    prepare_data(app.session_state, ["x"], ["y"], [60, 20, 20], 42, "minmax")
    app.sidebar.radio[0].set_value("Treinamento").run()
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert len(app.session_state["history"]) == 100
    assert app.session_state["history"][-1]["train_mse"] < app.session_state["initial_mse"]["train"]
    weights = network.layers[0].neurons[0].weights.copy()
    bias = network.layers[0].neurons[0].bias
    app.sidebar.radio[0].set_value("Avaliação").run()
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state["evaluation"] is not None
    app.sidebar.radio[0].set_value("Modelos e Previsões").run()
    next(button for button in app.button if button.label == "Realizar previsão").click().run()
    assert not app.exception
    assert len(app.dataframe) == 1
    app.sidebar.radio[0].set_value("Modelos e Previsões").run()
    assert not app.exception
    assert not app.error
    assert app.session_state["network"] is network
    assert (network.layers[0].neurons[0].weights == weights).all()
    assert network.layers[0].neurons[0].bias == bias


def test_preparacao_pelo_formulario_e_erros_claros():
    app = AppTest.from_file(str(APP)).run()
    app.sidebar.radio[0].set_value("Arquitetura Neural").run()
    app.button[0].click().run()  # Rede padrão: 2 entradas, 1 saída.
    csv = b"a,b,y\n0,0,0\n1,1,2\n2,2,4\n3,3,6\n4,4,8\n5,5,10\n6,6,12\n7,7,14\n8,8,16\n9,9,18\n"
    replace_dataset(app.session_state, csv, "exemplo.csv")
    app.sidebar.radio[0].set_value("Dataset").run()
    assert not app.exception
    app.button[-1].click().run()
    assert app.error  # Nenhuma coluna selecionada.
    assert app.session_state["prepared"] is None
    app.multiselect[0].set_value(["a", "b"])
    app.multiselect[1].set_value(["y"])
    app.selectbox[0].set_value("minmax")
    app.button[-1].click().run()
    assert not app.exception
    assert not app.error
    assert app.session_state["prepared"] is not None
    assert app.session_state["network"].normalizer.method == "minmax"
    assert app.session_state["network"].normalizer is app.session_state["normalizer"]


def test_dashboard_sem_metricas_inventadas():
    app = AppTest.from_file(str(APP)).run()
    assert len(app.metric) == 0
    assert any(button.label == "Criar minha primeira rede" for button in app.button)
    assert not app.exception


def test_dashboard_com_rede_e_estado_preservado():
    app = AppTest.from_file(str(APP)).run()
    app.sidebar.radio[0].set_value("Arquitetura Neural").run()
    app.button[0].click().run()
    network = app.session_state["network"]
    app.sidebar.radio[0].set_value("Dashboard").run()
    assert not app.exception
    assert app.session_state["network"] is network
    assert app.metric[0].value == "2 → 3 → 1"
    assert app.metric[1].value == "4"
    assert app.metric[2].value == "2"
    assert app.metric[3].value == "Pronto para configurar dados"
    assert len(app.metric) == 4


def test_previa_atualiza_antes_de_criar_sem_instanciar_rede():
    app = AppTest.from_file(str(APP)).run()
    next(button for button in app.button if button.label == "Criar minha primeira rede").click().run()
    assert app.sidebar.radio[0].value == "Arquitetura Neural"
    assert app.session_state["network"] is None
    initial = json.loads(app.get("plotly_chart")[0].proto.spec)
    next(widget for widget in app.number_input if widget.label == "Quantidade de entradas").set_value(5).run()
    changed = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert len(initial["data"][1]["x"]) == 2
    assert len(changed["data"][1]["x"]) == 5
    assert app.session_state["network"] is None
    assert app.metric[1].value == "18"  # 5*3 + 3*1 conexões reais do rascunho.
    next(widget for widget in app.selectbox if widget.label == "Ativação · oculta 1").set_value("tanh").run()
    changed = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert any("Tanh" in annotation["text"] for annotation in changed["layout"]["annotations"])
    app.sidebar.radio[0].set_value("Dashboard").run()
    app.sidebar.radio[0].set_value("Arquitetura Neural").run()
    assert app.session_state["architecture_draft"]["input_size"] == 5
    assert app.session_state["network"] is None
    assert not app.exception


def test_rascunho_nao_altera_rede_treinada_e_resultados():
    app = AppTest.from_file(str(APP)).run()
    app.sidebar.radio[0].set_value("Arquitetura Neural").run()
    app.button[0].click().run()
    network = app.session_state["network"]
    csv = b"a,b,y\n0,0,0\n1,1,2\n2,2,4\n3,3,6\n4,4,8\n5,5,10\n6,6,12\n7,7,14\n8,8,16\n9,9,18\n"
    replace_dataset(app.session_state, csv, "dados.csv")
    prepare_data(app.session_state, ["a", "b"], ["y"], [60, 20, 20], 42, "minmax")
    app.sidebar.radio[0].set_value("Treinamento").run()
    app.number_input[0].set_value(1)
    app.button[0].click().run()
    original = [(n.weights.copy(), n.bias) for layer in network.layers for n in layer.neurons]
    history = list(app.session_state["history"])
    prepared = app.session_state["prepared"]
    app.sidebar.radio[0].set_value("Arquitetura Neural").run()
    next(widget for widget in app.number_input if widget.label == "Quantidade de entradas").set_value(1000).run()
    next(widget for widget in app.number_input if widget.label == "Neurônios · oculta 1").set_value(1000).run()
    assert not app.exception
    figure = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert len(figure["data"][1]["x"]) == 8
    assert app.session_state["network"] is network
    assert network.input_size == 2
    assert app.session_state["history"] == history
    assert app.session_state["prepared"] is prepared
    for neuron, (weights, bias) in zip([n for layer in network.layers for n in layer.neurons], original):
        assert (neuron.weights == weights).all()
        assert neuron.bias == bias


def test_uploads_csv_e_json_pelos_widgets_e_download():
    app = AppTest.from_file(str(APP), default_timeout=15).run()
    if not hasattr(app, "file_uploader"):
        pytest.skip("A versão instalada de AppTest não oferece interação com upload; auxiliares são cobertos separadamente.")
    app.sidebar.radio[0].set_value("Arquitetura Neural").run()
    app.button[0].click().run()
    app.sidebar.radio[0].set_value("Dataset").run()
    content = b"a,b,y\n0,0,0\n1,1,2\n2,2,4\n3,3,6\n4,4,8\n5,5,10\n6,6,12\n7,7,14\n8,8,16\n9,9,18\n"
    app.file_uploader[0].set_value(("dados.csv", content, "text/csv")).run()
    next(button for button in app.button if button.label == "Carregar CSV").click().run()
    assert not app.exception
    app.multiselect[0].set_value(["a", "b"])
    app.multiselect[1].set_value(["y"])
    app.number_input[0].set_value(60.)
    app.number_input[1].set_value(20.)
    app.number_input[2].set_value(20.)
    app.selectbox[0].set_value("minmax")
    next(button for button in app.button if button.label == "Validar e preparar conjuntos").click().run()
    assert not app.error
    app.sidebar.radio[0].set_value("Dashboard").run()
    app.sidebar.radio[0].set_value("Dataset").run()
    assert app.multiselect[0].value == ["a", "b"]
    assert app.number_input[0].value == 60.
    assert app.selectbox[0].value == "minmax"
    app.sidebar.radio[0].set_value("Treinamento").run()
    app.number_input[0].set_value(2)
    next(button for button in app.button if button.label == "Iniciar treinamento").click().run()
    network = app.session_state["network"]
    before = network.predict([1, 1])
    saved = export_json(network)
    app.sidebar.radio[0].set_value("Modelos e Previsões").run()
    app.download_button[0].click().run()
    assert app.session_state["network"] is network
    app.file_uploader[0].set_value(("rede.json", saved, "application/json")).run()
    next(button for button in app.button if button.label == "Carregar rede JSON").click().run()
    assert not app.exception
    assert not app.error
    assert app.session_state["prepared"] is None
    assert (app.session_state["network"].predict([1, 1]) == before).all()

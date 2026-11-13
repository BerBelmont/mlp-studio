"""Testes da navegação disponível nesta primeira etapa."""

import pytest

from mlp.cli import main


def executar_menu(monkeypatch, entradas):
    """Substitui a entrada do terminal por uma sequência de opções."""
    respostas = iter(entradas)
    monkeypatch.setattr("builtins.input", lambda _: next(respostas))
    main()


@pytest.mark.parametrize("opcao", ["1", "2", "3", "4", "5", "6", "7"])
def test_opcoes_informam_funcionalidade_pendente(monkeypatch, capsys, opcao):
    executar_menu(monkeypatch, [opcao, "0"])
    saida = capsys.readouterr().out
    assert "funcionalidade ainda não implementada" in saida
    assert "Aplicação encerrada." in saida


def test_opcao_invalida_permite_continuar(monkeypatch, capsys):
    executar_menu(monkeypatch, ["inválida", "0"])
    assert "Opção inválida" in capsys.readouterr().out


@pytest.mark.parametrize("erro", [EOFError, KeyboardInterrupt])
def test_encerramento_do_terminal(monkeypatch, capsys, erro):
    def interromper(_):
        raise erro

    monkeypatch.setattr("builtins.input", interromper)
    main()
    assert "Aplicação encerrada." in capsys.readouterr().out

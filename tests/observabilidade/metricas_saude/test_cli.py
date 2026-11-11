"""Testes da interface de linha de comando da coleta de saude."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from monitoramento.observabilidade.metricas_saude import cli

from .apoio import CATALOGO_REAL


def test_main_ajuda(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["metricas-saude", "--help"])
    with pytest.raises(SystemExit) as saida:
        cli.main()
    assert saida.value.code == 0
    assert "--catalog" in capsys.readouterr().out


def test_main_publica_saude_sem_execucoes(tmp_path: Path, monkeypatch, capsys) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    destino = tmp_path / "saude"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "metricas-saude",
            "--runs-dir",
            str(runs),
            "--output-dir",
            str(destino),
            "--catalog",
            str(CATALOGO_REAL),
        ],
    )
    with pytest.raises(SystemExit) as saida:
        cli.main()
    assert saida.value.code == 0
    assert "0 execucoes" in capsys.readouterr().out
    assert json.loads((destino / "saude.json").read_text())["execucoes"] == []


def test_main_diretorio_ausente(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "metricas-saude",
            "--runs-dir",
            str(tmp_path / "ausente"),
            "--catalog",
            str(CATALOGO_REAL),
            "--output-dir",
            str(tmp_path / "saude"),
        ],
    )
    with pytest.raises(SystemExit) as saida:
        cli.main()
    assert saida.value.code == 1
    assert "Falha ao coletar as metricas de saude:" in capsys.readouterr().out

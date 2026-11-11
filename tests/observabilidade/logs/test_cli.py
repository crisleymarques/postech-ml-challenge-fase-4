"""Testes da interface de linha de comando dos logs operacionais."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from monitoramento.observabilidade.logs import cli


def test_main_ajuda(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["logs", "--help"])
    with pytest.raises(SystemExit) as saida:
        cli.main()
    assert saida.value.code == 0
    assert "--runs-dir" in capsys.readouterr().out


def test_main_publica_logs_vazios(tmp_path: Path, monkeypatch, capsys) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    destino = tmp_path / "logs"
    monkeypatch.setattr(
        sys, "argv", ["logs", "--runs-dir", str(runs), "--output-dir", str(destino)]
    )
    with pytest.raises(SystemExit) as saida:
        cli.main()
    assert saida.value.code == 0
    assert "0 execucoes" in capsys.readouterr().out
    assert json.loads((destino / "resumo.json").read_text())["eventos"]["total"] == 0
    assert (destino / "operacional.jsonl").read_text() == ""


def test_main_diretorio_ausente(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["logs", "--runs-dir", str(tmp_path / "ausente")])
    with pytest.raises(SystemExit) as saida:
        cli.main()
    assert saida.value.code == 1
    assert "Falha ao consolidar os logs:" in capsys.readouterr().out

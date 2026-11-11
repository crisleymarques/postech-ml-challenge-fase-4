"""Testes de entradas das metricas de saude."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from monitoramento.observabilidade.metricas_saude import entradas

from .apoio import escrever_run, evento_drift


def test_listar_execucoes_manifesto_ou_eventos(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    escrever_run(runs_dir, "run-b", eventos=[])
    escrever_run(runs_dir, "run-a", manifesto={"run_id": "run-a"})
    (runs_dir / "run-vazio").mkdir()

    execucoes = entradas.listar_execucoes(runs_dir)

    assert [caminho.name for caminho in execucoes] == ["run-a", "run-b"]


def test_listar_execucoes_diretorio_ausente(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        entradas.listar_execucoes(tmp_path / "inexistente")


def test_ler_json_tolerante(tmp_path: Path) -> None:
    assert entradas.ler_json(tmp_path / "ausente.json") is None

    invalido = tmp_path / "invalido.json"
    invalido.write_text("{nao-eh-json}", encoding="utf-8")
    assert entradas.ler_json(invalido) is None


def test_ler_eventos_tolerante(tmp_path: Path) -> None:
    caminho = tmp_path / "events.jsonl"
    caminho.write_text(
        json.dumps(evento_drift("run-a", "deteccao")) + "\n{invalido}\n\n", encoding="utf-8"
    )

    eventos = entradas.ler_eventos(caminho)

    assert len(eventos) == 1
    assert eventos[0]["event"] == "drift_alert"

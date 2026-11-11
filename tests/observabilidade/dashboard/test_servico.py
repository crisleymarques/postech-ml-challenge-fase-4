"""Testes de servico do painel."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from monitoramento.observabilidade.dashboard import servico
from tests.observabilidade.dashboard.apoio import escrever_execucao, evento


def test_carregar_painel_disponibilidade(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    escrever_execucao(runs_dir, "run-a", manifesto={"run_id": "run-a", "status": "success"})
    logs = tmp_path / "resumo.json"
    logs.write_text(json.dumps({"generated_at": "2026-11-09T00:00:00.000000Z"}), encoding="utf-8")

    painel = servico.carregar_painel(
        runs_dir, logs, tmp_path / "saude.json", tmp_path / "alert.json"
    )

    assert painel["disponibilidade"] == {"logs": True, "saude": False, "alertas": False}
    assert painel["logs"] is not None
    assert painel["saude"] is None
    assert painel["fontes"]["runs_dir"] == runs_dir.as_posix()


@pytest.mark.parametrize("invalido", [None, [], 1, "texto", True])
def test_evento_nao_objeto_nao_impede_outras_execucoes(tmp_path, invalido):
    runs = tmp_path / "runs"
    ruim = escrever_execucao(runs, "ruim")
    (ruim / "events.jsonl").write_text(json.dumps(invalido) + "\n", encoding="utf-8")
    escrever_execucao(
        runs,
        "boa",
        manifesto={
            "contract_version": "1.0",
            "run_id": "boa",
            "stage": "predicao",
            "status": "success",
        },
        eventos=[evento("boa", "predicao", "completed")],
    )
    painel = servico.carregar_painel(
        runs, tmp_path / "logs", tmp_path / "saude", tmp_path / "alertas"
    )
    assert painel["resumo_execucoes"]["por_status"] == {"success": 1, "incomplete": 1}
    assert painel["execucoes"][1]["event_count"] == 0

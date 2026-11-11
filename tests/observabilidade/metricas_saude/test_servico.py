"""Testes de servico das metricas de saude."""

from __future__ import annotations

from pathlib import Path

import pytest

from monitoramento.observabilidade.metricas_saude import classificacao, servico

from .apoio import CATALOGO_REAL, escrever_run, evento_drift


def test_gerar_ponta_a_ponta(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    escrever_run(
        runs_dir,
        "run-deteccao",
        manifesto={
            "contract_version": "1.0",
            "run_id": "run-deteccao",
            "stage": "deteccao",
            "status": "success",
            "started_at": "2026-11-09T02:50:49.872109Z",
            "finished_at": "2026-11-09T02:51:05.727987Z",
            "inputs": [{"row_count": 60000}, {"row_count": 10000}],
        },
        drift={
            "summary": {
                "dataset_drift": True,
                "drifted_columns": ["a", "b"],
                "drift_share": 0.5,
            }
        },
        eventos=[
            evento_drift("run-deteccao", "deteccao"),
            {"run_id": "run-deteccao", "stage": "deteccao", "event": "completed"},
        ],
    )
    escrever_run(
        runs_dir,
        "run-validacao",
        manifesto={
            "run_id": "run-validacao",
            "stage": "validacao",
            "status": "rejected",
            "started_at": "2026-11-09T02:12:35.456240Z",
            "finished_at": "2026-11-09T02:12:35.971287Z",
            "inputs": [{"row_count": 601}],
        },
        quality={"approved": False, "violations": [{"rule_id": "R_A", "count": 2}]},
    )

    output_dir = tmp_path / "saude"
    saude = servico.gerar(runs_dir, output_dir, CATALOGO_REAL)

    assert (output_dir / "saude.json").is_file()
    assert saude["resumo"]["execucoes"] == 2

    por_run = {execucao["run_id"]: execucao for execucao in saude["execucoes"]}
    ids_deteccao = {medida["id"] for medida in por_run["run-deteccao"]["metricas"]}
    ids_validacao = {medida["id"] for medida in por_run["run-validacao"]["metricas"]}

    assert "drift.dataset_drift" in ids_deteccao
    assert "qualidade.violacoes_total" not in ids_deteccao
    assert "qualidade.violacoes_total" in ids_validacao
    assert "drift.dataset_drift" not in ids_validacao

    medidas_deteccao = {medida["id"]: medida for medida in por_run["run-deteccao"]["metricas"]}
    assert medidas_deteccao["drift.dataset_drift"]["situacao"] == classificacao.SITUACAO_ATENCAO
    assert medidas_deteccao["pipeline.status_execucao"]["situacao"] == classificacao.SITUACAO_OK

    medidas_validacao = {medida["id"]: medida for medida in por_run["run-validacao"]["metricas"]}
    assert (
        medidas_validacao["pipeline.status_execucao"]["situacao"] == classificacao.SITUACAO_ATENCAO
    )
    assert medidas_validacao["qualidade.violacoes_total"]["valor"] == 2
    assert (
        medidas_validacao["qualidade.violacoes_total"]["situacao"] == classificacao.SITUACAO_ATENCAO
    )

    agregados = {medida["id"]: medida for medida in saude["agregados"]}
    assert agregados["drift.alertas_total"]["valor"] == 1
    assert agregados["pipeline.falhas_total"]["valor"] == 0
    assert agregados["qualidade.lotes_rejeitados"]["valor"] == 1
    assert agregados["infra.escrita_ok"]["valor"] is True

    assert saude["resumo"]["por_situacao"][classificacao.SITUACAO_ATENCAO] >= 4


def test_gerar_sem_execucoes(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    saude = servico.gerar(runs_dir, tmp_path / "saude", CATALOGO_REAL)

    assert saude["execucoes"] == []
    assert saude["resumo"]["execucoes"] == 0
    assert saude["resumo"]["medidas"] == len(saude["agregados"])


def test_publicacao_parcial_nao_alimenta_sucesso_ou_metricas_de_drift(tmp_path):
    escrever_run(
        tmp_path / "runs",
        "parcial",
        manifesto={
            "contract_version": "1.0",
            "run_id": "parcial",
            "stage": "deteccao",
            "status": "success",
        },
        drift={"summary": {"dataset_drift": False, "drift_share": 0}},
        eventos=[{"run_id": "parcial", "stage": "deteccao", "event": "started"}],
    )
    saude = servico.gerar(tmp_path / "runs", tmp_path / "saude", CATALOGO_REAL)
    execucao = saude["execucoes"][0]
    assert execucao["status"] == "incomplete"
    medidas = {m["id"]: m for m in execucao["metricas"]}
    assert medidas["pipeline.status_execucao"]["situacao"] == classificacao.SITUACAO_CRITICO
    assert medidas["drift.dataset_drift"]["disponibilidade"] == classificacao.INDISPONIVEL
    taxa = next(m for m in saude["agregados"] if m["id"] == "pipeline.taxa_sucesso")
    assert taxa["valor"] == 0


def test_gerar_runs_dir_inexistente(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        servico.gerar(tmp_path / "inexistente", tmp_path / "saude", CATALOGO_REAL)

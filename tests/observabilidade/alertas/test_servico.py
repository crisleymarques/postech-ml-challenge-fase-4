"""Testes de servico dos alertas."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from monitoramento.observabilidade.alertas import servico as al

from .apoio import CONFIG_REAL, medida, resumo_logs, saude


def test_gerar_ponta_a_ponta(tmp_path: Path) -> None:
    saude_path = tmp_path / "saude.json"
    logs_path = tmp_path / "resumo.json"
    saude_path.write_text(
        json.dumps(
            saude(
                [
                    {
                        "run_id": "run-validacao",
                        "stage": "validacao",
                        "status": "rejected",
                        "metricas": [
                            medida(
                                "pipeline.status_execucao",
                                "pipeline",
                                "atencao",
                                "rejected",
                                "categoria",
                            ),
                            medida("qualidade.violacoes_total", "qualidade", "atencao", 7),
                        ],
                    },
                    {
                        "run_id": "run-deteccao",
                        "stage": "deteccao",
                        "status": "success",
                        "metricas": [
                            medida("drift.dataset_drift", "drift", "atencao", True, "booleano"),
                            medida(
                                "drift.variaveis_com_drift", "drift", "atencao", 13, "variaveis"
                            ),
                            medida("drift.proporcao_drift", "drift", "critico", 0.59, "proporcao"),
                        ],
                    },
                ],
                agregados=[medida("drift.alertas_total", "drift", "ok", 1, "alertas")],
            )
        ),
        encoding="utf-8",
    )
    logs_path.write_text(
        json.dumps(
            resumo_logs(
                falhas=[
                    {
                        "run_id": "run-validacao",
                        "stage": "validacao",
                        "categoria": "validacao",
                        "event": "rejected",
                        "error_code": "QUALIDADE_REPROVADA",
                        "timestamp": "2026-11-09T00:00:00.000000Z",
                    }
                ],
                alertas_drift=[
                    {
                        "run_id": "run-deteccao",
                        "stage": "deteccao",
                        "scenario_id": "cenario-01",
                        "batch_id": "lote-01",
                        "timestamp": "2026-11-09T00:00:00.000000Z",
                    }
                ],
            )
        ),
        encoding="utf-8",
    )

    saida = tmp_path / "alertas"
    relatorio = al.gerar(saude_path, logs_path, CONFIG_REAL, saida)

    assert relatorio["resumo"]["total"] == 7
    assert relatorio["resumo"]["por_severidade"] == {"warning": 5, "critical": 2}
    assert relatorio["drift"]["dataset_drift"] is True
    assert relatorio["drift"]["variaveis_com_drift"] == 13
    assert relatorio["drift"]["alertas_total"] == 1
    assert (saida / "alertas.json").is_file()

    gravado = json.loads((saida / "alertas.json").read_text(encoding="utf-8"))
    assert gravado["resumo"]["total"] == 7
    assert len(gravado["alertas"]) == 7
    assert len(gravado["resumo"]["criticos"]) == 2


def test_gerar_saude_ausente(tmp_path: Path) -> None:
    logs_path = tmp_path / "resumo.json"
    logs_path.write_text(json.dumps(resumo_logs()), encoding="utf-8")
    with pytest.raises(ValueError, match="nao encontrado"):
        al.gerar(tmp_path / "saude.json", logs_path, CONFIG_REAL, tmp_path / "alertas")


def test_gerar_logs_ausente(tmp_path: Path) -> None:
    saude_path = tmp_path / "saude.json"
    saude_path.write_text(json.dumps(saude([])), encoding="utf-8")
    with pytest.raises(ValueError, match="nao encontrado"):
        al.gerar(saude_path, tmp_path / "resumo.json", CONFIG_REAL, tmp_path / "alertas")

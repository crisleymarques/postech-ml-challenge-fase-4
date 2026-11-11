"""Testes e apoio de test resumo."""

from __future__ import annotations

from pathlib import Path

from monitoramento.observabilidade.logs import classificacao
from monitoramento.observabilidade.logs import resumo as consolidacao

from .apoio import evento


def test_montar_resumo_conta_por_status_e_categoria(tmp_path: Path) -> None:
    eventos = [
        classificacao.classificar_evento(
            evento(run_id="run-a", stage="preparacao", tipo="started")
        ),
        classificacao.classificar_evento(
            evento(run_id="run-a", stage="preparacao", tipo="completed")
        ),
        classificacao.classificar_evento(
            evento(
                run_id="run-b",
                stage="validacao",
                tipo="rejected",
                nivel="ERROR",
                error_code="QUALIDADE_REPROVADA",
            )
        ),
        classificacao.classificar_evento(
            evento(
                run_id="run-c",
                stage="simulacao",
                tipo="drift_alert",
                scenario_id="data-drift",
                batch_id="batch-f201",
            )
        ),
    ]
    execucoes = [
        {
            "run_id": "run-a",
            "stage": "preparacao",
            "status": "success",
            "resultado": "sucesso",
            "transformations": ["trim_espacos"],
            "error_code": None,
        },
        {
            "run_id": "run-b",
            "stage": "validacao",
            "status": "rejected",
            "resultado": "rejeicao",
            "transformations": [],
            "error_code": "QUALIDADE_REPROVADA",
        },
        {
            "run_id": "run-c",
            "stage": "simulacao",
            "status": "success",
            "resultado": "sucesso",
            "transformations": [],
            "error_code": None,
        },
        {
            "run_id": "run-d",
            "stage": "treinamento",
            "status": "incomplete",
            "resultado": "incompleto",
            "transformations": [],
            "error_code": None,
        },
    ]

    resumo = consolidacao.montar_resumo(
        runs_dir=tmp_path / "runs",
        output_dir=tmp_path / "logs",
        execucoes=execucoes,
        eventos=eventos,
    )

    assert resumo["execucoes"]["total"] == 4
    assert resumo["execucoes"]["por_status"] == {"success": 2, "rejected": 1, "incomplete": 1}
    assert resumo["execucoes"]["incompletas"] == ["run-d"]
    assert resumo["eventos"]["total"] == 4
    # A preparacao alimenta ingestao e transformacao.
    assert resumo["por_categoria"]["ingestao"]["inicio"] == 1
    assert resumo["por_categoria"]["transformacao"]["sucesso"] == 1
    assert resumo["por_categoria"]["validacao"]["rejeicao"] == 1
    assert resumo["por_categoria"]["simulacao"]["alerta"] == 1


def test_montar_resumo_reune_falhas_alertas_e_transformacoes(tmp_path: Path) -> None:
    eventos = [
        classificacao.classificar_evento(
            evento(
                run_id="run-b",
                stage="validacao",
                tipo="rejected",
                nivel="ERROR",
                error_code="QUALIDADE_REPROVADA",
            )
        ),
        classificacao.classificar_evento(
            evento(
                run_id="run-c",
                stage="simulacao",
                tipo="drift_alert",
                scenario_id="data-drift",
                batch_id="batch-f201",
            )
        ),
    ]
    execucoes = [
        {
            "run_id": "run-a",
            "stage": "preparacao",
            "status": "success",
            "resultado": "sucesso",
            "transformations": ["trim_espacos", "binarizar_alvo"],
            "error_code": None,
        },
        {
            "run_id": "run-b",
            "stage": "validacao",
            "status": "rejected",
            "resultado": "rejeicao",
            "transformations": [],
            "error_code": "QUALIDADE_REPROVADA",
        },
    ]

    execucoes.append(
        {
            "run_id": "run-c",
            "stage": "simulacao",
            "status": "success",
            "resultado": "sucesso",
            "transformations": [],
            "error_code": None,
        }
    )
    resumo = consolidacao.montar_resumo(
        runs_dir=tmp_path / "runs",
        output_dir=tmp_path / "logs",
        execucoes=execucoes,
        eventos=eventos,
    )

    assert len(resumo["falhas"]) == 1
    assert resumo["falhas"][0]["error_code"] == "QUALIDADE_REPROVADA"
    assert resumo["falhas"][0]["run_id"] == "run-b"
    assert len(resumo["alertas"]) == 1
    assert resumo["alertas"][0]["scenario_id"] == "data-drift"
    assert resumo["transformacoes"] == [
        {"run_id": "run-a", "nomes": ["trim_espacos", "binarizar_alvo"], "resultado": "sucesso"}
    ]

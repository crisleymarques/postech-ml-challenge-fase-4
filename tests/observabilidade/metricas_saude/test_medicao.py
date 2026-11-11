"""Testes de medicao das metricas de saude."""

from __future__ import annotations

from monitoramento.observabilidade.metricas_saude import classificacao, medicao

from .apoio import metrica


def test_medir_execucao_ignora_etapa_nao_aplicavel() -> None:
    restrita = metrica(id="drift.dataset_drift", etapas=["deteccao"])

    assert medicao.medir_execucao(restrita, {"stage": "treinamento"}) is None


def test_medir_execucao_coleta_valor() -> None:
    metrica_drift = metrica(id="drift.dataset_drift", etapas=["deteccao"])
    contexto = {"stage": "deteccao", "drift": {"summary": {"dataset_drift": True}}}

    medida = medicao.medir_execucao(metrica_drift, contexto)

    assert medida is not None
    assert medida["valor"] is True
    assert medida["disponibilidade"] == classificacao.DISPONIVEL
    assert medida["situacao"] == classificacao.SITUACAO_ATENCAO


def test_medir_execucao_indisponivel_sem_artefato() -> None:
    metrica_drift = metrica(id="drift.dataset_drift", etapas=["deteccao"])
    contexto = {"stage": "deteccao", "drift": None}

    medida = medicao.medir_execucao(metrica_drift, contexto)

    assert medida is not None
    assert medida["disponibilidade"] == classificacao.INDISPONIVEL
    assert medida["situacao"] == classificacao.SITUACAO_INDISPONIVEL

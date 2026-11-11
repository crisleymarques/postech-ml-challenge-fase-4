"""Testes de classificacao das metricas de saude."""

from __future__ import annotations

from monitoramento.observabilidade.metricas_saude import classificacao

from .apoio import metrica


def test_situacao_numerica_maior_pior() -> None:
    direcao = metrica(
        id="pipeline.duracao_s",
        direcao="maior_pior",
        limiar_degracao=60.0,
        limiar_falha=300.0,
    )

    assert classificacao.situacao_medida(direcao, 10.0) == classificacao.SITUACAO_OK
    assert classificacao.situacao_medida(direcao, 60.0) == classificacao.SITUACAO_ATENCAO
    assert classificacao.situacao_medida(direcao, 300.0) == classificacao.SITUACAO_CRITICO
    assert classificacao.situacao_medida(direcao, None) == classificacao.SITUACAO_INDISPONIVEL


def test_situacao_numerica_menor_pior() -> None:
    taxa = metrica(
        id="pipeline.taxa_sucesso",
        direcao="menor_pior",
        limiar_degracao=0.9,
        limiar_falha=0.5,
    )

    assert classificacao.situacao_medida(taxa, 0.95) == classificacao.SITUACAO_OK
    assert classificacao.situacao_medida(taxa, 0.7) == classificacao.SITUACAO_ATENCAO
    assert classificacao.situacao_medida(taxa, 0.4) == classificacao.SITUACAO_CRITICO


def test_situacao_sem_direcao_fica_ok() -> None:
    assert classificacao.situacao_medida(metrica(valor=None), 5) == classificacao.SITUACAO_OK


def test_situacao_especiais() -> None:
    status = metrica(id="pipeline.status_execucao")
    assert classificacao.situacao_medida(status, "success") == classificacao.SITUACAO_OK
    assert classificacao.situacao_medida(status, "rejected") == classificacao.SITUACAO_ATENCAO
    assert classificacao.situacao_medida(status, "failed") == classificacao.SITUACAO_CRITICO
    assert classificacao.situacao_medida(status, "incomplete") == classificacao.SITUACAO_CRITICO
    assert classificacao.situacao_medida(status, "outro") == classificacao.SITUACAO_INDISPONIVEL

    drift = metrica(id="drift.dataset_drift")
    assert classificacao.situacao_medida(drift, False) == classificacao.SITUACAO_OK
    assert classificacao.situacao_medida(drift, True) == classificacao.SITUACAO_ATENCAO

    escrita = metrica(id="infra.escrita_ok")
    assert classificacao.situacao_medida(escrita, True) == classificacao.SITUACAO_OK
    assert classificacao.situacao_medida(escrita, False) == classificacao.SITUACAO_CRITICO

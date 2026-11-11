"""Testes e apoio de test classificacao."""

from __future__ import annotations

from monitoramento.observabilidade.logs import classificacao

from .apoio import evento


def test_categoria_da_etapa_mapeia_etapas_conhecidas() -> None:
    assert classificacao.categoria_da_etapa("preparacao") == "ingestao"
    assert classificacao.categoria_da_etapa("validacao") == "validacao"
    assert classificacao.categoria_da_etapa("deteccao") == "drift"
    assert classificacao.categoria_da_etapa("analise") == "drift"
    assert classificacao.categoria_da_etapa("avaliacao") == "avaliacao"


def test_categoria_da_etapa_desconhecida() -> None:
    assert classificacao.categoria_da_etapa(None) == classificacao.CATEGORIA_DESCONHECIDA
    assert classificacao.categoria_da_etapa("etapa-nova") == classificacao.CATEGORIA_DESCONHECIDA


def test_categorias_da_etapa_preparacao_alimenta_ingestao_e_transformacao() -> None:
    assert classificacao.categorias_da_etapa("preparacao") == ("ingestao", "transformacao")
    assert classificacao.categorias_da_etapa("treinamento") == ("treinamento",)


def test_resultado_do_evento() -> None:
    assert classificacao.resultado_do_evento("started") == "inicio"
    assert classificacao.resultado_do_evento("completed") == "sucesso"
    assert classificacao.resultado_do_evento("rejected") == "rejeicao"
    assert classificacao.resultado_do_evento("failed") == "falha"
    assert classificacao.resultado_do_evento("skipped") == "ignorado"
    assert classificacao.resultado_do_evento("drift_alert") == "alerta"
    assert classificacao.resultado_do_evento("outro") == classificacao.RESULTADO_DESCONHECIDO


def test_resultado_do_status() -> None:
    assert classificacao.resultado_do_status("success") == "sucesso"
    assert classificacao.resultado_do_status("rejected") == "rejeicao"
    assert classificacao.resultado_do_status("failed") == "falha"
    assert classificacao.resultado_do_status(None) == "incompleto"


def test_inferir_status_por_evento_terminal() -> None:
    eventos = [
        evento(run_id="run-a", stage="treinamento", tipo="started"),
        evento(run_id="run-a", stage="treinamento", tipo="completed"),
    ]
    assert classificacao.inferir_status(eventos) == "success"

    rejeitado = [
        evento(run_id="run-a", stage="validacao", tipo="started"),
        evento(run_id="run-a", stage="validacao", tipo="rejected", nivel="ERROR"),
    ]
    assert classificacao.inferir_status(rejeitado) == "rejected"

    incompleto = [evento(run_id="run-a", stage="treinamento", tipo="started")]
    assert classificacao.inferir_status(incompleto) == classificacao.STATUS_INCOMPLETO


def test_classificar_evento_adiciona_categoria_e_resultado() -> None:
    classificado = classificacao.classificar_evento(
        evento(run_id="run-a", stage="preparacao", tipo="completed")
    )

    assert classificado["categoria"] == "ingestao"
    assert classificado["resultado"] == "sucesso"
    assert classificado["stage"] == "preparacao"

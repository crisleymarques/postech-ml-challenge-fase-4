"""Testes de coletores das metricas de saude."""

from __future__ import annotations

from pathlib import Path

import pytest

from monitoramento.observabilidade.metricas_saude import coletores

from .apoio import evento_drift


def test_coletar_duracao() -> None:
    manifesto = {
        "started_at": "2026-11-09T02:12:35.456240Z",
        "finished_at": "2026-11-09T02:12:35.971287Z",
    }

    assert coletores.coletar_duracao(manifesto) == pytest.approx(0.515047, abs=1e-6)


def test_coletar_duracao_invalida() -> None:
    assert coletores.coletar_duracao({}) is None
    assert coletores.coletar_duracao({"started_at": "invalido", "finished_at": "x"}) is None


def test_coletar_volume_prefere_saidas() -> None:
    manifesto = {
        "outputs": [{"row_count": 100}],
        "inputs": [{"row_count": 999}],
    }

    assert coletores.coletar_volume(manifesto) == 100


def test_coletar_volume_usa_entradas_sem_row_count_nas_saidas() -> None:
    manifesto = {
        "outputs": [{"path": "quality.json"}],
        "inputs": [{"row_count": 361}, {"row_count": 120}],
    }

    assert coletores.coletar_volume(manifesto) == 481


def test_coletar_volume_ausente() -> None:
    assert coletores.coletar_volume({"outputs": [], "inputs": []}) is None


def test_coletar_violacoes_do_relatorio() -> None:
    quality = {"violations": [{"rule_id": "R_A", "count": 1}, {"rule_id": "R_B", "count": 2}]}

    assert coletores.coletar_violacoes(quality, None) == 3


def test_coletar_violacoes_sem_relatorio_usa_metrics() -> None:
    metrics = {
        "measurements": [
            {"name": "linhas_verificadas", "value": 601},
            {"name": "violacoes_total", "value": 7},
        ]
    }

    assert coletores.coletar_violacoes(None, metrics) == 7


def test_coletar_violacoes_ausente() -> None:
    assert coletores.coletar_violacoes(None, None) is None


def test_coletar_valor_drift_prefere_detectcao() -> None:
    drift = {"summary": {"drift_share": 0.59}}
    analise = {"summary": {"change_share": 0.36}}

    assert coletores.coletar_valor_drift(drift, analise, "drift_share", "change_share") == 0.59


def test_coletar_valor_drift_usa_analise_como_fallback() -> None:
    analise = {"summary": {"changed_columns": ["a", "b"]}}

    assert coletores.coletar_valor_drift(None, analise, "drifted_columns", "changed_columns") == [
        "a",
        "b",
    ]


def test_coletar_variaveis_drift_conta_lista() -> None:
    drift = {"summary": {"drifted_columns": ["a", "b", "c"]}}

    assert coletores.coletar_variaveis_drift(drift, None) == 3


def test_coletar_variaveis_drift_aceita_contagem() -> None:
    drift = {"summary": {"drifted_columns": 13}}

    assert coletores.coletar_variaveis_drift(drift, None) == 13


def test_coletar_variaveis_drift_usa_analise() -> None:
    analise = {"summary": {"changed_columns": 8}}

    assert coletores.coletar_variaveis_drift(None, analise) == 8


def test_taxa_sucesso_e_contagens() -> None:
    execucoes = [
        {"run_id": "a", "stage": "preparacao", "status": "success"},
        {"run_id": "b", "stage": "validacao", "status": "rejected"},
        {"run_id": "c", "stage": "validacao", "status": "rejected"},
        {"run_id": "d", "stage": "treinamento", "status": "failed"},
    ]

    assert coletores.taxa_sucesso(execucoes) == 0.25
    assert coletores.contar_por_status(execucoes, "failed") == 1
    assert coletores.contar_lotes_rejeitados(execucoes) == 2


def test_taxa_sucesso_sem_execucoes() -> None:
    assert coletores.taxa_sucesso([]) is None


def test_contar_alertas() -> None:
    eventos = [evento_drift("a", "deteccao"), {"event": "completed"}, evento_drift("b", "deteccao")]

    assert coletores.contar_alertas(eventos) == 2


def test_espaco_livre_e_escrita(tmp_path: Path) -> None:
    livre = coletores.espaco_livre(tmp_path)

    assert isinstance(livre, float)
    assert livre > 0
    assert coletores.escrita_disponivel(tmp_path / "saida") is True
    assert not (tmp_path / "saida" / ".escrita_saude").exists()

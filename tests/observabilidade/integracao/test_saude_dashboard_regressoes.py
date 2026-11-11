"""Propagacao da classificacao e da publicacao de saude ate o dashboard."""

from __future__ import annotations

import json

import pytest

from monitoramento.observabilidade import alertas, dashboard
from monitoramento.observabilidade.dashboard.indicadores import medidas_de_saude
from monitoramento.observabilidade.metricas_saude.resultados import gravar_saude
from tests.observabilidade.integracao.test_saude_alertas_regressoes import RAIZ, gerar_fontes


@pytest.mark.parametrize(
    "auc,situacao,severidade", [(0.6, "ok", None), (0.5, "atencao", "warning")]
)
def test_dashboard_preserva_classificacao_dos_limites(tmp_path, auc, situacao, severidade) -> None:
    gerar_fontes(tmp_path, auc)
    alertas.gerar(
        tmp_path / "saude/saude.json",
        tmp_path / "logs/resumo.json",
        RAIZ / "configs/alertas.json",
        tmp_path / "alertas",
    )

    painel = dashboard.carregar_painel(
        tmp_path / "runs",
        tmp_path / "logs/resumo.json",
        tmp_path / "saude/saude.json",
        tmp_path / "alertas/alertas.json",
    )

    medida = next(m for m in medidas_de_saude(painel["saude"]) if m["id"] == "modelo.roc_auc")
    assert medida["situacao"] == situacao
    assert medida["valor"] == auc
    por_auc = [a for a in painel["alertas"]["alertas"] if a.get("metrica") == "modelo.roc_auc"]
    assert [a["severidade"] for a in por_auc] == ([] if severidade is None else [severidade])


def test_dashboard_mantem_saude_disponivel_durante_escrita(tmp_path, monkeypatch) -> None:
    anterior = gerar_fontes(tmp_path, 0.49)
    destino = tmp_path / "saude/saude.json"
    dump_original = json.dump
    observados = []

    def escrever_com_leitura(conteudo, arquivo, **kwargs):
        arquivo.write("{")
        arquivo.flush()
        painel = dashboard.carregar_painel(
            tmp_path / "runs",
            tmp_path / "logs/resumo.json",
            destino,
            tmp_path / "alertas/alertas.json",
        )
        observados.append(painel["saude"])
        arquivo.seek(0)
        arquivo.truncate()
        dump_original(conteudo, arquivo, **kwargs)

    monkeypatch.setattr(json, "dump", escrever_com_leitura)
    gravar_saude(destino, anterior)

    assert observados == [anterior]

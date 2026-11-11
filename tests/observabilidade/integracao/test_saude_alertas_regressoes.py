"""Integracao dos limites e da publicacao de saude com o consumidor de alertas."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from monitoramento.observabilidade import alertas, logs, metricas_saude
from monitoramento.observabilidade.alertas.entradas import ler_json
from monitoramento.observabilidade.metricas_saude.resultados import gravar_saude

RAIZ = Path(__file__).resolve().parents[3]


def gerar_fontes(tmp_path: Path, auc: float) -> dict:
    """Publica uma avaliacao concluida e consolida logs e metricas de saude."""
    run_dir = tmp_path / "runs" / "avaliacao"
    run_dir.mkdir(parents=True)
    manifesto = {
        "contract_version": "1.0",
        "run_id": "avaliacao",
        "stage": "avaliacao",
        "status": "success",
        "started_at": "2026-11-09T00:00:00Z",
        "finished_at": "2026-11-09T00:00:01Z",
        "inputs": [{"row_count": 1}],
    }
    performance = {
        "comparison": {"degradation_detected": False},
        "contrast": {"concept_drift": {"metrics": {"roc_auc": auc}}},
    }
    for nome, conteudo in (("manifest.json", manifesto), ("performance.json", performance)):
        (run_dir / nome).write_text(json.dumps(conteudo), encoding="utf-8")
    evento = {"run_id": "avaliacao", "stage": "avaliacao", "event": "completed"}
    (run_dir / "events.jsonl").write_text(json.dumps(evento) + "\n", encoding="utf-8")
    logs.gerar(tmp_path / "runs", tmp_path / "logs")
    return metricas_saude.gerar(
        tmp_path / "runs", tmp_path / "saude", RAIZ / "configs/metricas_saude.json"
    )


@pytest.mark.parametrize("auc,severidade", [(0.6, None), (0.5, "warning"), (0.49, "critical")])
def test_limites_de_saude_geram_severidade_correta(tmp_path, auc, severidade) -> None:
    gerar_fontes(tmp_path, auc)

    relatorio = alertas.gerar(
        tmp_path / "saude/saude.json",
        tmp_path / "logs/resumo.json",
        RAIZ / "configs/alertas.json",
        tmp_path / "alertas",
    )

    alertas_auc = [a for a in relatorio["alertas"] if a.get("metrica") == "modelo.roc_auc"]
    assert [a["severidade"] for a in alertas_auc] == ([] if severidade is None else [severidade])
    assert not any(a.get("metrica") == "pipeline.volume_linhas" for a in relatorio["alertas"])


def test_alertas_leem_ultima_saude_completa_durante_atualizacao(tmp_path, monkeypatch) -> None:
    anterior = gerar_fontes(tmp_path, 0.49)
    destino = tmp_path / "saude/saude.json"
    novo = {**anterior, "resumo": {**anterior["resumo"], "medidas": 999}}
    dump_original = json.dump
    observados = []

    def escrever_com_leitura(conteudo, arquivo, **kwargs):
        arquivo.write("{")
        arquivo.flush()
        observados.append(ler_json(destino, "Saude operacional"))
        arquivo.seek(0)
        arquivo.truncate()
        dump_original(conteudo, arquivo, **kwargs)

    monkeypatch.setattr(json, "dump", escrever_com_leitura)
    gravar_saude(destino, novo)

    assert observados == [anterior]
    assert ler_json(destino, "Saude operacional") == novo

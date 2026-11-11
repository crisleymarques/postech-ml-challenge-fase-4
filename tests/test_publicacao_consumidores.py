"""Regressao integrada da leitura de resultados completos e parciais."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from monitoramento import dashboard, logs, metricas_saude
from monitoramento.drift.deteccao.servico import processar
from tests.drift.deteccao.apoio import montar_config, preparar_manifestos


@pytest.mark.parametrize("remover", [None, "manifesto", "completed", "drift_alert"])
def test_consumidores_concordam_sobre_publicacao_da_deteccao(tmp_path, remover):
    entradas = preparar_manifestos(tmp_path)
    runs = tmp_path / "runs"
    output = runs / "deteccao"
    assert processar(tmp_path / "config.json", montar_config(), entradas, "deteccao", output) == 0
    if remover == "manifesto":
        (output / "manifest.json").unlink()
    elif remover:
        caminho = output / "events.jsonl"
        eventos = [json.loads(linha) for linha in caminho.read_text().splitlines()]
        caminho.write_text("".join(json.dumps(e) + "\n" for e in eventos if e["event"] != remover))
    esperado = "success" if remover is None else "incomplete"
    resumo = logs.gerar(runs, tmp_path / "logs")
    saude = metricas_saude.gerar(
        runs,
        tmp_path / "saude",
        Path(__file__).resolve().parents[1] / "configs/metricas_saude.json",
    )
    painel = dashboard.carregar_execucoes(runs)
    assert resumo["execucoes"]["por_status"] == {esperado: 1}
    assert len(resumo["alertas"]) == (1 if remover is None else 0)
    assert saude["execucoes"][0]["status"] == esperado
    assert painel[0]["status"] == esperado
    medidas = {m["id"]: m for m in saude["execucoes"][0]["metricas"]}
    assert medidas["drift.dataset_drift"]["valor"] is (True if remover is None else None)
    alertas = next(m for m in saude["agregados"] if m["id"] == "drift.alertas_total")
    assert alertas["valor"] == (1 if remover is None else 0)

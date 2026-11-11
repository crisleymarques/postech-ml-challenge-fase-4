"""Fluxo persistido de Concept Drift ate o carregamento do dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from monitoramento import alertas, dashboard, logs, metricas_saude
from monitoramento.avaliacao.concept_drift import servico
from monitoramento.infraestrutura.arquivos import sha256_arquivo
from tests.avaliacao.concept_drift.apoio import montar_config, preparar_manifestos


@pytest.mark.parametrize("avaliavel", [True, False])
def test_dashboard_preserva_alerta_falha_e_indisponibilidade(
    tmp_path: Path, avaliavel: bool
) -> None:
    manifestos = preparar_manifestos(tmp_path)
    if not avaliavel:
        for item in manifestos[1]["outputs"]:
            if item["artifact_type"] in ("labels", "labels_reference"):
                base = pd.read_csv(item["path"])
                base["loan_status"] = 0
                base.to_csv(item["path"], index=False)
                item["sha256"] = sha256_arquivo(Path(item["path"]))
    runs = tmp_path / "runs"
    config = montar_config()
    assert servico.processar(tmp_path / "config.json", config, manifestos, "ok", runs / "ok") == 0
    assert (
        servico.processar(tmp_path / "config.json", config, manifestos[:2], "falha", runs / "falha")
        == 1
    )
    resumo = logs.gerar(runs, tmp_path / "logs")
    assert resumo["execucoes"]["por_status"] == {"failed": 1, "success": 1}
    assert len(resumo["alertas"]) == int(avaliavel)
    assert any(f["run_id"] == "falha" for f in resumo["falhas"])
    metricas_saude.gerar(runs, tmp_path / "saude", Path("configs/metricas_saude.json"))
    integrado = alertas.gerar(
        tmp_path / "saude/saude.json",
        tmp_path / "logs/resumo.json",
        Path("configs/alertas.json"),
        tmp_path / "alertas",
    )
    assert any(a["origem"] == "falha" for a in integrado["alertas"])
    if not avaliavel:
        assert not any(a["origem"] == "ok" for a in integrado["alertas"])
    painel = dashboard.carregar_painel(
        runs,
        tmp_path / "logs/resumo.json",
        tmp_path / "saude/saude.json",
        tmp_path / "alertas/alertas.json",
    )
    assert painel["resumo_execucoes"]["por_status"] == {"failed": 1, "success": 1}
    medidas = next(e for e in painel["saude"]["execucoes"] if e["run_id"] == "ok")["metricas"]
    medida = next(m for m in medidas if m["id"] == "drift.degradacao_conceito")
    assert medida["situacao"] == ("atencao" if avaliavel else "indisponivel")

"""Integracao de Concept Drift com metricas de saude, incluindo indisponibilidade."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from monitoramento.avaliacao.concept_drift import servico
from monitoramento.infraestrutura.arquivos import sha256_arquivo
from monitoramento.observabilidade import metricas_saude
from tests.avaliacao.concept_drift.apoio import montar_config, preparar_manifestos


@pytest.mark.parametrize("estado", ["degradacao", "sem_degradacao", "indisponivel"])
def test_saude_preserva_tres_estados_da_avaliacao(tmp_path: Path, estado: str) -> None:
    manifestos = preparar_manifestos(tmp_path, inverter=estado == "degradacao")
    if estado == "indisponivel":
        for item in manifestos[1]["outputs"]:
            if item["artifact_type"] in ("labels", "labels_reference"):
                base = pd.read_csv(item["path"])
                base["loan_status"] = 0
                base.to_csv(item["path"], index=False)
                item["sha256"] = sha256_arquivo(Path(item["path"]))
    runs = tmp_path / "runs"
    assert (
        servico.processar(tmp_path / "config.json", montar_config(), manifestos, "r", runs / "r")
        == 0
    )
    saude = metricas_saude.gerar(runs, tmp_path / "saude", Path("configs/metricas_saude.json"))
    medidas = {m["id"]: m for m in saude["execucoes"][0]["metricas"]}
    medida = medidas["drift.degradacao_conceito"]
    valor, situacao = {
        "degradacao": (True, "atencao"),
        "sem_degradacao": (False, "ok"),
        "indisponivel": (None, "indisponivel"),
    }[estado]
    assert medida["valor"] is valor
    assert medida["situacao"] == situacao
    if estado == "indisponivel":
        assert medidas["modelo.roc_auc"]["situacao"] == "indisponivel"
    assert (
        json.loads((tmp_path / "saude/saude.json").read_text())["execucoes"][0]["status"]
        == "success"
    )

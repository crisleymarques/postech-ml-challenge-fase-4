"""Testes de publicacao de Concept Drift."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from monitoramento.infraestrutura import arquivos
from monitoramento.simulacao.concept_drift import artefatos, servico
from monitoramento.simulacao.concept_drift import config as configuracao

from .apoio import SEMENTE, escrever_config, gerar_base, montar_config


def test_carregar_base_rejeita_arquivo_ausente(tmp_path: Path) -> None:
    with pytest.raises(configuracao.ErroSimulacao, match="nao encontrada"):
        artefatos.carregar_base(tmp_path / "inexistente.csv")


def test_montar_medidas_descreve_cenario() -> None:
    descricao = {
        "segment_rows": 10,
        "flipped_rows": 4,
        "positive_rate_before": 0.2,
        "positive_rate_after": 0.3,
    }

    medidas = artefatos.montar_medidas(descricao, 100)
    nomes = {item["name"] for item in medidas}

    assert nomes == {
        "linhas_geradas",
        "registros_no_segmento",
        "rotulos_alterados",
        "taxa_positivos_antes",
        "taxa_positivos_depois",
    }


def test_gerar_publica_artefatos_e_manifesto(tmp_path: Path) -> None:
    config = montar_config()
    origem = tmp_path / "simulation_base.csv"
    gerar_base().to_csv(origem, index=False, encoding="utf-8")
    config["base"]["path"] = origem.as_posix()
    caminho_config = escrever_config(tmp_path, config)
    output_dir = tmp_path / "runs" / "run-concept"
    analysis_dir = tmp_path / "analysis"

    codigo = servico.gerar(
        config_path=caminho_config,
        source_path=None,
        analysis_dir=analysis_dir,
        batch_id="batch-1",
        run_id="run-concept",
        output_dir=output_dir,
        seed=SEMENTE,
    )

    assert codigo == configuracao.CODIGO_SUCESSO
    manifesto = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifesto["stage"] == "simulacao"
    assert manifesto["status"] == "success"
    assert manifesto["scenario_id"] == "concept-drift"
    assert manifesto["batch_id"] == "batch-1"
    assert manifesto["drift_type"] == "concept"
    assert manifesto["features_unchanged"] is True
    assert manifesto["concept_change"]["flipped_rows"] > 0

    saidas = {item["artifact_type"]: item for item in manifesto["outputs"]}
    assert set(saidas) == {"features", "labels", "labels_reference"}
    for item in saidas.values():
        assert Path(item["path"]).is_file()
        assert arquivos.sha256_arquivo(Path(item["path"])) == item["sha256"]

    features = pd.read_csv(saidas["features"]["path"])
    assert "loan_status" not in features.columns
    rotulos = pd.read_csv(saidas["labels"]["path"])
    assert list(rotulos.columns) == ["record_id", "loan_status"]
    referencia = pd.read_csv(saidas["labels_reference"]["path"])
    assert rotulos["loan_status"].sum() > referencia["loan_status"].sum()

    eventos = [
        json.loads(linha)
        for linha in (output_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [evento["event"] for evento in eventos] == ["started", "completed"]

    metricas = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metricas["run_id"] == "run-concept"
    assert any(item["name"] == "rotulos_alterados" for item in metricas["measurements"])

"""Testes migrados da etapa de predicao."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from monitoramento.infraestrutura import arquivos
from monitoramento.predicao import config as cfg
from monitoramento.predicao import inferencia, servico

from .apoio import (
    gerar_lote,
    montar_config,
    preparar_manifestos,
)


def test_processar_publica_predicoes_e_manifesto(tmp_path: Path) -> None:
    config = montar_config()
    caminho_config = tmp_path / "predicao.json"
    caminho_config.write_text(json.dumps(config), encoding="utf-8")
    manifestos = preparar_manifestos(tmp_path)
    output_dir = tmp_path / "runs" / "run-pred"

    codigo = servico.processar(
        caminho_config,
        config,
        manifestos,
        "run-pred",
        output_dir,
        analysis_dir=tmp_path / "analysis",
    )

    assert codigo == cfg.CODIGO_SUCESSO
    manifesto = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifesto["stage"] == "predicao"
    assert manifesto["status"] == "success"
    assert manifesto["model_id"] == "modelo-teste"
    assert manifesto["reference_id"] == "ref-teste"
    assert manifesto["scenario_id"] == "data-drift"
    assert manifesto["batch_id"] == "batch-1"
    assert manifesto["score"]["available"] is True
    assert manifesto["score"]["meaning"] == "probabilidade estimada da classe Charged Off"

    saidas = {item["artifact_type"]: item for item in manifesto["outputs"]}
    assert set(saidas) == {"predictions"}
    caminho_predicoes = Path(saidas["predictions"]["path"])
    assert caminho_predicoes.is_file()
    assert arquivos.sha256_arquivo(caminho_predicoes) == saidas["predictions"]["sha256"]

    predicoes = pd.read_csv(caminho_predicoes)
    assert list(predicoes.columns) == list(inferencia.COLUNAS_SAIDA)
    assert len(predicoes) == len(gerar_lote())
    assert set(predicoes["prediction"]).issubset({0, 1})
    assert predicoes["record_id"].is_unique

    assert caminho_predicoes.as_posix().endswith("data-drift/batch-1/modelo-teste/predictions.csv")
    metricas = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    nomes = {item["name"] for item in metricas["measurements"]}
    assert nomes == {"predicoes_geradas", "taxa_positivos", "score_medio"}

    eventos = [
        json.loads(linha)
        for linha in (output_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [evento["event"] for evento in eventos] == ["started", "completed"]


def test_processar_rejeita_dataset_divergente(tmp_path: Path) -> None:
    config = montar_config()
    config["dataset_id"] = "outro-dataset"
    caminho_config = tmp_path / "predicao.json"
    caminho_config.write_text(json.dumps(config), encoding="utf-8")
    manifestos = preparar_manifestos(tmp_path)

    codigo = servico.processar(
        caminho_config,
        config,
        manifestos,
        "run-pred",
        tmp_path / "runs" / "run-pred",
        analysis_dir=tmp_path / "analysis",
    )

    assert codigo == cfg.CODIGO_ERRO
    eventos = [
        json.loads(linha)
        for linha in (tmp_path / "runs" / "run-pred" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert eventos[-1]["event"] == "failed"
    assert eventos[-1]["error_code"] == "PREDICAO_INVALIDA"

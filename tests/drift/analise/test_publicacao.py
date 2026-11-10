"""Testes de publicacao da analise estatistica."""
from __future__ import annotations

import json
from pathlib import Path

from monitoramento.drift.analise import config as configuracao
from monitoramento.drift.analise import servico
from monitoramento.infraestrutura import arquivos

from .apoio import (
    FEATURES,
    QUANTIDADE,
    montar_config,
    preparar_manifestos,
)


def test_processar_publica_analise_e_manifesto(tmp_path: Path) -> None:
    config = montar_config()
    caminho_config = tmp_path / "analise.json"
    caminho_config.write_text(json.dumps(config), encoding="utf-8")
    manifestos = preparar_manifestos(tmp_path)
    output_dir = tmp_path / "runs" / "run-analise"

    codigo = servico.processar(caminho_config, config, manifestos, "run-analise", output_dir)

    assert codigo == configuracao.CODIGO_SUCESSO
    manifesto = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifesto["stage"] == "analise"
    assert manifesto["status"] == "success"
    assert manifesto["model_id"] == "modelo-teste"
    assert manifesto["reference_id"] == "ref-teste"
    assert manifesto["scenario_id"] == "data-drift"
    assert manifesto["batch_id"] == "batch-1"

    saidas = {item["artifact_type"]: item for item in manifesto["outputs"]}
    assert set(saidas) == {"analise_json"}
    for item in saidas.values():
        assert Path(item["path"]).is_file()
        assert arquivos.sha256_arquivo(Path(item["path"])) == item["sha256"]

    documento = json.loads((output_dir / "analise.json").read_text(encoding="utf-8"))
    assert documento["tool"]["name"] == "scipy"
    assert documento["method"]["psi_threshold"] == 0.25
    assert documento["method"]["ks_p_value_threshold"] == 0.05
    assert documento["samples"]["reference"]["row_count"] == QUANTIDADE
    assert documento["samples"]["current"]["row_count"] == QUANTIDADE
    assert {item["feature"] for item in documento["features"]} == set(FEATURES)
    assert documento["summary"]["changed_columns"] == len(FEATURES)
    assert set(documento["changed_variables"]["confirmed"]) == {"annual_inc", "dti", "revol_util"}
    assert documento["interpretation"]["conclusions"]

    metricas = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    nomes = {item["name"] for item in metricas["measurements"]}
    assert "psi_maximo" in nomes
    assert "variaveis_alteradas_confirmadas" in nomes


def test_processar_emite_alerta_de_drift(tmp_path: Path) -> None:
    config = montar_config()
    caminho_config = tmp_path / "analise.json"
    caminho_config.write_text(json.dumps(config), encoding="utf-8")
    manifestos = preparar_manifestos(tmp_path)
    output_dir = tmp_path / "runs" / "run-alerta"

    servico.processar(caminho_config, config, manifestos, "run-alerta", output_dir)

    eventos = [
        json.loads(linha)
        for linha in (output_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [evento["event"] for evento in eventos] == ["started", "drift_alert", "completed"]
    alerta = next(evento for evento in eventos if evento["event"] == "drift_alert")
    assert alerta["level"] == "WARNING"


def test_processar_rejeita_dataset_divergente(tmp_path: Path) -> None:
    config = montar_config()
    config["dataset_id"] = "outro-dataset"
    caminho_config = tmp_path / "analise.json"
    caminho_config.write_text(json.dumps(config), encoding="utf-8")
    manifestos = preparar_manifestos(tmp_path)
    output_dir = tmp_path / "runs" / "run-analise"

    codigo = servico.processar(caminho_config, config, manifestos, "run-analise", output_dir)

    assert codigo == configuracao.CODIGO_ERRO
    eventos = [
        json.loads(linha)
        for linha in (output_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert eventos[-1]["event"] == "failed"
    assert eventos[-1]["error_code"] == "ANALISE_INVALIDA"

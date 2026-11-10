"""Testes de publicacao de Concept Drift."""
from __future__ import annotations

import json
from pathlib import Path

from monitoramento.avaliacao.concept_drift import config as configuracao
from monitoramento.avaliacao.concept_drift import (
    interpretacao,
    resultados,
    servico,
)

from .apoio import (
    montar_config,
    preparar_manifestos,
)


def test_montar_medidas_registra_variacao() -> None:
    referencia = {"roc_auc": 0.70}
    alterado = {"roc_auc": 0.60}
    comparacao = interpretacao.comparar(referencia, alterado, "roc_auc", 0.02)

    medidas = resultados.montar_medidas(comparacao, referencia, alterado)
    nomes = {item["name"] for item in medidas}

    assert nomes == {
        "roc_auc_referencia",
        "roc_auc_concept_drift",
        "variacao_roc_auc",
        "degradacao_detectada",
    }


def test_processar_publica_performance_e_manifesto(tmp_path: Path) -> None:
    config = montar_config()
    caminho_config = tmp_path / "avaliacao.json"
    caminho_config.write_text(json.dumps(config), encoding="utf-8")
    manifestos = preparar_manifestos(tmp_path, inverter=True)
    output_dir = tmp_path / "runs" / "run-aval"

    codigo = servico.processar(caminho_config, config, manifestos, "run-aval", output_dir)

    assert codigo == configuracao.CODIGO_SUCESSO
    manifesto = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifesto["stage"] == "avaliacao"
    assert manifesto["status"] == "success"
    assert manifesto["model_id"] == "modelo-teste"
    assert manifesto["reference_id"] == "ref-teste"
    assert manifesto["scenario_id"] == "concept-drift"
    assert manifesto["batch_id"] == "batch-1"
    assert manifesto["comparison"]["degradation_detected"] is True

    documento = json.loads((output_dir / "performance.json").read_text(encoding="utf-8"))
    assert documento["comparison"]["metric"] == "roc_auc"
    assert (
        documento["contrast"]["reference"]["metrics"]["roc_auc"]
        > documento["contrast"]["concept_drift"]["metrics"]["roc_auc"]
    )
    texto = " ".join(documento["interpretation"]["conclusions"])
    assert "Data Drift" in texto
    assert "Concept Drift" in texto

    metricas = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    nomes = {item["name"] for item in metricas["measurements"]}
    assert "roc_auc_referencia" in nomes
    assert "degradacao_detectada" in nomes

    eventos = [
        json.loads(linha)
        for linha in (output_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [evento["event"] for evento in eventos] == ["started", "drift_alert", "completed"]
    alerta = next(evento for evento in eventos if evento["event"] == "drift_alert")
    assert alerta["level"] == "WARNING"


def test_processar_sem_degradacao_nao_emite_alerta(tmp_path: Path) -> None:
    config = montar_config()
    caminho_config = tmp_path / "avaliacao.json"
    caminho_config.write_text(json.dumps(config), encoding="utf-8")
    manifestos = preparar_manifestos(tmp_path, inverter=False)
    output_dir = tmp_path / "runs" / "run-sem-alerta"

    codigo = servico.processar(caminho_config, config, manifestos, "run-sem-alerta", output_dir)

    assert codigo == configuracao.CODIGO_SUCESSO
    manifesto = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifesto["comparison"]["degradation_detected"] is False

    eventos = [
        json.loads(linha)
        for linha in (output_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [evento["event"] for evento in eventos] == ["started", "completed"]


def test_processar_rejeita_dataset_divergente(tmp_path: Path) -> None:
    config = montar_config()
    config["dataset_id"] = "outro-dataset"
    caminho_config = tmp_path / "avaliacao.json"
    caminho_config.write_text(json.dumps(config), encoding="utf-8")
    manifestos = preparar_manifestos(tmp_path)
    output_dir = tmp_path / "runs" / "run-aval"

    codigo = servico.processar(caminho_config, config, manifestos, "run-aval", output_dir)

    assert codigo == configuracao.CODIGO_ERRO
    eventos = [
        json.loads(linha)
        for linha in (output_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert eventos[-1]["event"] == "failed"
    assert eventos[-1]["error_code"] == "AVALIACAO_INVALIDA"

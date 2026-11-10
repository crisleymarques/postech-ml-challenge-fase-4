"""Orquestracao da avaliacao de Concept Drift."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any

from monitoramento.infraestrutura import observabilidade
from monitoramento.infraestrutura.arquivos import escrever_json
from monitoramento.infraestrutura.observabilidade import agora_iso, escrever_eventos
from monitoramento.modelos.carregamento import carregar_pipeline

from .config import CODIGO_ERRO, CODIGO_SUCESSO, IDENTIFICADOR, STAGE, ErroAvaliacao
from .entradas import (
    artefato,
    juntar_rotulos,
    ler_lote,
    separar_manifestos,
    verificar_artefato,
    verificar_compatibilidade,
)
from .inferencia import prever
from .interpretacao import comparar
from .metricas import calcular_metricas
from .resultados import descritor_artefato, montar_documento, montar_manifesto, montar_medidas

registrar_evento = partial(observabilidade.registrar_evento, stage=STAGE)


def processar(
    config_path: Path,
    config: dict[str, Any],
    manifestos: list[dict[str, Any]],
    run_id: str,
    output_dir: Path,
) -> int:
    """Executa a avaliacao e publica manifesto, eventos, metricas e desempenho."""
    output_dir.mkdir(parents=True, exist_ok=True)
    iniciado = agora_iso()
    eventos: list[dict[str, Any]] = []
    registrar_evento(
        eventos,
        run_id=run_id,
        dataset_id=config.get("dataset_id"),
        model_id=None,
        reference_id=None,
        scenario_id=None,
        batch_id=None,
        evento="started",
        nivel="INFO",
    )
    try:
        manifesto_modelo, manifesto_lote = separar_manifestos(manifestos, config)
        verificar_compatibilidade(config, manifesto_modelo, manifesto_lote)

        modelo_id = manifesto_modelo.get("model_id")
        referencia_id = manifesto_modelo.get("reference_id")
        scenario_id = manifesto_lote.get("scenario_id")
        batch_id = manifesto_lote.get("batch_id")

        item_pipeline = artefato(manifesto_modelo, config["model"]["pipeline_artifact"])
        item_metadata = artefato(manifesto_modelo, config["model"]["metadata_artifact"])
        item_features = artefato(manifesto_lote, config["batch"]["features_artifact"])
        item_labels = artefato(manifesto_lote, config["batch"]["labels_artifact"])
        item_ref_labels = artefato(manifesto_lote, config["batch"]["reference_labels_artifact"])

        caminho_pipeline = verificar_artefato(item_pipeline)
        caminho_metadata = verificar_artefato(item_metadata)
        caminho_features = verificar_artefato(item_features)
        caminho_labels = verificar_artefato(item_labels)
        caminho_ref_labels = verificar_artefato(item_ref_labels)

        with caminho_metadata.open(encoding="utf-8") as arquivo:
            metadata = json.load(arquivo)

        alvo = metadata.get("target", {}).get("column")
        if not alvo:
            raise ErroAvaliacao("Metadados do modelo sem coluna alvo.")

        pipeline = carregar_pipeline(caminho_pipeline)
        lote = ler_lote(caminho_features)
        predito, score = prever(pipeline, metadata, lote)

        rotulos_ref = juntar_rotulos(lote[IDENTIFICADOR], caminho_ref_labels, alvo)
        rotulos_alt = juntar_rotulos(lote[IDENTIFICADOR], caminho_labels, alvo)

        metricas = tuple(config["metrics"])
        valores_ref = calcular_metricas(rotulos_ref, predito, score, metricas)
        valores_alt = calcular_metricas(rotulos_alt, predito, score, metricas)

        degradacao = config.get("degradation", {})
        metrica = degradacao.get("metric", "roc_auc")
        min_delta = float(degradacao.get("min_delta", 0.02))
        comparacao = comparar(valores_ref, valores_alt, metrica, min_delta)

        conceito = manifesto_lote.get("concept_change")
        documento = montar_documento(
            run_id=run_id,
            config=config,
            metadata=metadata,
            modelo_id=modelo_id,
            referencia_id=referencia_id,
            scenario_id=scenario_id,
            batch_id=batch_id,
            caminho_ref=caminho_ref_labels,
            caminho_alt=caminho_labels,
            linhas=len(lote),
            referencia=valores_ref,
            alterado=valores_alt,
            comparacao=comparacao,
            conceito=conceito,
        )
        caminho_json = output_dir / "performance.json"
        escrever_json(caminho_json, documento)

        entradas = [
            {**item_pipeline, "run_id_origem": manifesto_modelo.get("run_id")},
            {**item_metadata, "run_id_origem": manifesto_modelo.get("run_id")},
            {**item_features, "run_id_origem": manifesto_lote.get("run_id")},
            {**item_labels, "run_id_origem": manifesto_lote.get("run_id")},
            {**item_ref_labels, "run_id_origem": manifesto_lote.get("run_id")},
        ]
        saidas = [descritor_artefato("performance_json", caminho_json, run_id)]

        escrever_json(output_dir / "config.json", config)
        escrever_json(
            output_dir / "metrics.json",
            {
                "run_id": run_id,
                "measurements": montar_medidas(comparacao, valores_ref, valores_alt),
            },
        )
        escrever_json(
            output_dir / "manifest.json",
            montar_manifesto(
                run_id=run_id,
                iniciado=iniciado,
                finalizado=agora_iso(),
                config_path=config_path,
                config=config,
                manifesto_lote=manifesto_lote,
                modelo_id=modelo_id,
                referencia_id=referencia_id,
                scenario_id=scenario_id,
                batch_id=batch_id,
                entradas=entradas,
                saidas=saidas,
                comparacao=comparacao,
                linhas=len(lote),
            ),
        )
        if comparacao["degradation_detected"]:
            registrar_evento(
                eventos,
                run_id=run_id,
                dataset_id=config.get("dataset_id"),
                model_id=modelo_id,
                reference_id=referencia_id,
                scenario_id=scenario_id,
                batch_id=batch_id,
                evento="drift_alert",
                nivel="WARNING",
            )
        registrar_evento(
            eventos,
            run_id=run_id,
            dataset_id=config.get("dataset_id"),
            model_id=modelo_id,
            reference_id=referencia_id,
            scenario_id=scenario_id,
            batch_id=batch_id,
            evento="completed",
            nivel="INFO",
        )
        escrever_eventos(output_dir / "events.jsonl", eventos)
    except (ErroAvaliacao, ValueError, KeyError) as erro:
        registrar_evento(
            eventos,
            run_id=run_id,
            dataset_id=config.get("dataset_id"),
            model_id=None,
            reference_id=None,
            scenario_id=None,
            batch_id=None,
            evento="failed",
            nivel="ERROR",
            error_code="AVALIACAO_INVALIDA",
        )
        escrever_eventos(output_dir / "events.jsonl", eventos)
        print(f"Falha na avaliacao: {erro}")
        return CODIGO_ERRO

    metrica = comparacao["metric"]
    print(
        f"Avaliacao de Concept Drift: {metrica} "
        f"{valores_ref.get(metrica)} (referencia) -> {valores_alt.get(metrica)} "
        f"({comparacao['metric_delta']:+})"
    )
    print(f"  cenario {scenario_id} / lote {batch_id} (modelo {modelo_id})")
    print(f"  degradacao detectada: {comparacao['degradation_detected']}")
    print(f"  documento -> {caminho_json.as_posix()}")
    return CODIGO_SUCESSO

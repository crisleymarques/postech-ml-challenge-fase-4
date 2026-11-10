"""Orquestracao da predicao, publicacao e observabilidade."""
from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any

from monitoramento.infraestrutura import observabilidade
from monitoramento.infraestrutura.arquivos import escrever_json
from monitoramento.infraestrutura.observabilidade import agora_iso, escrever_eventos
from monitoramento.modelos.carregamento import carregar_pipeline

from .config import CODIGO_ERRO, CODIGO_SUCESSO, STAGE, ErroPredicao
from .entradas import (
    artefato,
    ler_lote,
    separar_manifestos,
    verificar_artefato,
    verificar_compatibilidade,
)
from .inferencia import prever
from .resultados import descritor_artefato, escrever_predicoes, montar_manifesto, montar_medidas

registrar_evento = partial(observabilidade.registrar_evento, stage=STAGE)


def processar(
    config_path: Path,
    config: dict[str, Any],
    manifestos: list[dict[str, Any]],
    run_id: str,
    output_dir: Path,
    *,
    analysis_dir: Path | None = None,
) -> int:
    """Executa a predicao e publica manifesto, eventos, metricas e predicoes."""
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

        caminho_pipeline = verificar_artefato(item_pipeline)
        caminho_metadata = verificar_artefato(item_metadata)
        caminho_features = verificar_artefato(item_features)

        with caminho_metadata.open(encoding="utf-8") as arquivo:
            metadata = json.load(arquivo)

        pipeline = carregar_pipeline(caminho_pipeline)
        lote = ler_lote(caminho_features)
        predicoes, score_disponivel = prever(pipeline, metadata, lote)

        if scenario_id is None or batch_id is None:
            raise ErroPredicao("O manifesto do lote nao informa scenario_id e batch_id.")

        destino = (
            (analysis_dir or Path(config["outputs"]["analysis_dir"]))
            / str(scenario_id)
            / str(batch_id)
            / str(modelo_id)
        )
        caminho_predicoes = destino / "predictions.csv"
        escrever_predicoes(caminho_predicoes, predicoes)

        entradas = [
            {**item_pipeline, "run_id_origem": manifesto_modelo.get("run_id")},
            {**item_metadata, "run_id_origem": manifesto_modelo.get("run_id")},
            {**item_features, "run_id_origem": manifesto_lote.get("run_id")},
        ]
        saidas = [descritor_artefato("predictions", caminho_predicoes, run_id, predicoes)]

        score = {
            "available": bool(score_disponivel),
            "column": "score",
            "meaning": metadata.get("score_meaning"),
            "positive_value": metadata.get("target", {}).get("positive_value"),
            "prediction_threshold": metadata.get("prediction_threshold"),
        }

        escrever_json(output_dir / "config.json", config)
        escrever_json(
            output_dir / "metrics.json",
            {
                "run_id": run_id,
                "measurements": montar_medidas(predicoes, score_disponivel),
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
                score=score,
                linhas=len(predicoes),
            ),
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
    except (ErroPredicao, ValueError, KeyError) as erro:
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
            error_code="PREDICAO_INVALIDA",
        )
        escrever_eventos(output_dir / "events.jsonl", eventos)
        print(f"Falha na predicao: {erro}")
        return CODIGO_ERRO

    print(f"Predicoes geradas: {len(predicoes)} linhas (modelo {modelo_id})")
    print(f"  cenario {scenario_id} / lote {batch_id}")
    print(f"  score disponivel: {score_disponivel} ({metadata.get('score_meaning')})")
    print(f"  predicoes -> {caminho_predicoes.as_posix()}")
    print(f"  manifesto -> {output_dir.as_posix()}/manifest.json")
    return CODIGO_SUCESSO

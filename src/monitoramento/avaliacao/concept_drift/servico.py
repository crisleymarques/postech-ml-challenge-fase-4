"""Orquestracao da avaliacao de Concept Drift."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any

from monitoramento.infraestrutura import observabilidade
from monitoramento.infraestrutura.arquivos import escrever_json_atomico as escrever_json
from monitoramento.infraestrutura.observabilidade import agora_iso, escrever_eventos
from monitoramento.modelos.carregamento import carregar_pipeline

from .config import (
    CODIGO_ERRO,
    CODIGO_SUCESSO,
    IDENTIFICADOR,
    STAGE,
    ErroAvaliacao,
    ErroConfiguracao,
    validar_config,
)
from .entradas import (
    artefato,
    carregar_manifesto,
    juntar_rotulos,
    ler_lote,
    separar_manifestos,
    verificar_aprovacao,
    verificar_artefato,
    verificar_compatibilidade,
    verificar_metadata,
)
from .inferencia import prever
from .interpretacao import comparar
from .metricas import calcular_metricas
from .resultados import descritor_artefato, montar_documento, montar_manifesto, montar_medidas

registrar_evento = partial(observabilidade.registrar_evento, stage=STAGE)


def processar(
    config_path: Path,
    config: dict[str, Any],
    manifestos: list[dict[str, Any] | Path],
    run_id: str,
    output_dir: Path,
) -> int:
    """Executa a avaliacao e publica manifesto, eventos, metricas e desempenho."""
    try:
        validar_config(config)
        output_dir.mkdir(parents=True, exist_ok=False)
    except (ErroConfiguracao, OSError):
        print("Falha ao iniciar avaliacao: configuracao ou destino invalido.")
        return CODIGO_ERRO
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
    modelo_id = referencia_id = scenario_id = batch_id = None
    entradas: list[dict[str, Any]] = []
    validacao_run_id = None
    try:
        escrever_eventos(output_dir / "events.jsonl", eventos)
        escrever_json(output_dir / "config.json", config)
        manifestos = [carregar_manifesto(m) if isinstance(m, Path) else m for m in manifestos]
        manifesto_modelo, manifesto_lote, manifesto_validacao = separar_manifestos(
            manifestos, config
        )
        validacao_run_id = manifesto_validacao["run_id"]
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

        verificar_aprovacao(manifesto_validacao, manifesto_lote, item_features)
        caminho_pipeline = verificar_artefato(item_pipeline)
        caminho_metadata = verificar_artefato(item_metadata)
        caminho_features = verificar_artefato(item_features)
        caminho_labels = verificar_artefato(item_labels)
        caminho_ref_labels = verificar_artefato(item_ref_labels)

        with caminho_metadata.open(encoding="utf-8") as arquivo:
            metadata = json.load(arquivo)

        verificar_metadata(metadata, manifesto_modelo, manifesto_lote)
        alvo_config = metadata.get("target", {})
        if alvo_config.get("positive_value") != 1 or alvo_config.get("negative_value") != 0:
            raise ErroAvaliacao("O alvo deve declarar positive_value=1 e negative_value=0.")
        alvo = metadata.get("target", {}).get("column")
        if not alvo:
            raise ErroAvaliacao("Metadados do modelo sem coluna alvo.")

        pipeline = carregar_pipeline(caminho_pipeline)
        lote = ler_lote(caminho_features)
        rotulos_ref = juntar_rotulos(lote[IDENTIFICADOR], caminho_ref_labels, alvo)
        rotulos_alt = juntar_rotulos(lote[IDENTIFICADOR], caminho_labels, alvo)

        predito, score = prever(pipeline, metadata, lote)

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
        escrever_json(
            output_dir / "manifest.json",
            montar_manifesto(
                run_id=run_id,
                iniciado=iniciado,
                finalizado=agora_iso(),
                config_path=output_dir / "config.json",
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
                validacao_run_id=validacao_run_id,
            ),
        )
    except Exception as erro:  # Fronteira da etapa: falhas de entrada, inferencia e persistencia.
        eventos[:] = [e for e in eventos if e["event"] not in ("completed", "drift_alert")]
        registrar_evento(
            eventos,
            run_id=run_id,
            dataset_id=config.get("dataset_id"),
            model_id=modelo_id,
            reference_id=referencia_id,
            scenario_id=scenario_id,
            batch_id=batch_id,
            evento="failed",
            nivel="ERROR",
            error_code="AVALIACAO_INVALIDA",
        )
        falha = {
            "contract_version": "1.0",
            "run_id": run_id,
            "stage": STAGE,
            "status": "failed",
            "started_at": iniciado,
            "finished_at": agora_iso(),
            "config_path": (output_dir / "config.json").as_posix(),
            "dataset_id": config.get("dataset_id"),
            "model_id": modelo_id,
            "reference_id": referencia_id,
            "scenario_id": scenario_id,
            "batch_id": batch_id,
            "inputs": entradas,
            "outputs": [],
            "error_code": "AVALIACAO_INVALIDA",
            "validation_run_id": validacao_run_id,
        }
        for escrever, caminho, conteudo in (
            (escrever_json, output_dir / "config.json", config),
            (escrever_eventos, output_dir / "events.jsonl", eventos),
            (escrever_json, output_dir / "metrics.json", {"run_id": run_id, "measurements": []}),
            (escrever_json, output_dir / "manifest.json", falha),
        ):
            try:
                escrever(caminho, conteudo)
            except OSError:
                print(f"Nao foi possivel persistir {caminho.name} da falha.")
        print(f"Falha na avaliacao ({type(erro).__name__}).")
        return CODIGO_ERRO

    metrica = comparacao["metric"]
    delta = comparacao["metric_delta"]
    delta_texto = "indisponivel" if delta is None else f"{delta:+}"
    print(
        f"Avaliacao de Concept Drift: {metrica} "
        f"{valores_ref.get(metrica)} (referencia) -> {valores_alt.get(metrica)} "
        f"({delta_texto})"
    )
    print(f"  cenario {scenario_id} / lote {batch_id} (modelo {modelo_id})")
    print(f"  degradacao detectada: {comparacao['degradation_detected']}")
    print(f"  documento -> {caminho_json.as_posix()}")
    return CODIGO_SUCESSO

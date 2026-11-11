"""Orquestracao da deteccao, publicacao e observabilidade."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any

from monitoramento.drift.dados import ler_features
from monitoramento.infraestrutura import observabilidade
from monitoramento.infraestrutura.arquivos import escrever_json, escrever_json_atomico
from monitoramento.infraestrutura.observabilidade import agora_iso, escrever_eventos

from .adaptador_evidently import comparar_features
from .config import CODIGO_ERRO, CODIGO_SUCESSO, STAGE, ErroDeteccao
from .entradas import (
    artefato,
    carregar_manifesto,
    separar_manifestos,
    validar_features_esperadas,
    verificar_artefato,
    verificar_compatibilidade,
)
from .regras import resumir
from .relatorio import salvar_html
from .resultados import descritor_artefato, montar_documento, montar_manifesto, montar_medidas

registrar_evento = partial(observabilidade.registrar_evento, stage=STAGE)


def processar(
    config_path: Path,
    config: dict[str, Any],
    manifestos: list[dict[str, Any] | Path],
    run_id: str,
    output_dir: Path,
) -> int:
    """Executa a deteccao e publica manifesto, eventos, metricas e relatorios."""
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except OSError as erro:
        print(f"Falha ao reservar diretorio da deteccao: {erro}")
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
    manifesto_lote: dict[str, Any] = {}
    entradas: list[dict[str, Any]] = []
    resumo: dict[str, Any] = {}
    try:
        escrever_eventos(output_dir / "events.jsonl", eventos)
        escrever_json(output_dir / "config.json", config)
        manifestos = [carregar_manifesto(m) if isinstance(m, Path) else m for m in manifestos]
        manifesto_ref, manifesto_lote = separar_manifestos(manifestos, config)
        verificar_compatibilidade(config, manifesto_ref, manifesto_lote)

        modelo_id = manifesto_ref.get("model_id")
        referencia_id = manifesto_ref.get("reference_id")
        scenario_id = manifesto_lote.get("scenario_id")
        batch_id = manifesto_lote.get("batch_id")

        item_ref = artefato(manifesto_ref, config["reference"]["features_artifact"])
        item_meta = artefato(manifesto_ref, config["reference"]["metadata_artifact"])
        item_lote = artefato(manifesto_lote, config["batch"]["features_artifact"])

        caminho_ref = verificar_artefato(item_ref)
        caminho_meta = verificar_artefato(item_meta)
        caminho_lote = verificar_artefato(item_lote)

        entradas = [
            {**item_ref, "run_id_origem": manifesto_ref.get("run_id")},
            {**item_meta, "run_id_origem": manifesto_ref.get("run_id")},
            {**item_lote, "run_id_origem": manifesto_lote.get("run_id")},
        ]

        with caminho_meta.open(encoding="utf-8") as arquivo:
            metadata = json.load(arquivo)

        numericas = list(metadata.get("numeric_features", []))
        categoricas = list(metadata.get("categorical_features", []))
        colunas = list(metadata.get("features", []))
        if not colunas:
            raise ErroDeteccao("Metadados do modelo sem lista de features.")

        referencia = ler_features(caminho_ref, numericas, categoricas)
        atual = ler_features(caminho_lote, numericas, categoricas)

        validar_features_esperadas(referencia, atual, colunas)

        limiar_p = float(config["method"]["p_value_threshold"])
        limiar_dist = float(config["method"]["distance_threshold"])
        resultados = comparar_features(referencia, atual, colunas, limiar_p, limiar_dist)
        resumo = resumir(resultados)
        if not resumo["evaluated_columns"]:
            raise ErroDeteccao("Nenhuma feature avaliavel entre a referencia e o lote.")

        caminho_html = output_dir / "drift.html"

        documento = montar_documento(
            run_id=run_id,
            config=config,
            modelo_id=modelo_id,
            referencia_id=referencia_id,
            scenario_id=scenario_id,
            batch_id=batch_id,
            caminho_ref=caminho_ref,
            caminho_lote=caminho_lote,
            resultados=resultados,
            resumo=resumo,
            linhas_ref=len(referencia),
            linhas_lote=len(atual),
            alteradas=list(manifesto_lote.get("variaveis_alteradas", [])),
        )
        caminho_json = output_dir / "drift.json"
        escrever_json(caminho_json, documento)
        salvar_html(documento, caminho_html)
        saidas = [
            descritor_artefato("drift_json", caminho_json, run_id),
            descritor_artefato("drift_html", caminho_html, run_id),
        ]

        escrever_json(
            output_dir / "metrics.json",
            {"run_id": run_id, "measurements": montar_medidas(resumo)},
        )
        if resumo["dataset_drift"]:
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
        escrever_json_atomico(
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
                resumo=resumo,
            ),
        )
    except (ErroDeteccao, ValueError, KeyError, OSError, TypeError) as erro:
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
            error_code="DETECCAO_INVALIDA",
        )
        try:
            escrever_eventos(output_dir / "events.jsonl", eventos)
        except OSError:
            print("Nao foi possivel persistir os eventos de falha.")
        try:
            escrever_json_atomico(
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
                    saidas=[],
                    resumo={},
                    status="failed",
                    error_code="DETECCAO_INVALIDA",
                ),
            )
        except OSError:
            print("Nao foi possivel persistir o manifesto de falha.")
        print(f"Falha na deteccao: {erro}")
        return CODIGO_ERRO

    print(f"Drift detectado: {resumo['drifted_columns']}/{resumo['evaluated_columns']} variaveis")
    print(f"  cenario {scenario_id} / lote {batch_id}")
    for item in resultados:
        marca = {True: "DRIFT", False: "ok", None: "nao avaliavel"}[item["drift_detected"]]
        print(f"  [{marca}] {item['feature']} ({item['method']} = {item['statistic']})")
    print(f"  relatorio -> {caminho_json.as_posix()}")
    print(f"  html -> {caminho_html.as_posix()}")
    return CODIGO_SUCESSO

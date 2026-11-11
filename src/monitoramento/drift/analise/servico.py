"""Orquestracao da analise estatistica e publicacao dos resultados."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any

from monitoramento.drift.dados import ler_features, selecionar_features
from monitoramento.infraestrutura import observabilidade
from monitoramento.infraestrutura.arquivos import escrever_json
from monitoramento.infraestrutura.observabilidade import agora_iso, escrever_eventos

from .config import CODIGO_ERRO, CODIGO_SUCESSO, STAGE, ErroAnalise
from .entradas import artefato, separar_manifestos, verificar_artefato, verificar_compatibilidade
from .interpretacao import montar_resultados, resumir
from .resultados import descritor_artefato, montar_documento, montar_manifesto, montar_medidas

registrar_evento = partial(observabilidade.registrar_evento, stage=STAGE)


def processar(
    config_path: Path,
    config: dict[str, Any],
    manifestos: list[dict[str, Any]],
    run_id: str,
    output_dir: Path,
) -> int:
    """Executa a analise e publica manifesto, eventos, metricas e documento."""
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

        with caminho_meta.open(encoding="utf-8") as arquivo:
            metadata = json.load(arquivo)

        numericas = list(metadata.get("numeric_features", []))
        categoricas = list(metadata.get("categorical_features", []))
        colunas = list(metadata.get("features", []))
        if not colunas:
            raise ErroAnalise("Metadados do modelo sem lista de features.")

        referencia = ler_features(caminho_ref, numericas, categoricas)
        atual = ler_features(caminho_lote, numericas, categoricas)

        avaliadas = selecionar_features(referencia, atual, colunas)
        if not avaliadas:
            raise ErroAnalise("Nenhuma feature em comum entre a referencia e o lote.")

        metodo = config["method"]
        resultados = montar_resultados(
            referencia,
            atual,
            avaliadas,
            numericas,
            bins=int(metodo["bins"]),
            psi_moderado=float(metodo["psi_moderate"]),
            psi_limiar=float(metodo["psi_threshold"]),
            ks_p=float(metodo["ks_p_value_threshold"]),
            ks_limiar=float(metodo["ks_statistic_threshold"]),
        )
        resumo = resumir(resultados)
        alteradas = list(manifesto_lote.get("variaveis_alteradas", []))

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
            alteradas=alteradas,
        )
        caminho_json = output_dir / "analise.json"
        escrever_json(caminho_json, documento)

        entradas = [
            {**item_ref, "run_id_origem": manifesto_ref.get("run_id")},
            {**item_meta, "run_id_origem": manifesto_ref.get("run_id")},
            {**item_lote, "run_id_origem": manifesto_lote.get("run_id")},
        ]
        saidas = [descritor_artefato("analise_json", caminho_json, run_id)]

        escrever_json(output_dir / "config.json", config)
        escrever_json(
            output_dir / "metrics.json",
            {"run_id": run_id, "measurements": montar_medidas(resumo, alteradas)},
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
                resumo=resumo,
            ),
        )
        if resumo["changed_columns"]:
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
    except (ErroAnalise, ValueError, KeyError) as erro:
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
            error_code="ANALISE_INVALIDA",
        )
        escrever_eventos(output_dir / "events.jsonl", eventos)
        print(f"Falha na analise: {erro}")
        return CODIGO_ERRO

    print(
        f"Analise concluida: {resumo['changed_columns']}/{resumo['evaluated_columns']} "
        "variaveis com mudanca"
    )
    print(f"  cenario {scenario_id} / lote {batch_id}")
    for item in resultados:
        marca = {True: "MUDA", False: "ok", None: "nao avaliavel"}[item["drift_detected"]]
        estatistica = item["psi"]
        print(f"  [{marca}] {item['feature']} (PSI {estatistica}, faixa {item['psi_band']})")
    print(f"  documento -> {caminho_json.as_posix()}")
    return CODIGO_SUCESSO

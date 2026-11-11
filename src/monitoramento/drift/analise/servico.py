"""Orquestracao da analise estatistica e publicacao dos resultados."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any

from monitoramento.drift.dados import ler_features
from monitoramento.infraestrutura import observabilidade
from monitoramento.infraestrutura.arquivos import escrever_json, escrever_json_atomico
from monitoramento.infraestrutura.observabilidade import agora_iso, escrever_eventos

from .config import (
    CODIGO_ERRO,
    CODIGO_SUCESSO,
    STAGE,
    ErroAnalise,
    ErroConfiguracao,
    validar_config,
)
from .entradas import (
    artefato,
    carregar_manifesto,
    separar_manifestos,
    validar_features_esperadas,
    verificar_artefato,
    verificar_compatibilidade,
)
from .interpretacao import montar_resultados, resumir
from .resultados import descritor_artefato, montar_documento, montar_manifesto, montar_medidas

registrar_evento = partial(observabilidade.registrar_evento, stage=STAGE)


def processar(
    config_path: Path,
    config: dict[str, Any],
    manifestos: list[dict[str, Any] | Path],
    run_id: str,
    output_dir: Path,
) -> int:
    """Executa a analise e publica manifesto, eventos, metricas e documento."""
    try:
        validar_config(config)
        output_dir.mkdir(parents=True, exist_ok=False)
    except (ErroConfiguracao, OSError) as erro:
        print(f"Falha ao iniciar analise: {erro}")
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
    try:
        escrever_eventos(output_dir / "events.jsonl", eventos)
        escrever_json(output_dir / "config.json", config)
        manifestos = [carregar_manifesto(m) for m in manifestos]
        manifesto_ref, manifesto_lote = separar_manifestos(manifestos, config)

        modelo_id = manifesto_ref.get("model_id")
        referencia_id = manifesto_ref.get("reference_id")
        scenario_id = manifesto_lote.get("scenario_id")
        batch_id = manifesto_lote.get("batch_id")
        verificar_compatibilidade(config, manifesto_ref, manifesto_lote)

        for manifesto, tipo in (
            (manifesto_ref, config["reference"]["features_artifact"]),
            (manifesto_ref, config["reference"]["metadata_artifact"]),
            (manifesto_lote, config["batch"]["features_artifact"]),
        ):
            item = artefato(manifesto, tipo)
            entradas.append({**item, "run_id_origem": manifesto.get("run_id")})
        item_ref, item_meta, item_lote = entradas

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

        validar_features_esperadas(referencia, atual, colunas)

        metodo = config["method"]
        resultados = montar_resultados(
            referencia,
            atual,
            colunas,
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

        saidas = [descritor_artefato("analise_json", caminho_json, run_id)]

        escrever_json(
            output_dir / "metrics.json",
            {"run_id": run_id, "measurements": montar_medidas(resumo, alteradas)},
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
    except Exception as erro:  # Fronteira da execucao: registra falhas de leitura, calculo e IO.
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
            error_code="ANALISE_INVALIDA",
        )
        diagnosticos = [
            (escrever_json, output_dir / "config.json", config),
            (escrever_eventos, output_dir / "events.jsonl", eventos),
            (escrever_json, output_dir / "metrics.json", {"run_id": run_id, "measurements": []}),
        ]
        for escrever, caminho, conteudo in diagnosticos:
            try:
                escrever(caminho, conteudo)
            except OSError:
                print(f"Nao foi possivel persistir {caminho.name} da falha.")
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
                    status="failed",
                    error_code="ANALISE_INVALIDA",
                ),
            )
        except OSError:
            print("Nao foi possivel persistir o manifesto de falha.")
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

"""Orquestracao e publicacao da simulacao de Data Drift."""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd

from monitoramento.infraestrutura import observabilidade
from monitoramento.infraestrutura.arquivos import escrever_json, sha256_arquivo
from monitoramento.infraestrutura.observabilidade import agora_iso, escrever_eventos

from .artefatos import (
    carregar_base,
    descritor_artefato,
    escrever_artefato,
    montar_manifesto,
    montar_medidas,
)
from .config import CODIGO_ERRO, CODIGO_SUCESSO, STAGE, ErroSimulacao, carregar_config
from .transformacoes import (
    aplicar_drift,
    deslocar_periodo,
    resumo_periodo,
    selecionar_alteracoes_efetivas,
    separar_features_rotulos,
)

registrar_evento = partial(observabilidade.registrar_evento, stage=STAGE)


def gerar(
    *,
    config_path: Path,
    source_path: Path | None,
    analysis_dir: Path | None,
    batch_id: str,
    run_id: str,
    output_dir: Path,
    seed: int | None,
) -> int:
    """Gera o cenario de Data Drift e publica manifesto, eventos e metricas."""
    try:
        config = carregar_config(config_path)
        dataset_id = config["dataset_id"]
        scenario_id = config["scenario_id"]
        destino = (analysis_dir or Path(config["outputs"]["analysis_dir"])) / scenario_id / batch_id
        semente = int(seed if seed is not None else config["drift"]["seed"])
        if semente < 0:
            raise ValueError("A seed deve ser nao negativa.")
        origem = source_path or Path(config["base"]["path"])
        config["drift"]["seed"] = semente
        config["base"]["path"] = origem.as_posix()
        config["outputs"]["analysis_dir"] = (
            analysis_dir or Path(config["outputs"]["analysis_dir"])
        ).as_posix()
    except (ErroSimulacao, ValueError, TypeError, KeyError, OSError) as erro:
        print(f"Configuracao invalida ({type(erro).__name__}).", file=sys.stderr)
        return CODIGO_ERRO

    # A verificacao antecipa colisoes comuns; mkdir exclusivo protege tambem corridas.
    if output_dir.exists() or destino.exists():
        print("Diretorio de execucao ou lote ja existe.", file=sys.stderr)
        return CODIGO_ERRO
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except OSError as erro:
        print(f"Falha ao reservar execucao ({type(erro).__name__}).", file=sys.stderr)
        return CODIGO_ERRO

    iniciado = agora_iso()
    eventos: list[dict[str, Any]] = []
    config_efetiva = output_dir / "config.json"
    registrar_evento(eventos, run_id, dataset_id, scenario_id, batch_id, "started", "INFO")
    try:
        escrever_json(config_efetiva, config)
        escrever_eventos(output_dir / "events.jsonl", eventos)
        destino.mkdir(parents=True, exist_ok=False)
        base = carregar_base(origem)

        tempo = config["time"]
        coluna_tempo = tempo["column"]
        formato = tempo["format"]
        if coluna_tempo not in base.columns:
            raise ErroSimulacao(f"Coluna temporal ausente na base: {coluna_tempo}")

        jitter = float(config["drift"].get("jitter", 0.0))

        periodo_origem = resumo_periodo(base[coluna_tempo], formato)
        base_simulada = base.copy()
        base_simulada[coluna_tempo] = deslocar_periodo(
            base[coluna_tempo], formato, tempo["shift_months"]
        )
        base_simulada, descritas = aplicar_drift(
            base_simulada, config["drift"]["variables"], semente, jitter
        )
        descritas = selecionar_alteracoes_efetivas(
            base, base_simulada, descritas, config["target"]["column"], coluna_tempo
        )
        periodo_simulado = resumo_periodo(base_simulada[coluna_tempo], formato)

        alvo = config["target"]["column"]
        features, rotulos = separar_features_rotulos(base_simulada, alvo)

        caminho_features = destino / "features.csv"
        caminho_rotulos = destino / "labels.csv"
        escrever_artefato(caminho_features, features)
        escrever_artefato(caminho_rotulos, rotulos)

        entrada = {
            "artifact_type": config["base"].get("artifact_type", "origem"),
            "path": origem.as_posix(),
            "sha256": sha256_arquivo(origem),
            "row_count": int(len(base)),
            "columns": list(base.columns),
        }
        saidas = [
            descritor_artefato("features", caminho_features, run_id, features),
            descritor_artefato("labels", caminho_rotulos, run_id, rotulos),
        ]

        periodo = {
            "column": coluna_tempo,
            "format": formato,
            "origem": periodo_origem,
            "simulado": periodo_simulado,
            "shift_months": int(tempo["shift_months"]),
        }

        escrever_json(
            output_dir / "metrics.json",
            {
                "run_id": run_id,
                "measurements": montar_medidas(saidas, descritas, periodo["shift_months"]),
            },
        )
        manifesto = montar_manifesto(
            run_id=run_id,
            iniciado=iniciado,
            finalizado=agora_iso(),
            config_path=config_efetiva,
            config=config,
            batch_id=batch_id,
            seed=semente,
            entrada=entrada,
            saidas=saidas,
            periodo=periodo,
            descritas=descritas,
        )
        registrar_evento(eventos, run_id, dataset_id, scenario_id, batch_id, "completed", "INFO")
        escrever_eventos(output_dir / "events.jsonl", eventos)
        # O manifesto terminal fecha a publicacao: nenhuma persistencia ocorre depois.
        escrever_json(output_dir / "manifest.json", manifesto)
    except Exception as erro:
        eventos[:] = [evento for evento in eventos if evento["event"] != "completed"]
        error_code = (
            "ERRO_ENTRADA"
            if isinstance(
                erro, (ErroSimulacao, ValueError, pd.errors.ParserError, FileNotFoundError)
            )
            else "ERRO_TECNICO"
        )
        registrar_evento(
            eventos, run_id, dataset_id, scenario_id, batch_id, "failed", "ERROR", error_code
        )
        try:
            # O diretorio foi reservado por esta chamada; nunca toca execucoes anteriores.
            (output_dir / "manifest.json").unlink(missing_ok=True)
            escrever_json(config_efetiva, config)
            escrever_eventos(output_dir / "events.jsonl", eventos)
            escrever_json(output_dir / "metrics.json", {"run_id": run_id, "measurements": []})
            manifesto = montar_manifesto(
                run_id=run_id,
                iniciado=iniciado,
                finalizado=agora_iso(),
                config_path=config_efetiva,
                config=config,
                batch_id=batch_id,
                seed=semente,
                entrada=None,
                saidas=[],
                periodo=None,
                descritas=[],
                status="failed",
                error_code=error_code,
            )
            escrever_json(output_dir / "manifest.json", manifesto)
        except OSError as persist_erro:
            print(
                f"Falha ao persistir resultado de erro ({type(persist_erro).__name__}).",
                file=sys.stderr,
            )
            return CODIGO_ERRO
        print(
            f"Falha ao gerar o cenario simulado ({error_code}): {type(erro).__name__}.",
            file=sys.stderr,
        )
        return CODIGO_ERRO

    print(f"Cenario simulado ({scenario_id}): {len(features)} linhas")
    print(
        f"  periodo {periodo['origem']['min']}..{periodo['origem']['max']} ->"
        f" {periodo['simulado']['min']}..{periodo['simulado']['max']}"
    )
    for item in descritas:
        print(
            f"    {item['column']:<16} {item['operation']:<6} "
            f"media {item['mean_before']} -> {item['mean_after']}"
        )
    print(f"  features -> {caminho_features.as_posix()}")
    print(f"  labels   -> {caminho_rotulos.as_posix()}")
    print(f"  manifesto -> {output_dir.as_posix()}/manifest.json")
    return CODIGO_SUCESSO

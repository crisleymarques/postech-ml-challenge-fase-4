"""Orquestracao da simulacao de Concept Drift."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

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
from .transformacoes import aplicar_mudanca_conceito, separar_features_rotulos

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
    """Gera o cenario de Concept Drift e publica manifesto, eventos e metricas."""
    output_dir.mkdir(parents=True, exist_ok=True)
    iniciado = agora_iso()
    eventos: list[dict[str, Any]] = []

    try:
        config = carregar_config(config_path)
        dataset_id = config["dataset_id"]
        scenario_id = config["scenario_id"]
        registrar_evento(eventos, run_id, dataset_id, scenario_id, batch_id, "started", "INFO")

        origem = source_path or Path(config["base"]["path"])
        base = carregar_base(origem)

        alvo = config["target"]["column"]
        positivos = int(config["target"]["positive_value"])
        semente = int(seed if seed is not None else config["concept"].get("seed", 42))

        alterada, descricao = aplicar_mudanca_conceito(
            base, alvo, positivos, config["concept"], semente
        )

        features, rotulos_ref = separar_features_rotulos(base, alvo)
        _, rotulos_novos = separar_features_rotulos(alterada, alvo)

        destino = (analysis_dir or Path(config["outputs"]["analysis_dir"])) / scenario_id
        caminho_features = destino / batch_id / "features.csv"
        caminho_rotulos = destino / batch_id / "labels.csv"
        caminho_ref = destino / batch_id / "labels_reference.csv"
        escrever_artefato(caminho_features, features)
        escrever_artefato(caminho_rotulos, rotulos_novos)
        escrever_artefato(caminho_ref, rotulos_ref)

        entrada = {
            "artifact_type": config["base"].get("artifact_type", "origem"),
            "path": origem.as_posix(),
            "sha256": sha256_arquivo(origem),
            "row_count": int(len(base)),
            "columns": list(base.columns),
        }
        saidas = [
            descritor_artefato("features", caminho_features, run_id, features),
            descritor_artefato("labels", caminho_rotulos, run_id, rotulos_novos),
            descritor_artefato("labels_reference", caminho_ref, run_id, rotulos_ref),
        ]

        escrever_json(output_dir / "config.json", config)
        escrever_json(
            output_dir / "metrics.json",
            {"run_id": run_id, "measurements": montar_medidas(descricao, len(features))},
        )
        escrever_json(
            output_dir / "manifest.json",
            montar_manifesto(
                run_id=run_id,
                iniciado=iniciado,
                finalizado=agora_iso(),
                config_path=config_path,
                config=config,
                batch_id=batch_id,
                seed=semente,
                entrada=entrada,
                saidas=saidas,
                descricao=descricao,
            ),
        )
        registrar_evento(eventos, run_id, dataset_id, scenario_id, batch_id, "completed", "INFO")
        escrever_eventos(output_dir / "events.jsonl", eventos)
    except (ErroSimulacao, ValueError) as erro:
        print(f"Falha ao gerar o cenario de Concept Drift: {erro}")
        return CODIGO_ERRO

    print(f"Cenario de Concept Drift ({scenario_id}): {len(features)} linhas")
    print(
        f"  segmento {descricao['segment']['column']} {descricao['segment']['operator']} "
        f"{descricao['segment']['value']}: {descricao['segment_rows']} registros"
    )
    print(
        f"  rotulos alterados: {descricao['flipped_rows']} "
        f"(taxa de positivos {descricao['positive_rate_before']} -> "
        f"{descricao['positive_rate_after']})"
    )
    print(f"  features -> {caminho_features.as_posix()}")
    print(f"  labels   -> {caminho_rotulos.as_posix()}")
    print(f"  manifesto -> {output_dir.as_posix()}/manifest.json")
    return CODIGO_SUCESSO

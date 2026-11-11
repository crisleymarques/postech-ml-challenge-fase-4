"""Leitura dos artefatos e contexto de cada execucao."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from monitoramento.infraestrutura.observabilidade import status_publicado

STATUS_SUCESSO = "success"

EVENTO_ALERTA = "drift_alert"


def ler_json(caminho: Path) -> dict[str, Any] | None:
    """Le um arquivo JSON quando disponivel, sem interromper o processo."""
    if not caminho.is_file():
        return None
    try:
        with caminho.open(encoding="utf-8") as arquivo:
            conteudo = json.load(arquivo)
    except (json.JSONDecodeError, OSError):
        return None
    return conteudo if isinstance(conteudo, dict) else None


def ler_eventos(caminho: Path) -> list[dict[str, Any]]:
    """Le um `events.jsonl` de forma tolerante, ignorando linhas invalidas."""
    eventos: list[dict[str, Any]] = []
    if not caminho.is_file():
        return eventos
    with caminho.open(encoding="utf-8") as arquivo:
        for linha in arquivo:
            conteudo = linha.strip()
            if not conteudo:
                continue
            try:
                eventos.append(json.loads(conteudo))
            except json.JSONDecodeError:
                continue
    return eventos


def listar_execucoes(runs_dir: Path) -> list[Path]:
    """Lista os diretorios de execucao com manifesto ou eventos, em ordem."""
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Diretorio de execucoes nao encontrado: {runs_dir}")
    return sorted(
        (
            caminho
            for caminho in runs_dir.iterdir()
            if (caminho / "manifest.json").is_file() or (caminho / "events.jsonl").is_file()
        ),
        key=lambda caminho: caminho.name,
    )


def carregar_contexto(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Le os artefatos elegiveis e reune o contexto da execucao."""
    manifesto = ler_json(run_dir / "manifest.json") or {}
    eventos_run = ler_eventos(run_dir / "events.jsonl")
    manifesto = {**manifesto, "status": status_publicado(manifesto, eventos_run)}
    publicado = manifesto["status"] == STATUS_SUCESSO
    evento_inicial = eventos_run[0] if eventos_run else {}
    stage = manifesto.get("stage") or evento_inicial.get("stage")
    contexto = {
        "run_dir": run_dir,
        "manifesto": manifesto,
        "metrics": ler_json(run_dir / "metrics.json"),
        "quality": ler_json(run_dir / "quality.json"),
        "drift": ler_json(run_dir / "drift.json") if publicado else None,
        "analise": ler_json(run_dir / "analise.json") if publicado else None,
        "performance": ler_json(run_dir / "performance.json") if publicado else None,
        "stage": stage,
    }
    return contexto, eventos_run

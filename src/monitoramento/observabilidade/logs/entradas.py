"""Leitura dos eventos e metadados de cada execucao."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from monitoramento.infraestrutura.observabilidade import status_publicado

from .classificacao import categoria_da_etapa, resultado_do_status


def listar_execucoes(runs_dir: Path) -> list[Path]:
    """Lista execucoes com eventos ou manifesto, incluindo publicacoes incompletas."""
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Diretorio de execucoes nao encontrado: {runs_dir}")
    return sorted(
        (
            caminho
            for caminho in runs_dir.iterdir()
            if (caminho / "events.jsonl").is_file() or (caminho / "manifest.json").is_file()
        ),
        key=lambda caminho: caminho.name,
    )


def ler_eventos(caminho: Path) -> list[dict[str, Any]]:
    """Le um `events.jsonl` e retorna a lista de eventos."""
    eventos: list[dict[str, Any]] = []
    if not caminho.is_file():
        return eventos
    with caminho.open(encoding="utf-8") as arquivo:
        for numero, linha in enumerate(arquivo, start=1):
            conteudo = linha.strip()
            if not conteudo:
                continue
            try:
                eventos.append(json.loads(conteudo))
            except json.JSONDecodeError as erro:
                raise ValueError(f"Evento invalido em {caminho} (linha {numero}): {erro}") from erro
    return eventos


def ler_manifesto(caminho: Path) -> dict[str, Any] | None:
    """Le o manifesto da execucao quando disponivel, sem interromper o processo."""
    if not caminho.is_file():
        return None
    try:
        with caminho.open(encoding="utf-8") as arquivo:
            conteudo = json.load(arquivo)
    except (json.JSONDecodeError, OSError):
        return None
    return conteudo if isinstance(conteudo, dict) else None


def analisar_execucao(run_dir: Path, eventos: list[dict[str, Any]]) -> dict[str, Any]:
    """Reune os metadados de uma execucao a partir do manifesto e dos eventos."""
    manifesto = ler_manifesto(run_dir / "manifest.json") or {}
    stage = manifesto.get("stage") or (eventos[0].get("stage") if eventos else None)
    status = status_publicado(manifesto, eventos)
    transformacoes = manifesto.get("transformations")
    return {
        "run_id": manifesto.get("run_id") or run_dir.name,
        "run_dir": run_dir.as_posix(),
        "stage": stage,
        "categoria": categoria_da_etapa(stage),
        "status": status,
        "resultado": resultado_do_status(status),
        "error_code": manifesto.get("error_code"),
        "finished_at": manifesto.get("finished_at"),
        "event_count": len(eventos),
        "transformations": list(transformacoes) if isinstance(transformacoes, list) else [],
    }

"""Leitura dos artefatos persistidos e das execucoes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from monitoramento.infraestrutura.observabilidade import status_publicado

from .config import STATUS_INCOMPLETO, STATUS_TERMINAL
from .indicadores import resumo_alertas


def ler_json(caminho: Path) -> dict[str, Any] | None:
    """Le um artefato JSON quando disponivel, sem interromper a interface."""
    if not caminho.is_file():
        return None
    try:
        with caminho.open(encoding="utf-8") as arquivo:
            conteudo = json.load(arquivo)
    except (json.JSONDecodeError, OSError, UnicodeError):
        return None
    return conteudo if isinstance(conteudo, dict) else None


def ler_eventos(caminho: Path) -> list[dict[str, Any]]:
    """Le um `events.jsonl` de forma tolerante, ignorando linhas invalidas."""
    eventos: list[dict[str, Any]] = []
    if not caminho.is_file():
        return eventos
    try:
        with caminho.open(encoding="utf-8") as arquivo:
            for linha in arquivo:
                conteudo = linha.strip()
                if not conteudo:
                    continue
                try:
                    evento = json.loads(conteudo)
                except json.JSONDecodeError:
                    continue
                if isinstance(evento, dict):
                    eventos.append(evento)
    except (OSError, UnicodeError):
        # Leitura incompleta nao confirma a publicacao de sucesso.
        return []
    return eventos


def ler_alertas(caminho: Path) -> dict[str, Any] | None:
    """Le o relatorio de alertas, exigindo resumo e colecao utilizaveis."""
    artefato = ler_json(caminho)
    if not resumo_alertas(artefato) or not isinstance(artefato.get("alertas"), list):
        return None
    return artefato


def inferir_status(eventos: list[dict[str, Any]]) -> str:
    """Infere o status da execucao pelo ultimo evento terminal observado."""
    for evento in reversed(eventos):
        status = STATUS_TERMINAL.get(evento.get("event"))
        if status is not None:
            return status
    return STATUS_INCOMPLETO


def listar_execucoes(runs_dir: Path) -> list[Path]:
    """Lista os diretorios de execucao com manifesto ou eventos, em ordem."""
    if not runs_dir.is_dir():
        return []
    return sorted(
        (
            caminho
            for caminho in runs_dir.iterdir()
            if (caminho / "manifest.json").is_file() or (caminho / "events.jsonl").is_file()
        ),
        key=lambda caminho: caminho.name,
    )


def carregar_execucoes(runs_dir: Path) -> list[dict[str, Any]]:
    """Reune os metadados de cada execucao a partir do manifesto e dos eventos."""
    execucoes: list[dict[str, Any]] = []
    for run_dir in listar_execucoes(runs_dir):
        manifesto = ler_json(run_dir / "manifest.json") or {}
        eventos = ler_eventos(run_dir / "events.jsonl")
        stage = manifesto.get("stage") or (eventos[0].get("stage") if eventos else None)
        status = status_publicado(manifesto, eventos)
        execucoes.append(
            {
                "run_id": manifesto.get("run_id") or run_dir.name,
                "stage": stage,
                "status": status,
                "started_at": manifesto.get("started_at"),
                "finished_at": manifesto.get("finished_at"),
                "error_code": manifesto.get("error_code"),
                "event_count": len(eventos),
                "incompleta": status == STATUS_INCOMPLETO,
            }
        )
    return execucoes

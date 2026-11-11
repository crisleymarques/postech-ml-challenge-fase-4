"""Servico da integracao de alertas operacionais."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from monitoramento.infraestrutura.observabilidade import agora_iso

from .config import carregar_config
from .drift import consolidar_drift
from .entradas import CONTRACT_VERSION, ler_json, validar_logs, validar_saude
from .regras import alertas_de_drift, alertas_de_falhas, alertas_de_medidas
from .resultados import gravar_alertas, montar_resumo


def gerar(
    saude_path: Path,
    logs_resumo_path: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Integra as metricas de saude e os logs, e publica `alertas.json`."""
    config = carregar_config(config_path)
    saude = ler_json(saude_path, "Relatorio de saude")
    resumo_logs = ler_json(logs_resumo_path, "Resumo dos logs")
    validar_saude(saude)
    validar_logs(resumo_logs)

    alertas = (
        alertas_de_medidas(saude, config.get("severidade_por_situacao") or {})
        + alertas_de_falhas(resumo_logs, config["severidade_falha"])
        + alertas_de_drift(resumo_logs, config["severidade_alerta_drift"])
    )

    relatorio = {
        "contract_version": CONTRACT_VERSION,
        "generated_at": agora_iso(),
        "fontes": {
            "saude": saude_path.as_posix(),
            "logs": logs_resumo_path.as_posix(),
            "config": config_path.as_posix(),
        },
        "output_dir": output_dir.as_posix(),
        "drift": consolidar_drift(saude),
        "alertas": alertas,
        "resumo": montar_resumo(alertas),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    gravar_alertas(output_dir / "alertas.json", relatorio)
    return relatorio

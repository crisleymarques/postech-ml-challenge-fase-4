"""Construtores de fontes e configuracoes para os testes."""

import json
from pathlib import Path

CONFIG_REAL = Path(__file__).resolve().parents[3] / "configs" / "alertas.json"


def medida(
    id_metrica: str,
    categoria: str,
    situacao: str,
    valor: object = 1,
    unidade: str = "ocorrencias",
    nome: str = "Medida de teste",
) -> dict:
    """Medida de saude no formato publicado pela F3-02."""
    return {
        "id": id_metrica,
        "categoria": categoria,
        "nome": nome,
        "unidade": unidade,
        "valor": valor,
        "disponibilidade": "available",
        "situacao": situacao,
    }


def saude(execucoes: list[dict], agregados: list[dict] | None = None) -> dict:
    """Relatorio de saude minimo conforme a F3-02."""
    return {
        "contract_version": "1.0",
        "execucoes": execucoes,
        "agregados": agregados or [],
    }


def resumo_logs(falhas: list[dict] | None = None, alertas_drift: list[dict] | None = None) -> dict:
    """Resumo de logs minimo conforme a F3-01."""
    return {
        "contract_version": "1.0",
        "execucoes": {"total": 1, "por_status": {"success": 1}, "incompletas": []},
        "eventos": {"total": 1},
        "falhas": falhas or [],
        "alertas": alertas_drift or [],
    }


def config_minimo(**campos: object) -> dict:
    """Configuracao de alertas valida, com ajustes opcionais."""
    base = {
        "contract_version": "1.0",
        "severidades": ["warning", "critical"],
        "severidade_por_situacao": {"atencao": "warning", "critico": "critical"},
        "severidade_falha": "critical",
        "severidade_alerta_drift": "warning",
    }
    base.update(campos)
    return base


def escrever_config(tmp_path: Path, config: dict) -> Path:
    caminho = tmp_path / "alertas.json"
    caminho.write_text(json.dumps(config), encoding="utf-8")
    return caminho

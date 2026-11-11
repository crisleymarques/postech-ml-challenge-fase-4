"""Regras da integracao de alertas operacionais."""

from __future__ import annotations

from typing import Any, Iterable

ORIGEM_AGREGADO = "agregado"


ORIGEM_SAUDE = "saude"


ORIGEM_LOGS = "logs"


CATEGORIA_LOG = "log"


def medidas_com_situacao(
    saude: dict[str, Any],
) -> Iterable[tuple[str, str, dict[str, Any]]]:
    """Percorre as medidas de saude, rendendo (origem, escopo, medida)."""
    for execucao in saude.get("execucoes", []) or []:
        if not isinstance(execucao, dict):
            continue
        origem = execucao.get("run_id") or execucao.get("stage") or ORIGEM_AGREGADO
        for medida in execucao.get("metricas", []) or []:
            if isinstance(medida, dict):
                yield str(origem), "execucao", medida
    for medida in saude.get("agregados", []) or []:
        if isinstance(medida, dict):
            yield ORIGEM_AGREGADO, "agregado", medida


def alerta_de_medida(
    origem: str,
    escopo: str,
    medida: dict[str, Any],
    severidade: str,
) -> dict[str, Any]:
    """Converte uma medida fora do esperado em alerta de observabilidade."""
    return {
        "id": f"{ORIGEM_SAUDE}:{origem}:{medida.get('id')}",
        "severidade": severidade,
        "categoria": medida.get("categoria"),
        "escopo": escopo,
        "origem": origem,
        "fonte": ORIGEM_SAUDE,
        "metrica": medida.get("id"),
        "descricao": f"{medida.get('nome')} em '{medida.get('situacao')}'",
        "valor": medida.get("valor"),
        "unidade": medida.get("unidade"),
        "situacao": medida.get("situacao"),
        "error_code": None,
        "evidencia": {"origem": origem, "metrica": medida.get("id")},
    }


def alertas_de_medidas(
    saude: dict[str, Any],
    severidade_por_situacao: dict[str, Any],
) -> list[dict[str, Any]]:
    """Alertas derivados das medidas de saude em situacao de atencao ou critico."""
    alertas: list[dict[str, Any]] = []
    for origem, escopo, medida in medidas_com_situacao(saude):
        severidade = severidade_por_situacao.get(medida.get("situacao"))
        if severidade is None:
            continue
        alertas.append(alerta_de_medida(origem, escopo, medida, severidade))
    return alertas


def alertas_de_falhas(
    resumo_logs: dict[str, Any],
    severidade_falha: str,
) -> list[dict[str, Any]]:
    """Alertas derivados das falhas de execucao ou de contrato registradas nos logs."""
    alertas: list[dict[str, Any]] = []
    for indice, falha in enumerate(resumo_logs.get("falhas", []) or [], start=1):
        if not isinstance(falha, dict):
            continue
        run_id = falha.get("run_id") or ORIGEM_AGREGADO
        error_code = falha.get("error_code")
        rotulo = f"{falha.get('event')} ({error_code})" if error_code else falha.get("event")
        alertas.append(
            {
                "id": f"{ORIGEM_LOGS}:falha:{run_id}:{indice}",
                "severidade": severidade_falha,
                "categoria": falha.get("categoria") or CATEGORIA_LOG,
                "escopo": CATEGORIA_LOG,
                "origem": run_id,
                "fonte": ORIGEM_LOGS,
                "metrica": None,
                "descricao": f"Falha de execucao: {rotulo}",
                "valor": None,
                "unidade": None,
                "situacao": None,
                "error_code": error_code,
                "evidencia": {
                    "run_id": falha.get("run_id"),
                    "stage": falha.get("stage"),
                    "event": falha.get("event"),
                    "timestamp": falha.get("timestamp"),
                },
            }
        )
    return alertas


def alertas_de_drift(
    resumo_logs: dict[str, Any],
    severidade_alerta_drift: str,
) -> list[dict[str, Any]]:
    """Alertas derivados dos eventos `drift_alert` registrados nos logs."""
    alertas: list[dict[str, Any]] = []
    for indice, evento in enumerate(resumo_logs.get("alertas", []) or [], start=1):
        if not isinstance(evento, dict):
            continue
        run_id = evento.get("run_id") or ORIGEM_AGREGADO
        contexto = evento.get("scenario_id") or evento.get("batch_id")
        descricao = "Alerta de drift" + (f" (cenario {contexto})" if contexto else "")
        alertas.append(
            {
                "id": f"{ORIGEM_LOGS}:drift:{run_id}:{indice}",
                "severidade": severidade_alerta_drift,
                "categoria": "drift",
                "escopo": CATEGORIA_LOG,
                "origem": run_id,
                "fonte": ORIGEM_LOGS,
                "metrica": None,
                "descricao": descricao,
                "valor": None,
                "unidade": None,
                "situacao": None,
                "error_code": None,
                "evidencia": {
                    "run_id": evento.get("run_id"),
                    "stage": evento.get("stage"),
                    "scenario_id": evento.get("scenario_id"),
                    "batch_id": evento.get("batch_id"),
                    "timestamp": evento.get("timestamp"),
                },
            }
        )
    return alertas

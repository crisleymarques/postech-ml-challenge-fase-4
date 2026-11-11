"""Coleta dos valores operacionais nos artefatos persistidos."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .entradas import EVENTO_ALERTA, STATUS_SUCESSO

STATUS_FALHA = "failed"

STATUS_REJEICAO = "rejected"

ETAPA_VALIDACAO = "validacao"


def _parse_iso(texto: Any) -> datetime | None:
    """Interpreta um instante ISO 8601 UTC com sufixo Z opcional."""
    if not isinstance(texto, str):
        return None
    limpo = texto[:-1] if texto.endswith("Z") else texto
    for formato in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(limpo, formato)
        except ValueError:
            continue
    return None


def _aninhado(origem: Any, *chaves: str) -> Any:
    """Percorre um caminho de chaves em objetos aninhados."""
    atual = origem
    for chave in chaves:
        if not isinstance(atual, dict):
            return None
        atual = atual.get(chave)
    return atual


def coletar_duracao(manifesto: dict[str, Any]) -> float | None:
    """Duracao em segundos entre inicio e fim da execucao."""
    inicio = _parse_iso(manifesto.get("started_at"))
    fim = _parse_iso(manifesto.get("finished_at"))
    if inicio is None or fim is None:
        return None
    return round((fim - inicio).total_seconds(), 6)


def coletar_volume(manifesto: dict[str, Any]) -> int | None:
    """Soma de linhas dos artefatos de saida (ou entrada) da execucao."""
    for chave in ("outputs", "inputs"):
        itens = manifesto.get(chave)
        if not isinstance(itens, list):
            continue
        total = 0
        encontrou = False
        for item in itens:
            if not isinstance(item, dict):
                continue
            linhas = item.get("row_count")
            if isinstance(linhas, bool) or not isinstance(linhas, (int, float)):
                continue
            total += int(linhas)
            encontrou = True
        if encontrou:
            return total
    return None


def coletar_violacoes(quality: Any, metrics: Any) -> int | None:
    """Total de ocorrencias violadas do contrato de qualidade."""
    violacoes = _aninhado(quality, "violations")
    if isinstance(violacoes, list):
        return sum(int(item.get("count", 0)) for item in violacoes if isinstance(item, dict))
    for medida in (metrics or {}).get("measurements", []) or []:
        if isinstance(medida, dict) and medida.get("name") == "violacoes_total":
            valor = medida.get("value")
            if valor is not None:
                return int(valor)
    return None


def coletar_valor_drift(drift: Any, analise: Any, chave_drift: str, chave_analise: Any) -> Any:
    """Le um valor do resumo de drift.json e, na ausencia, de analise.json."""
    for fonte, chave in ((drift, chave_drift), (analise, chave_analise)):
        valor = _aninhado(fonte, "summary", chave) if chave else None
        if valor is not None:
            return valor
    return None


def coletar_variaveis_drift(drift: Any, analise: Any) -> int | None:
    """Quantidade de variaveis com mudanca entre referencia e lote."""
    valores = coletar_valor_drift(drift, analise, "drifted_columns", "changed_columns")
    if isinstance(valores, list):
        return len(valores)
    if isinstance(valores, bool) or not isinstance(valores, (int, float)):
        return None
    return int(valores)


def contar_por_status(execucoes: list[dict[str, Any]], status: str) -> int:
    """Quantidade de execucoes com o status informado."""
    return sum(1 for execucao in execucoes if execucao.get("status") == status)


def taxa_sucesso(execucoes: list[dict[str, Any]]) -> float | None:
    """Proporcao de execucoes concluidas com status 'success'."""
    if not execucoes:
        return None
    return round(contar_por_status(execucoes, STATUS_SUCESSO) / len(execucoes), 6)


def contar_alertas(eventos: list[dict[str, Any]]) -> int:
    """Quantidade de eventos de alerta de drift."""
    return sum(1 for evento in eventos if evento.get("event") == EVENTO_ALERTA)


def contar_lotes_rejeitados(execucoes: list[dict[str, Any]]) -> int:
    """Quantidade de execucoes de validacao com lote rejeitado."""
    return sum(
        1
        for execucao in execucoes
        if execucao.get("stage") == ETAPA_VALIDACAO and execucao.get("status") == STATUS_REJEICAO
    )


def espaco_livre(output_dir: Path) -> float | None:
    """Espaco livre em bytes no volume do diretorio de saida."""
    try:
        return float(shutil.disk_usage(str(output_dir)).free)
    except OSError:
        return None


def escrita_disponivel(output_dir: Path) -> bool:
    """Verifica a capacidade de escrita gravando e removendo um arquivo temporario."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        teste = output_dir / ".escrita_saude"
        teste.write_text("ok", encoding="utf-8")
        teste.unlink()
    except OSError:
        return False
    return True


COLETORES: dict[str, Callable[[dict[str, Any]], Any]] = {
    "pipeline.status_execucao": lambda c: (c.get("manifesto") or {}).get("status"),
    "pipeline.duracao_s": lambda c: coletar_duracao(c.get("manifesto") or {}),
    "pipeline.volume_linhas": lambda c: coletar_volume(c.get("manifesto") or {}),
    "pipeline.taxa_sucesso": lambda c: taxa_sucesso(c.get("execucoes") or []),
    "pipeline.falhas_total": lambda c: contar_por_status(c.get("execucoes") or [], STATUS_FALHA),
    "qualidade.violacoes_total": lambda c: coletar_violacoes(c.get("quality"), c.get("metrics")),
    "qualidade.lotes_rejeitados": lambda c: contar_lotes_rejeitados(c.get("execucoes") or []),
    "drift.dataset_drift": lambda c: coletar_valor_drift(
        c.get("drift"), None, "dataset_drift", None
    ),
    "drift.variaveis_com_drift": lambda c: coletar_variaveis_drift(
        c.get("drift"), c.get("analise")
    ),
    "drift.proporcao_drift": lambda c: coletar_valor_drift(
        c.get("drift"), c.get("analise"), "drift_share", "change_share"
    ),
    "drift.alertas_total": lambda c: contar_alertas(c.get("eventos") or []),
    "drift.degradacao_conceito": lambda c: _aninhado(
        c.get("performance"), "comparison", "degradation_detected"
    ),
    "modelo.roc_auc": lambda c: _aninhado(
        c.get("performance"), "contrast", "concept_drift", "metrics", "roc_auc"
    ),
    "infra.espaco_livre_bytes": lambda c: espaco_livre(c["output_dir"]),
    "infra.escrita_ok": lambda c: escrita_disponivel(c["output_dir"]),
}


def coletar_metrica(metrica: dict[str, Any], contexto: dict[str, Any]) -> Any:
    """Coleta o valor de uma metrica a partir do contexto de artefatos."""
    coletor = COLETORES.get(metrica["id"])
    if coletor is None:
        return None
    try:
        return coletor(contexto)
    except (KeyError, TypeError):
        return None

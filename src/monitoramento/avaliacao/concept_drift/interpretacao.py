"""Comparacao e interpretacao dos resultados de Concept Drift."""

from __future__ import annotations

from typing import Any


def comparar(
    referencia: dict[str, float | None],
    alterado: dict[str, float | None],
    metrica: str,
    min_delta: float,
) -> dict[str, Any]:
    """Compara os conceitos e aplica o criterio de degradacao declarado."""
    deltas: dict[str, float | None] = {}
    for nome, valor_ref in referencia.items():
        valor_alt = alterado.get(nome)
        if valor_ref is None or valor_alt is None:
            deltas[nome] = None
        else:
            deltas[nome] = round(valor_alt - valor_ref, 6)

    variacao = deltas.get(metrica)
    detectada = variacao is not None and variacao <= -float(min_delta)
    return {
        "metric": metrica,
        "min_delta": float(min_delta),
        "deltas": deltas,
        "metric_delta": variacao,
        "degradation_detected": bool(detectada),
    }


def montar_conclusoes(
    referencia: dict[str, float | None],
    alterado: dict[str, float | None],
    comparacao: dict[str, Any],
    conceito: dict[str, Any] | None,
) -> list[str]:
    """Registra as conclusoes tecnicas da avaliacao, sem inferir causalidade."""
    conclusoes: list[str] = [
        "O contraste usa os mesmos registros e as mesmas features: apenas a relacao entre "
        "perfil e desfecho difere entre o conceito original e o cenario de Concept Drift, "
        "isolando a mudanca de conceito (P(y|X)) da mudanca de distribuicao (P(X))."
    ]

    auc_ref = referencia.get("roc_auc")
    auc_alt = alterado.get("roc_auc")
    if auc_ref is not None and auc_alt is not None:
        conclusoes.append(
            f"No conceito original o modelo tem ROC AUC {auc_ref:.4f}; no cenario de "
            f"Concept Drift, {auc_alt:.4f} (variacao {auc_alt - auc_ref:+.4f})."
        )

    metrica = comparacao["metric"]
    variacao = comparacao["metric_delta"]
    if variacao is None:
        conclusoes.append(
            f"A metrica de comparacao '{metrica}' nao pode ser calculada para os dois conceitos."
        )
    elif comparacao["degradation_detected"]:
        conclusoes.append(
            f"A variacao de {metrica} ({variacao:+.4f}) atinge o criterio declarado de queda "
            f"(<= -{comparacao['min_delta']}), evidenciando degradacao de desempenho associada "
            "a mudanca de conceito."
        )
    else:
        conclusoes.append(
            f"A variacao de {metrica} ({variacao:+.4f}) nao atinge o criterio de degradacao "
            f"declarado (<= -{comparacao['min_delta']})."
        )

    conclusoes.append(
        "Data Drift e Concept Drift sao distintos: o Data Drift muda a distribuicao das "
        "entradas (P(X)) e e detectado sem rotulos; o Concept Drift muda a relacao entre "
        "entradas e alvo (P(y|X)) e so pode ser medido com rotulos compativeis, como neste "
        "contraste. A queda de desempenho e uma consequencia possivel do Concept Drift, nao "
        "a sua definicao."
    )

    if conceito and conceito.get("description"):
        conclusoes.append(f"Mudanca simulada: {conceito['description']}")
    return conclusoes

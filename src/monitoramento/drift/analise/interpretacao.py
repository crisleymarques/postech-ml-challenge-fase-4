"""Decisao por variavel e interpretacao da mudanca de distribuicao."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .config import TIPO_CATEGORICO, TIPO_NUMERICO
from .kolmogorov_smirnov import teste_ks
from .psi import faixa_psi, psi_categorico, psi_numerico


def analisar_variavel(
    nome: str,
    tipo: str,
    referencia: pd.Series,
    atual: pd.Series,
    *,
    bins: int,
    psi_moderado: float,
    psi_limiar: float,
    ks_p: float,
    ks_limiar: float,
) -> dict[str, Any]:
    """Aplica os criterios declarados a uma variavel e descreve a decisao."""
    resultado: dict[str, Any] = {
        "feature": nome,
        "kind": tipo,
        "psi": None,
        "psi_band": None,
        "ks_statistic": None,
        "ks_p_value": None,
        "drift_detected": None,
        "reasons": [],
    }

    if tipo == TIPO_NUMERICO:
        valor_psi = psi_numerico(referencia, atual, bins)
        teste = teste_ks(referencia, atual)
    else:
        valor_psi = psi_categorico(referencia, atual)
        teste = None

    if valor_psi is None:
        resultado["reasons"] = ["PSI nao calculavel por falta de dados validos"]
        return resultado

    resultado["psi"] = round(valor_psi, 6)
    resultado["psi_band"] = faixa_psi(valor_psi, psi_moderado, psi_limiar)

    motivos: list[str] = []
    if valor_psi >= psi_limiar:
        motivos.append(f"PSI {valor_psi:.4f} >= limiar {psi_limiar}")

    if teste is not None:
        estatistica, pvalor = teste
        resultado["ks_statistic"] = round(estatistica, 6)
        resultado["ks_p_value"] = round(pvalor, 6)
        if pvalor < ks_p and estatistica >= ks_limiar:
            motivos.append(
                f"K-S p-valor {pvalor:.4g} < {ks_p} com "
                f"estatistica {estatistica:.4f} >= {ks_limiar}"
            )

    resultado["drift_detected"] = bool(motivos)
    resultado["reasons"] = motivos or ["sem evidencia de mudanca pelos criterios declarados"]
    return resultado


def montar_resultados(
    referencia: pd.DataFrame,
    atual: pd.DataFrame,
    colunas: list[str],
    numericas: list[str],
    *,
    bins: int,
    psi_moderado: float,
    psi_limiar: float,
    ks_p: float,
    ks_limiar: float,
) -> list[dict[str, Any]]:
    """Analisa cada variavel preservando a ordem declarada das features."""
    resultados: list[dict[str, Any]] = []
    for coluna in colunas:
        tipo = TIPO_NUMERICO if coluna in numericas else TIPO_CATEGORICO
        resultados.append(
            analisar_variavel(
                coluna,
                tipo,
                referencia[coluna],
                atual[coluna],
                bins=bins,
                psi_moderado=psi_moderado,
                psi_limiar=psi_limiar,
                ks_p=ks_p,
                ks_limiar=ks_limiar,
            )
        )
    return resultados


def resumir(resultados: list[dict[str, Any]]) -> dict[str, Any]:
    """Resume a analise com contagem e proporcao de variaveis com mudanca."""
    avaliaveis = [item for item in resultados if item["drift_detected"] is not None]
    com_drift = [item for item in avaliaveis if item["drift_detected"]]
    total = len(avaliaveis)
    proporcao = round(len(com_drift) / total, 6) if total else None
    significativas = [item for item in avaliaveis if item["psi_band"] == "significativa"]
    psi_maximo = max((item["psi"] for item in avaliaveis if item["psi"] is not None), default=None)
    return {
        "evaluated_columns": total,
        "changed_columns": len(com_drift),
        "change_share": proporcao,
        "significant_columns": len(significativas),
        "psi_maximo": round(psi_maximo, 6) if psi_maximo is not None else None,
        "changed_features": [item["feature"] for item in com_drift],
    }


def montar_conclusoes(
    resumo: dict[str, Any],
    alteradas: list[str],
    *,
    psi_limiar: float,
    ks_p: float,
) -> list[str]:
    """Registra as conclusoes tecnicas da analise, sem inferir causalidade."""
    conclusoes: list[str] = []
    if resumo["evaluated_columns"]:
        conclusoes.append(
            f"Os criterios declarados classificaram {resumo['changed_columns']} de "
            f"{resumo['evaluated_columns']} variaveis com mudanca de distribuicao, das quais "
            f"{resumo['significant_columns']} com PSI significativo (>= {psi_limiar})."
        )
    conclusoes.append(
        "A referencia (particao de treino) e o lote (base reservada de producao) provem de "
        "periodos distintos: parte das diferencas pode refletir a separacao temporal do dataset "
        "e nao apenas a alteracao injetada na simulacao."
    )
    if alteradas:
        faltantes = [coluna for coluna in alteradas if coluna not in resumo["changed_features"]]
        if faltantes:
            conclusoes.append(
                "Variaveis alteradas intencionalmente nao confirmadas pelos criterios: "
                + ", ".join(faltantes)
                + "."
            )
        else:
            conclusoes.append(
                f"Todas as {len(alteradas)} variaveis alteradas na simulacao foram confirmadas "
                "pela analise estatistica, o que liga a mudanca observada a uma causa conhecida "
                "do cenario."
            )
    conclusoes.append(
        "A mudanca de distribuicao observada nao comprova, por si so, degradacao do desempenho "
        f"nem causalidade: os testes (PSI, K-S p-valor < {ks_p}) evidenciam diferenca entre as "
        "amostras, e a comparacao de desempenho depende de rotulos compativeis em etapa propria."
    )
    return conclusoes

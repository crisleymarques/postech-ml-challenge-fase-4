"""Transformacoes deterministas e verificacao do cenario de Data Drift."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import IDENTIFICADOR, MINIMO_VARIAVEIS, ErroSimulacao


def resumo_periodo(serie: pd.Series, formato: str) -> dict[str, str]:
    """Descreve o menor e o maior periodo de uma coluna temporal."""
    datas = pd.to_datetime(serie, format=formato, errors="raise")
    return {"min": datas.min().strftime(formato), "max": datas.max().strftime(formato)}


def deslocar_periodo(serie: pd.Series, formato: str, meses: int) -> pd.Series:
    """Avanca a coluna temporal em um numero fixo de meses."""
    datas = pd.to_datetime(serie, format=formato, errors="raise")
    return (datas + pd.DateOffset(months=int(meses))).dt.strftime(formato)


def aplicar_drift(
    base: pd.DataFrame,
    variaveis: dict[str, Any],
    seed: int,
    jitter: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Altera a distribuicao das variaveis escolhidas e descreve cada mudanca."""
    ajustado = base.copy()
    gerador = np.random.default_rng(seed)
    descritas: list[dict[str, Any]] = []
    for coluna, especificacao in variaveis.items():
        if coluna not in ajustado.columns:
            raise ErroSimulacao(f"Coluna da alteracao ausente na base: {coluna}")

        originais = ajustado[coluna]
        ausentes = originais.isna() | originais.eq("")
        valores = pd.to_numeric(originais, errors="coerce")
        invalidos = ~ausentes & (valores.isna() | ~np.isfinite(valores))
        if invalidos.any():
            raise ErroSimulacao(f"Valores numericos invalidos na coluna {coluna}.")

        ruido = gerador.uniform(1 - jitter, 1 + jitter, size=len(valores))
        operacao = especificacao["operation"]
        if operacao == "scale":
            parametro = float(especificacao["factor"])
            transformados = valores * parametro * ruido
        else:
            parametro = float(especificacao["delta"])
            transformados = valores + parametro * ruido

        minimo, maximo = especificacao["clip"]
        resultado = transformados.clip(float(minimo), float(maximo)).astype(object)
        # Ausencias permanecem ausentes: a simulacao nao inventa valor onde nao havia.
        resultado[ausentes] = ""
        ajustado[coluna] = resultado
        descritas.append(
            {
                "column": coluna,
                "operation": operacao,
                "parameter": parametro,
                "clip": [float(minimo), float(maximo)],
                "missing_preserved": int(ausentes.sum()),
                "mean_before": round(float(valores.mean()), 4),
                "mean_after": round(float(pd.to_numeric(ajustado[coluna]).mean()), 4),
            }
        )
    return ajustado, descritas


def selecionar_alteracoes_efetivas(
    base: pd.DataFrame,
    base_simulada: pd.DataFrame,
    descritas: list[dict[str, Any]],
    alvo: str,
    coluna_tempo: str,
) -> list[dict[str, Any]]:
    """Exige mudanca empirica em pelo menos duas features do cenario."""
    # Contar configuracoes nao comprova mudanca: zeros, ausencias e clipping
    # podem neutralizar uma operacao. Compare as distribuicoes empiricas,
    # ignorando a ordem das linhas e diferencas apenas de representacao.
    descritas = [
        item
        for item in descritas
        if item["column"] not in (IDENTIFICADOR, alvo, coluna_tempo)
        and not np.array_equal(
            np.sort(pd.to_numeric(base[item["column"]]).to_numpy(dtype=float)),
            np.sort(pd.to_numeric(base_simulada[item["column"]]).to_numpy(dtype=float)),
            equal_nan=True,
        )
    ]
    if len(descritas) < MINIMO_VARIAVEIS:
        raise ErroSimulacao(
            "A simulacao exige alteracao efetiva da distribuicao de ao menos "
            f"{MINIMO_VARIAVEIS} features."
        )
    return descritas


def separar_features_rotulos(base: pd.DataFrame, alvo: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa as features dos rotulos, mantendo o identificador em ambos."""
    if alvo not in base.columns:
        raise ErroSimulacao(f"Coluna alvo ausente na base: {alvo}")
    if IDENTIFICADOR not in base.columns:
        raise ErroSimulacao(f"Coluna obrigatoria ausente: {IDENTIFICADOR}")

    features = base.drop(columns=[alvo])
    rotulos = base[[IDENTIFICADOR, alvo]]
    return features, rotulos

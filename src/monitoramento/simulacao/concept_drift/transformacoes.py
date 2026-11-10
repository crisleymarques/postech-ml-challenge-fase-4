"""Mudanca de conceito sem publicacao de arquivos."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import IDENTIFICADOR, OPERACAO, ErroSimulacao


def calcular_perigo(valores: pd.Series, mascara: np.ndarray, perigo: dict[str, Any]) -> np.ndarray:
    """Calcula a probabilidade extra de inadimplencia por registro do segmento.

    A probabilidade extra cresce com o excedente da variavel de risco em relacao
    ao valor de referencia, limitada em zero e no maximo declarado. Fora do
    segmento a probabilidade extra e zero.
    """
    referencia = float(perigo["reference_value"])
    inclinacao = float(perigo["slope"])
    escala = float(perigo["scale"])
    maximo = float(perigo["max_probability"])

    excedente = (valores - referencia).clip(lower=0.0) / escala
    bruto = (inclinacao * excedente).clip(lower=0.0, upper=maximo)

    return bruto.where(pd.Series(mascara, index=valores.index), 0.0).fillna(0.0).to_numpy()


def mascara_segmento(valores: pd.Series, segmento: dict[str, Any]) -> np.ndarray:
    """Devolve a mascara booleana dos registros pertencentes ao segmento."""
    limite = float(segmento["value"])
    operador = segmento["operator"]
    if operador == ">=":
        mascara = valores >= limite
    elif operador == ">":
        mascara = valores > limite
    elif operador == "<=":
        mascara = valores <= limite
    else:
        mascara = valores < limite
    return mascara.fillna(False).to_numpy()


def aplicar_mudanca_conceito(
    base: pd.DataFrame,
    alvo: str,
    positivo: int,
    conceito: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Altera a relacao perfil-desfecho no segmento e descreve a mudanca.

    Somente rotulos negativos dentro do segmento podem virar positivos, com
    probabilidade dada pela regra de perigo. A escolha dos registros usa uma
    semente fixa para ser reproduzivel.
    """
    segmento = conceito["segment"]
    coluna = segmento["column"]
    if coluna not in base.columns:
        raise ErroSimulacao(f"Coluna do segmento ausente na base: {coluna}")
    if alvo not in base.columns:
        raise ErroSimulacao(f"Coluna alvo ausente na base: {alvo}")
    if IDENTIFICADOR not in base.columns:
        raise ErroSimulacao(f"Coluna obrigatoria ausente: {IDENTIFICADOR}")

    rotulos = pd.to_numeric(base[alvo], errors="coerce")
    if rotulos.isna().any():
        raise ErroSimulacao("A base contem rotulos ausentes ou nao numericos.")

    positivos = int(positivo)
    valores = pd.to_numeric(base[coluna], errors="coerce")
    mascara = mascara_segmento(valores, segmento)
    perigo = calcular_perigo(valores, mascara, conceito["hazard"])

    gerador = np.random.default_rng(seed)
    sorteio = gerador.uniform(0.0, 1.0, size=len(base))

    elegiveis = (rotulos.to_numpy() != positivos) & mascara
    vira = elegiveis & (sorteio < perigo)

    novos = rotulos.to_numpy().astype(int).copy()
    novos[vira] = positivos

    alterada = base.copy()
    alterada[alvo] = novos.astype(str)

    taxa_antes = float((rotulos.to_numpy() == positivos).mean())
    taxa_depois = float((novos == positivos).mean())
    descricao = {
        "description": conceito["rationale"],
        "segment": {
            "column": coluna,
            "operator": segmento["operator"],
            "value": float(segmento["value"]),
        },
        "operation": OPERACAO,
        "parameters": {
            "reference_value": float(conceito["hazard"]["reference_value"]),
            "slope": float(conceito["hazard"]["slope"]),
            "scale": float(conceito["hazard"]["scale"]),
            "max_probability": float(conceito["hazard"]["max_probability"]),
            "positive_value": positivos,
        },
        "segment_rows": int(mascara.sum()),
        "flipped_rows": int(vira.sum()),
        "positive_rate_before": round(taxa_antes, 6),
        "positive_rate_after": round(taxa_depois, 6),
    }
    return alterada, descricao


def separar_features_rotulos(base: pd.DataFrame, alvo: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa as features dos rotulos, mantendo o identificador em ambos."""
    features = base.drop(columns=[alvo])
    rotulos = base[[IDENTIFICADOR, alvo]]
    return features, rotulos

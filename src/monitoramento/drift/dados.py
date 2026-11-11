"""Leitura e selecao das features comparaveis de drift."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def ler_features(caminho: Path, numericas: list[str], categoricas: list[str]) -> pd.DataFrame:
    """Le as features como texto e aplica os tipos declarados nos metadados."""
    base = pd.read_csv(caminho, dtype=str, na_filter=False).replace("", np.nan)
    for coluna in numericas:
        if coluna in base.columns:
            base[coluna] = pd.to_numeric(base[coluna], errors="coerce")
    for coluna in categoricas:
        if coluna in base.columns:
            base[coluna] = base[coluna].astype("object")
    return base


def selecionar_features(
    referencia: pd.DataFrame, atual: pd.DataFrame, colunas: list[str]
) -> list[str]:
    """Devolve as features comparaveis presentes nos dois conjuntos."""
    return [
        coluna for coluna in colunas if coluna in referencia.columns and coluna in atual.columns
    ]

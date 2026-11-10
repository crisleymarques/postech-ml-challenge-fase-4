"""Conversores usados pelo pipeline serializado de treinamento."""

from __future__ import annotations

from typing import Any

import pandas as pd


def converter_numericas(tabela: Any) -> Any:
    """Converte valores textuais em numericos; vazios ou invalidos viram ausentes."""
    return pd.DataFrame(tabela).apply(pd.to_numeric, errors="coerce").to_numpy()


def converter_categoricas(tabela: Any) -> Any:
    """Trata campo vazio como ausencia antes da imputacao categorica."""
    return pd.DataFrame(tabela).replace("", float("nan")).to_numpy()

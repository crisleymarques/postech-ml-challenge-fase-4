"""Testes da deteccao de Data Drift por responsabilidade."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from monitoramento.drift import dados


def test_ler_features_aplica_tipos(tmp_path: Path) -> None:
    caminho = tmp_path / "features.csv"
    pd.DataFrame(
        {
            "record_id": ["LC1", "LC2"],
            "annual_inc": ["60000.5", ""],
            "term": ["36 months", "60 months"],
        }
    ).to_csv(caminho, index=False, encoding="utf-8")

    base = dados.ler_features(caminho, ["annual_inc"], ["term"])

    assert pd.api.types.is_numeric_dtype(base["annual_inc"])
    assert np.isnan(base.loc[1, "annual_inc"])
    assert base["term"].dtype == object

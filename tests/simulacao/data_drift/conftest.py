"""Bases de teste da simulacao de Data Drift."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def base() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_id": ["LC0000001", "LC0000002", "LC0000003", "LC0000004"],
            "issue_d": ["Jul-2017"] * 4,
            "loan_status": ["0", "1", "0", "1"],
            "annual_inc": ["100000", "80000", "90000", "110000"],
            "dti": ["10", "12", "14", "16"],
            "revol_util": ["20", "30", "40", "50"],
            "fico_range_low": ["700"] * 4,
            "fico_range_high": ["704"] * 4,
        }
    )


@pytest.fixture
def config() -> dict:
    caminho = Path(__file__).resolve().parents[3] / "configs" / "simulacao.json"
    return json.loads(caminho.read_text(encoding="utf-8"))

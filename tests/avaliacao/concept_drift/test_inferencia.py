"""Contrato de inferencia preservado ao separar a avaliacao da predicao."""

from __future__ import annotations

import numpy as np
import pandas as pd

from monitoramento.avaliacao.concept_drift import inferencia


def test_prever_aplica_limiar_dos_metadados() -> None:
    class Pipeline:
        classes_ = np.array([0, 1])

        def predict(self, entrada):
            return np.zeros(len(entrada), dtype=int)

        def predict_proba(self, entrada):
            return np.array([[0.4, 0.6], [0.1, 0.9]])

    metadata = {
        "features": ["dti"],
        "target": {"column": "loan_status", "positive_value": 1},
        "prediction_threshold": 0.75,
    }
    lote = pd.DataFrame({"record_id": ["LC1", "LC2"], "dti": [10, 20]})

    predito, score = inferencia.prever(Pipeline(), metadata, lote)

    assert predito.tolist() == [0, 1]
    assert score.tolist() == [0.6, 0.9]

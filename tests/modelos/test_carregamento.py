"""Compatibilidade do carregador extraido com artefatos persistidos."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]


def test_carrega_conversor_legacy_serializado_em_main(tmp_path: Path) -> None:
    caminho = tmp_path / "pipeline.pkl"
    ambiente = os.environ.copy()
    ambiente["PYTHONPATH"] = str(RAIZ / "src")
    codigo = """
import joblib
from sklearn.preprocessing import FunctionTransformer
from monitoramento import treinamento
converter_numericas = treinamento.converter_numericas
converter_numericas.__module__ = '__main__'
joblib.dump(FunctionTransformer(converter_numericas), CAMINHO)
""".replace("CAMINHO", repr(str(caminho)))
    subprocess.run([sys.executable, "-c", codigo], env=ambiente, check=True)
    leitura = """
import numpy as np
from monitoramento.modelos.carregamento import carregar_pipeline
pipeline = carregar_pipeline(CAMINHO)
np.testing.assert_array_equal(pipeline.transform([["1", "2"]]), [[1, 2]])
""".replace("CAMINHO", repr(str(caminho)))
    subprocess.run([sys.executable, "-c", leitura], env=ambiente, check=True)

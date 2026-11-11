"""Carregamento de pipelines e compatibilidade com conversores persistidos."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib

CONVERSORES_DO_TREINO = ("converter_numericas", "converter_categoricas")


def registrar_conversores_do_treino() -> None:
    """Disponibiliza os conversores do treinamento no namespace corrente.

    Quando o treinamento roda como script, o pipeline serializa os conversores
    de pre-processamento pelo modulo `__main__`. Registrando esses nomes a
    partir do modulo de treinamento, o artefato ja persistido pode ser lido em
    outro processo sem qualquer alteracao no produtor.
    """
    namespace = sys.modules.get("__main__")
    if namespace is None:
        return
    from monitoramento import treinamento

    for nome in CONVERSORES_DO_TREINO:
        if not hasattr(namespace, nome):
            setattr(namespace, nome, getattr(treinamento, nome))


def carregar_pipeline(caminho: Path) -> Any:
    """Carrega o pipeline persistido, ajustando o namespace se necessario."""
    try:
        return joblib.load(caminho)
    except (AttributeError, ImportError):
        registrar_conversores_do_treino()
        return joblib.load(caminho)

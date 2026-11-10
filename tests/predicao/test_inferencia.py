"""Testes migrados da etapa de predicao."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sklearn.svm import LinearSVC

from monitoramento.predicao import config as cfg
from monitoramento.predicao import inferencia

from .apoio import (
    FEATURES,
    criar_pipeline,
    gerar_lote,
    montar_metadata,
)


def test_prever_rejeita_alvo_no_lote(tmp_path: Path) -> None:
    pipeline = criar_pipeline()
    pipeline.fit(
        gerar_lote(com_alvo=True)[FEATURES],
        pd.to_numeric(gerar_lote(com_alvo=True)["loan_status"]).astype(int),
    )

    with pytest.raises(cfg.ErroPredicao, match="nao deve conter o alvo"):
        inferencia.prever(pipeline, montar_metadata(), gerar_lote(com_alvo=True))


def test_prever_rejeita_features_ausentes() -> None:
    lote = gerar_lote().drop(columns=["dti", "term"])
    pipeline = criar_pipeline()

    with pytest.raises(cfg.ErroPredicao, match="Features ausentes"):
        inferencia.prever(pipeline, montar_metadata(), lote)


def test_prever_rejeita_identificador_repetido(tmp_path: Path) -> None:
    pipeline = criar_pipeline()
    treino = gerar_lote(com_alvo=True)
    pipeline.fit(treino[FEATURES], pd.to_numeric(treino["loan_status"]).astype(int))
    lote = gerar_lote()
    lote.loc[1, "record_id"] = lote.loc[0, "record_id"]

    with pytest.raises(cfg.ErroPredicao, match="identificadores repetidos"):
        inferencia.prever(pipeline, montar_metadata(), lote)


def test_prever_marca_score_indisponivel_sem_probabilidade(tmp_path: Path) -> None:
    lote = gerar_lote()
    pipeline = criar_pipeline(LinearSVC(max_iter=5000))
    treino = gerar_lote(com_alvo=True)
    pipeline.fit(treino[FEATURES], pd.to_numeric(treino["loan_status"]).astype(int))

    predicoes, score_disponivel = inferencia.prever(pipeline, montar_metadata(), lote)

    assert score_disponivel is False
    assert predicoes["score"].isna().all()
    assert list(predicoes.columns) == list(inferencia.COLUNAS_SAIDA)

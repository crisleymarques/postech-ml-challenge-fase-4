"""Aprovacao de producao e vinculo com os bytes consumidos pela inferencia."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from monitoramento import validacao
from monitoramento.infraestrutura.arquivos import sha256_arquivo
from monitoramento.predicao import servico

from .apoio import gerar_lote, montar_config, preparar_manifestos


@pytest.mark.parametrize(
    "caso",
    [
        "ausente",
        "rejeitada",
        "nao_aprovada",
        "hash",
        "path",
        "producer",
        "dataset_id",
        "schema_version",
        "scenario_id",
        "batch_id",
        "resumo",
        "duplicado",
        "extra",
        "alterado",
    ],
)
def test_aprovacao_incompativel_bloqueia_inferencia(tmp_path, monkeypatch, caso):
    manifestos = preparar_manifestos(tmp_path)
    aprovacao = manifestos[2]
    if caso == "ausente":
        manifestos.pop()
    elif caso == "rejeitada":
        aprovacao["status"] = "rejected"
    elif caso == "nao_aprovada":
        aprovacao["quality"]["approved"] = False
    elif caso in ("dataset_id", "schema_version", "scenario_id", "batch_id"):
        aprovacao[caso] = "outro"
    elif caso == "hash":
        aprovacao["inputs"][0]["sha256"] = "0" * 64
    elif caso == "path":
        aprovacao["inputs"][0]["path"] = str(tmp_path / "outro.csv")
    elif caso == "producer":
        aprovacao["inputs"][0]["producer_run_id"] = "outro"
    elif caso == "resumo":
        aprovacao["quality"]["artifacts"][0]["approved"] = False
    elif caso == "duplicado":
        manifestos.append(copy.deepcopy(aprovacao))
    elif caso == "extra":
        manifestos.append({"stage": "outra", "status": "success", "contract_version": "1.0"})
    elif caso == "alterado":
        with (tmp_path / "features.csv").open("a") as arquivo:
            arquivo.write("\n")
    monkeypatch.setattr(servico, "carregar_pipeline", lambda _: pytest.fail("nao pode inferir"))
    output = tmp_path / "run"
    assert (
        servico.processar(
            tmp_path / "config.json",
            montar_config(),
            manifestos,
            "bloqueado",
            output,
            analysis_dir=tmp_path / "analysis",
        )
        == 1
    )
    manifesto = json.loads((output / "manifest.json").read_text())
    assert manifesto["status"] == "failed"
    assert manifesto["outputs"] == []
    assert not (tmp_path / "analysis").exists()


def aprovar_lote(tmp_path: Path, lote: pd.DataFrame, manifestos: list[dict]) -> dict:
    """Executa o produtor real de validacao com tres regras rigidas."""
    caminho = tmp_path / "features.csv"
    lote.to_csv(caminho, index=False)
    item = manifestos[1]["outputs"][0]
    item.update(sha256=sha256_arquivo(caminho), row_count=len(lote), columns=list(lote.columns))
    config = {
        "contract_version": "1.0",
        "stage": "validacao",
        "profile": "production",
        "dataset_id": "teste",
        "schema_version": "1.0",
        "artifacts": ["features"],
        "rules": {
            "identifier": {"column": "record_id", "pattern": "^LC[0-9]{7}$"},
            "target": {"column": "loan_status", "domain": [0, 1]},
            "not_null": ["record_id", "annual_inc", "term"],
            "unique": ["record_id"],
            "ranges": {"annual_inc": [0, 10000000]},
            "categories": {"term": ["36 months", "60 months"]},
        },
    }
    output = tmp_path / "validacao"
    output.mkdir()
    validacao.processar(tmp_path / "config-val.json", config, manifestos[1], "val-real", output)
    return json.loads((output / "manifest.json").read_text())


@pytest.mark.parametrize("erro", [None, "renda", "nulo", "duplicado"])
def test_predicao_respeita_pandera_real(tmp_path: Path, monkeypatch, erro):
    manifestos = preparar_manifestos(tmp_path)
    lote = gerar_lote()
    if erro == "renda":
        lote.loc[0, "annual_inc"] = "-1"
    elif erro == "nulo":
        lote.loc[0, "annual_inc"] = ""
    elif erro == "duplicado":
        lote.loc[0, "record_id"] = lote.loc[1, "record_id"]
    manifestos[2] = aprovar_lote(tmp_path, lote, manifestos)
    if erro:
        assert manifestos[2]["status"] == "rejected"
        monkeypatch.setattr(servico, "carregar_pipeline", lambda _: pytest.fail("lote rejeitado"))
    output = tmp_path / "predicao"
    codigo = servico.processar(
        tmp_path / "config.json",
        montar_config(),
        list(reversed(manifestos)),
        "pred-real",
        output,
        analysis_dir=tmp_path / "analysis",
    )
    assert codigo == (1 if erro else 0)
    manifesto = json.loads((output / "manifest.json").read_text())
    assert manifesto["status"] == ("failed" if erro else "success")
    if not erro:
        assert manifesto["quality_validation_run_id"] == "val-real"

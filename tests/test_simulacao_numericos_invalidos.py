"""Regressoes para impedir que a simulacao oculte numeros invalidos."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from monitoramento import simulacao_drift as sd
from monitoramento import validacao


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
    caminho = Path(__file__).resolve().parents[1] / "configs" / "simulacao.json"
    return json.loads(caminho.read_text(encoding="utf-8"))


@pytest.mark.parametrize("ausente", ["", None, np.nan, pd.NA])
def test_aplicar_drift_preserva_apenas_ausencias_originais(base, config, ausente) -> None:
    base.loc[0, "dti"] = ausente

    ajustada, descritas = sd.aplicar_drift(base, config["drift"]["variables"], 42, 0.05)

    assert ajustada.at[0, "dti"] == ""
    assert ajustada.loc[1:, "dti"].astype(float).gt(base.loc[1:, "dti"].astype(float)).all()
    descricao = next(item for item in descritas if item["column"] == "dti")
    assert descricao["missing_preserved"] == 1


@pytest.mark.parametrize("coluna", ["dti", "revol_util"])
@pytest.mark.parametrize("invalido", ["dado-sigiloso", "NaN", "inf", "-inf", " "])
def test_numero_invalido_nao_vira_ausencia_aprovada(
    tmp_path: Path, capsys, base, config, coluna: str, invalido: str
) -> None:
    base.loc[0, coluna] = invalido
    origem = tmp_path / "origem.csv"
    base.to_csv(origem, index=False)
    hash_origem = sd.sha256_arquivo(origem)
    # Estas features permitem ausencia, mas devem rejeitar conteudo invalido.
    contrato = validacao.construir_contrato(
        {
            "profile": "production",
            "rules": {
                "identifier": {"column": "record_id", "pattern": "^LC[0-9]{7}$"},
                "target": {"column": "loan_status", "domain": [0, 1]},
                "not_null": ["record_id"],
                "unique": ["record_id"],
                "ranges": {"dti": [0, 500], "revol_util": [0, 1000]},
                "categories": {},
            },
        }
    )
    _, violacoes = validacao.validar_artefato(origem, contrato)
    # O leitor da validacao considera "NaN" ausencia; a simulacao le o texto
    # literal com na_filter=False e tambem deve recusar esse numero nao finito.
    if invalido != "NaN":
        assert violacoes
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output_dir = tmp_path / "run"

    codigo = sd.gerar(
        config_path=config_path,
        source_path=origem,
        analysis_dir=tmp_path / "analysis",
        batch_id="invalid-numeric",
        run_id="invalid-numeric",
        output_dir=output_dir,
        seed=None,
    )

    assert codigo == sd.CODIGO_ERRO
    manifesto = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifesto["status"] == "failed"
    assert manifesto["error_code"] == "ERRO_ENTRADA"
    assert manifesto["outputs"] == []
    assert not list((tmp_path / "analysis").rglob("*.csv"))
    eventos = [
        json.loads(linha)
        for linha in (output_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [evento["event"] for evento in eventos] == ["started", "failed"]
    assert eventos[-1]["level"] == "ERROR"
    assert eventos[-1]["error_code"] == "ERRO_ENTRADA"
    assert json.loads((output_dir / "metrics.json").read_text())["measurements"] == []
    assert sd.sha256_arquivo(origem) == hash_origem
    saida = capsys.readouterr()
    assert saida.out == ""
    assert "ERRO_ENTRADA" in saida.err
    assert "dado-sigiloso" not in saida.err
    assert "dado-sigiloso" not in (output_dir / "events.jsonl").read_text()

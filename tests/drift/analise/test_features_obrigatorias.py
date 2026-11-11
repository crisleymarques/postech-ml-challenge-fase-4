"""Regressoes de cobertura das features esperadas na analise."""
import json
from pathlib import Path

import pandas as pd
import pytest

from monitoramento.drift.analise import servico
from monitoramento.infraestrutura.arquivos import sha256_arquivo

from .apoio import montar_config, preparar_manifestos


@pytest.mark.parametrize("papel", [0, 1], ids=["referencia", "lote"])
def test_feature_ausente_bloqueia_analise(tmp_path: Path, papel: int, capsys) -> None:
    config = montar_config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    manifestos = preparar_manifestos(tmp_path)
    item = manifestos[papel]["outputs"][0]
    caminho = Path(item["path"])
    pd.read_csv(caminho).drop(columns=["annual_inc"]).to_csv(caminho, index=False)
    item["sha256"] = sha256_arquivo(caminho)
    output = tmp_path / "run"

    assert servico.processar(config_path, config, manifestos, "run", output) == 1
    assert not (output / "analise.json").exists()
    mensagem = capsys.readouterr().out
    assert "annual_inc" in mensagem
    assert ("referencia" if papel == 0 else "lote") in mensagem

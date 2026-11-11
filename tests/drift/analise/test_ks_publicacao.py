"""K-S disponivel e colunas nao avaliaveis na publicacao completa."""

import json
from pathlib import Path

import pandas as pd

from monitoramento.drift.analise import servico
from monitoramento.infraestrutura.arquivos import sha256_arquivo

from .apoio import montar_config, preparar_manifestos


def test_ks_sem_psi_emite_alerta_e_nao_marca_ausente_como_ok(tmp_path: Path, capsys) -> None:
    config = montar_config()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    manifestos = preparar_manifestos(tmp_path)
    for indice, manifesto in enumerate(manifestos):
        item = manifesto["outputs"][0]
        csv = Path(item["path"])
        base = pd.read_csv(csv)
        base["annual_inc"] = indice
        base["dti"] = None
        base.to_csv(csv, index=False)
        item["sha256"] = sha256_arquivo(csv)
    output = tmp_path / "run"
    assert servico.processar(path, config, manifestos, "run", output) == 0
    doc = json.loads((output / "analise.json").read_text())
    annual = next(f for f in doc["features"] if f["feature"] == "annual_inc")
    assert annual["psi"] is None
    assert annual["ks_statistic"] == 1.0
    assert annual["drift_detected"] is True
    assert "annual_inc" in doc["changed_variables"]["confirmed"]
    assert doc["summary"]["evaluated_columns"] == 4
    eventos = [json.loads(linha) for linha in (output / "events.jsonl").read_text().splitlines()]
    assert any(e["event"] == "drift_alert" for e in eventos)
    assert "[nao avaliavel] dti" in capsys.readouterr().out

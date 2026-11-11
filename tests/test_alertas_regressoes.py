"""Regressoes de consistencia e validacao da integracao F3-03."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from monitoramento import alertas, logs, metricas_saude

RAIZ = Path(__file__).resolve().parents[1]
CONFIG = RAIZ / "configs/alertas.json"


def medida(identificador: str, valor: object, situacao: str = "ok") -> dict:
    return {
        "id": identificador,
        "categoria": "drift",
        "nome": identificador,
        "unidade": "proporcao",
        "valor": valor,
        "situacao": situacao,
        "disponibilidade": "available" if valor is not None else "not_available",
    }


def execucao(run_id: str, presente: bool | None, quantidade: int | None, proporcao) -> dict:
    return {
        "run_id": run_id,
        "stage": "deteccao",
        "status": "success",
        "metricas": [
            medida("drift.dataset_drift", presente),
            medida("drift.variaveis_com_drift", quantidade),
            medida("drift.proporcao_drift", proporcao),
        ],
    }


@pytest.fixture
def fontes() -> tuple[dict, dict]:
    return (
        {"contract_version": "1.0", "execucoes": [], "agregados": []},
        {
            "contract_version": "1.0",
            "execucoes": {"total": 0, "por_status": {}, "incompletas": []},
            "eventos": {"total": 0},
            "falhas": [],
            "alertas": [],
        },
    )


def gravar_fontes(tmp_path: Path, fontes: tuple[dict, dict]) -> tuple[Path, Path]:
    caminhos = (tmp_path / "saude.json", tmp_path / "logs.json")
    for caminho, conteudo in zip(caminhos, fontes):
        caminho.write_text(json.dumps(conteudo), encoding="utf-8")
    return caminhos


def test_drift_independe_da_ordem_e_preserva_a_execucao(fontes) -> None:
    saude, _ = fontes
    com_drift = execucao("a-drift", True, 13, 0.59)
    controle = execucao("z-controle", False, 0, 0.0)
    saude["execucoes"] = [com_drift, controle]
    resultado = alertas.consolidar_drift(saude)
    saude["execucoes"].reverse()
    assert alertas.consolidar_drift(saude) == resultado
    assert resultado["dataset_drift"] is True
    assert resultado["variaveis_com_drift"] == 13
    assert resultado["proporcao_drift"] == 0.59
    assert resultado["execucao_resumo"] == "a-drift"
    assert {item["run_id"]: item["dataset_drift"] for item in resultado["por_execucao"]} == {
        "a-drift": True,
        "z-controle": False,
    }


def test_resumo_nao_combina_maximos_de_deteccao_e_analise(fontes) -> None:
    saude, _ = fontes
    analise = execucao("analise", None, 2, 1.0)
    analise["stage"] = "analise"
    analise["metricas"] = analise["metricas"][1:]
    saude["execucoes"] = [execucao("deteccao", True, 13, 0.59), analise]
    resultado = alertas.consolidar_drift(saude)
    assert resultado["execucao_resumo"] == "analise"
    assert resultado["dataset_drift"] is None
    assert resultado["variaveis_com_drift"] == 2
    assert resultado["proporcao_drift"] == 1.0
    assert resultado["por_execucao"][0]["stage"] == "analise"


def test_drift_sem_resultados_permanece_indisponivel(fontes) -> None:
    saude, _ = fontes
    saude["execucoes"] = [execucao("incompleta", None, None, None)]
    resultado = alertas.consolidar_drift(saude)
    assert resultado["execucao_resumo"] is None
    assert resultado["dataset_drift"] is None
    assert resultado["variaveis_com_drift"] is None
    assert resultado["proporcao_drift"] is None


@pytest.mark.parametrize(
    "mapa",
    [
        {"critico": "critical"},
        {"atencao": "warning"},
        {"atencao": "warning", "critico": "critcal"},
        {"atencao": None, "critico": "critical"},
        {"atencao": [], "critico": "critical"},
        {"atencao": "warning", "critico": "critical", "ok": "warning"},
    ],
)
def test_config_rejeita_mapeamento_que_perde_ou_inventa_alertas(tmp_path, mapa) -> None:
    config = json.loads(CONFIG.read_text())
    config["severidade_por_situacao"] = mapa
    caminho = tmp_path / "config.json"
    caminho.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="severidade_por_situacao"):
        alertas.carregar_config(caminho)


@pytest.mark.parametrize("severidades", [["critical"], ["warning", "critical", "urgente"]])
def test_config_rejeita_severidades_nao_suportadas(tmp_path, severidades) -> None:
    config = json.loads(CONFIG.read_text())
    config["severidades"] = severidades
    caminho = tmp_path / "config.json"
    caminho.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="severidades"):
        alertas.carregar_config(caminho)


@pytest.mark.parametrize("indice", [0, 1], ids=["saude", "logs"])
@pytest.mark.parametrize("versao", [None, "2.0", "1.invalida", 1])
def test_geracao_rejeita_versao_ausente_ou_invalida(tmp_path, fontes, indice, versao) -> None:
    fontes[indice]["contract_version"] = versao
    caminhos = gravar_fontes(tmp_path, fontes)
    saida = tmp_path / "saida"
    with pytest.raises(ValueError, match="contract_version"):
        alertas.gerar(*caminhos, CONFIG, saida)
    assert not saida.exists()


@pytest.mark.parametrize(
    "indice,chave,valor",
    [
        (0, "execucoes", {}),
        (0, "execucoes", [None]),
        (0, "agregados", None),
        (0, "agregados", [False]),
        (1, "falhas", {}),
        (1, "falhas", [None]),
        (1, "alertas", None),
        (1, "alertas", [False]),
        (1, "execucoes", []),
        (1, "eventos", []),
    ],
)
def test_geracao_rejeita_estrutura_invalida(tmp_path, fontes, indice, chave, valor) -> None:
    fontes[indice][chave] = valor
    caminhos = gravar_fontes(tmp_path, fontes)
    with pytest.raises(ValueError, match=chave):
        alertas.gerar(*caminhos, CONFIG, tmp_path / "saida")
    assert not (tmp_path / "saida").exists()


@pytest.mark.parametrize(
    "chave,valor",
    [
        ("id", None),
        ("situacao", "critcal"),
        ("valor", []),
        ("disponibilidade", "talvez"),
    ],
)
def test_geracao_nao_ignora_medida_malformada(tmp_path, fontes, chave, valor) -> None:
    saude, _ = fontes
    item = medida("drift.proporcao_drift", 0.8, "critico")
    item[chave] = valor
    saude["agregados"] = [item]
    caminhos = gravar_fontes(tmp_path, fontes)
    with pytest.raises(ValueError, match=chave):
        alertas.gerar(*caminhos, CONFIG, tmp_path / "saida")


@pytest.mark.parametrize("indice", [0, 1], ids=["saude", "logs"])
def test_cli_rejeita_objeto_vazio_sem_sobrescrever_relatorio(tmp_path, fontes, indice) -> None:
    entradas = list(deepcopy(fontes))
    entradas[indice] = {}
    caminhos = gravar_fontes(tmp_path, tuple(entradas))
    saida = tmp_path / "saida"
    saida.mkdir()
    anterior = saida / "alertas.json"
    anterior.write_text('{"anterior":true}')
    resultado = subprocess.run(
        [
            sys.executable,
            str(RAIZ / "src/monitoramento/alertas.py"),
            "--saude",
            str(caminhos[0]),
            "--logs-resumo",
            str(caminhos[1]),
            "--config",
            str(CONFIG),
            "--output-dir",
            str(saida),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert resultado.returncode == 1
    assert "Falha ao integrar" in resultado.stdout
    assert "Traceback" not in resultado.stderr
    assert anterior.read_text() == '{"anterior":true}'


def test_relatorios_reais_dos_produtores_vazios_sao_validos(tmp_path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    logs_dir, saude_dir = tmp_path / "logs", tmp_path / "saude"
    logs.gerar(runs, logs_dir)
    metricas_saude.gerar(runs, saude_dir, RAIZ / "configs/metricas_saude.json")
    resultado = alertas.gerar(
        saude_dir / "saude.json",
        logs_dir / "resumo.json",
        CONFIG,
        tmp_path / "saida",
    )
    assert resultado["resumo"]["total"] == 0
    assert resultado["drift"]["dataset_drift"] is None
    assert resultado["drift"]["execucoes"] == []


def test_integra_contratos_dos_produtores_com_drift_e_falha(tmp_path) -> None:
    runs = tmp_path / "runs"
    for run_id, stage, status in (
        ("run-drift", "deteccao", "success"),
        ("run-falha", "validacao", "failed"),
    ):
        destino = runs / run_id
        destino.mkdir(parents=True)
        identidade = {"run_id": run_id, "stage": stage}
        manifesto = {
            "contract_version": "1.0",
            **identidade,
            "status": status,
            "started_at": "2026-11-09T00:00:00Z",
            "finished_at": "2026-11-09T00:00:01Z",
            "inputs": [{"row_count": 10}],
            "error_code": "FALHA_TESTE" if status == "failed" else None,
        }
        (destino / "manifest.json").write_text(json.dumps(manifesto))
        eventos = []
        if status == "success":
            (destino / "drift.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "dataset_drift": True,
                            "drifted_columns": ["renda", "idade"],
                            "drift_share": 0.5,
                        }
                    }
                )
            )
            eventos.append({**identidade, "event": "drift_alert", "level": "WARNING"})
        eventos.append(
            {
                **identidade,
                "event": "completed" if status == "success" else "failed",
                "level": "INFO" if status == "success" else "ERROR",
                "error_code": manifesto["error_code"],
            }
        )
        (destino / "events.jsonl").write_text("\n".join(json.dumps(e) for e in eventos))

    logs.gerar(runs, tmp_path / "logs")
    metricas_saude.gerar(runs, tmp_path / "saude", RAIZ / "configs/metricas_saude.json")
    resultado = alertas.gerar(
        tmp_path / "saude/saude.json",
        tmp_path / "logs/resumo.json",
        CONFIG,
        tmp_path / "saida",
    )
    assert resultado["drift"]["dataset_drift"] is True
    assert resultado["drift"]["variaveis_com_drift"] == 2
    assert resultado["drift"]["alertas_total"] == 1
    falha = next(a for a in resultado["alertas"] if a["error_code"] == "FALHA_TESTE")
    assert falha["origem"] == "run-falha"
    assert falha["severidade"] == "critical"
    assert resultado["resumo"]["total"] == sum(resultado["resumo"]["por_severidade"].values())

"""Testes da integracao de drift, metricas e alertas (issue F3-03)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from monitoramento import alertas as al

CONFIG_REAL = Path(__file__).resolve().parents[1] / "configs" / "alertas.json"


def medida(
    id_metrica: str,
    categoria: str,
    situacao: str,
    valor: object = 1,
    unidade: str = "ocorrencias",
    nome: str = "Medida de teste",
) -> dict:
    """Medida de saude no formato publicado pela F3-02."""
    return {
        "id": id_metrica,
        "categoria": categoria,
        "nome": nome,
        "unidade": unidade,
        "valor": valor,
        "disponibilidade": "available",
        "situacao": situacao,
    }


def saude(execucoes: list[dict], agregados: list[dict] | None = None) -> dict:
    """Relatorio de saude minimo conforme a F3-02."""
    return {
        "contract_version": "1.0",
        "execucoes": execucoes,
        "agregados": agregados or [],
    }


def resumo_logs(falhas: list[dict] | None = None, alertas_drift: list[dict] | None = None) -> dict:
    """Resumo de logs minimo conforme a F3-01."""
    return {
        "contract_version": "1.0",
        "execucoes": {"total": 1, "por_status": {"success": 1}, "incompletas": []},
        "eventos": {"total": 1},
        "falhas": falhas or [],
        "alertas": alertas_drift or [],
    }


def config_minimo(**campos: object) -> dict:
    """Configuracao de alertas valida, com ajustes opcionais."""
    base = {
        "contract_version": "1.0",
        "severidades": ["warning", "critical"],
        "severidade_por_situacao": {"atencao": "warning", "critico": "critical"},
        "severidade_falha": "critical",
        "severidade_alerta_drift": "warning",
    }
    base.update(campos)
    return base


def escrever_config(tmp_path: Path, config: dict) -> Path:
    caminho = tmp_path / "alertas.json"
    caminho.write_text(json.dumps(config), encoding="utf-8")
    return caminho


def test_carregar_config_real() -> None:
    config = al.carregar_config(CONFIG_REAL)
    assert config["severidade_por_situacao"]["critico"] == "critical"
    assert "critical" in config["severidades"]


def test_carregar_config_ausente(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nao encontrada"):
        al.carregar_config(tmp_path / "inexistente.json")


def test_carregar_config_versao_incompativel(tmp_path: Path) -> None:
    caminho = escrever_config(tmp_path, config_minimo(contract_version="2.0"))
    with pytest.raises(ValueError, match="incompativel"):
        al.carregar_config(caminho)


def test_carregar_config_sem_severidades(tmp_path: Path) -> None:
    caminho = escrever_config(tmp_path, config_minimo(severidades=[]))
    with pytest.raises(ValueError, match="severidades"):
        al.carregar_config(caminho)


def test_carregar_config_sem_critico(tmp_path: Path) -> None:
    caminho = escrever_config(tmp_path, config_minimo(severidades=["warning"]))
    with pytest.raises(ValueError, match="critical"):
        al.carregar_config(caminho)


def test_carregar_config_mapa_invalido(tmp_path: Path) -> None:
    caminho = escrever_config(
        tmp_path, config_minimo(severidade_por_situacao={"atencao": "warning"})
    )
    with pytest.raises(ValueError, match="severidade_por_situacao"):
        al.carregar_config(caminho)


def test_carregar_config_severidade_falha_invalida(tmp_path: Path) -> None:
    caminho = escrever_config(tmp_path, config_minimo(severidade_falha="urgente"))
    with pytest.raises(ValueError, match="severidade_falha"):
        al.carregar_config(caminho)


def test_ler_json_ausente(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nao encontrado"):
        al.ler_json(tmp_path / "saude.json", "Relatorio de saude")


def test_ler_json_invalido(tmp_path: Path) -> None:
    caminho = tmp_path / "saude.json"
    caminho.write_text("{nao e json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalido"):
        al.ler_json(caminho, "Relatorio de saude")


def test_alertas_de_medidas_por_situacao() -> None:
    relatorio = saude(
        [
            {
                "run_id": "run-a",
                "stage": "validacao",
                "status": "rejected",
                "metricas": [
                    medida("pipeline.status_execucao", "pipeline", "atencao", "rejected"),
                    medida("qualidade.violacoes_total", "qualidade", "ok", 0),
                ],
            },
            {
                "run_id": "run-b",
                "stage": "deteccao",
                "status": "success",
                "metricas": [
                    medida("drift.proporcao_drift", "drift", "critico", 0.59, "proporcao")
                ],
            },
        ]
    )
    mapa = {"atencao": "warning", "critico": "critical"}
    alertas = al.alertas_de_medidas(relatorio, mapa)

    assert len(alertas) == 2
    assert alertas[0]["severidade"] == "warning"
    assert alertas[0]["origem"] == "run-a"
    assert alertas[0]["fonte"] == "saude"
    assert alertas[1]["severidade"] == "critical"
    assert alertas[1]["categoria"] == "drift"


def test_alertas_de_medidas_ignora_ok() -> None:
    relatorio = saude(
        [
            {
                "run_id": "run-a",
                "stage": "validacao",
                "status": "success",
                "metricas": [medida("pipeline.duracao_s", "pipeline", "ok", 1.0, "segundos")],
            }
        ]
    )
    assert al.alertas_de_medidas(relatorio, {"atencao": "warning", "critico": "critical"}) == []


def test_alertas_de_falhas() -> None:
    resumo = resumo_logs(
        falhas=[
            {
                "run_id": "run-f104-validacao",
                "stage": "validacao",
                "categoria": "validacao",
                "event": "rejected",
                "level": "WARNING",
                "error_code": "QUALIDADE_REPROVADA",
                "timestamp": "2026-11-09T00:00:00.000000Z",
            }
        ]
    )
    alertas = al.alertas_de_falhas(resumo, "critical")

    assert len(alertas) == 1
    alerta = alertas[0]
    assert alerta["severidade"] == "critical"
    assert alerta["error_code"] == "QUALIDADE_REPROVADA"
    assert alerta["fonte"] == "logs"
    assert "QUALIDADE_REPROVADA" in alerta["descricao"]


def test_alertas_de_drift() -> None:
    resumo = resumo_logs(
        alertas_drift=[
            {
                "run_id": "run-f203-deteccao",
                "stage": "deteccao",
                "scenario_id": "cenario-01",
                "batch_id": "lote-01",
                "timestamp": "2026-11-09T00:00:00.000000Z",
            }
        ]
    )
    alertas = al.alertas_de_drift(resumo, "warning")

    assert len(alertas) == 1
    assert alertas[0]["severidade"] == "warning"
    assert alertas[0]["categoria"] == "drift"
    assert alertas[0]["evidencia"]["scenario_id"] == "cenario-01"


def test_consolidar_drift() -> None:
    relatorio = saude(
        [
            {
                "run_id": "run-b",
                "stage": "deteccao",
                "status": "success",
                "metricas": [
                    medida("drift.dataset_drift", "drift", "atencao", True, "booleano"),
                    medida("drift.variaveis_com_drift", "drift", "atencao", 13, "variaveis"),
                    medida("drift.proporcao_drift", "drift", "critico", 0.59, "proporcao"),
                ],
            }
        ],
        agregados=[medida("drift.alertas_total", "drift", "ok", 1, "alertas")],
    )
    drift = al.consolidar_drift(relatorio)

    assert drift["dataset_drift"] is True
    assert drift["variaveis_com_drift"] == 13
    assert drift["proporcao_drift"] == 0.59
    assert drift["alertas_total"] == 1
    assert drift["execucoes"] == ["run-b"]


def test_consolidar_drift_sem_dados() -> None:
    drift = al.consolidar_drift(saude([]))
    assert drift["dataset_drift"] is None
    assert drift["variaveis_com_drift"] is None
    assert drift["alertas_total"] is None
    assert drift["execucoes"] == []


def test_montar_resumo() -> None:
    alertas = [
        {"severidade": "warning", "categoria": "drift", "fonte": "saude", "id": "a"},
        {"severidade": "critical", "categoria": "drift", "fonte": "saude", "id": "b"},
        {"severidade": "critical", "categoria": "validacao", "fonte": "logs", "id": "c"},
    ]
    resumo = al.montar_resumo(alertas)

    assert resumo["total"] == 3
    assert resumo["por_severidade"] == {"warning": 1, "critical": 2}
    assert resumo["por_categoria"] == {"drift": 2, "validacao": 1}
    assert resumo["por_origem"] == {"saude": 2, "logs": 1}
    assert resumo["criticos"] == ["b", "c"]


def test_gerar_ponta_a_ponta(tmp_path: Path) -> None:
    saude_path = tmp_path / "saude.json"
    logs_path = tmp_path / "resumo.json"
    saude_path.write_text(
        json.dumps(
            saude(
                [
                    {
                        "run_id": "run-validacao",
                        "stage": "validacao",
                        "status": "rejected",
                        "metricas": [
                            medida(
                                "pipeline.status_execucao",
                                "pipeline",
                                "atencao",
                                "rejected",
                                "categoria",
                            ),
                            medida("qualidade.violacoes_total", "qualidade", "atencao", 7),
                        ],
                    },
                    {
                        "run_id": "run-deteccao",
                        "stage": "deteccao",
                        "status": "success",
                        "metricas": [
                            medida("drift.dataset_drift", "drift", "atencao", True, "booleano"),
                            medida(
                                "drift.variaveis_com_drift", "drift", "atencao", 13, "variaveis"
                            ),
                            medida("drift.proporcao_drift", "drift", "critico", 0.59, "proporcao"),
                        ],
                    },
                ],
                agregados=[medida("drift.alertas_total", "drift", "ok", 1, "alertas")],
            )
        ),
        encoding="utf-8",
    )
    logs_path.write_text(
        json.dumps(
            resumo_logs(
                falhas=[
                    {
                        "run_id": "run-validacao",
                        "stage": "validacao",
                        "categoria": "validacao",
                        "event": "rejected",
                        "error_code": "QUALIDADE_REPROVADA",
                        "timestamp": "2026-11-09T00:00:00.000000Z",
                    }
                ],
                alertas_drift=[
                    {
                        "run_id": "run-deteccao",
                        "stage": "deteccao",
                        "scenario_id": "cenario-01",
                        "batch_id": "lote-01",
                        "timestamp": "2026-11-09T00:00:00.000000Z",
                    }
                ],
            )
        ),
        encoding="utf-8",
    )

    saida = tmp_path / "alertas"
    relatorio = al.gerar(saude_path, logs_path, CONFIG_REAL, saida)

    assert relatorio["resumo"]["total"] == 7
    assert relatorio["resumo"]["por_severidade"] == {"warning": 5, "critical": 2}
    assert relatorio["drift"]["dataset_drift"] is True
    assert relatorio["drift"]["variaveis_com_drift"] == 13
    assert relatorio["drift"]["alertas_total"] == 1
    assert (saida / "alertas.json").is_file()

    gravado = json.loads((saida / "alertas.json").read_text(encoding="utf-8"))
    assert gravado["resumo"]["total"] == 7
    assert len(gravado["alertas"]) == 7
    assert len(gravado["resumo"]["criticos"]) == 2


def test_gerar_saude_ausente(tmp_path: Path) -> None:
    logs_path = tmp_path / "resumo.json"
    logs_path.write_text(json.dumps(resumo_logs()), encoding="utf-8")
    with pytest.raises(ValueError, match="nao encontrado"):
        al.gerar(tmp_path / "saude.json", logs_path, CONFIG_REAL, tmp_path / "alertas")


def test_gerar_logs_ausente(tmp_path: Path) -> None:
    saude_path = tmp_path / "saude.json"
    saude_path.write_text(json.dumps(saude([])), encoding="utf-8")
    with pytest.raises(ValueError, match="nao encontrado"):
        al.gerar(saude_path, tmp_path / "resumo.json", CONFIG_REAL, tmp_path / "alertas")


def test_ids_de_alerta_sao_unicos() -> None:
    relatorio = saude(
        [
            {
                "run_id": "run-a",
                "stage": "validacao",
                "status": "rejected",
                "metricas": [medida("qualidade.violacoes_total", "qualidade", "atencao", 7)],
            }
        ]
    )
    resumo = resumo_logs(
        falhas=[
            {
                "run_id": "run-a",
                "stage": "validacao",
                "event": "rejected",
                "error_code": "QUALIDADE_REPROVADA",
            }
        ],
        alertas_drift=[{"run_id": "run-b", "stage": "deteccao"}],
    )
    alertas = (
        al.alertas_de_medidas(relatorio, {"atencao": "warning", "critico": "critical"})
        + al.alertas_de_falhas(resumo, "critical")
        + al.alertas_de_drift(resumo, "warning")
    )
    ids = [alerta["id"] for alerta in alertas]
    assert len(ids) == len(set(ids))

"""Testes da deteccao de Data Drift por responsabilidade."""

from __future__ import annotations

from monitoramento.drift.deteccao import adaptador_evidently

from .apoio import FEATURES, SnapshotFalso, gerar_features, metrica_drift


def test_montar_resultados_marca_variavel_nao_avaliavel() -> None:
    snapshot = SnapshotFalso([metrica_drift("annual_inc", 0.001)])

    resultados = adaptador_evidently.montar_resultados(snapshot, ["annual_inc", "dti"], 0.05, 0.1)

    por_feature = {item["feature"]: item for item in resultados}
    assert por_feature["annual_inc"]["drift_detected"] is True
    assert por_feature["annual_inc"]["threshold"] == 0.05
    assert por_feature["dti"]["drift_detected"] is None
    assert por_feature["dti"]["statistic"] is None
    assert "reason" in por_feature["dti"]


def test_montar_resultados_marca_nulo_como_nao_avaliavel() -> None:
    snapshot = SnapshotFalso([metrica_drift("annual_inc", None)])

    resultados = adaptador_evidently.montar_resultados(snapshot, ["annual_inc"], 0.05, 0.1)

    assert resultados[0]["drift_detected"] is None
    assert "nao avaliavel" in resultados[0]["reason"]


def test_adaptador_real_executa() -> None:
    referencia = gerar_features(deslocado=False)[FEATURES]
    atual = gerar_features(deslocado=True)[FEATURES]

    snapshot = adaptador_evidently.executar_evidently(referencia, atual)
    resultados = adaptador_evidently.montar_resultados(snapshot, FEATURES, 0.05, 0.1)

    assert [item["feature"] for item in resultados] == FEATURES
    assert all(item["drift_detected"] is True for item in resultados)

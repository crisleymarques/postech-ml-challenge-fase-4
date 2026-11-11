"""Testes de resultados dos alertas."""

from __future__ import annotations

from monitoramento.observabilidade.alertas import resultados as al


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

"""Painel de monitoramento do pipeline em Streamlit (issue F3-04).

O painel evidencia a saude do pipeline, da infraestrutura e do modelo a partir
dos artefatos ja persistidos pelas etapas anteriores:

- execucoes em `reports/runs/<run_id>/` (manifesto e `events.jsonl`);
- logs operacionais consolidados (`reports/logs/resumo.json`, F3-01);
- metricas de saude operacional (`reports/saude/saude.json`, F3-02);
- alertas integrados de drift, metricas e falhas
  (`reports/alertas/alertas.json`, F3-03).

A interface organiza os resultados em abas por tema (visao geral, saude
operacional, drift, alertas e logs). Cada artefato ausente aparece como
indisponivel sem impedir a exibicao dos demais; execucao sem evento terminal e
apresentada como incompleta, sem inferir sucesso.

Uso:
    uv run streamlit run src/monitoramento/dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from monitoramento.infraestrutura.observabilidade import status_publicado

RUNS_DIR_PADRAO = Path("reports/runs")
LOGS_RESUMO_PADRAO = Path("reports/logs/resumo.json")
SAUDE_PADRAO = Path("reports/saude/saude.json")
ALERTAS_PADRAO = Path("reports/alertas/alertas.json")

STATUS_SUCESSO = "success"
STATUS_REJEICAO = "rejected"
STATUS_FALHA = "failed"
STATUS_INCOMPLETO = "incomplete"
STATUS_TERMINAL = {
    "completed": STATUS_SUCESSO,
    "rejected": STATUS_REJEICAO,
    "failed": STATUS_FALHA,
}

SITUACAO_ATENCAO = "atencao"
SITUACAO_CRITICO = "critico"
SEVERIDADE_WARNING = "warning"
SEVERIDADE_CRITICAL = "critical"

CATEGORIA_EXECUCAO = "execucao"
CATEGORIA_AGREGADO = "agregado"

NAO_DISPONIVEL = "indisponivel"


def ler_json(caminho: Path) -> dict[str, Any] | None:
    """Le um artefato JSON quando disponivel, sem interromper a interface."""
    if not caminho.is_file():
        return None
    try:
        with caminho.open(encoding="utf-8") as arquivo:
            conteudo = json.load(arquivo)
    except (json.JSONDecodeError, OSError, UnicodeError):
        return None
    return conteudo if isinstance(conteudo, dict) else None


def ler_eventos(caminho: Path) -> list[dict[str, Any]]:
    """Le um `events.jsonl` de forma tolerante, ignorando linhas invalidas."""
    eventos: list[dict[str, Any]] = []
    if not caminho.is_file():
        return eventos
    try:
        with caminho.open(encoding="utf-8") as arquivo:
            for linha in arquivo:
                conteudo = linha.strip()
                if not conteudo:
                    continue
                try:
                    evento = json.loads(conteudo)
                except json.JSONDecodeError:
                    continue
                if isinstance(evento, dict):
                    eventos.append(evento)
    except (OSError, UnicodeError):
        # Leitura incompleta nao confirma a publicacao de sucesso.
        return []
    return eventos


def resumo_alertas(artefato: dict[str, Any] | None) -> dict[str, Any]:
    """Retorna contagens confirmadas; um resumo ausente ou invalido nao e zero."""
    resumo = (artefato or {}).get("resumo")
    if not isinstance(resumo, dict):
        return {}
    severidades = resumo.get("por_severidade")
    if not isinstance(severidades, dict):
        return {}
    contagens = [
        resumo.get("total"),
        *(severidades.get(s) for s in (SEVERIDADE_WARNING, SEVERIDADE_CRITICAL)),
    ]
    if any(type(valor) is not int or valor < 0 for valor in contagens):
        return {}
    return resumo


def ler_alertas(caminho: Path) -> dict[str, Any] | None:
    """Le o relatorio de alertas, exigindo resumo e colecao utilizaveis."""
    artefato = ler_json(caminho)
    if not resumo_alertas(artefato) or not isinstance(artefato.get("alertas"), list):
        return None
    return artefato


def inferir_status(eventos: list[dict[str, Any]]) -> str:
    """Infere o status da execucao pelo ultimo evento terminal observado."""
    for evento in reversed(eventos):
        status = STATUS_TERMINAL.get(evento.get("event"))
        if status is not None:
            return status
    return STATUS_INCOMPLETO


def listar_execucoes(runs_dir: Path) -> list[Path]:
    """Lista os diretorios de execucao com manifesto ou eventos, em ordem."""
    if not runs_dir.is_dir():
        return []
    return sorted(
        (
            caminho
            for caminho in runs_dir.iterdir()
            if (caminho / "manifest.json").is_file() or (caminho / "events.jsonl").is_file()
        ),
        key=lambda caminho: caminho.name,
    )


def carregar_execucoes(runs_dir: Path) -> list[dict[str, Any]]:
    """Reune os metadados de cada execucao a partir do manifesto e dos eventos."""
    execucoes: list[dict[str, Any]] = []
    for run_dir in listar_execucoes(runs_dir):
        manifesto = ler_json(run_dir / "manifest.json") or {}
        eventos = ler_eventos(run_dir / "events.jsonl")
        stage = manifesto.get("stage") or (eventos[0].get("stage") if eventos else None)
        status = status_publicado(manifesto, eventos)
        execucoes.append(
            {
                "run_id": manifesto.get("run_id") or run_dir.name,
                "stage": stage,
                "status": status,
                "started_at": manifesto.get("started_at"),
                "finished_at": manifesto.get("finished_at"),
                "error_code": manifesto.get("error_code"),
                "event_count": len(eventos),
                "incompleta": status == STATUS_INCOMPLETO,
            }
        )
    return execucoes


def resumir_execucoes(execucoes: list[dict[str, Any]]) -> dict[str, Any]:
    """Consolida as execucoes por status e por etapa."""
    por_status: dict[str, int] = {}
    por_etapa: dict[str, int] = {}
    for execucao in execucoes:
        status = execucao["status"]
        por_status[status] = por_status.get(status, 0) + 1
        etapa = execucao.get("stage") or "desconhecida"
        por_etapa[etapa] = por_etapa.get(etapa, 0) + 1
    return {
        "total": len(execucoes),
        "por_status": por_status,
        "por_etapa": dict(sorted(por_etapa.items())),
        "incompletas": [execucao["run_id"] for execucao in execucoes if execucao["incompleta"]],
    }


def taxa_sucesso(execucoes: list[dict[str, Any]]) -> float | None:
    """Proporcao de execucoes concluidas com status 'success'."""
    if not execucoes:
        return None
    sucessos = sum(1 for execucao in execucoes if execucao["status"] == STATUS_SUCESSO)
    return round(sucessos / len(execucoes), 6)


def carregar_painel(
    runs_dir: Path,
    logs_path: Path,
    saude_path: Path,
    alertas_path: Path,
) -> dict[str, Any]:
    """Carrega as execucoes e os artefatos consolidados para o painel."""
    execucoes = carregar_execucoes(runs_dir)
    logs = ler_json(logs_path)
    saude = ler_json(saude_path)
    alertas = ler_alertas(alertas_path)
    return {
        "execucoes": execucoes,
        "resumo_execucoes": resumir_execucoes(execucoes),
        "logs": logs,
        "saude": saude,
        "alertas": alertas,
        "fontes": {
            "runs_dir": runs_dir.as_posix(),
            "logs": logs_path.as_posix(),
            "saude": saude_path.as_posix(),
            "alertas": alertas_path.as_posix(),
        },
        "disponibilidade": {
            "logs": logs is not None,
            "saude": saude is not None,
            "alertas": alertas is not None,
        },
    }


def contar_situacoes(saude: dict[str, Any] | None) -> dict[str, int]:
    """Contagens de situacao das medidas de saude, quando disponiveis."""
    resumo = (saude or {}).get("resumo") or {}
    por_situacao = resumo.get("por_situacao")
    return por_situacao if isinstance(por_situacao, dict) else {}


def indicadores(painel: dict[str, Any]) -> dict[str, Any]:
    """Indicadores sinteticos usados na visao geral do painel."""
    execucoes = painel["execucoes"]
    por_status = painel["resumo_execucoes"]["por_status"]
    alertas = painel.get("alertas") or {}
    resumo = resumo_alertas(alertas)
    por_severidade = resumo.get("por_severidade") or {}
    drift = alertas.get("drift") or {}
    return {
        "execucoes": len(execucoes),
        "taxa_sucesso": taxa_sucesso(execucoes),
        "sucessos": por_status.get(STATUS_SUCESSO, 0),
        "rejeicoes": por_status.get(STATUS_REJEICAO, 0),
        "falhas": por_status.get(STATUS_FALHA, 0),
        "incompletas": len(painel["resumo_execucoes"]["incompletas"]),
        "alertas_total": resumo.get("total"),
        "alertas_warning": por_severidade.get(SEVERIDADE_WARNING),
        "alertas_critical": por_severidade.get(SEVERIDADE_CRITICAL),
        "dataset_drift": drift.get("dataset_drift"),
        "variaveis_com_drift": drift.get("variaveis_com_drift"),
    }


def medidas_de_saude(saude: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Reune as medidas de saude por execucao e os agregados, com origem e escopo."""
    linhas: list[dict[str, Any]] = []
    for execucao in (saude or {}).get("execucoes", []) or []:
        if not isinstance(execucao, dict):
            continue
        for medida in execucao.get("metricas", []) or []:
            if isinstance(medida, dict):
                linhas.append(
                    {
                        "origem": execucao.get("run_id"),
                        "escopo": CATEGORIA_EXECUCAO,
                        "id": medida.get("id"),
                        "categoria": medida.get("categoria"),
                        "valor": medida.get("valor"),
                        "unidade": medida.get("unidade"),
                        "situacao": medida.get("situacao"),
                    }
                )
    for medida in (saude or {}).get("agregados", []) or []:
        if isinstance(medida, dict):
            linhas.append(
                {
                    "origem": "agregado",
                    "escopo": CATEGORIA_AGREGADO,
                    "id": medida.get("id"),
                    "categoria": medida.get("categoria"),
                    "valor": medida.get("valor"),
                    "unidade": medida.get("unidade"),
                    "situacao": medida.get("situacao"),
                }
            )
    return linhas


def _tabela(linhas: list[dict[str, Any]]) -> pd.DataFrame:
    """Converte uma lista de registros em DataFrame, tolerando lista vazia."""
    return pd.DataFrame(linhas) if linhas else pd.DataFrame()


def _origem_artefato(artefato: dict[str, Any] | None, disponivel: bool) -> str:
    """Descreve a origem e o horario de atualizacao de um artefato."""
    if not disponivel or artefato is None:
        return NAO_DISPONIVEL
    return str(artefato.get("generated_at") or "sem horario")


def render_visao_geral(painel: dict[str, Any]) -> None:
    """Aba de visao geral: indicadores, saude dos artefatos e execucoes."""
    ind = indicadores(painel)
    colunas = st.columns(5)
    colunas[0].metric("Execuções", ind["execucoes"])
    taxa = ind["taxa_sucesso"]
    colunas[1].metric("Taxa de sucesso", "—" if taxa is None else f"{taxa:.1%}")
    colunas[2].metric("Falhas", ind["falhas"])
    colunas[3].metric("Rejeições", ind["rejeicoes"])
    colunas[4].metric("Alertas", "—" if ind["alertas_total"] is None else ind["alertas_total"])

    st.subheader("Alertas e drift")
    colunas = st.columns(4)
    colunas[0].metric(
        "Alertas críticos", "—" if ind["alertas_critical"] is None else ind["alertas_critical"]
    )
    colunas[1].metric(
        "Alertas de atenção", "—" if ind["alertas_warning"] is None else ind["alertas_warning"]
    )
    drift = ind["dataset_drift"]
    colunas[2].metric("Data drift", "—" if drift is None else ("sim" if drift else "não"))
    variaveis = ind["variaveis_com_drift"]
    colunas[3].metric("Variáveis com drift", "—" if variaveis is None else int(variaveis))

    st.subheader("Saúde dos artefatos")
    disponibilidade = painel["disponibilidade"]
    st.dataframe(
        _tabela(
            [
                {
                    "artefato": "logs (F3-01)",
                    "situação": _origem_artefato(painel["logs"], disponibilidade["logs"]),
                },
                {
                    "artefato": "saúde (F3-02)",
                    "situação": _origem_artefato(painel["saude"], disponibilidade["saude"]),
                },
                {
                    "artefato": "alertas (F3-03)",
                    "situação": _origem_artefato(painel["alertas"], disponibilidade["alertas"]),
                },
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Execuções por etapa")
    por_etapa = painel["resumo_execucoes"]["por_etapa"]
    if por_etapa:
        st.bar_chart(pd.DataFrame({"execuções": por_etapa}))
    else:
        st.info("Nenhuma execução registrada.")

    incompletas = painel["resumo_execucoes"]["incompletas"]
    if incompletas:
        st.warning("Execuções sem evento terminal (incompletas): " + ", ".join(incompletas))

    execucoes = painel["execucoes"]
    if execucoes:
        st.subheader("Histórico de execuções")
        st.dataframe(_tabela(execucoes), use_container_width=True, hide_index=True)


def render_saude(painel: dict[str, Any]) -> None:
    """Aba de saude operacional: medidas por execucao e agregados."""
    saude = painel["saude"]
    if saude is None:
        st.info("Relatório de saúde (F3-02) indisponível.")
        return

    situacoes = contar_situacoes(saude)
    if situacoes:
        colunas = st.columns(3)
        colunas[0].metric("Medidas ok", situacoes.get("ok", 0))
        colunas[1].metric("Em atenção", situacoes.get(SITUACAO_ATENCAO, 0))
        colunas[2].metric("Críticas", situacoes.get(SITUACAO_CRITICO, 0))

    medidas = medidas_de_saude(saude)
    if not medidas:
        st.info("Nenhuma medida de saúde coletada.")
        return

    categorias = sorted({medida["categoria"] for medida in medidas if medida["categoria"]})
    selecionadas = st.multiselect("Filtrar por categoria", categorias, default=categorias)
    selecionadas_ou_todas = selecionadas or categorias

    st.subheader("Medidas por execução")
    execucao = [
        medida
        for medida in medidas
        if medida["escopo"] == CATEGORIA_EXECUCAO and medida["categoria"] in selecionadas_ou_todas
    ]
    st.dataframe(_tabela(execucao), use_container_width=True, hide_index=True)

    st.subheader("Agregados")
    agregados = [
        medida
        for medida in medidas
        if medida["escopo"] == CATEGORIA_AGREGADO and medida["categoria"] in selecionadas_ou_todas
    ]
    st.dataframe(_tabela(agregados), use_container_width=True, hide_index=True)


def render_drift(painel: dict[str, Any]) -> None:
    """Aba de drift: bloco consolidado e referencias aos relatorios."""
    alertas = painel["alertas"]
    if alertas is None or not isinstance(alertas.get("drift"), dict):
        st.info("Resultados de drift indisponíveis (artefato de alertas F3-03 ausente).")
        return

    drift = alertas["drift"]
    destaque = drift.get("execucao_resumo")
    if destaque:
        st.caption(
            f"Destaque histórico: execução {destaque}. Maior proporção de variáveis com drift, "
            "com desempate por quantidade e execução. Alertas: total histórico."
        )
    colunas = st.columns(4)
    presente = drift.get("dataset_drift")
    colunas[0].metric("Data drift", "—" if presente is None else ("sim" if presente else "não"))
    variaveis = drift.get("variaveis_com_drift")
    colunas[1].metric("Variáveis com drift", "—" if variaveis is None else int(variaveis))
    proporcao = drift.get("proporcao_drift")
    colunas[2].metric("Proporção", "—" if proporcao is None else f"{proporcao:.3f}")
    alertas_total = drift.get("alertas_total")
    colunas[3].metric("Alertas de drift", "—" if alertas_total is None else int(alertas_total))

    por_execucao = drift.get("por_execucao") or []
    if por_execucao:
        st.subheader("Drift por execução")
        st.dataframe(
            _tabela(por_execucao).rename(
                columns={
                    "run_id": "Execução",
                    "stage": "Etapa",
                    "dataset_drift": "Drift no conjunto",
                    "variaveis_com_drift": "Variáveis com drift",
                    "proporcao_drift": "Proporção",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    execucoes = drift.get("execucoes") or []
    if execucoes:
        st.caption("Execuções com medidas de drift: " + ", ".join(str(item) for item in execucoes))

    st.caption(
        "Relatórios detalhados: `reports/runs/<run_id>/drift.json`, `analise.json`, "
        "`drift.html` e `performance.json`."
    )


def render_alertas(painel: dict[str, Any]) -> None:
    """Aba de alertas: lista por severidade, categoria e origem."""
    alertas = painel["alertas"]
    if not resumo_alertas(alertas):
        st.info("Relatório de alertas (F3-03) indisponível.")
        return

    resumo = alertas.get("resumo") or {}
    por_severidade = resumo.get("por_severidade") or {}
    colunas = st.columns(3)
    colunas[0].metric("Total", resumo.get("total", 0))
    colunas[1].metric("Críticos", por_severidade.get(SEVERIDADE_CRITICAL, 0))
    colunas[2].metric("Atenção", por_severidade.get(SEVERIDADE_WARNING, 0))

    lista = [alerta for alerta in alertas.get("alertas", []) or [] if isinstance(alerta, dict)]
    if not lista:
        st.info("Nenhum alerta registrado.")
        return

    severidades = sorted({alerta.get("severidade") for alerta in lista if alerta.get("severidade")})
    categorias = sorted({alerta.get("categoria") for alerta in lista if alerta.get("categoria")})
    colunas = st.columns(2)
    filtro_sev = colunas[0].multiselect("Severidade", severidades, default=severidades)
    filtro_cat = colunas[1].multiselect("Categoria", categorias, default=categorias)

    selecionados = [
        alerta
        for alerta in lista
        if alerta.get("severidade") in (filtro_sev or severidades)
        and alerta.get("categoria") in (filtro_cat or categorias)
    ]
    st.dataframe(_tabela(selecionados), use_container_width=True, hide_index=True)


def render_logs(painel: dict[str, Any]) -> None:
    """Aba de logs: contagens por categoria/resultado e falhas."""
    logs = painel["logs"]
    if logs is None:
        st.info("Resumo dos logs (F3-01) indisponível.")
        return

    execucoes = logs.get("execucoes") or {}
    eventos = logs.get("eventos") or {}
    colunas = st.columns(3)
    colunas[0].metric("Execuções", execucoes.get("total", 0))
    colunas[1].metric("Eventos", eventos.get("total", 0))
    colunas[2].metric("Falhas", len(logs.get("falhas", []) or []))

    por_categoria = logs.get("por_categoria") or {}
    if por_categoria:
        st.subheader("Eventos por categoria")
        linhas = [
            {"categoria": categoria, **contagens}
            for categoria, contagens in sorted(por_categoria.items())
            if isinstance(contagens, dict)
        ]
        st.dataframe(_tabela(linhas), use_container_width=True, hide_index=True)

    falhas = [falha for falha in logs.get("falhas", []) or [] if isinstance(falha, dict)]
    if falhas:
        st.subheader("Falhas registradas")
        st.dataframe(_tabela(falhas), use_container_width=True, hide_index=True)
    else:
        st.success("Nenhuma falha de execução ou de contrato registrada.")


def main() -> None:
    st.set_page_config(page_title="Monitoramento — Fase 4", layout="wide")
    st.title("Monitoramento do pipeline de Credit Scoring")
    st.caption(
        "Saúde do pipeline, infraestrutura, drift e alertas a partir dos artefatos persistidos."
    )

    st.sidebar.header("Fontes de dados")
    runs_dir = Path(st.sidebar.text_input("Execuções", RUNS_DIR_PADRAO.as_posix()))
    logs_path = Path(st.sidebar.text_input("Logs (F3-01)", LOGS_RESUMO_PADRAO.as_posix()))
    saude_path = Path(st.sidebar.text_input("Saúde (F3-02)", SAUDE_PADRAO.as_posix()))
    alertas_path = Path(st.sidebar.text_input("Alertas (F3-03)", ALERTAS_PADRAO.as_posix()))

    painel = carregar_painel(runs_dir, logs_path, saude_path, alertas_path)

    if not painel["execucoes"] and not any(painel["disponibilidade"].values()):
        st.info("Nenhuma execução ou artefato encontrado nos caminhos informados.")
        return

    abas = st.tabs(["Visão geral", "Saúde operacional", "Drift", "Alertas", "Logs e falhas"])
    with abas[0]:
        render_visao_geral(painel)
    with abas[1]:
        render_saude(painel)
    with abas[2]:
        render_drift(painel)
    with abas[3]:
        render_alertas(painel)
    with abas[4]:
        render_logs(painel)


if __name__ == "__main__":
    main()

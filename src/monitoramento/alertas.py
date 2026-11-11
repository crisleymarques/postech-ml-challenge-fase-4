"""Integracao de drift, metricas e alertas de observabilidade (issue F3-03).

Consolida os artefatos ja publicados pelas etapas anteriores — as metricas de
saude operacional (`reports/saude/saude.json`, F3-02) e os logs operacionais
(`reports/logs/resumo.json`, F3-01) — e aplica os criterios de alerta declarados
em `configs/alertas.json` para sinalizar falhas e degradacao:

- medida de saude com situacao `atencao` gera alerta `warning`;
- medida de saude com situacao `critico` gera alerta `critical`;
- falha de execucao ou de contrato registrada nos logs gera alerta `critical`;
- evento `drift_alert` registrado nos logs gera alerta `warning`.

O componente publica `reports/alertas/alertas.json` com a lista de alertas
(identificavel e contabilizavel), um bloco consolidado de resultados de drift e
um resumo por severidade, categoria e origem.

Este componente apenas integra e sinaliza; nao relê os artefatos brutos por
execucao e nao interrompe o pipeline. Alerta de drift ou de degradacao nao e
erro de execucao.

Codigos de saida: 0 sucesso, 1 erro tecnico.

Uso:
    uv run python src/monitoramento/alertas.py \
        --saude reports/saude/saude.json \
        --logs-resumo reports/logs/resumo.json \
        --config configs/alertas.json \
        --output-dir reports/alertas
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

CONTRACT_VERSION = "1.0"

SAUDE_PADRAO = Path("reports/saude/saude.json")
LOGS_RESUMO_PADRAO = Path("reports/logs/resumo.json")
CONFIG_PADRAO = Path("configs/alertas.json")
OUTPUT_DIR_PADRAO = Path("reports/alertas")

CODIGO_SUCESSO = 0
CODIGO_ERRO = 1

SEVERIDADE_WARNING = "warning"
SEVERIDADE_CRITICAL = "critical"
ORDEM_SEVERIDADES = (SEVERIDADE_WARNING, SEVERIDADE_CRITICAL)

SITUACAO_ATENCAO = "atencao"
SITUACAO_CRITICO = "critico"

ORIGEM_AGREGADO = "agregado"
ORIGEM_SAUDE = "saude"
ORIGEM_LOGS = "logs"
CATEGORIA_LOG = "log"

ID_DATASET_DRIFT = "drift.dataset_drift"
ID_VARIAVEIS_DRIFT = "drift.variaveis_com_drift"
ID_PROPORCAO_DRIFT = "drift.proporcao_drift"
ID_ALERTAS_TOTAL = "drift.alertas_total"
IDS_DRIFT_EXECUCAO = (ID_DATASET_DRIFT, ID_VARIAVEIS_DRIFT, ID_PROPORCAO_DRIFT)


def agora_iso() -> str:
    """Retorna o instante atual em ISO 8601 UTC com sufixo Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def versao_maior(versao: Any) -> str:
    """Extrai o componente principal de uma versao de contrato."""
    return str(versao or "").split(".")[0]


def carregar_config(caminho: Path) -> dict[str, Any]:
    """Le e valida minimamente os criterios de alerta."""
    if not caminho.is_file():
        raise ValueError(f"Configuracao de alertas nao encontrada: {caminho}")
    with caminho.open(encoding="utf-8") as arquivo:
        config = json.load(arquivo)

    if versao_maior(config.get("contract_version")) != versao_maior(CONTRACT_VERSION):
        raise ValueError("Versao principal do contrato de alertas incompativel.")

    severidades = config.get("severidades")
    if not isinstance(severidades, list) or not severidades:
        raise ValueError("A configuracao deve declarar as severidades.")
    if SEVERIDADE_CRITICAL not in severidades:
        raise ValueError(f"A severidade '{SEVERIDADE_CRITICAL}' deve constar na configuracao.")

    mapa = config.get("severidade_por_situacao")
    if not isinstance(mapa, dict) or SITUACAO_CRITICO not in mapa:
        raise ValueError("Mapeamento 'severidade_por_situacao' invalido ou incompleto.")

    for chave in ("severidade_falha", "severidade_alerta_drift"):
        if config.get(chave) not in severidades:
            raise ValueError(f"Campo '{chave}' deve referenciar uma severidade declarada.")
    return config


def ler_json(caminho: Path, rotulo: str) -> dict[str, Any]:
    """Le um artefato JSON obrigatorio, interrompendo em caso de ausencia."""
    if not caminho.is_file():
        raise ValueError(f"{rotulo} nao encontrado: {caminho}")
    try:
        with caminho.open(encoding="utf-8") as arquivo:
            conteudo = json.load(arquivo)
    except (json.JSONDecodeError, OSError) as erro:
        raise ValueError(f"{rotulo} invalido em {caminho}: {erro}") from erro
    if not isinstance(conteudo, dict):
        raise ValueError(f"{rotulo} invalido em {caminho}: esperado objeto JSON.")
    return conteudo


def medidas_com_situacao(
    saude: dict[str, Any],
) -> Iterable[tuple[str, str, dict[str, Any]]]:
    """Percorre as medidas de saude, rendendo (origem, escopo, medida)."""
    for execucao in saude.get("execucoes", []) or []:
        if not isinstance(execucao, dict):
            continue
        origem = execucao.get("run_id") or execucao.get("stage") or ORIGEM_AGREGADO
        for medida in execucao.get("metricas", []) or []:
            if isinstance(medida, dict):
                yield str(origem), "execucao", medida
    for medida in saude.get("agregados", []) or []:
        if isinstance(medida, dict):
            yield ORIGEM_AGREGADO, "agregado", medida


def alerta_de_medida(
    origem: str,
    escopo: str,
    medida: dict[str, Any],
    severidade: str,
) -> dict[str, Any]:
    """Converte uma medida fora do esperado em alerta de observabilidade."""
    return {
        "id": f"{ORIGEM_SAUDE}:{origem}:{medida.get('id')}",
        "severidade": severidade,
        "categoria": medida.get("categoria"),
        "escopo": escopo,
        "origem": origem,
        "fonte": ORIGEM_SAUDE,
        "metrica": medida.get("id"),
        "descricao": f"{medida.get('nome')} em '{medida.get('situacao')}'",
        "valor": medida.get("valor"),
        "unidade": medida.get("unidade"),
        "situacao": medida.get("situacao"),
        "error_code": None,
        "evidencia": {"origem": origem, "metrica": medida.get("id")},
    }


def alertas_de_medidas(
    saude: dict[str, Any],
    severidade_por_situacao: dict[str, Any],
) -> list[dict[str, Any]]:
    """Alertas derivados das medidas de saude em situacao de atencao ou critico."""
    alertas: list[dict[str, Any]] = []
    for origem, escopo, medida in medidas_com_situacao(saude):
        severidade = severidade_por_situacao.get(medida.get("situacao"))
        if severidade is None:
            continue
        alertas.append(alerta_de_medida(origem, escopo, medida, severidade))
    return alertas


def alertas_de_falhas(
    resumo_logs: dict[str, Any],
    severidade_falha: str,
) -> list[dict[str, Any]]:
    """Alertas derivados das falhas de execucao ou de contrato registradas nos logs."""
    alertas: list[dict[str, Any]] = []
    for indice, falha in enumerate(resumo_logs.get("falhas", []) or [], start=1):
        if not isinstance(falha, dict):
            continue
        run_id = falha.get("run_id") or ORIGEM_AGREGADO
        error_code = falha.get("error_code")
        rotulo = f"{falha.get('event')} ({error_code})" if error_code else falha.get("event")
        alertas.append(
            {
                "id": f"{ORIGEM_LOGS}:falha:{run_id}:{indice}",
                "severidade": severidade_falha,
                "categoria": falha.get("categoria") or CATEGORIA_LOG,
                "escopo": CATEGORIA_LOG,
                "origem": run_id,
                "fonte": ORIGEM_LOGS,
                "metrica": None,
                "descricao": f"Falha de execucao: {rotulo}",
                "valor": None,
                "unidade": None,
                "situacao": None,
                "error_code": error_code,
                "evidencia": {
                    "run_id": falha.get("run_id"),
                    "stage": falha.get("stage"),
                    "event": falha.get("event"),
                    "timestamp": falha.get("timestamp"),
                },
            }
        )
    return alertas


def alertas_de_drift(
    resumo_logs: dict[str, Any],
    severidade_alerta_drift: str,
) -> list[dict[str, Any]]:
    """Alertas derivados dos eventos `drift_alert` registrados nos logs."""
    alertas: list[dict[str, Any]] = []
    for indice, evento in enumerate(resumo_logs.get("alertas", []) or [], start=1):
        if not isinstance(evento, dict):
            continue
        run_id = evento.get("run_id") or ORIGEM_AGREGADO
        contexto = evento.get("scenario_id") or evento.get("batch_id")
        descricao = "Alerta de drift" + (f" (cenario {contexto})" if contexto else "")
        alertas.append(
            {
                "id": f"{ORIGEM_LOGS}:drift:{run_id}:{indice}",
                "severidade": severidade_alerta_drift,
                "categoria": "drift",
                "escopo": CATEGORIA_LOG,
                "origem": run_id,
                "fonte": ORIGEM_LOGS,
                "metrica": None,
                "descricao": descricao,
                "valor": None,
                "unidade": None,
                "situacao": None,
                "error_code": None,
                "evidencia": {
                    "run_id": evento.get("run_id"),
                    "stage": evento.get("stage"),
                    "scenario_id": evento.get("scenario_id"),
                    "batch_id": evento.get("batch_id"),
                    "timestamp": evento.get("timestamp"),
                },
            }
        )
    return alertas


def _ultimo_valor(medidas: list[dict[str, Any]], id_metrica: str, chave: str) -> Any:
    """Ultimo valor nao nulo de uma metrica, na ordem em que aparece."""
    valor = None
    for medida in medidas:
        if medida.get("id") == id_metrica and medida.get(chave) is not None:
            valor = medida.get(chave)
    return valor


def _maior_valor(medidas: list[dict[str, Any]], id_metrica: str) -> float | None:
    """Maior valor numerico de uma metrica entre as execucoes."""
    valores = [
        medida.get("valor")
        for medida in medidas
        if medida.get("id") == id_metrica
        and isinstance(medida.get("valor"), (int, float))
        and not isinstance(medida.get("valor"), bool)
    ]
    return max(valores) if valores else None


def consolidar_drift(saude: dict[str, Any]) -> dict[str, Any]:
    """Reune as evidencias de drift das medidas de saude e dos alertas dos logs."""
    execucoes: list[str] = []
    medidas: list[dict[str, Any]] = []
    for execucao in saude.get("execucoes", []) or []:
        if not isinstance(execucao, dict):
            continue
        metricas = [item for item in execucao.get("metricas", []) or [] if isinstance(item, dict)]
        medidas.extend(metricas)
        if any(item.get("id") in IDS_DRIFT_EXECUCAO for item in metricas):
            execucoes.append(execucao.get("run_id") or execucao.get("stage") or ORIGEM_AGREGADO)

    agregados = [item for item in saude.get("agregados", []) or [] if isinstance(item, dict)]
    return {
        "dataset_drift": _ultimo_valor(medidas, ID_DATASET_DRIFT, "valor"),
        "variaveis_com_drift": _maior_valor(medidas, ID_VARIAVEIS_DRIFT),
        "proporcao_drift": _maior_valor(medidas, ID_PROPORCAO_DRIFT),
        "alertas_total": _ultimo_valor(agregados, ID_ALERTAS_TOTAL, "valor"),
        "execucoes": execucoes,
    }


def montar_resumo(alertas: list[dict[str, Any]]) -> dict[str, Any]:
    """Consolida as contagens de alertas por severidade, categoria e origem."""
    por_severidade = {severidade: 0 for severidade in ORDEM_SEVERIDADES}
    por_categoria: dict[str, int] = {}
    por_origem = {ORIGEM_SAUDE: 0, ORIGEM_LOGS: 0}
    for alerta in alertas:
        severidade = alerta["severidade"]
        por_severidade[severidade] = por_severidade.get(severidade, 0) + 1
        categoria = alerta.get("categoria") or "desconhecida"
        por_categoria[categoria] = por_categoria.get(categoria, 0) + 1
        origem = alerta.get("fonte")
        por_origem[origem] = por_origem.get(origem, 0) + 1
    return {
        "total": len(alertas),
        "por_severidade": por_severidade,
        "por_categoria": dict(sorted(por_categoria.items())),
        "por_origem": por_origem,
        "criticos": [
            alerta["id"] for alerta in alertas if alerta["severidade"] == SEVERIDADE_CRITICAL
        ],
    }


def gerar(
    saude_path: Path,
    logs_resumo_path: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Integra as metricas de saude e os logs, e publica `alertas.json`."""
    config = carregar_config(config_path)
    saude = ler_json(saude_path, "Relatorio de saude")
    resumo_logs = ler_json(logs_resumo_path, "Resumo dos logs")

    alertas = (
        alertas_de_medidas(saude, config.get("severidade_por_situacao") or {})
        + alertas_de_falhas(resumo_logs, config["severidade_falha"])
        + alertas_de_drift(resumo_logs, config["severidade_alerta_drift"])
    )

    relatorio = {
        "contract_version": CONTRACT_VERSION,
        "generated_at": agora_iso(),
        "fontes": {
            "saude": saude_path.as_posix(),
            "logs": logs_resumo_path.as_posix(),
            "config": config_path.as_posix(),
        },
        "output_dir": output_dir.as_posix(),
        "drift": consolidar_drift(saude),
        "alertas": alertas,
        "resumo": montar_resumo(alertas),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    destino = output_dir / "alertas.json"
    with destino.open("w", encoding="utf-8") as arquivo:
        json.dump(relatorio, arquivo, ensure_ascii=False, indent=2)
        arquivo.write("\n")
    return relatorio


def imprimir_relatorio(relatorio: dict[str, Any]) -> None:
    """Imprime um resumo legivel dos alertas integrados."""
    resumo = relatorio["resumo"]
    severidades = ", ".join(
        f"{severidade}={quantidade}"
        for severidade, quantidade in resumo["por_severidade"].items()
        if quantidade
    )
    drift = relatorio["drift"]
    print(f"Alertas integrados: {resumo['total']} alertas")
    print(f"  severidades: {severidades or 'nenhuma'}")
    print(f"  por categoria: {resumo['por_categoria'] or 'nenhuma'}")
    print(
        "  drift: "
        f"dataset_drift={drift['dataset_drift']}, "
        f"variaveis_com_drift={drift['variaveis_com_drift']}, "
        f"proporcao_drift={drift['proporcao_drift']}, "
        f"alertas={drift['alertas_total']}"
    )
    for identificador in resumo["criticos"]:
        print(f"    critico: {identificador}")
    print(f"  {relatorio['output_dir']}/alertas.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Integra drift, metricas de saude e alertas de observabilidade."
    )
    parser.add_argument(
        "--saude",
        type=Path,
        default=SAUDE_PADRAO,
        help="Relatorio de saude da F3-02 (padrao: reports/saude/saude.json).",
    )
    parser.add_argument(
        "--logs-resumo",
        type=Path,
        default=LOGS_RESUMO_PADRAO,
        help="Resumo dos logs da F3-01 (padrao: reports/logs/resumo.json).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PADRAO,
        help="Configuracao dos criterios de alerta (padrao: configs/alertas.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR_PADRAO,
        help="Diretorio de saida do relatorio de alertas (padrao: reports/alertas).",
    )
    args = parser.parse_args()

    try:
        relatorio = gerar(args.saude, args.logs_resumo, args.config, args.output_dir)
    except ValueError as erro:
        print(f"Falha ao integrar os alertas: {erro}")
        sys.exit(CODIGO_ERRO)

    imprimir_relatorio(relatorio)
    sys.exit(CODIGO_SUCESSO)


if __name__ == "__main__":
    main()

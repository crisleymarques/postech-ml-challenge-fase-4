"""Consolidacao dos logs operacionais do pipeline (issue F3-01).

Cada etapa do pipeline publica um `events.jsonl` proprio em
`reports/runs/<run_id>/`, seguindo o contrato comum de evento (`timestamp`,
`run_id`, `stage`, `event`, `level`, ids e `error_code`). Este componente le
todos esses arquivos, classifica cada evento por categoria operacional
(ingestao, transformacao, validacao, treinamento, simulacao, predicao, drift e
avaliacao) e por resultado (inicio, sucesso, rejeicao, falha, alerta e
ignorado), e produz um log operacional unico:

- `operacional.jsonl`: todos os eventos de todas as execucoes, ordenados por
  horario e enriquecidos com `categoria` e `resultado`.
- `resumo.json`: contagens por status de execucao, por categoria operacional,
  alem das falhas (com `error_code`) e dos alertas de drift, para diagnostico.

A etapa `preparacao` realiza ingestao e transformacao de forma acoplada; o
desfecho de suas execucoes alimenta as categorias `ingestao` e `transformacao`,
e as transformacoes aplicadas sao detalhadas a partir do manifesto.

Nao altera as etapas existentes: apenas agrega o que ja e emitido.

Codigos de saida: 0 sucesso, 1 erro tecnico.

Uso:
    uv run python src/monitoramento/logs.py --runs-dir reports/runs --output-dir reports/logs
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from monitoramento.infraestrutura.observabilidade import status_publicado

CONTRACT_VERSION = "1.0"

RUNS_DIR_PADRAO = Path("reports/runs")
OUTPUT_DIR_PADRAO = Path("reports/logs")

CODIGO_SUCESSO = 0
CODIGO_ERRO = 1

# Categoria operacional principal de cada etapa.
CATEGORIA_POR_ETAPA = {
    "preparacao": "ingestao",
    "validacao": "validacao",
    "treinamento": "treinamento",
    "simulacao": "simulacao",
    "predicao": "predicao",
    "deteccao": "drift",
    "analise": "drift",
    "avaliacao": "avaliacao",
}
# A preparacao cobre ingestao e transformacao; seu desfecho alimenta ambas.
CATEGORIAS_POR_ETAPA = {"preparacao": ("ingestao", "transformacao")}
CATEGORIAS_CONHECIDAS = (
    "ingestao",
    "transformacao",
    "validacao",
    "treinamento",
    "simulacao",
    "predicao",
    "drift",
    "avaliacao",
)
CATEGORIA_DESCONHECIDA = "desconhecida"

RESULTADO_POR_EVENTO = {
    "started": "inicio",
    "completed": "sucesso",
    "rejected": "rejeicao",
    "failed": "falha",
    "skipped": "ignorado",
    "drift_alert": "alerta",
}
RESULTADO_DESCONHECIDO = "desconhecido"
ORDEM_RESULTADOS = ("inicio", "sucesso", "rejeicao", "falha", "alerta", "ignorado", "desconhecido")

STATUS_TERMINAL = {"completed": "success", "rejected": "rejected", "failed": "failed"}
STATUS_INCOMPLETO = "incomplete"
RESULTADO_POR_STATUS = {
    "success": "sucesso",
    "rejected": "rejeicao",
    "failed": "falha",
    "incomplete": "incompleto",
}
EVENTOS_DE_FALHA = frozenset({"rejected", "failed"})


def agora_iso() -> str:
    """Retorna o instante atual em ISO 8601 UTC com sufixo Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def categoria_da_etapa(stage: str | None) -> str:
    """Categoria operacional principal de uma etapa."""
    if stage is None:
        return CATEGORIA_DESCONHECIDA
    return CATEGORIA_POR_ETAPA.get(stage, CATEGORIA_DESCONHECIDA)


def categorias_da_etapa(stage: str | None) -> tuple[str, ...]:
    """Categorias operacionais que uma etapa alimenta."""
    if stage is None:
        return (CATEGORIA_DESCONHECIDA,)
    return CATEGORIAS_POR_ETAPA.get(stage, (categoria_da_etapa(stage),))


def resultado_do_evento(evento: str | None) -> str:
    """Resultado operacional derivado do tipo de evento."""
    if evento is None:
        return RESULTADO_DESCONHECIDO
    return RESULTADO_POR_EVENTO.get(evento, RESULTADO_DESCONHECIDO)


def resultado_do_status(status: str | None) -> str:
    """Resultado operacional derivado do status de uma execucao."""
    if status is None:
        return RESULTADO_POR_STATUS[STATUS_INCOMPLETO]
    return RESULTADO_POR_STATUS.get(status, status)


def listar_execucoes(runs_dir: Path) -> list[Path]:
    """Lista execucoes com eventos ou manifesto, incluindo publicacoes incompletas."""
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Diretorio de execucoes nao encontrado: {runs_dir}")
    return sorted(
        (
            caminho
            for caminho in runs_dir.iterdir()
            if (caminho / "events.jsonl").is_file() or (caminho / "manifest.json").is_file()
        ),
        key=lambda caminho: caminho.name,
    )


def ler_eventos(caminho: Path) -> list[dict[str, Any]]:
    """Le um `events.jsonl` e retorna a lista de eventos."""
    eventos: list[dict[str, Any]] = []
    if not caminho.is_file():
        return eventos
    with caminho.open(encoding="utf-8") as arquivo:
        for numero, linha in enumerate(arquivo, start=1):
            conteudo = linha.strip()
            if not conteudo:
                continue
            try:
                eventos.append(json.loads(conteudo))
            except json.JSONDecodeError as erro:
                raise ValueError(f"Evento invalido em {caminho} (linha {numero}): {erro}") from erro
    return eventos


def ler_manifesto(caminho: Path) -> dict[str, Any] | None:
    """Le o manifesto da execucao quando disponivel, sem interromper o processo."""
    if not caminho.is_file():
        return None
    try:
        with caminho.open(encoding="utf-8") as arquivo:
            conteudo = json.load(arquivo)
    except (json.JSONDecodeError, OSError):
        return None
    return conteudo if isinstance(conteudo, dict) else None


def inferir_status(eventos: list[dict[str, Any]]) -> str:
    """Infere o status da execucao pelo ultimo evento terminal observado."""
    for evento in reversed(eventos):
        status = STATUS_TERMINAL.get(evento.get("event"))
        if status is not None:
            return status
    return STATUS_INCOMPLETO


def classificar_evento(evento: dict[str, Any]) -> dict[str, Any]:
    """Acrescenta `categoria` e `resultado` ao evento, sem remove-lo do contrato."""
    classificado = dict(evento)
    classificado["categoria"] = categoria_da_etapa(evento.get("stage"))
    classificado["resultado"] = resultado_do_evento(evento.get("event"))
    return classificado


def analisar_execucao(run_dir: Path, eventos: list[dict[str, Any]]) -> dict[str, Any]:
    """Reune os metadados de uma execucao a partir do manifesto e dos eventos."""
    manifesto = ler_manifesto(run_dir / "manifest.json") or {}
    stage = manifesto.get("stage") or (eventos[0].get("stage") if eventos else None)
    status = status_publicado(manifesto, eventos)
    transformacoes = manifesto.get("transformations")
    return {
        "run_id": manifesto.get("run_id") or run_dir.name,
        "run_dir": run_dir.as_posix(),
        "stage": stage,
        "categoria": categoria_da_etapa(stage),
        "status": status,
        "resultado": resultado_do_status(status),
        "error_code": manifesto.get("error_code"),
        "finished_at": manifesto.get("finished_at"),
        "event_count": len(eventos),
        "transformations": list(transformacoes) if isinstance(transformacoes, list) else [],
    }


def reunir_falhas(
    execucoes: list[dict[str, Any]], eventos: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Preserva falhas dos manifestos quando falta o evento terminal equivalente."""
    falhas = [
        {
            "run_id": evento.get("run_id"),
            "stage": evento.get("stage"),
            "categoria": categoria_da_etapa(evento.get("stage")),
            "event": evento.get("event"),
            "level": evento.get("level"),
            "error_code": evento.get("error_code"),
            "timestamp": evento.get("timestamp"),
        }
        for evento in eventos
        if evento.get("event") in EVENTOS_DE_FALHA or evento.get("level") == "ERROR"
    ]
    registradas = {(falha["run_id"], falha["stage"], falha["event"]) for falha in falhas}
    for execucao in execucoes:
        status = execucao["status"]
        identidade = (execucao["run_id"], execucao["stage"], status)
        if status not in EVENTOS_DE_FALHA or identidade in registradas:
            continue
        falhas.append(
            {
                "run_id": execucao["run_id"],
                "stage": execucao["stage"],
                "categoria": categoria_da_etapa(execucao["stage"]),
                "event": status,
                "level": "ERROR",
                "error_code": execucao.get("error_code"),
                "timestamp": execucao.get("finished_at"),
            }
        )
        registradas.add(identidade)
    return falhas


def montar_resumo(
    *,
    runs_dir: Path,
    output_dir: Path,
    execucoes: list[dict[str, Any]],
    eventos: list[dict[str, Any]],
) -> dict[str, Any]:
    """Consolida contagens de status, categorias, falhas e alertas."""
    por_status: dict[str, int] = {}
    for execucao in execucoes:
        status = execucao["status"]
        por_status[status] = por_status.get(status, 0) + 1

    por_categoria: dict[str, dict[str, int]] = {
        categoria: {resultado: 0 for resultado in ORDEM_RESULTADOS}
        for categoria in CATEGORIAS_CONHECIDAS
    }
    for evento in eventos:
        resultado = evento.get("resultado", RESULTADO_DESCONHECIDO)
        for categoria in categorias_da_etapa(evento.get("stage")):
            contagem = por_categoria.setdefault(categoria, {valor: 0 for valor in ORDEM_RESULTADOS})
            contagem[resultado] = contagem.get(resultado, 0) + 1

    publicadas = {
        (execucao["run_id"], execucao["stage"])
        for execucao in execucoes
        if execucao["status"] == "success"
    }
    alertas = [
        {
            "run_id": evento.get("run_id"),
            "stage": evento.get("stage"),
            "scenario_id": evento.get("scenario_id"),
            "batch_id": evento.get("batch_id"),
            "timestamp": evento.get("timestamp"),
        }
        for evento in eventos
        if evento.get("event") == "drift_alert"
        and (evento.get("run_id"), evento.get("stage")) in publicadas
    ]
    transformacoes = [
        {
            "run_id": execucao["run_id"],
            "nomes": execucao["transformations"],
            "resultado": execucao["resultado"],
        }
        for execucao in execucoes
        if execucao["transformations"]
    ]

    return {
        "contract_version": CONTRACT_VERSION,
        "generated_at": agora_iso(),
        "runs_dir": runs_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "execucoes": {
            "total": len(execucoes),
            "por_status": por_status,
            "incompletas": [
                execucao["run_id"]
                for execucao in execucoes
                if execucao["status"] == STATUS_INCOMPLETO
            ],
        },
        "eventos": {"total": len(eventos)},
        "por_categoria": por_categoria,
        "transformacoes": transformacoes,
        "falhas": reunir_falhas(execucoes, eventos),
        "alertas": alertas,
    }


def gravar_operacional(caminho: Path, eventos: list[dict[str, Any]]) -> None:
    """Grava o log operacional consolidado em JSON Lines, um evento por linha."""
    with caminho.open("w", encoding="utf-8") as arquivo:
        for evento in eventos:
            arquivo.write(json.dumps(evento, ensure_ascii=False) + "\n")


def gravar_resumo(caminho: Path, resumo: dict[str, Any]) -> None:
    """Grava o resumo consolidado em JSON UTF-8."""
    with caminho.open("w", encoding="utf-8") as arquivo:
        json.dump(resumo, arquivo, ensure_ascii=False, indent=2)
        arquivo.write("\n")


def gerar(runs_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Consolida os logs de todas as execucoes e publica os artefatos."""
    execucoes: list[dict[str, Any]] = []
    eventos: list[dict[str, Any]] = []
    for run_dir in listar_execucoes(runs_dir):
        eventos_run = ler_eventos(run_dir / "events.jsonl")
        execucoes.append(analisar_execucao(run_dir, eventos_run))
        for evento in eventos_run:
            classificado = classificar_evento(evento)
            classificado["run_dir"] = run_dir.as_posix()
            eventos.append(classificado)

    eventos.sort(key=lambda item: (item.get("timestamp") or "", item.get("run_id") or ""))
    resumo = montar_resumo(
        runs_dir=runs_dir,
        output_dir=output_dir,
        execucoes=execucoes,
        eventos=eventos,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    gravar_operacional(output_dir / "operacional.jsonl", eventos)
    gravar_resumo(output_dir / "resumo.json", resumo)
    return resumo


def imprimir_resumo(resumo: dict[str, Any]) -> None:
    """Imprime um resumo legivel da consolidacao."""
    execucoes = resumo["execucoes"]
    por_status = execucoes["por_status"]
    detalhe = ", ".join(
        f"{status}={quantidade}" for status, quantidade in sorted(por_status.items())
    )
    print(
        f"Log operacional consolidado: {execucoes['total']} execucoes, "
        f"{resumo['eventos']['total']} eventos"
    )
    print(f"  por status: {detalhe or 'nenhuma execucao'}")
    print(f"  falhas: {len(resumo['falhas'])} | alertas: {len(resumo['alertas'])}")
    if execucoes["incompletas"]:
        print(f"  incompletas: {', '.join(execucoes['incompletas'])}")
    print(f"  {resumo['output_dir']}/operacional.jsonl")
    print(f"  {resumo['output_dir']}/resumo.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolida os logs operacionais do pipeline.")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS_DIR_PADRAO,
        help="Diretorio com as execucoes (padrao: reports/runs).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR_PADRAO,
        help="Diretorio de saida do log consolidado (padrao: reports/logs).",
    )
    args = parser.parse_args()

    try:
        resumo = gerar(args.runs_dir, args.output_dir)
    except (FileNotFoundError, ValueError) as erro:
        print(f"Falha ao consolidar os logs: {erro}")
        sys.exit(CODIGO_ERRO)

    imprimir_resumo(resumo)
    sys.exit(CODIGO_SUCESSO)


if __name__ == "__main__":
    main()

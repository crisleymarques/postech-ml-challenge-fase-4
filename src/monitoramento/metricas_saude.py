"""Metricas de saude operacional do pipeline (issue F3-02).

Le o catalogo `configs/metricas_saude.json`, coleta os valores a partir dos
artefatos persistidos por execucao (`manifest.json`, `metrics.json`,
`events.jsonl`, `quality.json`, `drift.json`, `analise.json` e
`performance.json`) e publica `reports/saude/saude.json` com os valores por
execucao, os agregados, a situacao de cada medida (`ok`, `atencao`, `critico`,
`indisponivel` ou `nao_aplicavel`) e um resumo.

O catalogo define a semantica, a unidade, a fonte de coleta e os limiares de
degradacao e falha de cada metrica. Este componente apenas mede e avalia os
criterios declarados; a sinalizacao e o alerta efetivo ficam para a issue
F3-03. Uma medida ruim nao interrompe o processo.

Codigos de saida: 0 sucesso, 1 erro tecnico.

Uso:
    uv run python src/monitoramento/metricas_saude.py \
        --runs-dir reports/runs --output-dir reports/saude \
        --catalog configs/metricas_saude.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from monitoramento.infraestrutura.arquivos import escrever_json_atomico
from monitoramento.infraestrutura.observabilidade import status_publicado

CONTRACT_VERSION = "1.0"

RUNS_DIR_PADRAO = Path("reports/runs")
OUTPUT_DIR_PADRAO = Path("reports/saude")
CATALOGO_PADRAO = Path("configs/metricas_saude.json")

CODIGO_SUCESSO = 0
CODIGO_ERRO = 1

ESCOPO_EXECUCAO = "execucao"
ESCOPO_AGREGADO = "agregado"
ESCOPOS = (ESCOPO_EXECUCAO, ESCOPO_AGREGADO)

DIRECAO_MAIOR_PIOR = "maior_pior"
DIRECAO_MENOR_PIOR = "menor_pior"

SITUACAO_OK = "ok"
SITUACAO_ATENCAO = "atencao"
SITUACAO_CRITICO = "critico"
SITUACAO_INDISPONIVEL = "indisponivel"
SITUACAO_NAO_APLICAVEL = "nao_aplicavel"
ORDEM_SITUACOES = (
    SITUACAO_OK,
    SITUACAO_ATENCAO,
    SITUACAO_CRITICO,
    SITUACAO_INDISPONIVEL,
    SITUACAO_NAO_APLICAVEL,
)

DISPONIVEL = "available"
INDISPONIVEL = "not_available"
NAO_APLICAVEL = "not_applicable"

STATUS_SUCESSO = "success"
STATUS_FALHA = "failed"
STATUS_REJEICAO = "rejected"

ETAPA_VALIDACAO = "validacao"
EVENTO_ALERTA = "drift_alert"

SITUACAO_POR_STATUS = {
    "success": SITUACAO_OK,
    "rejected": SITUACAO_ATENCAO,
    "failed": SITUACAO_CRITICO,
    "incomplete": SITUACAO_CRITICO,
}
SITUACAO_BOOLEANO_ALERTA = {False: SITUACAO_OK, True: SITUACAO_ATENCAO}
SITUACAO_BOOLEANO_FALHA = {False: SITUACAO_CRITICO, True: SITUACAO_OK}

CAMPOS_METRICA = ("id", "categoria", "nome", "unidade", "escopo", "coleta", "criterio")


def agora_iso() -> str:
    """Retorna o instante atual em ISO 8601 UTC com sufixo Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def versao_maior(versao: Any) -> str:
    """Extrai o componente principal de uma versao de contrato."""
    return str(versao or "").split(".")[0]


def carregar_catalogo(caminho: Path) -> dict[str, Any]:
    """Le e valida minimamente o catalogo de metricas de saude."""
    if not caminho.is_file():
        raise ValueError(f"Catalogo nao encontrado: {caminho}")
    with caminho.open(encoding="utf-8") as arquivo:
        catalogo = json.load(arquivo)

    if versao_maior(catalogo.get("contract_version")) != versao_maior(CONTRACT_VERSION):
        raise ValueError("Versao principal do contrato do catalogo incompativel.")

    metricas = catalogo.get("metricas")
    if not isinstance(metricas, list) or not metricas:
        raise ValueError("O catalogo deve declarar ao menos uma metrica.")

    ids: list[str] = []
    for metrica in metricas:
        if not isinstance(metrica, dict):
            raise ValueError(f"Metrica invalida no catalogo: {metrica}")
        ausentes = [campo for campo in CAMPOS_METRICA if not metrica.get(campo)]
        if ausentes:
            raise ValueError(f"Metrica {metrica.get('id')} sem campos: {ausentes}")
        if metrica["escopo"] not in ESCOPOS:
            raise ValueError(f"Escopo invalido na metrica {metrica['id']}: {metrica['escopo']}")
        etapas = metrica.get("etapas")
        if etapas is not None and (not isinstance(etapas, list) or not etapas):
            raise ValueError(f"'etapas' invalido na metrica {metrica['id']}.")
        ids.append(metrica["id"])

    if len(set(ids)) != len(ids):
        raise ValueError("Ids de metrica duplicados no catalogo.")
    return catalogo


def metricas_por_escopo(catalogo: dict[str, Any], escopo: str) -> list[dict[str, Any]]:
    """Metricas do catalogo com o escopo informado, na ordem declarada."""
    return [metrica for metrica in catalogo["metricas"] if metrica["escopo"] == escopo]


def aplicavel(metrica: dict[str, Any], stage: str | None) -> bool:
    """Indica se a metrica se aplica a etapa informada."""
    etapas = metrica.get("etapas")
    return etapas is None or stage in etapas


def ler_json(caminho: Path) -> dict[str, Any] | None:
    """Le um arquivo JSON quando disponivel, sem interromper o processo."""
    if not caminho.is_file():
        return None
    try:
        with caminho.open(encoding="utf-8") as arquivo:
            conteudo = json.load(arquivo)
    except (json.JSONDecodeError, OSError):
        return None
    return conteudo if isinstance(conteudo, dict) else None


def ler_eventos(caminho: Path) -> list[dict[str, Any]]:
    """Le um `events.jsonl` de forma tolerante, ignorando linhas invalidas."""
    eventos: list[dict[str, Any]] = []
    if not caminho.is_file():
        return eventos
    with caminho.open(encoding="utf-8") as arquivo:
        for linha in arquivo:
            conteudo = linha.strip()
            if not conteudo:
                continue
            try:
                eventos.append(json.loads(conteudo))
            except json.JSONDecodeError:
                continue
    return eventos


def listar_execucoes(runs_dir: Path) -> list[Path]:
    """Lista os diretorios de execucao com manifesto ou eventos, em ordem."""
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Diretorio de execucoes nao encontrado: {runs_dir}")
    return sorted(
        (
            caminho
            for caminho in runs_dir.iterdir()
            if (caminho / "manifest.json").is_file() or (caminho / "events.jsonl").is_file()
        ),
        key=lambda caminho: caminho.name,
    )


def _parse_iso(texto: Any) -> datetime | None:
    """Interpreta um instante ISO 8601 UTC com sufixo Z opcional."""
    if not isinstance(texto, str):
        return None
    limpo = texto[:-1] if texto.endswith("Z") else texto
    for formato in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(limpo, formato)
        except ValueError:
            continue
    return None


def _aninhado(origem: Any, *chaves: str) -> Any:
    """Percorre um caminho de chaves em objetos aninhados."""
    atual = origem
    for chave in chaves:
        if not isinstance(atual, dict):
            return None
        atual = atual.get(chave)
    return atual


def coletar_duracao(manifesto: dict[str, Any]) -> float | None:
    """Duracao em segundos entre inicio e fim da execucao."""
    inicio = _parse_iso(manifesto.get("started_at"))
    fim = _parse_iso(manifesto.get("finished_at"))
    if inicio is None or fim is None:
        return None
    return round((fim - inicio).total_seconds(), 6)


def coletar_volume(manifesto: dict[str, Any]) -> int | None:
    """Soma de linhas dos artefatos de saida (ou entrada) da execucao."""
    for chave in ("outputs", "inputs"):
        itens = manifesto.get(chave)
        if not isinstance(itens, list):
            continue
        total = 0
        encontrou = False
        for item in itens:
            if not isinstance(item, dict):
                continue
            linhas = item.get("row_count")
            if isinstance(linhas, bool) or not isinstance(linhas, (int, float)):
                continue
            total += int(linhas)
            encontrou = True
        if encontrou:
            return total
    return None


def coletar_violacoes(quality: Any, metrics: Any) -> int | None:
    """Total de ocorrencias violadas do contrato de qualidade."""
    violacoes = _aninhado(quality, "violations")
    if isinstance(violacoes, list):
        return sum(int(item.get("count", 0)) for item in violacoes if isinstance(item, dict))
    for medida in (metrics or {}).get("measurements", []) or []:
        if isinstance(medida, dict) and medida.get("name") == "violacoes_total":
            valor = medida.get("value")
            if valor is not None:
                return int(valor)
    return None


def coletar_valor_drift(drift: Any, analise: Any, chave_drift: str, chave_analise: Any) -> Any:
    """Le um valor do resumo de drift.json e, na ausencia, de analise.json."""
    for fonte, chave in ((drift, chave_drift), (analise, chave_analise)):
        valor = _aninhado(fonte, "summary", chave) if chave else None
        if valor is not None:
            return valor
    return None


def coletar_variaveis_drift(drift: Any, analise: Any) -> int | None:
    """Quantidade de variaveis com mudanca entre referencia e lote."""
    valores = coletar_valor_drift(drift, analise, "drifted_columns", "changed_columns")
    if isinstance(valores, list):
        return len(valores)
    if isinstance(valores, bool) or not isinstance(valores, (int, float)):
        return None
    return int(valores)


def contar_por_status(execucoes: list[dict[str, Any]], status: str) -> int:
    """Quantidade de execucoes com o status informado."""
    return sum(1 for execucao in execucoes if execucao.get("status") == status)


def taxa_sucesso(execucoes: list[dict[str, Any]]) -> float | None:
    """Proporcao de execucoes concluidas com status 'success'."""
    if not execucoes:
        return None
    return round(contar_por_status(execucoes, STATUS_SUCESSO) / len(execucoes), 6)


def contar_alertas(eventos: list[dict[str, Any]]) -> int:
    """Quantidade de eventos de alerta de drift."""
    return sum(1 for evento in eventos if evento.get("event") == EVENTO_ALERTA)


def contar_lotes_rejeitados(execucoes: list[dict[str, Any]]) -> int:
    """Quantidade de execucoes de validacao com lote rejeitado."""
    return sum(
        1
        for execucao in execucoes
        if execucao.get("stage") == ETAPA_VALIDACAO and execucao.get("status") == STATUS_REJEICAO
    )


def espaco_livre(output_dir: Path) -> float | None:
    """Espaco livre em bytes no volume do diretorio de saida."""
    try:
        return float(shutil.disk_usage(str(output_dir)).free)
    except OSError:
        return None


def escrita_disponivel(output_dir: Path) -> bool:
    """Verifica a capacidade de escrita gravando e removendo um arquivo temporario."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        teste = output_dir / ".escrita_saude"
        teste.write_text("ok", encoding="utf-8")
        teste.unlink()
    except OSError:
        return False
    return True


COLETORES: dict[str, Callable[[dict[str, Any]], Any]] = {
    "pipeline.status_execucao": lambda c: (c.get("manifesto") or {}).get("status"),
    "pipeline.duracao_s": lambda c: coletar_duracao(c.get("manifesto") or {}),
    "pipeline.volume_linhas": lambda c: coletar_volume(c.get("manifesto") or {}),
    "pipeline.taxa_sucesso": lambda c: taxa_sucesso(c.get("execucoes") or []),
    "pipeline.falhas_total": lambda c: contar_por_status(c.get("execucoes") or [], STATUS_FALHA),
    "qualidade.violacoes_total": lambda c: coletar_violacoes(c.get("quality"), c.get("metrics")),
    "qualidade.lotes_rejeitados": lambda c: contar_lotes_rejeitados(c.get("execucoes") or []),
    "drift.dataset_drift": lambda c: coletar_valor_drift(
        c.get("drift"), None, "dataset_drift", None
    ),
    "drift.variaveis_com_drift": lambda c: coletar_variaveis_drift(
        c.get("drift"), c.get("analise")
    ),
    "drift.proporcao_drift": lambda c: coletar_valor_drift(
        c.get("drift"), c.get("analise"), "drift_share", "change_share"
    ),
    "drift.alertas_total": lambda c: contar_alertas(c.get("eventos") or []),
    "drift.degradacao_conceito": lambda c: _aninhado(
        c.get("performance"), "comparison", "degradation_detected"
    ),
    "modelo.roc_auc": lambda c: _aninhado(
        c.get("performance"), "contrast", "concept_drift", "metrics", "roc_auc"
    ),
    "infra.espaco_livre_bytes": lambda c: espaco_livre(c["output_dir"]),
    "infra.escrita_ok": lambda c: escrita_disponivel(c["output_dir"]),
}

AVALIADORES: dict[str, Callable[[Any], str]] = {
    "pipeline.status_execucao": lambda valor: SITUACAO_POR_STATUS.get(valor, SITUACAO_INDISPONIVEL),
    "drift.dataset_drift": lambda valor: SITUACAO_BOOLEANO_ALERTA.get(valor, SITUACAO_INDISPONIVEL),
    "drift.degradacao_conceito": lambda valor: SITUACAO_BOOLEANO_ALERTA.get(
        valor, SITUACAO_INDISPONIVEL
    ),
    "infra.escrita_ok": lambda valor: SITUACAO_BOOLEANO_FALHA.get(valor, SITUACAO_INDISPONIVEL),
}


def situacao_medida(metrica: dict[str, Any], valor: Any) -> str:
    """Situacao da medida segundo os limiares e o criterio do catalogo."""
    if valor is None:
        return SITUACAO_INDISPONIVEL

    avaliador = AVALIADORES.get(metrica["id"])
    if avaliador is not None:
        return avaliador(valor)

    direcao = metrica.get("direcao")
    if direcao == DIRECAO_MAIOR_PIOR:

        def atingiu(limiar: Any) -> bool:
            return valor >= limiar

    elif direcao == DIRECAO_MENOR_PIOR:

        def atingiu(limiar: Any) -> bool:
            return valor < limiar

    else:
        return SITUACAO_OK

    limiar_falha = metrica.get("limiar_falha")
    if limiar_falha is not None and atingiu(limiar_falha):
        return SITUACAO_CRITICO
    limiar_degracao = metrica.get("limiar_degracao")
    if limiar_degracao is not None and atingiu(limiar_degracao):
        return SITUACAO_ATENCAO
    return SITUACAO_OK


def montar_medida(metrica: dict[str, Any], valor: Any) -> dict[str, Any]:
    """Descreve uma medida coletada conforme o catalogo."""
    return {
        "id": metrica["id"],
        "categoria": metrica["categoria"],
        "nome": metrica["nome"],
        "unidade": metrica["unidade"],
        "valor": valor,
        "disponibilidade": DISPONIVEL if valor is not None else INDISPONIVEL,
        "situacao": situacao_medida(metrica, valor),
    }


def coletar_metrica(metrica: dict[str, Any], contexto: dict[str, Any]) -> Any:
    """Coleta o valor de uma metrica a partir do contexto de artefatos."""
    coletor = COLETORES.get(metrica["id"])
    if coletor is None:
        return None
    try:
        return coletor(contexto)
    except (KeyError, TypeError):
        return None


def medir_execucao(metrica: dict[str, Any], contexto: dict[str, Any]) -> dict[str, Any] | None:
    """Mede uma metrica de execucao, ignorando etapas nao aplicaveis."""
    if not aplicavel(metrica, contexto.get("stage")):
        return None
    return montar_medida(metrica, coletar_metrica(metrica, contexto))


def medir_agregado(metrica: dict[str, Any], contexto: dict[str, Any]) -> dict[str, Any]:
    """Mede uma metrica de escopo agregado."""
    return montar_medida(metrica, coletar_metrica(metrica, contexto))


def referencia(run_id: str | None, medida: dict[str, Any]) -> dict[str, Any]:
    """Resume uma medida fora da faixa esperada para o resumo."""
    return {
        "run_id": run_id,
        "id": medida["id"],
        "valor": medida["valor"],
        "situacao": medida["situacao"],
    }


def montar_resumo(
    execucoes: list[dict[str, Any]],
    agregados: list[dict[str, Any]],
) -> dict[str, Any]:
    """Consolida as contagens de situacao e as medidas fora do esperado."""
    todas: list[tuple[str | None, dict[str, Any]]] = [
        (execucao["run_id"], medida) for execucao in execucoes for medida in execucao["metricas"]
    ]
    todas.extend((None, medida) for medida in agregados)

    por_situacao = {situacao: 0 for situacao in ORDEM_SITUACOES}
    for _, medida in todas:
        situacao = medida["situacao"]
        por_situacao[situacao] = por_situacao.get(situacao, 0) + 1

    return {
        "execucoes": len(execucoes),
        "medidas": len(todas),
        "por_situacao": por_situacao,
        "atencao": [
            referencia(run_id, medida)
            for run_id, medida in todas
            if medida["situacao"] == SITUACAO_ATENCAO
        ],
        "criticos": [
            referencia(run_id, medida)
            for run_id, medida in todas
            if medida["situacao"] == SITUACAO_CRITICO
        ],
    }


def gravar_saude(caminho: Path, saude: dict[str, Any]) -> None:
    """Publica o JSON completo, preservando o relatorio anterior se houver falha."""
    escrever_json_atomico(caminho, saude)


def gerar(runs_dir: Path, output_dir: Path, catalogo_path: Path) -> dict[str, Any]:
    """Coleta as metricas de saude e publica `saude.json`."""
    catalogo = carregar_catalogo(catalogo_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    execucoes_saida: list[dict[str, Any]] = []
    eventos: list[dict[str, Any]] = []
    resumos: list[dict[str, Any]] = []
    for run_dir in listar_execucoes(runs_dir):
        manifesto = ler_json(run_dir / "manifest.json") or {}
        eventos_run = ler_eventos(run_dir / "events.jsonl")
        manifesto = {**manifesto, "status": status_publicado(manifesto, eventos_run)}
        publicado = manifesto["status"] == STATUS_SUCESSO
        evento_inicial = eventos_run[0] if eventos_run else {}
        stage = manifesto.get("stage") or evento_inicial.get("stage")
        contexto = {
            "run_dir": run_dir,
            "manifesto": manifesto,
            "metrics": ler_json(run_dir / "metrics.json"),
            "quality": ler_json(run_dir / "quality.json"),
            "drift": ler_json(run_dir / "drift.json") if publicado else None,
            "analise": ler_json(run_dir / "analise.json") if publicado else None,
            "performance": ler_json(run_dir / "performance.json") if publicado else None,
            "stage": stage,
        }
        medidas = [
            medida
            for medida in (
                medir_execucao(metrica, contexto)
                for metrica in metricas_por_escopo(catalogo, ESCOPO_EXECUCAO)
            )
            if medida is not None
        ]
        run_id = manifesto.get("run_id") or run_dir.name
        execucoes_saida.append(
            {
                "run_id": run_id,
                "stage": stage,
                "status": manifesto.get("status"),
                "metricas": medidas,
            }
        )
        resumos.append({"run_id": run_id, "stage": stage, "status": manifesto.get("status")})
        eventos.extend(e for e in eventos_run if publicado or e.get("event") != EVENTO_ALERTA)

    contexto_agregado = {
        "execucoes": resumos,
        "eventos": eventos,
        "output_dir": output_dir,
    }
    agregados = [
        medir_agregado(metrica, contexto_agregado)
        for metrica in metricas_por_escopo(catalogo, ESCOPO_AGREGADO)
    ]

    saude = {
        "contract_version": CONTRACT_VERSION,
        "generated_at": agora_iso(),
        "runs_dir": runs_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "catalog_path": catalogo_path.as_posix(),
        "execucoes": execucoes_saida,
        "agregados": agregados,
        "resumo": montar_resumo(execucoes_saida, agregados),
    }
    gravar_saude(output_dir / "saude.json", saude)
    return saude


def imprimir_saude(saude: dict[str, Any]) -> None:
    """Imprime um resumo legivel da saude operacional."""
    resumo = saude["resumo"]
    situacoes = ", ".join(
        f"{situacao}={quantidade}"
        for situacao, quantidade in resumo["por_situacao"].items()
        if quantidade
    )
    print(f"Saude operacional: {resumo['execucoes']} execucoes, {resumo['medidas']} medidas")
    print(f"  situacoes: {situacoes or 'nenhuma'}")
    print(f"  atencao: {len(resumo['atencao'])} | criticos: {len(resumo['criticos'])}")
    for item in resumo["criticos"]:
        origem = item["run_id"] or "agregado"
        print(f"    critico: {item['id']} = {item['valor']} ({origem})")
    print(f"  {saude['output_dir']}/saude.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Coleta as metricas de saude operacional do pipeline."
    )
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
        help="Diretorio de saida do relatorio (padrao: reports/saude).",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=CATALOGO_PADRAO,
        help="Catalogo de metricas (padrao: configs/metricas_saude.json).",
    )
    args = parser.parse_args()

    try:
        saude = gerar(args.runs_dir, args.output_dir, args.catalog)
    except (FileNotFoundError, ValueError) as erro:
        print(f"Falha ao coletar as metricas de saude: {erro}")
        sys.exit(CODIGO_ERRO)

    imprimir_saude(saude)
    sys.exit(CODIGO_SUCESSO)


if __name__ == "__main__":
    main()

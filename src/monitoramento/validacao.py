"""Contrato de qualidade de dados com Pandera (issue F1-03).

Le o manifesto de um produtor (`preparacao` ou `simulacao`), confere os hashes
dos artefatos tabulares e aplica o contrato Pandera descrito na configuracao.
Publica o relatorio de qualidade (`quality.json`) e as saidas de execucao
(`config.json`, `events.jsonl`, `metrics.json`, `manifest.json`) em
`reports/runs/<run_id>/`. Nao reescreve nem descarta registros: uma violacao
bloqueia o conjunto inteiro.

Codigos de saida: 0 aprovado, 2 rejeitado por qualidade, 1 erro tecnico.

Uso:
    uv run python src/monitoramento/validacao.py \
        --config configs/validacao.json \
        --input-manifest reports/runs/<run_id>/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

CONTRACT_VERSION = "1.0"
STAGE = "validacao"
ETAPAS_PRODUTORAS = ("preparacao", "simulacao")

CODIGO_SUCESSO = 0
CODIGO_ERRO = 1
CODIGO_REJEICAO = 2

CHAVES_REGRAS = ("identifier", "target", "not_null", "unique", "ranges", "categories")
MINIMO_REGRAS = 3


class ErroArtefato(Exception):
    """Indica dependencia ausente, divergente ou incompativel."""


def agora_iso() -> str:
    """Retorna o instante atual em ISO 8601 UTC com sufixo Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def sha256_arquivo(caminho: Path) -> str:
    """Calcula o hash SHA-256 de um arquivo em blocos."""
    resumo = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            resumo.update(bloco)
    return resumo.hexdigest()


def escrever_json(caminho: Path, conteudo: dict[str, Any]) -> None:
    """Grava um objeto JSON em UTF-8."""
    with caminho.open("w", encoding="utf-8") as arquivo:
        json.dump(conteudo, arquivo, ensure_ascii=False, indent=2)
        arquivo.write("\n")


def versao_maior(versao: Any) -> str:
    """Extrai o componente principal de uma versao de contrato."""
    return str(versao or "").split(".")[0]


def carregar_config(caminho: Path) -> dict[str, Any]:
    """Le e valida minimamente a configuracao da etapa."""
    if not caminho.is_file():
        raise ValueError(f"Configuracao nao encontrada: {caminho}")
    with caminho.open(encoding="utf-8") as arquivo:
        config = json.load(arquivo)

    if versao_maior(config.get("contract_version")) != versao_maior(CONTRACT_VERSION):
        raise ValueError("Versao principal do contrato da configuracao incompativel.")

    obrigatorios = ("dataset_id", "schema_version", "rules")
    ausentes = [chave for chave in obrigatorios if chave not in config]
    if ausentes:
        raise ValueError(f"Chaves ausentes na configuracao: {ausentes}")
    if not config["schema_version"]:
        raise ValueError("schema_version obrigatorio para a validacao de dados.")

    perfil = config.get("profile", "reference")
    if perfil not in ("reference", "production"):
        raise ValueError("profile deve ser 'reference' ou 'production'.")
    config["profile"] = perfil
    config["artifacts"] = (
        ["features"]
        if perfil == "production"
        else config.get("artifacts", ["train", "validation", "test"])
    )

    regras = config["rules"]
    ausentes = [chave for chave in CHAVES_REGRAS if chave not in regras]
    if ausentes:
        raise ValueError(f"Regras ausentes na configuracao: {ausentes}")

    quantidade = (
        2
        + len(regras["not_null"])
        + len(regras["unique"])
        + len(regras["ranges"])
        + len(regras["categories"])
    )
    if quantidade < MINIMO_REGRAS:
        raise ValueError(f"O contrato exige ao menos {MINIMO_REGRAS} regras rigidas.")

    for coluna, faixa in regras["ranges"].items():
        if len(faixa) != 2 or faixa[0] > faixa[1]:
            raise ValueError(f"Faixa invalida para {coluna}: {faixa}")
    return config


def carregar_manifesto(caminho: Path) -> dict[str, Any]:
    """Le o manifesto de entrada e exige o produtor com sucesso."""
    if not caminho.is_file():
        raise ValueError(f"Manifesto de entrada nao encontrado: {caminho}")
    with caminho.open(encoding="utf-8") as arquivo:
        manifesto = json.load(arquivo)

    if versao_maior(manifesto.get("contract_version")) != versao_maior(CONTRACT_VERSION):
        raise ValueError("Versao principal do contrato do manifesto incompativel.")
    if manifesto.get("stage") not in ETAPAS_PRODUTORAS:
        raise ValueError(f"Etapa produtora inesperada: {manifesto.get('stage')}")
    if manifesto.get("status") != "success":
        raise ValueError(f"Dependencia sem sucesso: {manifesto.get('status')}")
    return manifesto


def resolver_artefatos(
    manifesto: dict[str, Any], config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Resolve e verifica os artefatos tabulares do manifesto produtor."""
    selecionados = config.get("artifacts")
    artefatos: dict[str, dict[str, Any]] = {}
    for item in manifesto.get("outputs", []):
        if "row_count" not in item or "columns" not in item:
            continue
        tipo = item.get("artifact_type")
        if selecionados and tipo not in selecionados:
            continue
        artefatos[tipo] = item

    if selecionados:
        faltantes = [tipo for tipo in selecionados if tipo not in artefatos]
        if faltantes:
            raise ErroArtefato(f"Artefatos ausentes no manifesto: {faltantes}")
    if not artefatos:
        raise ErroArtefato("Nenhum artefato tabular para validar no manifesto.")

    for tipo, item in artefatos.items():
        caminho = Path(item["path"])
        if not caminho.is_file():
            raise ErroArtefato(f"Arquivo do artefato ausente: {caminho}")
        if sha256_arquivo(caminho) != item.get("sha256"):
            raise ErroArtefato(f"Hash divergente para o artefato {tipo}.")
    return artefatos


def _formato(serie: pd.Series, padrao: str) -> pd.Series:
    """Sinaliza identificadores que seguem o padrao combinado."""
    return serie.isna() | serie.astype(str).str.fullmatch(padrao)


def _faixa(serie: pd.Series, minimo: float, maximo: float) -> pd.Series:
    """Aceita ausencias, mas rejeita valores presentes que nao sejam numericos."""
    valores = pd.to_numeric(serie, errors="coerce")
    ausente_original = serie.isna()
    return (ausente_original & valores.isna()) | valores.between(minimo, maximo)


def _dominio_numerico(serie: pd.Series, dominio: list[Any]) -> pd.Series:
    """Aceita ausencias, mas rejeita valores presentes fora do dominio numerico."""
    valores = pd.to_numeric(serie, errors="coerce")
    ausente_original = serie.isna()
    return (ausente_original & valores.isna()) | valores.isin(dominio)


def _dominio_texto(serie: pd.Series, dominio: list[str]) -> pd.Series:
    """Sinaliza categorias pertencentes ao dominio esperado."""
    return serie.isna() | serie.isin(dominio)


def _nao_duplicado(serie: pd.Series) -> pd.Series:
    """Sinaliza linhas envolvidas em duplicidade; ausente nao participa."""
    return ~serie.duplicated(keep=False)


def construir_contrato(config: dict[str, Any]) -> pa.DataFrameSchema:
    """Monta o contrato Pandera reutilizavel a partir da configuracao."""
    regras = config["rules"]
    perfil = config.get("profile", "reference")
    identificador = regras["identifier"]["column"]
    alvo = regras["target"]["column"]
    exige_alvo = perfil == "reference"
    obrigatorias = set(regras["not_null"]) | {identificador}
    if exige_alvo:
        obrigatorias.add(alvo)

    checks: dict[str, list[pa.Check]] = {}

    def adicionar(coluna: str, check: pa.Check) -> None:
        checks.setdefault(coluna, []).append(check)

    adicionar(
        identificador,
        pa.Check(lambda s: _formato(s, regras["identifier"]["pattern"]), name="R_ID_FORMATO"),
    )
    if exige_alvo:
        adicionar(
            alvo,
            pa.Check(
                lambda s: _dominio_numerico(s, regras["target"]["domain"]),
                name="R_ALVO_DOMINIO",
            ),
        )

    for coluna in regras["unique"]:
        adicionar(coluna, pa.Check(_nao_duplicado, name=f"R_UNICO_{coluna}"))

    for coluna, (minimo, maximo) in regras["ranges"].items():
        adicionar(
            coluna,
            pa.Check(lambda s, lo=minimo, hi=maximo: _faixa(s, lo, hi), name=f"R_FAIXA_{coluna}"),
        )

    for coluna, dominio in regras["categories"].items():
        adicionar(
            coluna,
            pa.Check(lambda s, d=dominio: _dominio_texto(s, d), name=f"R_CATEGORIA_{coluna}"),
        )

    colunas_declaradas = [
        identificador,
        *([alvo] if exige_alvo else []),
        *regras["unique"],
        *regras["ranges"],
        *regras["categories"],
        *(coluna for coluna in regras["not_null"] if exige_alvo or coluna != alvo),
    ]
    colunas = {
        coluna: pa.Column(str, checks=checks.get(coluna, []), nullable=coluna not in obrigatorias)
        for coluna in dict.fromkeys(colunas_declaradas)
    }
    return pa.DataFrameSchema(columns=colunas, strict=False)


def agregar_violacoes(falhas: pd.DataFrame) -> list[dict[str, Any]]:
    """Consolida as falhas do Pandera em regra violada e contagem."""
    if falhas is None or falhas.empty:
        return []
    registros = falhas.copy()
    registros["rule_id"] = registros["check"].astype(str)
    if "column" in registros.columns:
        nulos = registros["rule_id"] == "not_nullable"
        registros.loc[nulos, "rule_id"] = "R_NAO_NULO_" + registros.loc[nulos, "column"].astype(str)
    contagens = registros["rule_id"].value_counts()
    return [
        {"rule_id": str(regra), "count": int(quantidade)}
        for regra, quantidade in sorted(contagens.items())
    ]


def validar_artefato(caminho: Path, schema: pa.DataFrameSchema) -> tuple[int, list[dict[str, Any]]]:
    """Aplica o contrato a um artefato tabular e devolve linhas e violacoes."""
    base = pd.read_csv(caminho, dtype=str)
    try:
        schema.validate(base, lazy=True)
    except SchemaErrors as erro:
        return len(base), agregar_violacoes(erro.failure_cases)
    return len(base), []


def descritor_artefato(tipo: str, caminho: Path, run_id: str) -> dict[str, Any]:
    """Descreve um artefato conforme o manifesto comum."""
    return {
        "artifact_type": tipo,
        "path": caminho.as_posix(),
        "sha256": sha256_arquivo(caminho),
        "producer_run_id": run_id,
    }


def registrar_evento(
    eventos: list[dict[str, Any]],
    run_id: str,
    dataset_id: str,
    evento: str,
    nivel: str,
    error_code: str | None = None,
    scenario_id: str | None = None,
    batch_id: str | None = None,
) -> None:
    """Acrescenta um evento ao contrato comum de observabilidade."""
    eventos.append(
        {
            "timestamp": agora_iso(),
            "run_id": run_id,
            "stage": STAGE,
            "event": evento,
            "level": nivel,
            "dataset_id": dataset_id,
            "model_id": None,
            "reference_id": None,
            "scenario_id": scenario_id,
            "batch_id": batch_id,
            "error_code": error_code,
        }
    )


def escrever_eventos(caminho: Path, eventos: list[dict[str, Any]]) -> None:
    """Grava os eventos em JSON Lines, um objeto por linha."""
    with caminho.open("w", encoding="utf-8") as arquivo:
        for evento in eventos:
            arquivo.write(json.dumps(evento, ensure_ascii=False) + "\n")


def montar_manifesto(
    *,
    run_id: str,
    status: str,
    iniciado: str,
    finalizado: str,
    config_path: Path,
    config: dict[str, Any],
    entradas: list[dict[str, Any]],
    saidas: list[dict[str, Any]],
    relatorio: dict[str, Any] | None,
    error_code: str | None,
) -> dict[str, Any]:
    """Reune o manifesto comum da etapa de validacao."""
    resumo: dict[str, Any] | None = None
    if relatorio is not None:
        resumo = {
            "dataset_id": relatorio["dataset_id"],
            "schema_version": relatorio["schema_version"],
            "approved": relatorio["approved"],
            "checked_rows": relatorio["checked_rows"],
            "violations": relatorio["violations"],
            "artifacts": [
                {
                    "artifact_type": item["artifact_type"],
                    "sha256": item["sha256"],
                    "approved": item["approved"],
                }
                for item in relatorio["artifacts"]
            ],
        }
    return {
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "stage": STAGE,
        "status": status,
        "started_at": iniciado,
        "finished_at": finalizado,
        "config_path": config_path.as_posix(),
        "inputs": entradas,
        "outputs": saidas,
        "dataset_id": config.get("dataset_id"),
        "schema_version": config.get("schema_version"),
        "target_contract": relatorio.get("target_contract") if relatorio else None,
        "model_id": None,
        "reference_id": None,
        "scenario_id": next(
            (item.get("scenario_id") for item in entradas if item.get("scenario_id") is not None),
            None,
        ),
        "batch_id": next(
            (item.get("batch_id") for item in entradas if item.get("batch_id") is not None),
            None,
        ),
        "error_code": error_code,
        "quality": resumo,
    }


def montar_medidas(relatorio: dict[str, Any]) -> list[dict[str, Any]]:
    """Converte o relatorio de qualidade em medidas do contrato de metricas."""
    total_violacoes = sum(item["count"] for item in relatorio["violations"])
    return [
        {
            "name": "linhas_verificadas",
            "value": relatorio["checked_rows"],
            "unit": "linhas",
            "availability": "available",
        },
        {
            "name": "violacoes_total",
            "value": total_violacoes,
            "unit": "ocorrencias",
            "availability": "available",
        },
        {
            "name": "regras_violadas",
            "value": len(relatorio["violations"]),
            "unit": "regras",
            "availability": "available",
        },
    ]


def finalizar(
    codigo: int,
    status: str,
    error_code: str | None,
    mensagem: str | None,
    run_id: str,
    iniciado: str,
    config_path: Path,
    config: dict[str, Any],
    eventos: list[dict[str, Any]],
    entradas: list[dict[str, Any]],
    saidas: list[dict[str, Any]],
    relatorio: dict[str, Any] | None,
    output_dir: Path,
) -> int:
    """Persiste config, eventos, metricas e manifesto e imprime o desfecho."""
    escrever_json(output_dir / "config.json", config)
    escrever_eventos(output_dir / "events.jsonl", eventos)

    medidas = montar_medidas(relatorio) if relatorio is not None else []
    escrever_json(output_dir / "metrics.json", {"run_id": run_id, "measurements": medidas})

    manifesto = montar_manifesto(
        run_id=run_id,
        status=status,
        iniciado=iniciado,
        finalizado=agora_iso(),
        config_path=config_path,
        config=config,
        entradas=entradas,
        saidas=saidas,
        relatorio=relatorio,
        error_code=error_code,
    )
    escrever_json(output_dir / "manifest.json", manifesto)

    if status == "success":
        print(f"Validacao aprovada: {relatorio['checked_rows']:,} linhas verificadas")
        for artefato in relatorio["artifacts"]:
            print(f"  {artefato['artifact_type']:<14} {artefato['checked_rows']:>7} linhas")
        print(f"  quality_report -> {output_dir.as_posix()}/quality.json")
    elif status == "rejected":
        print(f"Validacao rejeitada: {mensagem}")
        for item in relatorio["violations"]:
            print(f"  {item['rule_id']:<28} {item['count']:>6} ocorrencias")
    else:
        print(f"Validacao {status}: {mensagem}")
    return codigo


def processar(
    config_path: Path,
    config: dict[str, Any],
    manifesto: dict[str, Any],
    run_id: str,
    output_dir: Path,
) -> int:
    """Executa a validacao completa e retorna o codigo de saida."""
    config = dict(config)
    perfil = config.get("profile", "reference")
    if perfil not in ("reference", "production"):
        raise ValueError("profile deve ser 'reference' ou 'production'.")
    config["profile"] = perfil
    config["artifacts"] = (
        ["features"]
        if perfil == "production"
        else config.get("artifacts", ["train", "validation", "test"])
    )
    iniciado = agora_iso()
    dataset_id = config.get("dataset_id", "desconhecido")
    eventos: list[dict[str, Any]] = []
    ids_evento = {
        "scenario_id": manifesto.get("scenario_id"),
        "batch_id": manifesto.get("batch_id"),
    }
    registrar_evento(eventos, run_id, dataset_id, "started", "INFO", **ids_evento)

    entradas: list[dict[str, Any]] = []
    saidas: list[dict[str, Any]] = []
    relatorio: dict[str, Any] | None = None

    try:
        if manifesto.get("dataset_id") != config["dataset_id"]:
            raise ErroArtefato("dataset_id divergente entre configuracao e manifesto.")

        target_contract = manifesto.get("target")
        if config.get("profile", "reference") == "reference":
            if not isinstance(target_contract, dict):
                raise ErroArtefato("Manifesto de preparacao sem contrato semantico do alvo.")
            regras_alvo = config["rules"]["target"]
            dominio_esperado = {str(valor) for valor in regras_alvo["domain"]}
            dominio_origem = {
                str(target_contract.get("positive_value")),
                str(target_contract.get("negative_value")),
            }
            if (
                target_contract.get("column") != regras_alvo["column"]
                or None
                in (
                    target_contract.get("positive_class"),
                    target_contract.get("negative_class"),
                )
                or target_contract.get("positive_class") == target_contract.get("negative_class")
                or dominio_origem != dominio_esperado
            ):
                raise ErroArtefato("Contrato semantico do alvo incompativel com a validacao.")

        artefatos = resolver_artefatos(manifesto, config)
        schema = construir_contrato(config)

        resultados: list[dict[str, Any]] = []
        violacoes_agregadas: dict[str, int] = {}
        linhas_verificadas = 0
        for tipo, artefato in artefatos.items():
            caminho = Path(artefato["path"])
            linhas, violacoes = validar_artefato(caminho, schema)
            linhas_verificadas += linhas
            for item in violacoes:
                violacoes_agregadas[item["rule_id"]] = (
                    violacoes_agregadas.get(item["rule_id"], 0) + item["count"]
                )
            resultados.append(
                {
                    "artifact_type": tipo,
                    "path": artefato["path"],
                    "sha256": artefato["sha256"],
                    "checked_rows": linhas,
                    "approved": not violacoes,
                    "violations": violacoes,
                }
            )
            entradas.append(
                {
                    "artifact_type": tipo,
                    "path": artefato["path"],
                    "sha256": artefato["sha256"],
                    "producer_run_id": artefato.get("producer_run_id"),
                    "row_count": artefato.get("row_count"),
                    "columns": artefato.get("columns"),
                    "scenario_id": manifesto.get("scenario_id"),
                    "batch_id": manifesto.get("batch_id"),
                }
            )

        violacoes = [
            {"rule_id": regra, "count": quantidade}
            for regra, quantidade in sorted(violacoes_agregadas.items())
        ]
        aprovado = all(item["approved"] for item in resultados)
        relatorio = {
            "dataset_id": config["dataset_id"],
            "schema_version": config["schema_version"],
            "target_contract": (
                target_contract if config.get("profile", "reference") == "reference" else None
            ),
            "approved": aprovado,
            "checked_rows": linhas_verificadas,
            "violations": violacoes,
            "artifacts": resultados,
        }
        caminho_relatorio = output_dir / "quality.json"
        escrever_json(caminho_relatorio, relatorio)
        saidas.append(descritor_artefato("quality_report", caminho_relatorio, run_id))
    except ErroArtefato as erro:
        registrar_evento(
            eventos,
            run_id,
            dataset_id,
            "failed",
            "ERROR",
            error_code="ARTEFATO_INCOMPATIVEL",
            **ids_evento,
        )
        return finalizar(
            CODIGO_ERRO,
            "failed",
            "ARTEFATO_INCOMPATIVEL",
            str(erro),
            run_id,
            iniciado,
            config_path,
            config,
            eventos,
            entradas,
            saidas,
            relatorio,
            output_dir,
        )
    except Exception as erro:  # noqa: BLE001 - entrada invalida e erro tecnico
        registrar_evento(
            eventos,
            run_id,
            dataset_id,
            "failed",
            "ERROR",
            error_code="ERRO_TECNICO",
            **ids_evento,
        )
        return finalizar(
            CODIGO_ERRO,
            "failed",
            "ERRO_TECNICO",
            str(erro),
            run_id,
            iniciado,
            config_path,
            config,
            eventos,
            entradas,
            saidas,
            relatorio,
            output_dir,
        )

    if not aprovado:
        registrar_evento(
            eventos,
            run_id,
            dataset_id,
            "rejected",
            "ERROR",
            error_code="QUALIDADE_REPROVADA",
            **ids_evento,
        )
        return finalizar(
            CODIGO_REJEICAO,
            "rejected",
            "QUALIDADE_REPROVADA",
            "Dados reprovados pelo contrato de qualidade.",
            run_id,
            iniciado,
            config_path,
            config,
            eventos,
            entradas,
            saidas,
            relatorio,
            output_dir,
        )

    registrar_evento(eventos, run_id, dataset_id, "completed", "INFO", **ids_evento)
    return finalizar(
        CODIGO_SUCESSO,
        "success",
        None,
        None,
        run_id,
        iniciado,
        config_path,
        config,
        eventos,
        entradas,
        saidas,
        relatorio,
        output_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Valida dados contra o contrato de qualidade.")
    parser.add_argument("--config", type=Path, required=True, help="JSON de configuracao da etapa.")
    parser.add_argument(
        "--input-manifest",
        type=Path,
        action="append",
        required=True,
        help="Manifesto do produtor com os artefatos a validar.",
    )
    parser.add_argument("--run-id", default=None, help="Identificador unico da execucao.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Diretorio novo da execucao.")
    parser.add_argument(
        "--profile",
        choices=("reference", "production"),
        default=None,
        help="Perfil do contrato; production valida features sem exigir alvo.",
    )
    args = parser.parse_args()

    if len(args.input_manifest) != 1:
        parser.error("A validacao consome exatamente um manifesto produtor.")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ")
    output_dir = args.output_dir or Path("reports/runs") / run_id

    try:
        config = carregar_config(args.config)
        if args.profile:
            config["profile"] = args.profile
        config["artifacts"] = (
            ["features"]
            if config["profile"] == "production"
            else config.get("artifacts", ["train", "validation", "test"])
        )
        manifesto = carregar_manifesto(args.input_manifest[0])
    except Exception as erro:  # noqa: BLE001 - entrada invalida e erro tecnico
        print(f"Entrada invalida: {erro}")
        sys.exit(CODIGO_ERRO)

    if output_dir.exists():
        print(f"Diretorio de execucao ja existe: {output_dir}")
        sys.exit(CODIGO_ERRO)
    output_dir.mkdir(parents=True)

    sys.exit(processar(args.config, config, manifesto, run_id, output_dir))


if __name__ == "__main__":
    main()

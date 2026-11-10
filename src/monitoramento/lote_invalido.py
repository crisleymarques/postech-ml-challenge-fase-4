"""Gera um lote propositalmente invalido (issue F1-04).

Parte de uma base de origem, monta um lote com anomalias intencionais e
deterministicas (uma para cada tipo de regra do contrato de qualidade) e
publica o manifesto do produtor. O lote e submetido em seguida a validacao,
que deve reprova-lo e impedir o consumo pelas etapas seguintes do pipeline.

Codigos de saida: 0 sucesso, 1 erro tecnico.

Uso:
    uv run python src/monitoramento/lote_invalido.py \
        --config configs/validacao.json \
        --source data/analysis/lending_club_100k_v1/simulation_base.csv
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

CONTRACT_VERSION = "1.0"
STAGE = "simulacao"
SCENARIO_ID = "lote-invalido"

CODIGO_SUCESSO = 0
CODIGO_ERRO = 1

PARTICOES = ("train", "validation", "test")
PROPORCOES = {"train": 0.6, "validation": 0.2, "test": 0.2}

# Anomalia intencional por particao: (regra esperada, coluna, valor).
# O valor None indica copia da linha, produzindo a violacao de unicidade.
ANOMALIAS: dict[str, tuple[tuple[str, str, str | None], ...]] = {
    "train": (
        ("R_ID_FORMATO", "record_id", "LC12345"),
        ("R_UNICO_record_id", "record_id", None),
    ),
    "validation": (
        ("R_NAO_NULO_loan_amnt", "loan_amnt", ""),
        ("R_FAIXA_loan_amnt", "loan_amnt", "999999"),
    ),
    "test": (
        ("R_CATEGORIA_term", "term", "48 months"),
        ("R_ALVO_DOMINIO", "loan_status", "3"),
    ),
}


class ErroLote(Exception):
    """Indica entrada invalida para a geracao do lote."""


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
    """Le a configuracao de validacao que o lote devera satisfazer."""
    if not caminho.is_file():
        raise ErroLote(f"Configuracao nao encontrada: {caminho}")
    with caminho.open(encoding="utf-8") as arquivo:
        config = json.load(arquivo)

    if not isinstance(config, dict):
        raise ValueError("A configuracao deve ser um objeto JSON.")
    if versao_maior(config.get("contract_version")) != versao_maior(CONTRACT_VERSION):
        raise ValueError("Versao principal do contrato da configuracao incompativel.")

    obrigatorios = ("dataset_id", "schema_version", "artifacts", "rules", "target_contract")
    ausentes = [chave for chave in obrigatorios if not config.get(chave)]
    if ausentes:
        raise ValueError(f"Chaves ausentes na configuracao: {ausentes}")
    if config.get("profile") != "reference":
        raise ValueError("O gerador aceita somente profile 'reference'.")
    artifacts = config["artifacts"]
    if (
        not isinstance(artifacts, list)
        or any(not isinstance(nome, str) for nome in artifacts)
        or len(artifacts) != len(set(artifacts))
        or tuple(artifacts) != PARTICOES
    ):
        raise ValueError("artifacts deve conter exatamente train, validation e test, nessa ordem.")
    target = config["target_contract"]
    regra_alvo = config["rules"].get("target", {}) if isinstance(config["rules"], dict) else {}
    if (
        not isinstance(target, dict)
        or not isinstance(regra_alvo, dict)
        or target.get("column") != regra_alvo.get("column")
        or {str(target.get("positive_value")), str(target.get("negative_value"))}
        != {str(valor) for valor in regra_alvo.get("domain", [])}
        or not isinstance(target.get("positive_class"), str)
        or not target["positive_class"].strip()
        or not isinstance(target.get("negative_class"), str)
        or not target["negative_class"].strip()
        or target["positive_class"] == target["negative_class"]
        or type(target.get("positive_value")) is not int
        or type(target.get("negative_value")) is not int
    ):
        raise ValueError("target_contract incompativel com rules.target.")
    return config


def carregar_base(caminho: Path) -> pd.DataFrame:
    """Le a base de origem mantendo os valores como texto."""
    if not caminho.is_file():
        raise ErroLote(f"Base de origem nao encontrada: {caminho}")
    return pd.read_csv(caminho, dtype=str, na_filter=False)


def dividir_lote(
    base: pd.DataFrame, particoes: tuple[str, ...], total: int, seed: int
) -> dict[str, pd.DataFrame]:
    """Amostra a base e divide o lote nas particoes previstas."""
    if len(particoes) > total:
        raise ErroLote(f"Lote de {total} linhas menor que o numero de particoes.")
    if total > len(base):
        raise ErroLote(f"Base com {len(base)} linhas insuficiente para o lote de {total}.")

    amostra = base.sample(n=total, random_state=seed).reset_index(drop=True)
    tamanhos: dict[str, int] = {}
    acumulado = 0
    for posicao, nome in enumerate(particoes):
        if posicao == len(particoes) - 1:
            tamanhos[nome] = total - acumulado
        else:
            tamanhos[nome] = max(1, int(round(total * PROPORCOES.get(nome, 1 / len(particoes)))))
            acumulado += tamanhos[nome]
    if acumulado > total or any(tamanho < 1 for tamanho in tamanhos.values()):
        raise ErroLote(f"Proporcoes invalidas para o lote de {total} linhas.")

    fatias: dict[str, pd.DataFrame] = {}
    inicio = 0
    for nome in particoes:
        fim = inicio + tamanhos[nome]
        fatias[nome] = amostra.iloc[inicio:fim].reset_index(drop=True)
        inicio = fim
    return fatias


def aplicar_anomalias(
    base: pd.DataFrame, anomalias: tuple[tuple[str, str, str | None], ...]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Injeta as anomalias intencionais e descreve cada uma delas."""
    if len(base) < len(anomalias):
        raise ErroLote("Particao com linhas insuficientes para as anomalias previstas.")

    ajustado = base.reset_index(drop=True).copy()
    descritas: list[dict[str, Any]] = []
    duplicar: list[int] = []
    for posicao, (regra, coluna, valor) in enumerate(anomalias):
        if coluna not in ajustado.columns:
            raise ErroLote(f"Coluna da anomalia ausente na particao: {coluna}")
        registro = {"rule_id": regra, "column": coluna, "row": posicao}
        if valor is None:
            duplicar.append(posicao)
            registro["injected"] = "duplicata"
        else:
            ajustado.at[posicao, coluna] = valor
            registro["injected"] = valor
        descritas.append(registro)

    if duplicar:
        copias = ajustado.iloc[duplicar]
        ajustado = pd.concat([ajustado, copias], ignore_index=True)
    return ajustado, descritas


def escrever_particoes(destino: Path, particoes: dict[str, pd.DataFrame]) -> dict[str, Path]:
    """Grava as particoes do lote em CSV."""
    destino.mkdir(parents=True, exist_ok=False)
    caminhos: dict[str, Path] = {}
    for nome, base in particoes.items():
        caminho = destino / f"{nome}.csv"
        base.to_csv(caminho, index=False, encoding="utf-8")
        caminhos[nome] = caminho
    return caminhos


def descritor_artefato(tipo: str, caminho: Path, run_id: str, base: pd.DataFrame) -> dict[str, Any]:
    """Descreve um artefato conforme o manifesto comum."""
    return {
        "artifact_type": tipo,
        "path": caminho.as_posix(),
        "sha256": sha256_arquivo(caminho),
        "producer_run_id": run_id,
        "row_count": int(len(base)),
        "columns": list(base.columns),
    }


def registrar_evento(
    eventos: list[dict[str, Any]],
    run_id: str,
    dataset_id: str,
    evento: str,
    nivel: str,
    batch_id: str,
    error_code: str | None = None,
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
            "scenario_id": SCENARIO_ID,
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
    iniciado: str,
    finalizado: str,
    config_path: Path,
    config: dict[str, Any],
    batch_id: str,
    entrada: dict[str, Any] | None,
    saidas: list[dict[str, Any]],
    anomalias: list[dict[str, Any]],
    status: str = "success",
    error_code: str | None = None,
) -> dict[str, Any]:
    """Reune o manifesto comum do produtor do lote invalido."""
    return {
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "stage": STAGE,
        "status": status,
        "started_at": iniciado,
        "finished_at": finalizado,
        "config_path": config_path.as_posix(),
        "inputs": [entrada] if entrada else [],
        "outputs": saidas,
        "dataset_id": config.get("dataset_id"),
        "schema_version": config.get("schema_version"),
        "model_id": None,
        "reference_id": None,
        "scenario_id": SCENARIO_ID,
        "batch_id": batch_id,
        "seed": config.get("_seed"),
        "transformacoes": ["injecao de anomalias intencionais"],
        "variaveis_alteradas": sorted({item["column"] for item in anomalias}),
        "anomalies": anomalias,
        "error_code": error_code,
    }


def montar_medidas(saidas: list[dict[str, Any]], anomalias: list[dict[str, Any]]) -> list[dict]:
    """Converte o lote gerado em medidas do contrato de metricas."""
    linhas = sum(item["row_count"] for item in saidas)
    return [
        {
            "name": "linhas_geradas",
            "value": linhas,
            "unit": "linhas",
            "availability": "available",
        },
        {
            "name": "anomalias_total",
            "value": len(anomalias),
            "unit": "ocorrencias",
            "availability": "available",
        },
        {
            "name": "regras_exercitadas",
            "value": len({item["rule_id"] for item in anomalias}),
            "unit": "regras",
            "availability": "available",
        },
    ]


def gerar(
    *,
    config_path: Path,
    source_path: Path,
    analysis_dir: Path,
    batch_id: str,
    run_id: str,
    output_dir: Path,
    total: int,
    seed: int,
) -> int:
    """Gera o lote invalido e publica manifesto, eventos e metricas."""
    try:
        config = carregar_config(config_path)
    except (ErroLote, ValueError, TypeError, OSError) as erro:
        print(f"Configuracao invalida: {erro}")
        return CODIGO_ERRO
    if output_dir.exists():
        print(f"Diretorio de execucao ja existe: {output_dir}")
        return CODIGO_ERRO
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except OSError as erro:
        print(f"Falha ao reservar diretorio de execucao: {erro}")
        return CODIGO_ERRO
    iniciado = agora_iso()
    eventos: list[dict[str, Any]] = []
    config["_seed"] = seed
    dataset_id = str(config.get("dataset_id", "desconhecido"))
    registrar_evento(eventos, run_id, dataset_id, "started", "INFO", batch_id)
    error_code: str | None = None
    try:
        base = carregar_base(source_path)
        target_contract = config["target_contract"]
        particoes = list(config["artifacts"])
        fatias = dividir_lote(base, tuple(particoes), total, seed)

        ajustadas: dict[str, pd.DataFrame] = {}
        anomalias: list[dict[str, Any]] = []
        for nome in particoes:
            ajustada, descritas = aplicar_anomalias(fatias[nome], ANOMALIAS[nome])
            ajustadas[nome] = ajustada
            for item in descritas:
                anomalias.append({"artifact_type": nome, **item})

        destino = analysis_dir / batch_id
        caminhos = escrever_particoes(destino, ajustadas)

        entrada = {
            "artifact_type": "origem",
            "path": source_path.as_posix(),
            "sha256": sha256_arquivo(source_path),
            "producer_run_id": None,
            "row_count": int(len(base)),
            "columns": list(base.columns),
        }
        saidas = [
            descritor_artefato(nome, caminhos[nome], run_id, ajustadas[nome]) for nome in particoes
        ]

        manifesto_saida = {
            "run_id": run_id,
            "measurements": montar_medidas(saidas, anomalias),
        }
        registrar_evento(eventos, run_id, dataset_id, "completed", "INFO", batch_id)
        escrever_json(output_dir / "config.json", config)
        escrever_eventos(output_dir / "events.jsonl", eventos)
        escrever_json(output_dir / "metrics.json", manifesto_saida)
        manifesto = montar_manifesto(
            run_id=run_id,
            iniciado=iniciado,
            finalizado=agora_iso(),
            config_path=output_dir / "config.json",
            config=config,
            batch_id=batch_id,
            entrada=entrada,
            saidas=saidas,
            anomalias=anomalias,
        )
        manifesto["target"] = target_contract
        escrever_json(output_dir / "manifest.json", manifesto)
    except Exception as erro:
        # Os artefatos operacionais registram apenas a classe/codigo, nunca dados da origem.
        if eventos[-1]["event"] == "completed":
            eventos.pop()
        error_code = (
            "ERRO_ENTRADA"
            if isinstance(erro, (ErroLote, ValueError, pd.errors.ParserError, FileNotFoundError))
            else "ERRO_TECNICO"
        )
        registrar_evento(eventos, run_id, dataset_id, "failed", "ERROR", batch_id, error_code)
        try:
            config_path_efetiva = output_dir / "config.json"
            escrever_json(config_path_efetiva, config)
            escrever_eventos(output_dir / "events.jsonl", eventos)
            escrever_json(output_dir / "metrics.json", {"run_id": run_id, "measurements": []})
            manifesto = montar_manifesto(
                run_id=run_id,
                iniciado=iniciado,
                finalizado=agora_iso(),
                config_path=config_path_efetiva,
                config=config,
                batch_id=batch_id,
                entrada=None,
                saidas=[],
                anomalias=[],
                status="failed",
                error_code=error_code,
            )
            escrever_json(output_dir / "manifest.json", manifesto)
        except OSError as persist_erro:
            print(f"Falha ao persistir resultado de erro ({type(persist_erro).__name__}).")
            return CODIGO_ERRO
        print(f"Falha ao gerar o lote ({error_code}): {type(erro).__name__}")
        return CODIGO_ERRO

    print(f"Lote invalido gerado: {sum(len(f) for f in ajustadas.values())} linhas")
    print(f"  anomalias intencionais -> {len(anomalias)}")
    for item in anomalias:
        print(f"    {item['rule_id']:<26} {item['artifact_type']:<11} {item['column']}")
    print(f"  manifesto -> {output_dir.as_posix()}/manifest.json")
    return CODIGO_SUCESSO


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera um lote propositalmente invalido.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/validacao.json"), help="JSON do contrato."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/analysis/lending_club_100k_v1/simulation_base.csv"),
        help="Base de origem do lote.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("data/analysis/lending_club_100k_v1/lote-invalido"),
        help="Diretorio onde o lote sera gravado.",
    )
    parser.add_argument("--batch-id", default=None, help="Identificador do lote gerado.")
    parser.add_argument("--size", type=int, default=600, help="Linhas do lote.")
    parser.add_argument("--seed", type=int, default=42, help="Semente da amostragem.")
    parser.add_argument("--run-id", default=None, help="Identificador unico da execucao.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Diretorio novo da execucao.")
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ")
    batch_id = args.batch_id or f"batch-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    output_dir = args.output_dir or Path("reports/runs") / run_id

    if output_dir.exists():
        print(f"Diretorio de execucao ja existe: {output_dir}")
        sys.exit(CODIGO_ERRO)

    sys.exit(
        gerar(
            config_path=args.config,
            source_path=args.source,
            analysis_dir=args.analysis_dir,
            batch_id=batch_id,
            run_id=run_id,
            output_dir=output_dir,
            total=args.size,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()

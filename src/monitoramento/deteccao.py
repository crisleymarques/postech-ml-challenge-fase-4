"""Deteccao de Data Drift entre referencia e producao simulada (issue F2-03).

Le o manifesto do treinamento (referencia congelada das features de entrada do
modelo) e o manifesto da simulacao (lote de producao), confere a versao de
contrato, o `dataset_id` e o hash dos artefatos, e compara as distribuicoes das
features com o Evidently (`DataDriftPreset`). O Evidently calcula a estatistica
de cada variavel; o limiar declarado na configuracao decide `drift_detected`,
de modo que a regra de deteccao fica explicita e reproduzivel.

O resultado por variavel registra `feature`, `method`, `statistic`, `threshold`
e `drift_detected`; variaveis nao avaliaveis registram motivo e `drift_detected`
nulo, sem converter ausencia de evidencia em ausencia de drift. Publica
`drift.json` e `drift.html` no diretorio da execucao. Um alerta de drift nao e
erro: a execucao termina em `success` e registra o evento `drift_alert`.

Codigos de saida: 0 sucesso, 1 erro tecnico ou incompatibilidade de artefatos.

Uso:
    uv run python src/monitoramento/deteccao.py \
        --config configs/deteccao.json \
        --input-manifest reports/runs/<treinamento>/manifest.json \
        --input-manifest reports/runs/<simulacao>/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import evidently
import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

CONTRACT_VERSION = "1.0"
STAGE = "deteccao"
IDENTIFICADOR = "record_id"

CODIGO_SUCESSO = 0
CODIGO_ERRO = 1

PRESET = "DataDriftPreset"
TIPO_DRIFT = "evidently:metric_v2:ValueDrift"
LIMIARES_CONFIG = ("p_value_threshold", "distance_threshold")


class ErroDeteccao(Exception):
    """Indica dependencia ausente, divergente ou incompativel."""


class ErroConfiguracao(Exception):
    """Indica configuracao ausente ou invalida."""


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
        raise ErroConfiguracao(f"Configuracao nao encontrada: {caminho}")
    with caminho.open(encoding="utf-8") as arquivo:
        config = json.load(arquivo)

    if versao_maior(config.get("contract_version")) != versao_maior(CONTRACT_VERSION):
        raise ErroConfiguracao("Versao principal do contrato da configuracao incompativel.")

    obrigatorios = ("dataset_id", "reference", "batch", "method")
    ausentes = [chave for chave in obrigatorios if not config.get(chave)]
    if ausentes:
        raise ErroConfiguracao(f"Chaves ausentes na configuracao: {ausentes}")

    for papel in ("reference", "batch"):
        if not config[papel].get("producer_stage"):
            raise ErroConfiguracao(f"O papel '{papel}' exige a chave 'producer_stage'.")
    if config["reference"]["producer_stage"] == config["batch"]["producer_stage"]:
        raise ErroConfiguracao("As etapas produtoras da referencia e do lote devem ser diferentes.")

    if not config["method"].get("name"):
        raise ErroConfiguracao("O metodo de deteccao exige a chave 'name'.")
    for chave in LIMIARES_CONFIG:
        limiar = config["method"].get(chave)
        if not isinstance(limiar, (int, float)) or not 0 < float(limiar) < 1:
            raise ErroConfiguracao(f"'{chave}' deve ser um numero entre 0 e 1.")
    return config


def carregar_manifesto(caminho: Path) -> dict[str, Any]:
    """Le um manifesto de entrada e exige o produtor com sucesso."""
    if not caminho.is_file():
        raise ErroDeteccao(f"Manifesto de entrada nao encontrado: {caminho}")
    with caminho.open(encoding="utf-8") as arquivo:
        manifesto = json.load(arquivo)

    if manifesto.get("status") != "success":
        raise ErroDeteccao(f"Produtor {manifesto.get('run_id')} nao concluiu com sucesso.")
    if versao_maior(manifesto.get("contract_version")) != versao_maior(CONTRACT_VERSION):
        raise ErroDeteccao("Versao principal do contrato do manifesto incompativel.")
    return manifesto


def separar_manifestos(
    manifestos: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Identifica o manifesto da referencia e o do lote pela etapa produtora."""
    etapa_ref = config["reference"]["producer_stage"]
    etapa_lote = config["batch"]["producer_stage"]
    esperadas = (etapa_ref, etapa_lote)

    encontrados: dict[str, dict[str, Any]] = {}
    for manifesto in manifestos:
        etapa = manifesto.get("stage")
        if etapa in esperadas:
            if etapa in encontrados:
                raise ErroDeteccao(f"Manifesto duplicado para a etapa {etapa}.")
            encontrados[etapa] = manifesto

    ausentes = [etapa for etapa in esperadas if etapa not in encontrados]
    if ausentes:
        raise ErroDeteccao(f"Manifestos ausentes para as etapas: {ausentes}")
    return encontrados[etapa_ref], encontrados[etapa_lote]


def artefato(manifesto: dict[str, Any], tipo: str) -> dict[str, Any]:
    """Devolve o descritor de um artefato de saida pelo tipo."""
    for item in manifesto.get("outputs", []):
        if item.get("artifact_type") == tipo:
            return item
    raise ErroDeteccao(f"Artefato '{tipo}' ausente no manifesto {manifesto.get('run_id')}.")


def verificar_artefato(item: dict[str, Any]) -> Path:
    """Confere existencia e integridade de um artefato antes de consumi-lo."""
    caminho = Path(item.get("path", ""))
    if not caminho.is_file():
        raise ErroDeteccao(f"Artefato nao encontrado: {caminho}")
    esperado = item.get("sha256")
    if esperado and sha256_arquivo(caminho) != esperado:
        raise ErroDeteccao(f"Hash divergente para o artefato: {caminho}")
    return caminho


def verificar_compatibilidade(
    config: dict[str, Any], manifesto_ref: dict[str, Any], manifesto_lote: dict[str, Any]
) -> None:
    """Confere que referencia e lote pertencem ao mesmo dataset da configuracao."""
    dataset_id = config["dataset_id"]
    for manifesto in (manifesto_ref, manifesto_lote):
        if manifesto.get("dataset_id") != dataset_id:
            raise ErroDeteccao(
                f"dataset_id divergente no manifesto {manifesto.get('run_id')}: "
                f"{manifesto.get('dataset_id')} != {dataset_id}"
            )


def ler_features(caminho: Path, numericas: list[str], categoricas: list[str]) -> pd.DataFrame:
    """Le as features como texto e aplica os tipos declarados nos metadados."""
    base = pd.read_csv(caminho, dtype=str, na_filter=False).replace("", np.nan)
    for coluna in numericas:
        if coluna in base.columns:
            base[coluna] = pd.to_numeric(base[coluna], errors="coerce")
    for coluna in categoricas:
        if coluna in base.columns:
            base[coluna] = base[coluna].astype("object")
    return base


def selecionar_features(
    referencia: pd.DataFrame, atual: pd.DataFrame, colunas: list[str]
) -> list[str]:
    """Devolve as features comparaveis presentes nos dois conjuntos."""
    return [
        coluna for coluna in colunas if coluna in referencia.columns and coluna in atual.columns
    ]


def avaliar(
    estatistica: float, metodo: str | None, limiar_p: float, limiar_dist: float
) -> tuple[float, bool]:
    """Aplica a regra de deteccao e devolve o limiar usado e o resultado.

    Metodos de p-valor indicam drift quando a estatistica fica abaixo do limiar
    de p-valor; metodos de distancia, quando fica acima do limiar de distancia.
    """
    texto = str(metodo or "").lower()
    if "p_value" in texto or "p-value" in texto:
        return limiar_p, estatistica < limiar_p
    return limiar_dist, estatistica > limiar_dist


def executar_evidently(referencia: pd.DataFrame, atual: pd.DataFrame) -> Any:
    """Executa o preset de Data Drift do Evidently sobre os dois conjuntos."""
    relatorio = Report([DataDriftPreset()], include_tests=False)
    return relatorio.run(reference_data=referencia, current_data=atual)


def extrair_estatisticas(
    snapshot: Any, limiar_p: float, limiar_dist: float
) -> dict[str, dict[str, Any]]:
    """Extrai o resultado de drift por variavel a partir do snapshot do Evidently."""
    encontrados: dict[str, dict[str, Any]] = {}
    for item in snapshot.dict().get("metrics", []):
        config = item.get("config", {})
        if config.get("type") != TIPO_DRIFT:
            continue
        coluna = config.get("column")
        metodo = config.get("method")
        valor = item.get("value")
        if valor is None:
            encontrados[coluna] = {
                "feature": coluna,
                "method": metodo,
                "statistic": None,
                "threshold": None,
                "drift_detected": None,
                "reason": "variavel nao avaliavel pelo Evidently",
            }
            continue
        estatistica = float(valor)
        limiar, drift = avaliar(estatistica, metodo, limiar_p, limiar_dist)
        encontrados[coluna] = {
            "feature": coluna,
            "method": metodo,
            "statistic": round(estatistica, 6),
            "threshold": limiar,
            "drift_detected": drift,
        }
    return encontrados


def montar_resultados(
    snapshot: Any, colunas: list[str], limiar_p: float, limiar_dist: float
) -> list[dict[str, Any]]:
    """Monta o resultado por variavel preservando a ordem das features."""
    encontrados = extrair_estatisticas(snapshot, limiar_p, limiar_dist)
    resultados: list[dict[str, Any]] = []
    for coluna in colunas:
        if coluna in encontrados:
            resultados.append(encontrados[coluna])
        else:
            resultados.append(
                {
                    "feature": coluna,
                    "method": None,
                    "statistic": None,
                    "threshold": None,
                    "drift_detected": None,
                    "reason": "sem resultado de drift do Evidently para a variavel",
                }
            )
    return resultados


def resumir(resultados: list[dict[str, Any]]) -> dict[str, Any]:
    """Resume a deteccao com a contagem e a proporcao de variaveis com drift."""
    avaliaveis = [item for item in resultados if item["drift_detected"] is not None]
    com_drift = [item for item in avaliaveis if item["drift_detected"]]
    total = len(avaliaveis)
    proporcao = round(len(com_drift) / total, 6) if total else None
    return {
        "evaluated_columns": total,
        "drifted_columns": len(com_drift),
        "drift_share": proporcao,
        "dataset_drift": bool(proporcao is not None and proporcao > 0.5),
        "drifted_features": [item["feature"] for item in com_drift],
    }


def montar_medidas(resumo: dict[str, Any]) -> list[dict[str, Any]]:
    """Converte o resumo da deteccao em medidas do contrato de metricas."""
    disponibilidade = "available" if resumo["evaluated_columns"] else "not_available"
    return [
        {
            "name": "variaveis_avaliadas",
            "value": resumo["evaluated_columns"],
            "unit": "variaveis",
            "availability": disponibilidade,
        },
        {
            "name": "variaveis_com_drift",
            "value": resumo["drifted_columns"],
            "unit": "variaveis",
            "availability": disponibilidade,
        },
        {
            "name": "proporcao_drift",
            "value": resumo["drift_share"],
            "unit": "proporcao",
            "availability": "available" if resumo["drift_share"] is not None else "not_available",
        },
    ]


def descritor_artefato(tipo: str, caminho: Path, run_id: str) -> dict[str, Any]:
    """Descreve um artefato de saida nao tabular conforme o manifesto comum."""
    return {
        "artifact_type": tipo,
        "path": caminho.as_posix(),
        "sha256": sha256_arquivo(caminho),
        "producer_run_id": run_id,
    }


def registrar_evento(
    eventos: list[dict[str, Any]],
    *,
    run_id: str,
    dataset_id: str | None,
    model_id: str | None,
    reference_id: str | None,
    scenario_id: str | None,
    batch_id: str | None,
    evento: str,
    nivel: str,
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
            "model_id": model_id,
            "reference_id": reference_id,
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


def montar_documento(
    *,
    run_id: str,
    config: dict[str, Any],
    modelo_id: str | None,
    referencia_id: str | None,
    scenario_id: str | None,
    batch_id: str | None,
    caminho_ref: Path,
    caminho_lote: Path,
    resultados: list[dict[str, Any]],
    resumo: dict[str, Any],
    linhas_ref: int,
    linhas_lote: int,
    alteradas: list[str],
) -> dict[str, Any]:
    """Reune o documento de drift com ferramenta, metodo, amostras e resultados."""
    detectadas = [variavel for variavel in alteradas if variavel in resumo["drifted_features"]]
    nao_detectadas = [
        variavel for variavel in alteradas if variavel not in resumo["drifted_features"]
    ]
    return {
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "dataset_id": config["dataset_id"],
        "model_id": modelo_id,
        "reference_id": referencia_id,
        "scenario_id": scenario_id,
        "batch_id": batch_id,
        "tool": {"name": "evidently", "version": evidently.__version__},
        "method": {
            "name": config["method"]["name"],
            "preset": config["method"].get("preset", PRESET),
            "p_value_threshold": float(config["method"]["p_value_threshold"]),
            "distance_threshold": float(config["method"]["distance_threshold"]),
        },
        "samples": {
            "reference": {"path": caminho_ref.as_posix(), "row_count": int(linhas_ref)},
            "current": {"path": caminho_lote.as_posix(), "row_count": int(linhas_lote)},
        },
        "features": resultados,
        "summary": resumo,
        "changed_variables": {
            "configured": list(alteradas),
            "detected": detectadas,
            "not_detected": nao_detectadas,
        },
    }


def montar_manifesto(
    *,
    run_id: str,
    iniciado: str,
    finalizado: str,
    config_path: Path,
    config: dict[str, Any],
    manifesto_lote: dict[str, Any],
    modelo_id: str | None,
    referencia_id: str | None,
    scenario_id: str | None,
    batch_id: str | None,
    entradas: list[dict[str, Any]],
    saidas: list[dict[str, Any]],
    resumo: dict[str, Any],
) -> dict[str, Any]:
    """Reune o manifesto comum do produtor da deteccao."""
    return {
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "stage": STAGE,
        "status": "success",
        "started_at": iniciado,
        "finished_at": finalizado,
        "config_path": config_path.as_posix(),
        "inputs": entradas,
        "outputs": saidas,
        "dataset_id": config["dataset_id"],
        "schema_version": manifesto_lote.get("schema_version"),
        "model_id": modelo_id,
        "reference_id": referencia_id,
        "scenario_id": scenario_id,
        "batch_id": batch_id,
        "tool": {"name": "evidently", "version": evidently.__version__},
        "drift": resumo,
        "error_code": None,
    }


def processar(
    config_path: Path,
    config: dict[str, Any],
    manifestos: list[dict[str, Any]],
    run_id: str,
    output_dir: Path,
) -> int:
    """Executa a deteccao e publica manifesto, eventos, metricas e relatorios."""
    output_dir.mkdir(parents=True, exist_ok=True)
    iniciado = agora_iso()
    eventos: list[dict[str, Any]] = []
    registrar_evento(
        eventos,
        run_id=run_id,
        dataset_id=config.get("dataset_id"),
        model_id=None,
        reference_id=None,
        scenario_id=None,
        batch_id=None,
        evento="started",
        nivel="INFO",
    )
    try:
        manifesto_ref, manifesto_lote = separar_manifestos(manifestos, config)
        verificar_compatibilidade(config, manifesto_ref, manifesto_lote)

        modelo_id = manifesto_ref.get("model_id")
        referencia_id = manifesto_ref.get("reference_id")
        scenario_id = manifesto_lote.get("scenario_id")
        batch_id = manifesto_lote.get("batch_id")

        item_ref = artefato(manifesto_ref, config["reference"]["features_artifact"])
        item_meta = artefato(manifesto_ref, config["reference"]["metadata_artifact"])
        item_lote = artefato(manifesto_lote, config["batch"]["features_artifact"])

        caminho_ref = verificar_artefato(item_ref)
        caminho_meta = verificar_artefato(item_meta)
        caminho_lote = verificar_artefato(item_lote)

        with caminho_meta.open(encoding="utf-8") as arquivo:
            metadata = json.load(arquivo)

        numericas = list(metadata.get("numeric_features", []))
        categoricas = list(metadata.get("categorical_features", []))
        colunas = list(metadata.get("features", []))
        if not colunas:
            raise ErroDeteccao("Metadados do modelo sem lista de features.")

        referencia = ler_features(caminho_ref, numericas, categoricas)
        atual = ler_features(caminho_lote, numericas, categoricas)

        avaliadas = selecionar_features(referencia, atual, colunas)
        if not avaliadas:
            raise ErroDeteccao("Nenhuma feature em comum entre a referencia e o lote.")

        limiar_p = float(config["method"]["p_value_threshold"])
        limiar_dist = float(config["method"]["distance_threshold"])
        relatorio = executar_evidently(referencia[avaliadas], atual[avaliadas])
        resultados = montar_resultados(relatorio, avaliadas, limiar_p, limiar_dist)
        resumo = resumir(resultados)

        caminho_html = output_dir / "drift.html"
        relatorio.save_html(str(caminho_html))

        documento = montar_documento(
            run_id=run_id,
            config=config,
            modelo_id=modelo_id,
            referencia_id=referencia_id,
            scenario_id=scenario_id,
            batch_id=batch_id,
            caminho_ref=caminho_ref,
            caminho_lote=caminho_lote,
            resultados=resultados,
            resumo=resumo,
            linhas_ref=len(referencia),
            linhas_lote=len(atual),
            alteradas=list(manifesto_lote.get("variaveis_alteradas", [])),
        )
        caminho_json = output_dir / "drift.json"
        escrever_json(caminho_json, documento)

        entradas = [
            {**item_ref, "run_id_origem": manifesto_ref.get("run_id")},
            {**item_meta, "run_id_origem": manifesto_ref.get("run_id")},
            {**item_lote, "run_id_origem": manifesto_lote.get("run_id")},
        ]
        saidas = [
            descritor_artefato("drift_json", caminho_json, run_id),
            descritor_artefato("drift_html", caminho_html, run_id),
        ]

        escrever_json(output_dir / "config.json", config)
        escrever_json(
            output_dir / "metrics.json",
            {"run_id": run_id, "measurements": montar_medidas(resumo)},
        )
        escrever_json(
            output_dir / "manifest.json",
            montar_manifesto(
                run_id=run_id,
                iniciado=iniciado,
                finalizado=agora_iso(),
                config_path=config_path,
                config=config,
                manifesto_lote=manifesto_lote,
                modelo_id=modelo_id,
                referencia_id=referencia_id,
                scenario_id=scenario_id,
                batch_id=batch_id,
                entradas=entradas,
                saidas=saidas,
                resumo=resumo,
            ),
        )
        if resumo["dataset_drift"]:
            registrar_evento(
                eventos,
                run_id=run_id,
                dataset_id=config.get("dataset_id"),
                model_id=modelo_id,
                reference_id=referencia_id,
                scenario_id=scenario_id,
                batch_id=batch_id,
                evento="drift_alert",
                nivel="WARNING",
            )
        registrar_evento(
            eventos,
            run_id=run_id,
            dataset_id=config.get("dataset_id"),
            model_id=modelo_id,
            reference_id=referencia_id,
            scenario_id=scenario_id,
            batch_id=batch_id,
            evento="completed",
            nivel="INFO",
        )
        escrever_eventos(output_dir / "events.jsonl", eventos)
    except (ErroDeteccao, ValueError, KeyError) as erro:
        registrar_evento(
            eventos,
            run_id=run_id,
            dataset_id=config.get("dataset_id"),
            model_id=None,
            reference_id=None,
            scenario_id=None,
            batch_id=None,
            evento="failed",
            nivel="ERROR",
            error_code="DETECCAO_INVALIDA",
        )
        escrever_eventos(output_dir / "events.jsonl", eventos)
        print(f"Falha na deteccao: {erro}")
        return CODIGO_ERRO

    print(f"Drift detectado: {resumo['drifted_columns']}/{resumo['evaluated_columns']} variaveis")
    print(f"  cenario {scenario_id} / lote {batch_id}")
    for item in resultados:
        marca = "DRIFT" if item["drift_detected"] else "ok"
        print(f"  [{marca}] {item['feature']} ({item['method']} = {item['statistic']})")
    print(f"  relatorio -> {caminho_json.as_posix()}")
    print(f"  html -> {caminho_html.as_posix()}")
    return CODIGO_SUCESSO


def main() -> None:
    parser = argparse.ArgumentParser(description="Detecta Data Drift entre referencia e producao.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/deteccao.json"), help="JSON da etapa."
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        action="append",
        default=None,
        help="Manifesto de entrada (treinamento e simulacao). Repetivel.",
    )
    parser.add_argument("--run-id", default=None, help="Identificador unico da execucao.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Diretorio novo da execucao.")
    args = parser.parse_args()

    try:
        config = carregar_config(args.config)
    except ErroConfiguracao as erro:
        print(f"Falha na deteccao: {erro}")
        sys.exit(CODIGO_ERRO)

    entradas = args.input_manifest or []
    if len(entradas) < 2:
        print("Informe dois manifestos via --input-manifest (um do treinamento e um da simulacao).")
        sys.exit(CODIGO_ERRO)

    try:
        manifestos = [carregar_manifesto(caminho) for caminho in entradas]
    except ErroDeteccao as erro:
        print(f"Falha na deteccao: {erro}")
        sys.exit(CODIGO_ERRO)

    run_id = args.run_id or datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ")
    output_dir = args.output_dir or Path("reports/runs") / run_id
    if output_dir.exists():
        print(f"Diretorio de execucao ja existe: {output_dir}")
        sys.exit(CODIGO_ERRO)

    sys.exit(processar(args.config, config, manifestos, run_id, output_dir))


if __name__ == "__main__":
    main()

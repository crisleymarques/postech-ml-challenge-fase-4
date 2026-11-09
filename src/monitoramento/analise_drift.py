"""Analise estatistica e interpretacao do Data Drift (issue F2-04).

Le o manifesto do treinamento (referencia congelada das features de entrada do
modelo) e o manifesto da simulacao (lote de producao), confere a versao de
contrato, o `dataset_id` e o hash dos artefatos, e aplica testes estatisticos
explicitos por variavel: o PSI (Population Stability Index) e, para variaveis
numericas, o teste de Kolmogorov-Smirnov (K-S). O Evidently da deteccao (F2-03)
identifica a mudanca; esta etapa evidencia a diferenca com testes classicos,
registra os criterios de decisao e interpreta o resultado, separando a mudanca
observada de seus efeitos presumidos.

Cada variavel recebe `feature`, `kind`, `psi`, `psi_band`, `ks_statistic`,
`ks_p_value` e `drift_detected`. A decisao combina os criterios declarados na
configuracao: PSI acima do limiar ou K-S significante (p-valor abaixo do limiar
com estatistica de efeito relevante). Variaveis nao avaliaveis registram motivo
e resultado nulo, sem converter ausencia de evidencia em ausencia de mudanca.

O documento `analise.json` inclui tambem as variaveis alteradas pela simulacao
(confirmadas e nao confirmadas) e as conclusoes tecnicas. A mudanca de
distribuicao nao comprova, por si so, degradacao do modelo nem causalidade.

Codigos de saida: 0 sucesso, 1 erro tecnico ou incompatibilidade de artefatos.

Uso:
    uv run python src/monitoramento/analise_drift.py \
        --config configs/analise_drift.json \
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

import numpy as np
import pandas as pd
import scipy
from scipy import stats

CONTRACT_VERSION = "1.0"
STAGE = "analise"
IDENTIFICADOR = "record_id"

CODIGO_SUCESSO = 0
CODIGO_ERRO = 1

METODO = "psi_ks"
LIMIARES_CONFIG = (
    "psi_moderate",
    "psi_threshold",
    "ks_p_value_threshold",
    "ks_statistic_threshold",
)
TIPO_NUMERICO = "numeric"
TIPO_CATEGORICO = "categorical"

# Piso aplicado as proporcoes para evitar logaritmo de zero no PSI.
EPSILON = 1e-6

CRITERIO_PSI = (
    "PSI >= 'psi_threshold' indica mudanca significativa; PSI entre "
    "'psi_moderate' e 'psi_threshold' indica mudanca moderada; abaixo disso, "
    "insignificante."
)
CRITERIO_KS = (
    "K-S (variaveis numericas) indica mudanca quando o p-valor fica abaixo de "
    "'ks_p_value_threshold' e a estatistica atinge 'ks_statistic_threshold'."
)


class ErroAnalise(Exception):
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
        raise ErroConfiguracao("O metodo de analise exige a chave 'name'.")
    bins = config["method"].get("bins")
    if not isinstance(bins, int) or isinstance(bins, bool) or bins < 2:
        raise ErroConfiguracao("'bins' deve ser um inteiro maior ou igual a 2.")
    for chave in LIMIARES_CONFIG:
        limiar = config["method"].get(chave)
        if not isinstance(limiar, (int, float)) or not 0 < float(limiar) < 1:
            raise ErroConfiguracao(f"'{chave}' deve ser um numero entre 0 e 1.")
    return config


def carregar_manifesto(caminho: Path) -> dict[str, Any]:
    """Le um manifesto de entrada e exige o produtor com sucesso."""
    if not caminho.is_file():
        raise ErroAnalise(f"Manifesto de entrada nao encontrado: {caminho}")
    with caminho.open(encoding="utf-8") as arquivo:
        manifesto = json.load(arquivo)

    if manifesto.get("status") != "success":
        raise ErroAnalise(f"Produtor {manifesto.get('run_id')} nao concluiu com sucesso.")
    if versao_maior(manifesto.get("contract_version")) != versao_maior(CONTRACT_VERSION):
        raise ErroAnalise("Versao principal do contrato do manifesto incompativel.")
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
                raise ErroAnalise(f"Manifesto duplicado para a etapa {etapa}.")
            encontrados[etapa] = manifesto

    ausentes = [etapa for etapa in esperadas if etapa not in encontrados]
    if ausentes:
        raise ErroAnalise(f"Manifestos ausentes para as etapas: {ausentes}")
    return encontrados[etapa_ref], encontrados[etapa_lote]


def artefato(manifesto: dict[str, Any], tipo: str) -> dict[str, Any]:
    """Devolve o descritor de um artefato de saida pelo tipo."""
    for item in manifesto.get("outputs", []):
        if item.get("artifact_type") == tipo:
            return item
    raise ErroAnalise(f"Artefato '{tipo}' ausente no manifesto {manifesto.get('run_id')}.")


def verificar_artefato(item: dict[str, Any]) -> Path:
    """Confere existencia e integridade de um artefato antes de consumi-lo."""
    caminho = Path(item.get("path", ""))
    if not caminho.is_file():
        raise ErroAnalise(f"Artefato nao encontrado: {caminho}")
    esperado = item.get("sha256")
    if esperado and sha256_arquivo(caminho) != esperado:
        raise ErroAnalise(f"Hash divergente para o artefato: {caminho}")
    return caminho


def verificar_compatibilidade(
    config: dict[str, Any], manifesto_ref: dict[str, Any], manifesto_lote: dict[str, Any]
) -> None:
    """Confere que referencia e lote pertencem ao mesmo dataset da configuracao."""
    dataset_id = config["dataset_id"]
    for manifesto in (manifesto_ref, manifesto_lote):
        if manifesto.get("dataset_id") != dataset_id:
            raise ErroAnalise(
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


def indice_psi(esperado: np.ndarray, observado: np.ndarray) -> float:
    """Calcula o PSI entre duas distribuicoes de proporcao alinhadas."""
    esperado = np.where(np.asarray(esperado, dtype=float) <= 0, EPSILON, esperado)
    observado = np.where(np.asarray(observado, dtype=float) <= 0, EPSILON, observado)
    return float(np.sum((observado - esperado) * np.log(observado / esperado)))


def psi_numerico(referencia: pd.Series, atual: pd.Series, bins: int) -> float | None:
    """Calcula o PSI de uma variavel numerica usando faixas por quantis."""
    esperado = pd.to_numeric(referencia, errors="coerce").dropna()
    observado = pd.to_numeric(atual, errors="coerce").dropna()
    if esperado.empty or observado.empty:
        return None

    limites = np.unique(np.quantile(esperado, np.linspace(0, 1, bins + 1)))
    if len(limites) < 3:
        return None
    limites = limites.astype(float)
    limites[0] = -np.inf
    limites[-1] = np.inf

    contagem_ref, _ = np.histogram(esperado, bins=limites)
    contagem_atual, _ = np.histogram(observado, bins=limites)
    prop_ref = contagem_ref / contagem_ref.sum()
    prop_atual = contagem_atual / contagem_atual.sum()
    return indice_psi(prop_ref, prop_atual)


def psi_categorico(referencia: pd.Series, atual: pd.Series) -> float | None:
    """Calcula o PSI de uma variavel categorica pelas proporcoes por categoria."""
    esperado = referencia.astype("object").fillna("<ausente>")
    observado = atual.astype("object").fillna("<ausente>")
    if esperado.empty or observado.empty:
        return None

    categorias = sorted(set(esperado.unique()) | set(observado.unique()))
    prop_ref = esperado.value_counts(normalize=True)
    prop_atual = observado.value_counts(normalize=True)
    vetor_ref = np.array([prop_ref.get(categoria, 0.0) for categoria in categorias])
    vetor_atual = np.array([prop_atual.get(categoria, 0.0) for categoria in categorias])
    return indice_psi(vetor_ref, vetor_atual)


def teste_ks(referencia: pd.Series, atual: pd.Series) -> tuple[float, float] | None:
    """Executa o teste de Kolmogorov-Smirnov de duas amostras."""
    esperado = pd.to_numeric(referencia, errors="coerce").dropna()
    observado = pd.to_numeric(atual, errors="coerce").dropna()
    if esperado.empty or observado.empty:
        return None
    resultado = stats.ks_2samp(esperado, observado)
    return float(resultado.statistic), float(resultado.pvalue)


def faixa_psi(valor: float, moderado: float, limiar: float) -> str:
    """Classifica o PSI em insignificante, moderada ou significativa."""
    if valor >= limiar:
        return "significativa"
    if valor >= moderado:
        return "moderada"
    return "insignificante"


def analisar_variavel(
    nome: str,
    tipo: str,
    referencia: pd.Series,
    atual: pd.Series,
    *,
    bins: int,
    psi_moderado: float,
    psi_limiar: float,
    ks_p: float,
    ks_limiar: float,
) -> dict[str, Any]:
    """Aplica os criterios declarados a uma variavel e descreve a decisao."""
    resultado: dict[str, Any] = {
        "feature": nome,
        "kind": tipo,
        "psi": None,
        "psi_band": None,
        "ks_statistic": None,
        "ks_p_value": None,
        "drift_detected": None,
        "reasons": [],
    }

    if tipo == TIPO_NUMERICO:
        valor_psi = psi_numerico(referencia, atual, bins)
        teste = teste_ks(referencia, atual)
    else:
        valor_psi = psi_categorico(referencia, atual)
        teste = None

    if valor_psi is None:
        resultado["reasons"] = ["PSI nao calculavel por falta de dados validos"]
        return resultado

    resultado["psi"] = round(valor_psi, 6)
    resultado["psi_band"] = faixa_psi(valor_psi, psi_moderado, psi_limiar)

    motivos: list[str] = []
    if valor_psi >= psi_limiar:
        motivos.append(f"PSI {valor_psi:.4f} >= limiar {psi_limiar}")

    if teste is not None:
        estatistica, pvalor = teste
        resultado["ks_statistic"] = round(estatistica, 6)
        resultado["ks_p_value"] = round(pvalor, 6)
        if pvalor < ks_p and estatistica >= ks_limiar:
            motivos.append(
                f"K-S p-valor {pvalor:.4g} < {ks_p} com "
                f"estatistica {estatistica:.4f} >= {ks_limiar}"
            )

    resultado["drift_detected"] = bool(motivos)
    resultado["reasons"] = motivos or ["sem evidencia de mudanca pelos criterios declarados"]
    return resultado


def montar_resultados(
    referencia: pd.DataFrame,
    atual: pd.DataFrame,
    colunas: list[str],
    numericas: list[str],
    *,
    bins: int,
    psi_moderado: float,
    psi_limiar: float,
    ks_p: float,
    ks_limiar: float,
) -> list[dict[str, Any]]:
    """Analisa cada variavel preservando a ordem declarada das features."""
    resultados: list[dict[str, Any]] = []
    for coluna in colunas:
        tipo = TIPO_NUMERICO if coluna in numericas else TIPO_CATEGORICO
        resultados.append(
            analisar_variavel(
                coluna,
                tipo,
                referencia[coluna],
                atual[coluna],
                bins=bins,
                psi_moderado=psi_moderado,
                psi_limiar=psi_limiar,
                ks_p=ks_p,
                ks_limiar=ks_limiar,
            )
        )
    return resultados


def resumir(resultados: list[dict[str, Any]]) -> dict[str, Any]:
    """Resume a analise com contagem e proporcao de variaveis com mudanca."""
    avaliaveis = [item for item in resultados if item["drift_detected"] is not None]
    com_drift = [item for item in avaliaveis if item["drift_detected"]]
    total = len(avaliaveis)
    proporcao = round(len(com_drift) / total, 6) if total else None
    significativas = [item for item in avaliaveis if item["psi_band"] == "significativa"]
    psi_maximo = max((item["psi"] for item in avaliaveis if item["psi"] is not None), default=None)
    return {
        "evaluated_columns": total,
        "changed_columns": len(com_drift),
        "change_share": proporcao,
        "significant_columns": len(significativas),
        "psi_maximo": round(psi_maximo, 6) if psi_maximo is not None else None,
        "changed_features": [item["feature"] for item in com_drift],
    }


def montar_conclusoes(
    resumo: dict[str, Any],
    alteradas: list[str],
    *,
    psi_limiar: float,
    ks_p: float,
) -> list[str]:
    """Registra as conclusoes tecnicas da analise, sem inferir causalidade."""
    conclusoes: list[str] = []
    if resumo["evaluated_columns"]:
        conclusoes.append(
            f"Os criterios declarados classificaram {resumo['changed_columns']} de "
            f"{resumo['evaluated_columns']} variaveis com mudanca de distribuicao, das quais "
            f"{resumo['significant_columns']} com PSI significativo (>= {psi_limiar})."
        )
    conclusoes.append(
        "A referencia (particao de treino) e o lote (base reservada de producao) provem de "
        "periodos distintos: parte das diferencas pode refletir a separacao temporal do dataset "
        "e nao apenas a alteracao injetada na simulacao."
    )
    if alteradas:
        faltantes = [coluna for coluna in alteradas if coluna not in resumo["changed_features"]]
        if faltantes:
            conclusoes.append(
                "Variaveis alteradas intencionalmente nao confirmadas pelos criterios: "
                + ", ".join(faltantes)
                + "."
            )
        else:
            conclusoes.append(
                f"Todas as {len(alteradas)} variaveis alteradas na simulacao foram confirmadas "
                "pela analise estatistica, o que liga a mudanca observada a uma causa conhecida "
                "do cenario."
            )
    conclusoes.append(
        "A mudanca de distribuicao observada nao comprova, por si so, degradacao do desempenho "
        f"nem causalidade: os testes (PSI, K-S p-valor < {ks_p}) evidenciam diferenca entre as "
        "amostras, e a comparacao de desempenho depende de rotulos compativeis em etapa propria."
    )
    return conclusoes


def montar_medidas(resumo: dict[str, Any], alteradas: list[str]) -> list[dict[str, Any]]:
    """Converte o resumo da analise em medidas do contrato de metricas."""
    confirmadas = [coluna for coluna in alteradas if coluna in resumo["changed_features"]]
    disponibilidade = "available" if resumo["evaluated_columns"] else "not_available"
    return [
        {
            "name": "variaveis_avaliadas",
            "value": resumo["evaluated_columns"],
            "unit": "variaveis",
            "availability": disponibilidade,
        },
        {
            "name": "variaveis_com_mudanca",
            "value": resumo["changed_columns"],
            "unit": "variaveis",
            "availability": disponibilidade,
        },
        {
            "name": "proporcao_mudanca",
            "value": resumo["change_share"],
            "unit": "proporcao",
            "availability": "available" if resumo["change_share"] is not None else "not_available",
        },
        {
            "name": "psi_maximo",
            "value": resumo["psi_maximo"],
            "unit": "indice",
            "availability": "available" if resumo["psi_maximo"] is not None else "not_available",
        },
        {
            "name": "variaveis_alteradas_confirmadas",
            "value": len(confirmadas),
            "unit": "variaveis",
            "availability": "available" if alteradas else "not_applicable",
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
    """Reune o documento da analise com criterios, resultados e interpretacao."""
    confirmadas = [coluna for coluna in alteradas if coluna in resumo["changed_features"]]
    nao_confirmadas = [coluna for coluna in alteradas if coluna not in resumo["changed_features"]]
    metodo = config["method"]
    return {
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "dataset_id": config["dataset_id"],
        "model_id": modelo_id,
        "reference_id": referencia_id,
        "scenario_id": scenario_id,
        "batch_id": batch_id,
        "tool": {"name": "scipy", "version": scipy.__version__},
        "method": {
            "name": metodo["name"],
            "bins": int(metodo["bins"]),
            "psi_moderate": float(metodo["psi_moderate"]),
            "psi_threshold": float(metodo["psi_threshold"]),
            "ks_p_value_threshold": float(metodo["ks_p_value_threshold"]),
            "ks_statistic_threshold": float(metodo["ks_statistic_threshold"]),
        },
        "criteria": {"psi": CRITERIO_PSI, "ks": CRITERIO_KS},
        "samples": {
            "reference": {"path": caminho_ref.as_posix(), "row_count": int(linhas_ref)},
            "current": {"path": caminho_lote.as_posix(), "row_count": int(linhas_lote)},
        },
        "features": resultados,
        "summary": resumo,
        "changed_variables": {
            "configured": list(alteradas),
            "confirmed": confirmadas,
            "not_confirmed": nao_confirmadas,
        },
        "interpretation": {
            "conclusions": montar_conclusoes(
                resumo,
                alteradas,
                psi_limiar=float(metodo["psi_threshold"]),
                ks_p=float(metodo["ks_p_value_threshold"]),
            )
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
    """Reune o manifesto comum do produtor da analise."""
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
        "tool": {"name": "scipy", "version": scipy.__version__},
        "analysis": resumo,
        "error_code": None,
    }


def processar(
    config_path: Path,
    config: dict[str, Any],
    manifestos: list[dict[str, Any]],
    run_id: str,
    output_dir: Path,
) -> int:
    """Executa a analise e publica manifesto, eventos, metricas e documento."""
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
            raise ErroAnalise("Metadados do modelo sem lista de features.")

        referencia = ler_features(caminho_ref, numericas, categoricas)
        atual = ler_features(caminho_lote, numericas, categoricas)

        avaliadas = selecionar_features(referencia, atual, colunas)
        if not avaliadas:
            raise ErroAnalise("Nenhuma feature em comum entre a referencia e o lote.")

        metodo = config["method"]
        resultados = montar_resultados(
            referencia,
            atual,
            avaliadas,
            numericas,
            bins=int(metodo["bins"]),
            psi_moderado=float(metodo["psi_moderate"]),
            psi_limiar=float(metodo["psi_threshold"]),
            ks_p=float(metodo["ks_p_value_threshold"]),
            ks_limiar=float(metodo["ks_statistic_threshold"]),
        )
        resumo = resumir(resultados)
        alteradas = list(manifesto_lote.get("variaveis_alteradas", []))

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
            alteradas=alteradas,
        )
        caminho_json = output_dir / "analise.json"
        escrever_json(caminho_json, documento)

        entradas = [
            {**item_ref, "run_id_origem": manifesto_ref.get("run_id")},
            {**item_meta, "run_id_origem": manifesto_ref.get("run_id")},
            {**item_lote, "run_id_origem": manifesto_lote.get("run_id")},
        ]
        saidas = [descritor_artefato("analise_json", caminho_json, run_id)]

        escrever_json(output_dir / "config.json", config)
        escrever_json(
            output_dir / "metrics.json",
            {"run_id": run_id, "measurements": montar_medidas(resumo, alteradas)},
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
        if resumo["changed_columns"]:
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
    except (ErroAnalise, ValueError, KeyError) as erro:
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
            error_code="ANALISE_INVALIDA",
        )
        escrever_eventos(output_dir / "events.jsonl", eventos)
        print(f"Falha na analise: {erro}")
        return CODIGO_ERRO

    print(
        f"Analise concluida: {resumo['changed_columns']}/{resumo['evaluated_columns']} "
        "variaveis com mudanca"
    )
    print(f"  cenario {scenario_id} / lote {batch_id}")
    for item in resultados:
        marca = "MUDA" if item["drift_detected"] else "ok"
        estatistica = item["psi"]
        print(f"  [{marca}] {item['feature']} (PSI {estatistica}, faixa {item['psi_band']})")
    print(f"  documento -> {caminho_json.as_posix()}")
    return CODIGO_SUCESSO


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analisa estatisticamente (PSI e K-S) o drift entre referencia e producao."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/analise_drift.json"),
        help="JSON da etapa.",
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
        print(f"Falha na analise: {erro}")
        sys.exit(CODIGO_ERRO)

    entradas = args.input_manifest or []
    if len(entradas) < 2:
        print("Informe dois manifestos via --input-manifest (um do treinamento e um da simulacao).")
        sys.exit(CODIGO_ERRO)

    try:
        manifestos = [carregar_manifesto(caminho) for caminho in entradas]
    except ErroAnalise as erro:
        print(f"Falha na analise: {erro}")
        sys.exit(CODIGO_ERRO)

    run_id = args.run_id or datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ")
    output_dir = args.output_dir or Path("reports/runs") / run_id
    if output_dir.exists():
        print(f"Diretorio de execucao ja existe: {output_dir}")
        sys.exit(CODIGO_ERRO)

    sys.exit(processar(args.config, config, manifestos, run_id, output_dir))


if __name__ == "__main__":
    main()

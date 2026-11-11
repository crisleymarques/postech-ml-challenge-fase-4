"""Entradas da integracao de alertas operacionais."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

CONTRACT_VERSION = "1.0"


SITUACAO_ATENCAO = "atencao"


SITUACAO_CRITICO = "critico"


ID_DATASET_DRIFT = "drift.dataset_drift"


ID_VARIAVEIS_DRIFT = "drift.variaveis_com_drift"


ID_PROPORCAO_DRIFT = "drift.proporcao_drift"


ID_ALERTAS_TOTAL = "drift.alertas_total"


def versao_maior(versao: Any) -> str:
    """Extrai o componente principal de uma versao de contrato."""
    return str(versao or "").split(".")[0]


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


def validar_versao(conteudo: dict[str, Any], rotulo: str) -> None:
    """Exige uma versao textual valida da mesma versao principal."""
    versao = conteudo.get("contract_version")
    partes = versao.split(".") if isinstance(versao, str) else []
    if (
        len(partes) not in (2, 3)
        or not all(parte.isascii() and parte.isdigit() for parte in partes)
        or versao_maior(versao) != versao_maior(CONTRACT_VERSION)
    ):
        raise ValueError(f"{rotulo}: contract_version ausente, invalida ou incompativel.")


def exigir_campo(objeto: dict[str, Any], chave: str, tipo: type, rotulo: str) -> Any:
    """Valida um campo obrigatorio sem converter valores malformados."""
    valor = objeto.get(chave)
    if not isinstance(valor, tipo) or (tipo is str and not valor.strip()):
        raise ValueError(f"{rotulo}: campo '{chave}' invalido ou ausente.")
    return valor


def objetos(objeto: dict[str, Any], chave: str, rotulo: str) -> list[dict[str, Any]]:
    """Valida uma lista obrigatoria de objetos; lista vazia e permitida."""
    itens = exigir_campo(objeto, chave, list, rotulo)
    if any(not isinstance(item, dict) for item in itens):
        raise ValueError(f"{rotulo}: '{chave}' deve conter somente objetos.")
    return itens


def validar_medida(medida: dict[str, Any], rotulo: str) -> None:
    """Rejeita medidas que seriam ignoradas ou produziriam alertas sem identidade."""
    for chave in ("id", "categoria", "nome", "unidade"):
        exigir_campo(medida, chave, str, rotulo)
    if medida.get("situacao") not in (
        "ok",
        SITUACAO_ATENCAO,
        SITUACAO_CRITICO,
        "indisponivel",
        "nao_aplicavel",
    ):
        raise ValueError(f"{rotulo}: situacao invalida ou ausente.")
    if medida.get("disponibilidade") not in ("available", "not_available"):
        raise ValueError(f"{rotulo}: disponibilidade invalida ou ausente.")
    if "valor" not in medida:
        raise ValueError(f"{rotulo}: valor ausente.")
    valor = medida["valor"]
    if valor is None:
        return
    if not isinstance(valor, (str, bool, int, float)) or (
        isinstance(valor, float) and not math.isfinite(valor)
    ):
        raise ValueError(f"{rotulo}: valor deve ser escalar e finito ou nulo.")
    identificador = medida["id"]
    if identificador == ID_DATASET_DRIFT and not isinstance(valor, bool):
        raise ValueError(f"{rotulo}: valor de dataset_drift deve ser booleano ou nulo.")
    if identificador in (ID_VARIAVEIS_DRIFT, ID_ALERTAS_TOTAL) and (
        type(valor) is not int or valor < 0
    ):
        raise ValueError(f"{rotulo}: valor deve ser contagem inteira nao negativa ou nulo.")
    if identificador == ID_PROPORCAO_DRIFT and (
        type(valor) not in (int, float) or not 0 <= valor <= 1
    ):
        raise ValueError(f"{rotulo}: valor de proporcao deve estar entre 0 e 1 ou ser nulo.")


def validar_saude(saude: dict[str, Any]) -> None:
    """Valida o contrato consumido da F3-02 antes de gerar qualquer saida."""
    rotulo = "Relatorio de saude"
    validar_versao(saude, rotulo)
    for execucao in objetos(saude, "execucoes", rotulo):
        for chave in ("run_id", "stage", "status"):
            exigir_campo(execucao, chave, str, f"{rotulo}.execucoes")
        for medida in objetos(execucao, "metricas", f"{rotulo}.execucoes"):
            validar_medida(medida, f"{rotulo}.execucoes.metricas")
    for medida in objetos(saude, "agregados", rotulo):
        validar_medida(medida, f"{rotulo}.agregados")


def validar_logs(resumo: dict[str, Any]) -> None:
    """Valida o contrato consumido da F3-01, incluindo falhas e alertas."""
    rotulo = "Resumo dos logs"
    validar_versao(resumo, rotulo)
    for chave in ("execucoes", "eventos"):
        bloco = exigir_campo(resumo, chave, dict, rotulo)
        total = bloco.get("total")
        if type(total) is not int or total < 0:
            raise ValueError(f"{rotulo}.{chave}: total deve ser inteiro nao negativo.")
    for chave in ("falhas", "alertas"):
        for evento in objetos(resumo, chave, rotulo):
            for campo in ("run_id", "stage"):
                exigir_campo(evento, campo, str, f"{rotulo}.{chave}")
            if chave == "falhas":
                exigir_campo(evento, "event", str, f"{rotulo}.{chave}")

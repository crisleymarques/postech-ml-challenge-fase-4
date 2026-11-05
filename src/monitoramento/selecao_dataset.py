"""Seleção e amostragem da base Lending Club (issue PREP-02).

Lê a origem bruta em lotes, com memória limitada, mantém apenas empréstimos com
desfecho definitivo e gera uma base derivada menor, preservando o eixo temporal
(`issue_d`) necessário à futura simulação de drift.

As colunas são lidas como texto (espelhamento bruto); a tipagem fica para as
camadas de modelagem. O identificador pessoal de origem (`id`) é descartado e
substituído por um `record_id` técnico, conforme as regras de governança.

Uso:
    uv run python src/monitoramento/selecao_dataset.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ALVO = "loan_status"
DESFECHOS_DEFINITIVOS = ("Fully Paid", "Charged Off")

# Colunas mantidas na base derivada.
COLUNAS = [
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "issue_d",
    "loan_status",
    "purpose",
    "addr_state",
    "dti",
    "delinq_2yrs",
    "earliest_cr_line",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "initial_list_status",
    "application_type",
    "mort_acc",
    "pub_rec_bankruptcies",
]


def colunas_disponiveis(origem: Path) -> list[str]:
    """Valida que todas as colunas esperadas existem na origem."""
    presentes = set(pd.read_csv(origem, nrows=0).columns)
    ausentes = [coluna for coluna in COLUNAS if coluna not in presentes]
    if ausentes:
        raise ValueError(f"Colunas ausentes na origem: {ausentes}")
    return COLUNAS


def coletar_definitivos(
    origem: Path,
    colunas: list[str],
    capacidade: int,
    chunksize: int,
    seed: int,
) -> tuple[pd.DataFrame, int, int]:
    """Coleta desfechos definitivos em lotes, com afinamento para limitar a memória.

    Retorna a base retida (no máximo `capacidade` linhas), o total de linhas lidas
    e o total de desfechos definitivos encontrados antes do afinamento.
    """
    rng = np.random.default_rng(seed)
    lotes: list[pd.DataFrame] = []
    total_lidas = 0
    total_definitivos = 0

    leitor = pd.read_csv(
        origem,
        usecols=colunas,
        dtype=str,
        na_filter=False,
        chunksize=chunksize,
        engine="c",
    )
    for lote in leitor:
        total_lidas += len(lote)
        aprovados = lote[lote[ALVO].isin(DESFECHOS_DEFINITIVOS)]
        if aprovados.empty:
            continue
        total_definitivos += len(aprovados)
        lotes.append(aprovados)
        if sum(len(item) for item in lotes) > capacidade:
            combinado = pd.concat(lotes, ignore_index=True)
            escolhidos = rng.choice(len(combinado), size=capacidade, replace=False)
            lotes = [combinado.iloc[np.sort(escolhidos)].reset_index(drop=True)]

    if not lotes:
        return pd.DataFrame(columns=colunas), total_lidas, total_definitivos
    return pd.concat(lotes, ignore_index=True), total_lidas, total_definitivos


def amostra_estratificada(base: pd.DataFrame, n_linhas: int, seed: int) -> pd.DataFrame:
    """Sorteia `n_linhas` preservando a proporção das classes do alvo."""
    if n_linhas >= len(base):
        return base.sample(frac=1, random_state=seed).reset_index(drop=True)

    proporcoes = base[ALVO].value_counts(normalize=True)
    partes = []
    for rotulo, proporcao in proporcoes.items():
        grupo = base[base[ALVO] == rotulo]
        n_grupo = max(1, round(n_linhas * proporcao))
        partes.append(grupo.sample(n=min(n_grupo, len(grupo)), random_state=seed))
    return pd.concat(partes).sample(frac=1, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera a base derivada do Lending Club.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/reference/raw/accepted_2007_to_2018Q4.csv"),
        help="Caminho da origem bruta.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/reference/sample/lending_club_sample_100k.csv"),
        help="Caminho da base derivada.",
    )
    parser.add_argument("--n-rows", type=int, default=100_000, help="Linhas da base derivada.")
    parser.add_argument("--seed", type=int, default=42, help="Semente do sorteio.")
    parser.add_argument(
        "--chunksize", type=int, default=200_000, help="Linhas por lote de leitura."
    )
    parser.add_argument(
        "--buffer-factor",
        type=int,
        default=2,
        help="Multiplicador do alvo usado como limite de memória antes do sorteio final.",
    )
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"Origem não encontrada: {args.source}")

    colunas = colunas_disponiveis(args.source)
    base, total_lidas, total_definitivos = coletar_definitivos(
        args.source,
        colunas,
        capacidade=args.n_rows * args.buffer_factor,
        chunksize=args.chunksize,
        seed=args.seed,
    )
    amostra = amostra_estratificada(base, args.n_rows, args.seed)

    amostra.insert(0, "record_id", [f"LC{i:07d}" for i in range(1, len(amostra) + 1)])

    if len(amostra) < args.n_rows:
        print(f"AVISO: base derivada com {len(amostra):,} linhas (alvo {args.n_rows:,}).")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    amostra.to_csv(args.output, index=False, encoding="utf-8")

    print(f"Linhas lidas na origem  : {total_lidas:,}")
    print(f"Desfechos definitivos   : {total_definitivos:,}")
    print(f"Linhas na base derivada : {len(amostra):,}")
    print(f"Colunas na base         : {amostra.shape[1]}")
    print(f"Salvo em                : {args.output}")
    print(amostra[ALVO].value_counts().to_string())


if __name__ == "__main__":
    main()

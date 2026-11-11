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
import gzip
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import kagglehub
import numpy as np
import pandas as pd

DATASET_KAGGLE = "wordsforthewise/lending-club"
ARQUIVO_KAGGLE = "accepted_2007_to_2018Q4.csv.gz"
ORIGEM_PADRAO = Path("data/reference/raw/accepted_2007_to_2018Q4.csv")

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


def garantir_origem(origem: Path) -> None:
    """Reutiliza a origem local ou baixa o arquivo padrão do Kaggle."""
    if origem.is_file():
        print(f"Origem local encontrada; download dispensado: {origem}")
        return

    if origem.exists():
        raise ValueError(f"O caminho de origem não é um arquivo: {origem}")

    if origem.resolve() != ORIGEM_PADRAO.resolve():
        raise FileNotFoundError(
            f"Origem personalizada não encontrada: {origem}. "
            "O download automático atende apenas ao caminho padrão."
        )

    origem.parent.mkdir(parents=True, exist_ok=True)
    print("Origem ausente; iniciando download do Lending Club.")

    try:
        with TemporaryDirectory(
            prefix="lending-club-",
            dir=origem.parent,
        ) as diretorio:
            temporario = Path(diretorio)

            compactado = Path(
                kagglehub.dataset_download(
                    DATASET_KAGGLE,
                    path=ARQUIVO_KAGGLE,
                    output_dir=str(temporario / "download"),
                )
            )

            if not compactado.is_file():
                raise FileNotFoundError("O download não retornou o arquivo esperado.")

            csv_temporario = temporario / origem.name

            with (
                gzip.open(compactado, "rb") as entrada,
                csv_temporario.open("wb") as saida,
            ):
                shutil.copyfileobj(entrada, saida)

            # Confere o cabeçalho antes de disponibilizar o CSV.
            colunas_disponiveis(csv_temporario)

            csv_temporario.replace(origem)

    except Exception as erro:
        raise RuntimeError(
            "Não foi possível preparar a origem do Lending Club. "
            "Verifique conexão, espaço em disco e acesso ao Kaggle."
        ) from erro

    print(f"Origem preparada: {origem}")


def coletar_definitivos(
    origem: Path,
    colunas: list[str],
    capacidade: int,
    chunksize: int,
    seed: int,
) -> tuple[pd.DataFrame, int, int]:
    """Coleta desfechos definitivos em lotes, com afinamento para limitar a memória.

    Atribui uma prioridade aleatória a cada registro elegível e mantém os
    registros com as menores prioridades, até o limite de `capacidade`.
    Isso evita favorecer os últimos lotes e limita a memória ao buffer
    retido e ao lote em processamento.

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

        aprovados = aprovados.copy()
        aprovados["_prioridade"] = rng.random(len(aprovados))
        lotes.append(aprovados)

        if sum(len(item) for item in lotes) > capacidade:
            combinado = pd.concat(lotes, ignore_index=True)
            lotes = [combinado.nsmallest(capacidade, "_prioridade").reset_index(drop=True)]

    if not lotes:
        return pd.DataFrame(columns=colunas), total_lidas, total_definitivos
    return (
        pd.concat(lotes, ignore_index=True).drop(columns="_prioridade"),
        total_lidas,
        total_definitivos,
    )


def amostra_estratificada(base: pd.DataFrame, n_linhas: int, seed: int) -> pd.DataFrame:
    """Amostra por classe, respeitando o total solicitado.

    Distribui as vagas proporcionalmente às classes da base recebida,
    atribuindo as vagas restantes às maiores partes decimais das cotas.
    Se a base tiver menos linhas que o solicitado, retorna todas.
    """
    if n_linhas <= 0:
        raise ValueError("n_linhas deve ser maior que zero.")

    if n_linhas >= len(base):
        return base.sample(frac=1, random_state=seed).reset_index(drop=True)

    contagens = base[ALVO].value_counts()
    cotas = n_linhas * contagens / len(base)
    quantidades = np.floor(cotas).astype(int)

    faltantes = n_linhas - int(quantidades.sum())
    prioridades = (cotas - quantidades).sort_values(ascending=False, kind="stable")
    quantidades.loc[prioridades.index[:faltantes]] += 1

    partes = []
    for rotulo, quantidade in quantidades.items():
        grupo = base[base[ALVO] == rotulo]
        partes.append(grupo.sample(n=int(quantidade), random_state=seed))
    return pd.concat(partes).sample(frac=1, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera a base derivada do Lending Club.")
    parser.add_argument(
        "--source",
        type=Path,
        default=ORIGEM_PADRAO,
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

    garantir_origem(args.source)

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

# Seleção e análise do dataset — Credit Scoring (PREP-02)

Documento da issue PREP-02: escolha do dataset público de classificação binária
financeira, análise do alvo, das principais features, da qualidade e dos campos
pessoais ou sensíveis. Define a origem dos dados usados no restante do projeto.

## Origem e licença

| Item | Valor |
|---|---|
| Fonte | Kaggle — `wordsforthewise/lending-club` |
| URL | https://www.kaggle.com/datasets/wordsforthewise/lending-club |
| Dados originais | LendingClub Loan Data (aceitos e recusados, 2007–2018Q4) |
| Arquivo baixado | `accepted_2007_to_2018Q4.csv.gz` |
| Arquivo bruto usado | `accepted_2007_to_2018Q4.csv` (151 colunas) |
| Amostras na origem | 2.260.701 linhas |
| Local bruto (local) | `data/reference/raw/` — fora do versionamento |
| Licença | a confirmar na página do dataset antes de publicação |

O arquivo `rejected_2007_to_2018Q4.csv` **não** é usado para modelagem: não possui
variável-alvo e o campo `Debt-To-Income Ratio` é texto com `%`. Ele fica reservado a
uma eventual análise de governança (viés de reprovação). O `lending_club_info.csv`
serve apenas como dicionário de colunas.

## Obtenção automática e reutilização local

Na raiz do projeto, com o ambiente preparado por `uv sync --locked`, execute:

```powershell
uv run python src/monitoramento/selecao_dataset.py
```

A função `garantir_origem()` verifica a origem antes da seleção:

1. Se o CSV existir, reutiliza-o sem chamar o download do KaggleHub.
2. Se o caminho padrão estiver ausente, baixa somente o arquivo de aceitos
   `.csv.gz` de `wordsforthewise/lending-club` em um diretório temporário
   dentro de `data/reference/raw/`.
3. Descompacta em disco e valida a presença das colunas esperadas no cabeçalho.
4. Move o CSV para o destino definitivo somente após essas etapas. O contexto
   temporário remove seus arquivos ao encerrar normalmente ou tratar uma exceção.

Falhas de download ou descompactação interrompem a execução. Se `--source`
apontar para um caminho personalizado inexistente, ocorre erro sem download.
Uma origem local existente passa pela conferência de colunas na sequência
de `main()`, mas não é comparada com a versão remota ou com um hash conhecido.
Essa conferência de cabeçalho não equivale a um contrato de qualidade dos registros.

Se for necessário autenticar no Kaggle, use o login interativo:

```powershell
uv run python -c "import kagglehub; kagglehub.login()"
```

Não inserir tokens no código, nos comandos versionados ou na documentação.
A dependência declarada é `kagglehub>=1.0.3,<2`.
O uso de `path` para baixar um único arquivo e de `output_dir` para definir
o destino está descrito na [documentação oficial do KaggleHub](https://github.com/Kaggle/kagglehub#download-dataset),
assim como a [autenticação](https://github.com/Kaggle/kagglehub#authenticate).

O identificador remoto não fixa uma versão. A semente torna o sorteio
determinístico para a mesma origem, configuração e ambiente; não garante
o mesmo resultado se o Kaggle publicar novos dados. Fixar a versão e registrar
o hash da origem continuam pendentes.

## Base derivada

Gerada de forma reprodutível por [selecao_dataset.py](../src/monitoramento/selecao_dataset.py):

| Parâmetro | Valor |
|---|---|
| Origem | `data/reference/raw/accepted_2007_to_2018Q4.csv` |
| Saída | `data/reference/sample/lending_club_sample_100k.csv` |
| Linhas | 100.000 |
| Colunas | 30 (`record_id` + 29 da origem) |
| Semente | 42 |
| Lote de leitura (`--chunksize`) | 200.000 linhas |
| Fator do buffer (`--buffer-factor`) | 2 (até 200.000 registros retidos para o sorteio final) |
| Filtro | apenas `loan_status` em `Fully Paid` / `Charged Off` |
| Amostragem | retenção por prioridade aleatória, seguida de amostragem estratificada pelo alvo sobre o buffer; mantém `issue_d` |
| Tamanho | a medir na amostra atual; estimativa anterior de ~19,4 MB não reconfirmada |

```powershell
uv run python src/monitoramento/selecao_dataset.py
```

Da origem, 1.345.310 de 2.260.701 linhas têm desfecho definitivo. As colunas são
lidas como **texto** (espelhamento bruto); a tipagem fica para a camada de modelagem.
O identificador pessoal `id` é descartado e substituído por um `record_id` técnico
(`LC0000001`...).

O buffer preserva um subconjunto aleatório dos registros elegíveis; as cotas
da estratificação são calculadas nesse subconjunto, não diretamente em toda
a origem. Manter `issue_d` preserva a coluna temporal, sem implicar ordenação
cronológica ou separação de treino, validação, teste e produção simulada.

A saída é sobrescrita a cada execução. Para preservar a amostra principal,
use `--output` com outro caminho. Também estão disponíveis `--source`,
`--n-rows`, `--seed`, `--chunksize` e `--buffer-factor`.

## Validações apresentadas em 07/11/2026

O responsável executou o script duas vezes e apresentou as saídas:

| Evidência | Primeira execução | Segunda execução |
|---|---|---|
| Obtenção | download e preparação da origem | origem local; download dispensado |
| Linhas lidas | 2.260.701 | 2.260.701 |
| Desfechos definitivos | 1.345.310 | 1.345.310 |
| Linhas da amostra | 100.000 | 100.000 |
| Colunas da amostra | 30 | 30 |
| Fully Paid | 80.116 | 80.116 |
| Charged Off | 19.884 | 19.884 |

O responsável também informou aprovação dos quatro testes em
`tests/test_download_dataset.py`, executáveis com:

```powershell
uv run pytest tests/test_download_dataset.py -q
```

Eles simulam arquivo local presente, download inicial seguido de reutilização,
falha sem origem parcial e caminho personalizado ausente. Não acessam a rede
nem registros reais. O destino temporário está configurado em `pyproject.toml`
como `.venv/pytest-tmp`, exclusivo dos testes e limpo pelo pytest a cada execução.

O assistente conferiu código, configuração e saídas apresentadas; não executou
esses testes nem leu o dataset. As contagens coincidentes não demonstram
identidade byte a byte. A causa da diferença para as contagens anteriores
(79.563 / 20.437) não foi determinada nesta revisão.

## Variável-alvo

`loan_status`, binária após o filtro:

| Classe | Linhas | % |
|---|---|---|
| Fully Paid | 80.116 | 80,12 |
| Charged Off | 19.884 | 19,88 |

Classe positiva sugerida (risco de inadimplência): **Charged Off**. O
desbalanceamento é de aproximadamente 4:1 e deve ser considerado na modelagem e
nas métricas (p.ex. precision/recall, ROC-AUC).

## Principais features

**Ressalva sobre a análise anterior:** as estatísticas numéricas, frequências
categóricas e cobertura temporal abaixo foram mantidas como registro da análise
anterior. Não foram recalculadas para a amostra que produziu 80.116 / 19.884.
Devem ser reconfirmadas antes de serem usadas como resultados da amostra atual.

### Numéricas de maior relevância

| Coluna | Descrição | min | p25 | mediana | p75 | max |
|---|---|---|---|---|---|---|
| `loan_amnt` | Valor solicitado | 500 | 7.300 | 12.000 | 20.000 | 40.000 |
| `int_rate` | Taxa de juros | 5,31 | 10,49 | 13,35 | 16,55 | 30,99 |
| `installment` | Parcela mensal | 15,69 | 240,92 | 371,59 | 578,85 | 1.598,26 |
| `annual_inc` | Renda anual | 0 | 46.000 | 65.000 | 90.000 | 8.300.000 |
| `dti` | Dívida/renda | 0 | 11,49 | 17,23 | 23,46 | 999 |
| `fico_range_low` | FICO mínimo | 630 | 675 | 690 | 715 | 845 |
| `fico_range_high` | FICO máximo | 634 | 679 | 694 | 719 | 850 |
| `revol_util` | Utilização do revolving | 0 | 32,40 | 51,70 | 70,40 | 139,90 |
| `revol_bal` | Saldo rotativo | — | — | — | — | — |
| `open_acc` / `total_acc` | Contas abertas / totais | — | — | — | — | — |
| `delinq_2yrs`, `inq_last_6mths`, `pub_rec` | Histórico de crédito | — | — | — | — | — |

Valores em formato decimal com ponto; a fonte armazena tudo como texto.

### Categóricas

| Coluna | Categorias distintas | Exemplares mais frequentes |
|---|---|---|
| `term` | 2 | `" 36 months"` (76.381), `" 60 months"` (23.619) |
| `grade` | 7 | B, C, D |
| `sub_grade` | 35 | C1, B5, B4 |
| `home_ownership` | 6 | MORTGAGE, RENT, OWN |
| `verification_status` | 3 | Source Verified, Verified, Not Verified |
| `purpose` | 14 | debt_consolidation, credit_card, home_improvement |
| `addr_state` | 50 | CA, TX, NY |
| `application_type` | 2 | Individual, Joint App |
| `initial_list_status` | 2 | w, f |
| `emp_length` | 12 | 10+ years, 2 years, 3 years |

### Eixo temporal

`issue_d` (mês de concessão) cobre **2007 a 2018** (12 anos). É a base para
separar Referência (treino) e Produção simulada e para simular drift ao longo do
tempo. Complementar: `earliest_cr_line` (mês de abertura da primeira linha de
crédito).

### Risco de vazamento

`grade`, `sub_grade`, `int_rate` e `installment` são atribuídos pela LendingClub na
origem do contrato. Se o objetivo for uma pontuação **pré-decisão**, essas colunas
devem ser tratadas com cuidado (ou removidas) para não introduzir vazamento. A
decisão cabe à issue de definição do schema.

## Qualidade dos dados

**Resultados da análise anterior, pendentes de recálculo na amostra atual.**
As contagens e observações abaixo não foram confirmadas pelas duas execuções,
que reportam apenas volumes e distribuição do alvo.

Na análise anterior, valores ausentes foram representados por string vazia:

| Coluna | Vazios | % |
|---|---|---|
| `emp_length` | 5.703 | 5,70 |
| `mort_acc` | 5.170 | 5,17 |
| `revol_util` | 83 | 0,08 |
| `pub_rec_bankruptcies` | 60 | 0,06 |
| `dti` | 39 | 0,04 |

Pontos de atenção identificados (insumo para as regras rígidas da próxima issue):

- `term` vem com espaço à esquerda (`" 36 months"`), exigindo normalização.
- `dti` apresenta valor **sentinela 999**, fora da faixa plausível (0–100).
- `annual_inc` tem extremos (0 e 8.300.000), compatíveis com erro ou outlier.
- Colunas numéricas são texto e podem trazer `%`/espaços, exigindo conversão explícita.
- As demais colunas do recorte não apresentam vazios.

## Candidatas à simulação de drift

Selecionadas por serem contínuas, relevantes ao negócio e sensíveis a mudanças
econômicas:

1. **`annual_inc`** — renda; desloca com inflação e mudança de perfil de cliente.
2. **`dti`** — comprometimento de renda; sobe em cenários de estresse econômico.
3. **`int_rate`** — taxa precificada; reflete política de crédito e risco percebido.

Complementares: `fico_range_low`/`fico_range_high` e `revol_util`.

Os testes estatísticos previstos são **PSI** e **Kolmogorov-Smirnov**, a serem
implementados na issue de detecção (evidência da mudança de distribuição).

## Campos pessoais ou sensíveis (LGPD)

| Campo | Natureza | Tratamento proposto |
|---|---|---|
| `annual_inc` | Dado financeiro pessoal | Não publicar valores por indivíduo em relatórios |
| `dti`, `revol_bal`, `revol_util` | Financeiro pessoal | Uso interno; agregar antes de expor |
| `addr_state` | Localização (quasi-identificador) | Manter agregado; não cruzar com outros dados |
| `earliest_cr_line`, `issue_d` | Histórico de crédito | Uso interno |
| `fico_range_low` / `fico_range_high` | Score de crédito | Dado sensível |
| `emp_length` | Ocupação/renda implícita | Uso interno |

Campos **excluídos** da base derivada e recomendados a permanecer fora: `id`,
`member_id` (identificadores diretos), `emp_title`, `title`, `desc` (texto livre) e
`zip_code` (localização granular). Essas exclusões reduzem risco de
reidentificação sob LGPD.

## Critérios de aceite (PREP-02)

| Critério | Evidência |
|---|---|
| Dataset financeiro com ≥5.000 amostras e origem documentada | 100.000 linhas; origem Kaggle/LendingClub documentada acima |
| Alvo e principais features identificados | `loan_status` (binário) e tabelas de features |
| Pelo menos duas variáveis candidatas a drift | `annual_inc`, `dti`, `int_rate` (+ FICO, `revol_util`) |
| Campos pessoais ou sensíveis identificados | Tabela LGPD |

## Pendências

- Confirmar a licença de uso do dataset na página do Kaggle.
- Fixar a versão remota e registrar o hash da origem para reprodução entre ambientes.
- Recalcular tamanho do arquivo, estatísticas numéricas, frequências categóricas,
  cobertura temporal e indicadores de qualidade na amostra atual.
- Comparar hashes das saídas caso seja necessário comprovar identidade byte a byte.
- Definir o schema de negócio (tipos, categorias, classe positiva) e as três ou
  mais regras rígidas de qualidade.
- Definir a estratégia de separação treino/validação/teste e os cenários de drift.

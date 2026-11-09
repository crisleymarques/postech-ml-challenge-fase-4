# postech-ml-challenge-fase-4

Projeto do Tech Challenge da Fase 4 da pós-graduação em Machine Learning Engineering da PosTech, desenvolvido em equipe de três integrantes.

Este projeto é a camada de sustentação e confiabilidade de um modelo de Credit
Scoring: contratos de qualidade de dados, simulação de degradação (Data e Concept
Drift), detecção com observabilidade e governança sob a LGPD. A documentação reside
em [docs/](docs/), com a [arquitetura e os contratos entre componentes](docs/ARQUITETURA.md)
e a [seleção e análise do dataset](docs/DATASET.md). Esses documentos são atualizados
conforme o grupo define escopo e implementação.

## Documentação

| Documento | Conteúdo |
|---|---|
| [Contexto do projeto](docs/CONTEXTO_PROJETO.md) | Requisitos acadêmicos, decisões e pendências |
| [Arquitetura e contratos](docs/ARQUITETURA.md) | Fluxo proposto, interfaces e artefatos entre componentes |
| [Dataset — PREP-02](docs/DATASET.md) | Origem, download, amostragem, análise e limitações dos dados |
| [Registro de alterações](docs/REGISTRO_DE_ALTERACOES.md) | Autorizações, alterações e validações documentadas |
| [Instruções do repositório](AGENTS.md) | Regras de orientação, execução e privacidade |

## Ambiente de desenvolvimento

O projeto utiliza Python 3.12 e uv para gerenciar o ambiente e as
dependências.

### Pré-requisitos

- Python 3.12.
- uv instalado.

### Reprodução do ambiente

Na raiz do repositório, execute:
```powershell
uv sync --locked
```
Esse comando sincroniza o ambiente .venv com as versões registradas
no uv.lock. Não é necessário ativar o ambiente ao usar uv run.

### Dependências

| Ferramenta | Finalidade |
|---|---|
| KaggleHub | Download do arquivo bruto do Lending Club quando ausente |
| NumPy e pandas | Manipulação e processamento de dados |
| SciPy | Testes estatísticos |
| scikit-learn | Preparação dos dados e modelo baseline |
| Pandera | Contratos e validação de dados |
| Evidently | Detecção de drift e relatórios |
| Streamlit | Dashboard de monitoramento |
| JupyterLab e ipykernel | Notebooks e experimentação |
| pytest | Testes automatizados |
| Ruff | Análise estática e formatação de código |

A instalação dessas ferramentas prepara o ambiente; não representa
a implementação dos componentes do pipeline.

### Configuração e reprodutibilidade

- .python-version: seleciona Python 3.12 para o ambiente local.
- pyproject.toml: declara dependências e configura Ruff e pytest.
- uv.lock: registra as versões resolvidas das dependências.
- .venv/: ambiente local, excluído do versionamento.

O pandas está limitado a >=2.2,<3.1 devido à incompatibilidade de
importação observada entre Evidently e pandas 3.1. A verificação de
importações passou com pandas 3.0.6.

## Estrutura inicial
```text
src/
└── monitoramento/    # Código do projeto
tests/               # Testes automatizados
notebooks/           # Exploração e experimentação
reports/             # Relatórios e evidências
models/
data/
├── reference/       # Dados de referência
└── analysis/        # Dados de análise/produção simulada
```
As pastas inicialmente vazias contêm .gitkeep para preservar a
estrutura no Git.

O conteúdo de data/reference/ e data/analysis/ está excluído do
versionamento, exceto pelos arquivos .gitkeep. Os arquivos brutos ficam em
data/reference/raw/ e a base derivada em data/reference/sample/.

## Verificações do ambiente

Execute no PowerShell, na raiz do projeto:
```powershell
uv pip check
uv run ruff check .
uv run python -c "import numpy, pandas, scipy, sklearn, pandera, evidently, streamlit; print('Importações OK')"
uv run pytest --version
```

Durante o setup, foram verificadas a consistência das dependências,
as importações das bibliotecas e a disponibilidade de Ruff e pytest.
A análise estática também passou.

Essas verificações não validam o pipeline completo. A obtenção e a separação
do dataset possuem testes isolados descritos abaixo; contratos de dados,
treinamento, experimentos de drift e dashboard permanecem pendentes.

## Arquitetura e contratos

A arquitetura definida para implementação é um pipeline local em lote:
preparação e separação → validação de dados → treinamento e avaliação;
em paralelo ao treino, a base reservada alimenta simulação → validação de
lotes → predição → detecção. Lotes inválidos são bloqueados antes de chegar
ao modelo. Todas as etapas exportam eventos e resultados para monitoramento.

O dashboard **Streamlit** foi escolhido para observabilidade, com eventos
JSONL, métricas JSON e relatórios HTML do Evidently. MLflow não integra a
arquitetura base. As métricas específicas e os limiares serão definidos pelo
grupo durante a implementação.

Os [contratos entre componentes](docs/ARQUITETURA.md) detalham o diagrama,
os formatos CSV/JSON/JSONL, manifestos, argumentos futuros dos scripts,
estados de execução e artefatos produzidos por etapa. Dados de treino,
validação e teste ficam separados da base e dos lotes de produção simulada;
a referência de drift é congelada a partir das features de treino.

Os scripts de obtenção/amostragem do dataset e de preparação e separação das
partições estão implementados. Contratos Pandera, treinamento, simulação,
detecção e dashboard ainda não estão implementados. A documentação de arquitetura
não substitui as evidências funcionais nem a governança exigidas na entrega.

## Dataset — PREP-02

O dataset escolhido é o **Lending Club Loan Data**, obtido no Kaggle
(`wordsforthewise/lending-club`). O arquivo de empréstimos aceitos possui
2.260.701 linhas e 151 colunas; usamos apenas os registros com desfecho
definitivo.

Para viabilizar o trabalho, o script
[src/monitoramento/selecao_dataset.py](src/monitoramento/selecao_dataset.py)
gera de forma reprodutível uma base derivada de **100.000 linhas** em
`data/reference/sample/lending_club_sample_100k.csv`, mantendo `Fully Paid` e
`Charged Off` e preservando o eixo temporal `issue_d`.

A análise completa — origem, alvo, principais features, qualidade, candidatas a
drift e campos pessoais/sensíveis (LGPD) — está em
[docs/DATASET.md](docs/DATASET.md).

### Download e geração da amostra

Após `uv sync --locked`, execute no PowerShell, na raiz do projeto:

```powershell
uv run python src/monitoramento/selecao_dataset.py
```

Se o CSV padrão estiver ausente, o script baixa apenas
`accepted_2007_to_2018Q4.csv.gz` via KaggleHub, descompacta em diretório
temporário e confere as colunas antes de disponibilizar a origem em
`data/reference/raw/accepted_2007_to_2018Q4.csv`.
Nas próximas execuções, reutiliza esse CSV sem novo download.
A amostra é gerada novamente a cada execução, sobrescrevendo a saída escolhida.
Um `--source` personalizado inexistente gera erro, sem download automático.

Para preservar a amostra principal ao experimentar, informe outra saída:

```powershell
uv run python src/monitoramento/selecao_dataset.py --output data/reference/sample/lending_club_teste.csv
```

Caso o Kaggle exija autenticação, configure-a interativamente, sem colocar
credenciais no código:

```powershell
uv run python -c "import kagglehub; kagglehub.login()"
```

As duas execuções apresentadas pelo responsável em 07/11/2026 confirmaram
download inicial e reutilização local. Ambas reportaram 2.260.701 linhas lidas,
1.345.310 desfechos definitivos e amostra de 100.000 linhas e 30 colunas:
80.116 `Fully Paid` e 19.884 `Charged Off`.
Contagens iguais não comprovam identidade dos arquivos byte a byte.
A versão remota do dataset ainda não está fixada; um novo download futuro
pode obter uma versão diferente.

### Testes da obtenção do dataset

```powershell
uv run pytest tests/test_download_dataset.py -q
```

Para executar todos os testes disponíveis:

```powershell
uv run pytest
```

Os quatro testes de download usam simulações e conteúdo sintético, sem
acessar a rede ou o dataset real. Cobrem origem local existente, download
inicial seguido de reutilização, falha sem origem parcial e origem
personalizada ausente. O responsável informou aprovação dos quatro testes.

O `pyproject.toml` configura `--basetemp=.venv/pytest-tmp`, evitando a pasta
temporária do sistema que apresentou erro de permissão no ambiente local.
Esse diretório é exclusivo dos testes: o pytest limpa seu conteúdo a cada
execução. Essa validação não cobre o pipeline completo nem substitui contratos
de qualidade, treinamento e monitoramento.

## Preparação e separação das partições — F1-01

O componente [src/monitoramento/preparacao.py](src/monitoramento/preparacao.py)
lê a base derivada, binariza o alvo (`Charged Off` = 1, `Fully Paid` = 0) e separa
quatro partições **disjuntas** pela ordem temporal de `issue_d`, conforme os
parâmetros de [configs/preparacao.json](configs/preparacao.json) (60% treino, 20%
validação, 10% teste, 10% produção simulada):

```powershell
uv run python src/monitoramento/preparacao.py --config configs/preparacao.json
```

| Artefato | Local |
|---|---|
| Treino, validação e teste | `data/reference/<preparation_id>/{train,validation,test}.csv` |
| Base de produção simulada | `data/analysis/<preparation_id>/simulation_base.csv` |
| Manifesto, eventos, métricas e config | `reports/runs/<run_id>/` |

A etapa não ajusta imputação, escala ou codificação sobre o conjunto completo:
apenas faz `trim` nos textos e mantém o alvo binarizado. Ela recusa sobrescrever
um diretório de partição já existente e encerra com código `0` (sucesso), `2`
(dados rejeitados) ou `1` (erro técnico). O contrato completo está em
[docs/ARQUITETURA.md](docs/ARQUITETURA.md) e os resultados da execução real em
[docs/DATASET.md](docs/DATASET.md).

```powershell
uv run pytest tests/test_preparacao.py -q
```

## Treinamento do modelo baseline — F1-02

O componente [src/monitoramento/treinamento.py](src/monitoramento/treinamento.py)
consome o manifesto da preparação, ajusta o pipeline no treino, seleciona pela
validação e avalia no teste, conforme os parâmetros de
[configs/treinamento.json](configs/treinamento.json):

```powershell
uv run python src/monitoramento/treinamento.py ^
  --config configs/treinamento.json ^
  --input-manifest reports/runs/<run_id_da_preparacao>/manifest.json
```

| Artefato | Local |
|---|---|
| Pipeline persistido | `models/<model_id>/pipeline.pkl` |
| Metadados do modelo | `models/<model_id>/metadata.json` |
| Métricas de avaliação | `models/<model_id>/evaluation.json` |
| Referência congelada | `data/reference/<reference_id>/features.csv` |
| Manifesto, eventos, métricas e config | `reports/runs/<run_id>/` |

O `model_id` deriva do `dataset_id`, do algoritmo e do hash SHA-256 do
descritor do artefato (features, alvo, hiperparâmetros e origem), o que o torna
reproduzível. A etapa aceita `--config`, `--input-manifest`, `--run-id` e
`--output-dir`, exige um manifesto de `preparacao` com `status` de sucesso e
confere o hash de cada partição de entrada. Encerra com código `0` (sucesso), `2`
(dados rejeitados) ou `1` (erro técnico). O contrato completo está em
[docs/ARQUITETURA.md](docs/ARQUITETURA.md).

```powershell
uv run pytest tests/test_treinamento.py -q
```

### Baseline: regressão logística

O baseline usa regressão logística com `class_weight="balanced"`, semente 42 e
22 features (14 numéricas e 8 categóricas) sobre as partições de referência. O
`model_id` é `baseline-logreg-820dbb61` e a referência `ref-820dbb61`. A seleção
do modelo usa `roc_auc` na validação. A classe positiva é `Charged Off` e o
limiar de decisão é `0.5`.

| Métrica | Treino | Validação | Teste |
|---|---|---|---|
| accuracy | 0.6600 | 0.6625 | 0.6626 |
| precision | 0.2965 | 0.3577 | 0.3641 |
| recall | 0.6262 | 0.5900 | 0.5658 |
| f1 | 0.4025 | 0.4454 | 0.4431 |
| roc_auc | 0.7029 | 0.6961 | 0.6891 |

As métricas são referência para comparação com modelos e cenários futuros. O
`roc_auc` próximo de 0,69 e a precisão baixa em relação ao recall indicam um
modelo simples, com muitas features sem tratamento e sem engenharia de
atributos; esses resultados serão usados como linha de base para medir a
degradação nos cenários de drift.

A pasta models/ armazena os modelos gerados localmente. Seus arquivos
não são versionados, exceto pelo .gitkeep.

## Validação do contrato de qualidade — F1-03

O componente [src/monitoramento/validacao.py](src/monitoramento/validacao.py)
consome o manifesto de um produtor (`preparacao` ou `simulacao`), confere os
hashes dos artefatos tabulares e aplica o contrato Pandera descrito em
[configs/validacao.json](configs/validacao.json):

```powershell
uv run python src/monitoramento/validacao.py ^
  --config configs/validacao.json ^
  --input-manifest reports/runs/<run_id_do_produtor>/manifest.json
```

| Artefato | Local |
|---|---|
| Relatório de qualidade | `reports/runs/<run_id>/quality.json` |
| Manifesto, eventos, métricas e config | `reports/runs/<run_id>/` |

O contrato exige ao menos três regras rígidas e cobre identificador (formato),
alvo (domínio `0`/`1`), campos obrigatórios não nulos, unicidade, faixas
numéricas e domínios categóricos. Uma violação bloqueia o conjunto inteiro: o
relatório registra `approved`, `checked_rows` e a lista de `violations` com
`rule_id` e `count`, sem reescrever nem descartar registros. A etapa aceita
`--config`, `--input-manifest`, `--run-id` e `--output-dir` e encerra com código
`0` (aprovado), `2` (dados rejeitados) ou `1` (erro técnico). O contrato completo
está em [docs/ARQUITETURA.md](docs/ARQUITETURA.md).

A execução real sobre o manifesto da preparação aprovou 90.000 linhas
(60.000 treino, 20.000 validação e 10.000 teste), com `approved: true` e nenhuma
violação registrada.

```powershell
uv run pytest tests/test_validacao.py -q
```

## Bloqueio de ingestão de dados inválidos — F1-04

O componente
[src/monitoramento/lote_invalido.py](src/monitoramento/lote_invalido.py) gera um
lote propositalmente inválido a partir da base de simulação. Ele amostra um
conjunto determinístico (semente fixa), divide em treino, validação e teste e
injeta uma anomalia para cada tipo de regra do contrato, publicando o manifesto
do produtor (`stage: simulacao`, `status: success`) junto do lote:

```powershell
uv run python src/monitoramento/lote_invalido.py ^
  --config configs/validacao.json ^
  --source data/analysis/lending_club_100k_v1/simulation_base.csv ^
  --batch-id batch-f104 ^
  --size 600
```

| Partição | Regra exercitada | Anomalia injetada |
|---|---|---|
| train | `R_ID_FORMATO` | identificador fora do padrão `LC#######` |
| train | `R_UNICO_record_id` | linha duplicada |
| validation | `R_NAO_NULO_loan_amnt` | `loan_amnt` vazio |
| validation | `R_FAIXA_loan_amnt` | `loan_amnt` acima do limite |
| test | `R_CATEGORIA_term` | `term` fora do domínio |
| test | `R_ALVO_DOMINIO` | `loan_status` fora de `0`/`1` |

O lote e o manifesto são gravados em
`data/analysis/<preparation_id>/lote-invalido/<batch_id>/` e
`reports/runs/<run_id>/`, ambos ignorados pelo Git. Em seguida, a validação do
lote comprova o bloqueio: o contrato reprova o conjunto inteiro e a etapa
encerra com código `2` e `error_code: QUALIDADE_REPROVADA`, registrando cada
regra violada por `rule_id`, sem aprovar nenhum artefato para consumo.

```powershell
uv run python src/monitoramento/validacao.py ^
  --config configs/validacao.json ^
  --input-manifest reports/runs/<run_id_do_lote>/manifest.json
```

Resultado real da demonstração (lote de 600 linhas, 601 após a duplicata):

| rule_id | ocorrências |
|---|---|
| `R_ALVO_DOMINIO` | 1 |
| `R_CATEGORIA_term` | 1 |
| `R_FAIXA_loan_amnt` | 1 |
| `R_ID_FORMATO` | 1 |
| `R_NAO_NULO_loan_amnt` | 1 |
| `R_UNICO_record_id` | 2 |

```powershell
uv run pytest tests/test_lote_invalido.py -q
```

## Convenção de commits

Os commits seguem o formato:

tipo: descrição curta

| Tipo | Uso |
|---|---|
| feat | Nova funcionalidade |
| fix | Correção de erro |
| docs | Alteração na documentação |
| test | Criação ou alteração de testes |
| refactor | Reorganização de código sem mudar o comportamento |
| chore | Configuração, dependências e manutenção |

Exemplos:

- chore: configura ambiente com uv
- feat: adiciona contrato de dados
- test: verifica rejeição de lote inválido

## Estado atual

Concluídos:

- Configuração do ambiente com uv e Python 3.12.
- Instalação das dependências de execução e desenvolvimento.
- Configuração inicial de Ruff e pytest.
- Criação da estrutura de pastas e das regras de exclusão dos dados.
- Seleção e análise do dataset na PREP-02, com geração da base derivada de
  100.000 linhas e documentação em docs/DATASET.md.
- Download automático do arquivo bruto ausente e reutilização local via KaggleHub,
  com quatro testes isolados aprovados segundo o responsável e duas execuções
  reais apresentadas na conversa.
- Definição da arquitetura, contratos entre etapas e estratégia
  de observabilidade com Streamlit.
- Preparação e separação das partições de treino, validação, teste e produção
  simulada (F1-01), com manifesto, eventos e métricas, e testes em
  `tests/test_preparacao.py`.
- Treinamento e avaliação do modelo baseline de Credit Scoring (F1-02), com
  registro das métricas de classificação por partição, modelo persistido para
  inferência, referência congelada e testes em `tests/test_treinamento.py`.
- Contrato de qualidade de dados com Pandera (F1-03), aplicado às partições de
  treino, validação e teste a partir do manifesto da preparação, com relatório
  `quality.json`, bloqueio do conjunto inteiro em caso de violação e testes em
  `tests/test_validacao.py`.
- Demonstração do bloqueio de ingestão de dados inválidos (F1-04), com geração
  de lote com anomalias intencionais, reprovação pelo contrato
  (`QUALIDADE_REPROVADA`) e testes em `tests/test_lote_invalido.py`.

Pendentes:

- Confirmação da licença de uso do dataset.
- Fixação da versão remota do dataset e revisão das estatísticas anteriores
  para a amostra atual, conforme ressalvas em docs/DATASET.md.
- Definição dos cenários de drift da produção simulada.
- Validação funcional integrada do pipeline com dados sintéticos, além dos
  testes isolados de obtenção, de preparação do dataset, de treinamento e de
  validação.
- Implementação das etapas restantes do pipeline (simulação, detecção, dashboard).

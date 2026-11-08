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

## Modelos

A pasta models/ armazena os modelos gerados localmente. Seus arquivos
não são versionados, exceto pelo .gitkeep. O procedimento de geração
será documentado quando o treinamento for implementado.

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

Pendentes:

- Confirmação da licença de uso do dataset.
- Fixação da versão remota do dataset e revisão das estatísticas anteriores
  para a amostra atual, conforme ressalvas em docs/DATASET.md.
- Definição do schema de negócio e das regras rígidas de qualidade.
- Definição dos cenários de drift da produção simulada.
- Validação funcional integrada do pipeline com dados sintéticos, além dos
  testes isolados de obtenção e de preparação do dataset.
- Implementação do pipeline (contratos, treino, simulação, detecção, dashboard).

# Análise de vieses e mitigação — F4-02

Documento da issue F4-02: identificação de fontes potenciais de viés no dataset e
no modelo de Credit Scoring, limitações da análise, impactos de decisões
enviesadas e medidas de mitigação propostas. Complementa a
[governança de dados e conformidade com a LGPD](LGPD.md) (F4-01), que trata a
não discriminação como requisito da decisão de crédito.

A análise é **descritiva**: mede diferenças de comportamento do modelo entre
grupos, mas **não** demonstra discriminação nem estabelece causalidade. As
atribuições protegidas (raça, gênero, idade, etc.) **não existem** no dataset;
os grupos abaixo são **proxies** socioeconômicos e geográficos.

## 1. Escopo e método

| Item | Definição |
|---|---|
| Modelo avaliado | `baseline-logreg-820dbb61` — regressão logística, `class_weight="balanced"`, limiar 0,5 |
| Artefatos | [models/baseline-logreg-820dbb61](../models/baseline-logreg-820dbb61/) (`metadata.json`, `evaluation.json`) |
| Features | 22 atributos de [configs/treinamento.json](../configs/treinamento.json) (14 numéricos e 8 categóricos) |
| Partição | `test` — 10.000 linhas, `2016-10` a `2017-07`, não usada em ajuste nem seleção |
| Alvo | `loan_status`; classe positiva `Charged Off` (inadimplência) |
| Métricas | taxa base, taxa de seleção (predição positiva), TPR (recall), FPR, ROC-AUC |

O limiar 0,5 faz a predição positiva significar "modelo sinaliza risco de
inadimplência". Nesse enquadramento, **FPR** é a fração de bons pagadores
sinalizados indevidamente e **TPR** é a fração de inadimplentes efetivamente
identificados.

As métricas agregadas desta análise (**recall 0,5658, ROC-AUC 0,6891, precisão
0,3641**) coincidem com as registradas em
[evaluation.json](../models/baseline-logreg-820dbb61/evaluation.json) para a
partição de teste, o que valida a reprodução. O procedimento consta do
Apêndice.

**Natureza dos grupos.** São proxies, não atributos protegidos. `addr_state`
guarda correlação histórica com composição demográfica; `home_ownership`,
`annual_inc` e `emp_length` aproximam condição socioeconômica; `application_type`
distingue proposta individual de conjunta. A ausência de atributos protegidos
**não elimina** o risco de discriminação indireta: proxies podem transportá-la.

## 2. Fontes potenciais de viés

### 2.1 Viés de seleção da amostra

- **Somente contratos concedidos.** O treino usa o arquivo de empréstimos
  **aceitos**. Quem foi recusado pela LendingClub não tem desfecho e não entra
  no modelo, de modo que o modelo aprende a política do credor, não o risco de
  toda a população. O arquivo `rejected_2007_to_2018Q4.csv` existe em
  `data/reference/raw/` mas **não é usado** (ver [DATASET.md](DATASET.md)).
- **Somente desfechos definitivos.** Registros `Current` e outros estados são
  descartados; há viés de maturidade, pois contratos recentes ainda sem
  desfecho saem da amostra.
- **Corte por FICO.** A partição de teste não contém nenhum registro com
  `fico_range_low` abaixo de 660 (ver Seção 3). O desempenho do modelo sobre a
  faixa de risco mais baixo é, portanto, **desconhecido** e não avaliado.
- **Período 2007–2018.** Inclui a crise de 2008; a relação entre perfil e
  desfecho não é estacionária no tempo.

### 2.2 Viés por atributos representados (proxies)

Presentes nas features do modelo:

| Feature | Proxy de | Observação |
|---|---|---|
| `addr_state` | Localização / composição demográfica | 50 categorias; correlação histórica com perfis populacionais |
| `home_ownership` | Condição socioeconômica | `RENT`/`OWN`/`MORTGAGE` associados a patrimônio |
| `annual_inc` | Renda | Extremos e assimetria já observados em [DATASET.md](DATASET.md) |
| `emp_length` | Ocupação/estabilidade | Ausente em parte dos registros; ausência não é aleatória |
| `application_type` | Arranjo do titular | `Joint App` sugere segundo proponente |
| `purpose` | Contexto do crédito | Pode sinalizar necessidade financeira |

### 2.3 Viés de rótulo e vazamento

`grade`, `sub_grade`, `int_rate` e `installment` são atribuídos pela LendingClub
na origem do contrato e carregam sua decisão precificada; [DATASET.md](DATASET.md)
já registra o risco de vazamento. **Essas colunas não integram as features** do
baseline — mitigação já aplicada. Permanecem, porém, atributos definidos na
concessão, como `term` e `verification_status`, que podem codificar a política
do credor. O rótulo (`Charged Off`) é o desfecho observado de contratos
aprovados, não uma medida neutra de risco.

### 2.4 Viés estatístico e desbalanceamento

- Classe positiva em **23,72%** na partição de teste (proporção ~3:1). Com
  `class_weight="balanced"`, o modelo privilegia o recall e eleva o FPR global
  para **30,73%** — cerca de três bons pagadores sinalizados a cada dez.
- Ausência de `emp_length` concentra maior risco real (taxa base 33,74%) e o
  pior ROC-AUC por grupo (0,6377); a imputação pela moda pode desfavorecer esse
  segmento.

## 3. Evidências quantitativas (partição de teste)

**Agregado (10.000 registros):**

| Métrica | Valor |
|---|---|
| Taxa base (inadimplência real) | 23,72% |
| Taxa de seleção (predito positivo) | 36,86% |
| TPR (recall) | 56,58% |
| FPR | 30,73% |
| Precisão | 36,41% |
| ROC-AUC | 0,6891 |

**Por grupo** (subgrupos com n ≥ 100; subgrupos residuais omitidos):

| Grupo | Subgrupo | n | Taxa base | Taxa seleção | TPR | FPR | ROC-AUC |
|---|---|---|---|---|---|---|---|
| `home_ownership` | MORTGAGE | 5119 | 18,62% | 29,67% | 48,16% | 25,44% | 0,6785 |
| `home_ownership` | RENT | 3749 | 30,41% | 47,21% | 64,47% | 39,67% | 0,6798 |
| `home_ownership` | OWN | 1116 | 24,64% | 35,13% | 53,09% | 29,25% | 0,6710 |
| `application_type` | Individual | 9520 | 23,79% | 36,67% | 56,56% | 30,46% | 0,6905 |
| `application_type` | Joint App | 480 | 22,29% | 40,62% | 57,01% | 35,92% | 0,6650 |
| `verification_status` | Verified | 2892 | 29,36% | 47,65% | 64,90% | 40,48% | 0,6644 |
| `verification_status` | Source Verified | 4025 | 24,35% | 36,99% | 54,08% | 31,49% | 0,6738 |
| `verification_status` | Not Verified | 3083 | 17,61% | 26,57% | 48,07% | 21,97% | 0,7078 |
| `term` | 36 months | 7715 | 20,48% | 22,85% | 38,80% | 18,74% | 0,6803 |
| `term` | 60 months | 2285 | 34,66% | 84,16% | 92,05% | 79,97% | 0,6608 |
| `emp_length` | < 1 ano | 752 | 24,87% | 37,23% | 53,48% | 31,86% | 0,6675 |
| `emp_length` | 1–4 anos | 3039 | 24,88% | 35,01% | 54,10% | 28,69% | 0,6902 |
| `emp_length` | 5–9 anos | 2127 | 23,27% | 36,48% | 58,38% | 29,84% | 0,7025 |
| `emp_length` | 10+ anos | 3424 | 20,79% | 38,17% | 60,53% | 32,30% | 0,7018 |
| `emp_length` | ausente | 658 | 33,74% | 39,36% | 50,90% | 33,49% | 0,6377 |
| Faixa de renda | Q1 (menor) | 2502 | 29,02% | 41,01% | 57,71% | 34,18% | 0,6645 |
| Faixa de renda | Q2 | 2514 | 24,82% | 40,97% | 58,81% | 35,08% | 0,6887 |
| Faixa de renda | Q3 | 2504 | 21,45% | 36,26% | 56,80% | 30,66% | 0,6776 |
| Faixa de renda | Q4 (maior) | 2480 | 19,56% | 29,11% | 51,75% | 23,61% | 0,7130 |
| Faixa de FICO | 660–699 | 6010 | 28,29% | 48,79% | 64,71% | 42,51% | 0,6555 |
| Faixa de FICO | 700–739 | 2831 | 19,00% | 23,63% | 39,78% | 19,84% | 0,6871 |
| Faixa de FICO | 740+ | 1159 | 11,56% | 7,33% | 20,90% | 5,56% | 0,7179 |
| Faixa de FICO | < 660 | 0 | — | — | — | — | — |

Quartis de `annual_inc` usados nos cortes: 48.000, 67.000 e 96.000.

**Por estado (`addr_state`, 28 estados com n ≥ 100):** a taxa de seleção varia
de **7,76%** (OR, n=116) a **48,82%** (AL, n=127) e o TPR de **15,79%** (OR) a
**72,73%** (MN, n=177). Extremos de taxa de seleção: OR 7,76%, CO 23,44%,
IL 26,51% (menores); AL 48,82%, NV 46,45%, NJ 45,23% (maiores).

### Leitura das evidências

- As diferenças de **taxa de seleção** acompanham, em boa parte, as diferenças
  de **taxa base** (risco real). Isso é esperado e não é, por si, evidência de
  viés: RENT tem risco real maior (30,41% vs 18,62%) e por isso é sinalizado
  mais.
- O ponto de atenção é a **divergência de erros**: RENT apresenta FPR de 39,67%
  contra 25,44% de MORTGAGE — bons pagadores que alugam são sinalizados como
  risco com mais frequência, o que pode restringir crédito de forma desigual.
- `term` domina a decisão: em 60 meses, 84,16% dos registros são sinalizados,
  com FPR de **79,97%**. A variável quase determina a saída, e é definida na
  concessão, não antes da decisão.
- A faixa de FICO mais baixa **não tem registros** no teste: o modelo é avaliado
  sobre uma população já filtrada pelo credor, reforçando o viés de seleção.

## 4. Limitações da análise

- **Sem atributos protegidos.** Só há proxies; não é possível calcular métricas
  formais de equidade sobre raça, gênero ou idade.
- **Grupos pequenos.** Subgrupos com n < 200 têm estimativas instáveis; por isso
  foram reportados apenas os com n ≥ 100.
- **Sem inferência estatística.** Não foram calculados intervalos de confiança
  nem testes de significância das diferenças; os valores são descritivos.
- **Ponto único.** A análise cobre apenas a partição de teste do baseline local;
  não cobre produção simulada, drift (F2-03/F2-04) nem concept drift (F2-05).
- **Sem ajuste nem calibração por grupo.** As diferenças observadas são do
  modelo como está, sem pós-processamento.
- **Correlação não é causalidade.** Nenhuma relação aqui estabelece causa; a
  discussão causal é objeto da F4-03.
- **Contexto.** Dataset histórico dos EUA; os resultados não descrevem o
  contexto brasileiro.

## 5. Impactos das decisões enviesadas

- **Distributivo (titular).** Grupos com FPR mais alto veem bons pagadores
  recusados ou sobretaxados com mais frequência — restrição de acesso a crédito.
- **Qualidade da decisão.** Grupos com TPR mais baixo têm inadimplentes menos
  identificados, elevando o risco da carteira aprovada.
- **Discriminação indireta.** Usar proxies como `addr_state`, `home_ownership` e
  `emp_length` pode reproduzir desigualdades históricas sem intenção explícita,
  contrariando os princípios da [LGPD](LGPD.md).
- **Decisão automatizada.** Se a pontuação embasar decisão de crédito, incide o
  Art. 20 da LGPD (revisão humana e transparência), já previsto no plano da
  F4-01.
- **Reputação e regulação.** Sinalizações desiguais expõem o projeto a risco
  reputacional e a questionamentos sobre a licitude do tratamento.

## 6. Medidas de mitigação propostas

| Medida | Ação | Situação |
|---|---|---|
| Exclusão de vazamento | Manter fora as colunas definidas na concessão (`grade`, `sub_grade`, `int_rate`, `installment`) | **Já aplicada** no baseline |
| Redução de proxies | Reavaliar `addr_state`; agrupar em regiões ou remover se o ganho de desempenho não compensar o risco | Proposta |
| Métricas de equidade | Calcular, por grupo, diferença de paridade demográfica, diferença de oportunidade igual (TPR) e razão de impacto (regra 4/5) | Proposta |
| Reponderação | Ajustar pesos de treino por subgrupo para reduzir divergência de erros | Proposta |
| Ajuste de limiar | Explorar limiares por subgrupo, com avaliação jurídica prévia | Proposta |
| Calibração | Calibrar probabilidades para que o score reflita risco real em todos os grupos | Proposta |
| Revisão humana | Revisar casos limítrofes e decisões automatizadas (Art. 20) | **Prevista** na F4-01 |
| Transparência | Registrar modelo, features e limitações (cartão de modelo) para o consumidor | Proposta |
| Monitoramento | Acompanhar desempenho e drift por segmento na observabilidade existente | **Já disponível** (F2/F3) |
| Governança | Proibir a introdução de atributos protegidos ou proxies diretos sem avaliação prévia | Proposta |

## 7. Critérios de aceite (F4-02)

| Critério | Seção | Evidência |
|---|---|---|
| Fontes potenciais de viés identificadas | 2 | Viés de seleção, proxies, rótulo/vazamento e desbalanceamento |
| Limitações do modelo e da análise explicadas | 4 | Modelo linear simples, sem atributos protegidos, sem IC, ponto único |
| Medidas de mitigação propostas | 6 | Tabela de medidas com situação |
| Impactos das decisões enviesadas discutidos | 5 | Distributivo, qualidade, discriminação indireta, Art. 20 |
| Conteúdo preparado para README | 8 | Bloco pronto para a consolidação da F4-04 |

## 8. Insumos para o README (F4-04)

Texto sugerido para consolidar na seção de governança:

> **Vieses e mitigação (F4-02).** A análise de vieses avalia o baseline
> `baseline-logreg-820dbb61` na partição de teste (10.000 registros). O dataset
> **não** contém atributos protegidos; o risco é avaliado por proxies
> (`addr_state`, `home_ownership`, `annual_inc`, `emp_length`,
> `application_type`). O modelo já exclui as colunas atribuídas na concessão
> (`grade`, `sub_grade`, `int_rate`, `installment`) por risco de vazamento.
> Há viés de seleção (apenas contratos aprovados e com desfecho definitivo) e a
> faixa de `fico_range_low` < 660 não aparece no teste. As taxas de seleção
> acompanham as taxas base de cada grupo, mas as taxas de erro divergem — por
> exemplo, `home_ownership=RENT` tem FPR de 39,67% contra 25,44% de `MORTGAGE`.
> As medidas propostas incluem métricas de equidade por grupo, reponderação,
> calibração, revisão humana de decisões automatizadas (LGPD, Art. 20) e
> monitoramento de drift por segmento. Detalhes em
> [docs/VIESES.md](docs/VIESES.md).

## 9. Pendências

- Implementar as métricas de equidade em componente próprio (script, config e
  testes) quando priorizado; hoje a análise foi executada de forma pontual.
- Avaliar o viés de reprovação com `rejected_2007_to_2018Q4.csv`, hoje não usado.
- Recalcular as métricas caso o baseline seja retreinado (a análise vale para o
  modelo `baseline-logreg-820dbb61`).
- Revisar limiares e medidas de mitigação com a equipe antes da entrega final.

## Referências

- Lei nº 13.709/2018 (LGPD), arts. 5º, 7º e 20 — não discriminação e decisões
  automatizadas.
- [LGPD.md](LGPD.md) — governança de dados, PII e base legal da decisão de
  crédito.
- [DATASET.md](DATASET.md) — origem, limitações dos dados e risco de vazamento.
- [models/baseline-logreg-820dbb61/metadata.json](../models/baseline-logreg-820dbb61/metadata.json)
  e [evaluation.json](../models/baseline-logreg-820dbb61/evaluation.json) — modelo
  avaliado e métricas de referência.
- [configs/treinamento.json](../configs/treinamento.json) — features, alvo e
  limiar.

## Apêndice — como reproduzir

As métricas foram calculadas carregando o pipeline persistido e a partição de
teste, sem retreinar. Trecho essencial (executar na raiz do projeto):

```python
import json
from pathlib import Path

import numpy as np
import pandas as pd
from monitoramento.modelos.carregamento import carregar_pipeline

diretorio = Path("models/baseline-logreg-820dbb61")
meta = json.loads((diretorio / "metadata.json").read_text(encoding="utf-8"))

pipeline = carregar_pipeline(diretorio / "pipeline.pkl")
base = pd.read_csv(
    "data/reference/lending_club_100k_v1/test.csv", dtype=str, na_filter=False
).replace("", np.nan)

y = pd.to_numeric(base["loan_status"]).astype(int)
proba = pipeline.predict_proba(base[meta["features"]])[:, 1]
predito = (proba >= meta["prediction_threshold"]).astype(int)

# Agrupar por qualquer proxy (ex.: home_ownership) e calcular,
# por grupo: taxa base, taxa de seleção, TPR, FPR e ROC-AUC.
grupo = base["home_ownership"] == "RENT"
print(y[grupo].mean(), predito[grupo].mean())
```

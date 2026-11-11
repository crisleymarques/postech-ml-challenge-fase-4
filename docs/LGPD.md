# Proteção de dados e conformidade com a LGPD — F4-01

Documento da issue F4-01: plano de proteção de dados do pipeline de Credit
Scoring, cobrindo identificação de dados pessoais, tratamento e proteção de PII,
base legal da decisão de crédito, retenção e descarte e a adoção de Privacy by
Design e by Default.

A referência legal é a **Lei nº 13.709/2018 (LGPD)**, complementada, quando
pertinente à decisão de crédito, pela Lei nº 12.414/2011 (Cadastro Positivo). O
projeto é de natureza acadêmica (Tech Challenge) e utiliza um dataset público; as
práticas abaixo evitam, desde a origem, o tratamento de dados pessoais além do
necessário.

## 1. Escopo e contexto

| Item | Definição |
|---|---|
| Sistema | Pipeline local em lote de Credit Scoring (Fase 4 — sustentação e confiabilidade) |
| Origem dos dados | LendingClub Loan Data, via Kaggle (`wordsforthewise/lending-club`) — ver [DATASET.md](DATASET.md) |
| Titulares | Tomadores de crédito presentes na origem; na base derivada são **pseudonimizados** |
| Finalidade | Estimar risco de inadimplência (classe positiva `Charged Off`) e monitorar degradação do modelo (Data e Concept Drift) |
| Natureza | Pesquisa/ensino; dados públicos, sem coleta direta de titulares |
| Ambientes | Execução local; arquivos como interface; sem banco de dados nem serviços externos de rastreamento |

O dataset de origem não possui identificadores diretos utilizáveis (`id`,
`member_id`), mas contém dados financeiros pessoais e quasi-identificadores.
Por isso é tratado como contendo **dados pessoais**, com proteção proporcional.

## 2. Inventário e classificação dos dados

Classificação conforme os conceitos do Art. 5º da LGPD. O detalhamento coluna a
coluna consta em [DATASET.md](DATASET.md#campos-pessoais-ou-sensiveis-lgpd).

| Grupo | Exemplos no recorte | Classificação LGPD | Finalidade | Observação |
|---|---|---|---|---|
| Identificadores diretos | `id`, `member_id` | Dado pessoal | — | **Excluídos** da base derivada; não entram no projeto |
| Texto livre / localização granular | `emp_title`, `title`, `desc`, `zip_code` | Dado pessoal (potencial reidentificação) | — | **Excluídos**; reduzem risco de reidentificação |
| Dados financeiros pessoais | `annual_inc`, `loan_amnt`, `installment`, `dti`, `revol_bal`, `revol_util` | Dado pessoal | Features de risco | Uso interno; expor apenas agregado |
| Score de crédito | `fico_range_low`, `fico_range_high` | Dado pessoal (proteção reforçada) | Feature de risco | Não integra o rol de **dado sensível** do Art. 5º, II, mas recebe proteção reforçada por impacto na decisão |
| Quasi-identificadores | `addr_state`, `issue_d`, `earliest_cr_line`, `emp_length` | Dado pessoal (indireto) | Feature / eixo temporal | Manter agregado; não cruzar com fontes externas |
| Alvo | `loan_status` | Derivado de dado pessoal | Rótulo de treino | `Charged Off` = 1, `Fully Paid` = 0 |

**Nota técnica:** a LGPD define como sensível (Art. 5º, II) dados de origem
racial/étnica, convicção religiosa, opinião política, filiação sindical/religiosa/
filosófica, saúde, biometria e genética. O score de crédito **não** integra esse
rol; é dado pessoal comum. Ainda assim, por orientar decisão sobre o titular,
adota-se proteção reforçada equivalente (ver Seção 4).

## 3. Base legal para a decisão de crédito

A LGPD exige uma das hipóteses do Art. 7º para tratar dados pessoais. Para a
análise e a proteção do crédito aplicam-se, de forma combinada:

| Hipótese | Dispositivo | Aplicação ao projeto |
|---|---|---|
| Proteção do crédito | Art. 7º, X | Tratar dados estritamente necessários à avaliação e à proteção do risco de crédito, nos termos da legislação pertinente (inclusive Lei 12.414/2011) |
| Legítimo interesse | Art. 7º, IX | Interesse legítimo do controlador na classificação de risco, mediante teste de balanceamento |
| Cumprimento de obrigação legal/regulatória | Art. 7º, II | Quando houver exigência regulatória de registro e auditoria de modelos |

**Teste de balanceamento (legítimo interesse).** A finalidade (estimar risco de
inadimplência) é legítima, específica e explícita; os dados usados limitam-se ao
necessário (minimização — Seção 5); o tratamento não se destina a finalidade
incompatível; e há salvaguardas (pseudonimização, ausência de identificadores
diretos, decisão com revisão humana). Conclui-se que o interesse legítimo
prevalece, sem prejuízo dos direitos dos titulares.

**Decisão automatizada (Art. 20).** O modelo produz uma pontuação de risco que
pode embasar decisão de crédito — decisão unicamente automatizada sensível ao
titular. Portanto, o titular tem direito a solicitar **revisão humana** e a
informações sobre os critérios da decisão. No projeto, esse direito é atendido
por: (i) documentação da lógica do modelo e das features; (ii) disponibilização
de métricas de desempenho; (iii) previsão de revisão humana no processo de
decisão (fora do escopo de código deste pipeline).

**Não discriminação.** O uso de features correlacionadas a atributos protegidos
(ex.: localização, ocupação) deve ser avaliado quanto a vieses — tratado na issue
F4-02 (análise de vieses e mitigação).

## 4. Tratamento e proteção de PII

Princípios de proteção adotados, com a medida correspondente:

| Medida | Implementação no projeto |
|---|---|
| Minimização de dados | Base derivada com 30 colunas; identificadores diretos e textos livres não entram (`selecao_dataset.py`) |
| Pseudonimização | `id`/`member_id` descartados; `record_id` técnico (`LC0000001`…) substitui o identificador na amostra |
| Limitação de finalidade | Dados usados apenas para treino, avaliação e detecção de drift — sem finalidade secundária |
| Segurança no armazenamento | Dados brutos e derivados em `data/`, fora do Git; `reports/` (logs, métricas, alertas) ignorados pelo versionamento |
| Ausência de segredos | Credenciais do Kaggle nunca versionadas nem em logs; autenticação apenas interativa |
| Não exposição de PII em logs | Eventos e métricas registram contagens e agregados, **não** linhas, features individuais ou PII |
| Agregação na observabilidade | Dashboard prioriza resultados agregados; predições não revelam rótulos reais |
| Revisão de relatórios HTML | Relatórios do Evidently podem conter valores do dataset; revisão obrigatória antes de compartilhar |
| Integridade e rastreabilidade | Manifestos com `sha256` dos artefatos; `run_id` por execução |

## 5. Privacy by Design e by Default

**Privacy by Design** (Art. 46, §2º) — incorporado desde a concepção do pipeline:

1. **Proativo, não reativo:** exclusão de identificadores na construção da amostra,
   antes de qualquer etapa de modelagem.
2. **Privacidade como padrão:** a configuração padrão da amostragem já produz uma
   base pseudonimizada, sem exigir ação adicional do usuário.
3. **Privacidade embutida no design:** o schema da arquitetura separa dados de
   treino/produção e remove campos técnicos da matriz de features
   (ver [ARQUITETURA.md](ARQUITETURA.md)).
4. **Funcionalidade plena (soma positiva):** a utilidade analítica é preservada
   sem os identificadores diretos.
5. **Segurança de ponta a ponta:** proteção em todas as etapas, do download ao
   artefato final (minimização, pseudonimização, versionamento seletivo).
6. **Visibilidade e transparência:** documentação de origem, tratamento e
   finalidade dos dados neste documento e em [DATASET.md](DATASET.md).
7. **Respeito ao titular:** previsão de revisão humana das decisões automatizadas
   (Art. 20).

**Privacy by Default** (Art. 46, §2º) — padrões que já operam sem escolha do usuário:

- a base derivada **não** contém identificadores diretos nem texto livre;
- os diretórios de dados e relatórios são **ignorados pelo Git** por padrão;
- logs e métricas registram apenas **agregados**, nunca registros individuais;
- o dashboard parte de artefatos **consolidados**, sem exibir PII.

## 6. Plano de retenção e descarte

Prazos propostos, alinhados à finalidade e ao contexto acadêmico, ajustáveis pela
equipe. A contagem parte do término do ciclo de avaliação do projeto.

| Ativo | Local | Retenção proposta | Método de descarte |
|---|---|---|---|
| Dados brutos | `data/reference/raw/` | Até 12 meses | Exclusão dos arquivos e limpeza de temporários |
| Base derivada e partições | `data/reference/`, `data/analysis/` | Até 12 meses | Exclusão dos diretórios de dados |
| Modelos e metadados | `models/` | Até 12 meses (ou enquanto sustentado) | Exclusão das pastas versionadas por hash |
| Logs operacionais | `reports/logs/` | 12 meses | Exclusão dos arquivos JSONL/JSON |
| Métricas, saúde e alertas | `reports/saude/`, `reports/alertas/` | 12 meses | Exclusão dos artefatos |
| Relatórios de análise e HTML | `reports/runs/` | 24 meses (evidência para defesa) | Exclusão dos diretórios de execução |
| Predições simuladas | `reports/runs/<run_id>/predictions.csv` | Até o fim da avaliação | Exclusão do arquivo |
| Caches de teste | `.venv/pytest-tmp` | Efêmero | Remoção automática a cada execução do pytest |

**Critérios de descarte:** eliminação segura (exclusão dos arquivos e remoção de
cópias), com verificação de que caches e diretórios temporários também foram
limpos. O descarte é registrado quando aplicável, conforme a governança da equipe.

## 7. Papéis e responsabilidades

| Papel | Responsabilidade |
|---|---|
| Equipe do projeto | Adotar este plano, revisar relatórios antes de compartilhar e executar o descarte |
| Responsável técnico (issue owner) | Manter a documentação de governança atualizada e zelar pelas medidas de PII |
| Encarregado (DPO) | No contexto acadêmico, papel assumido coletivamente pela equipe; ponto de contato para dúvidas e solicitações de titulares |

## 8. Critérios de aceite (F4-01)

| Critério | Seção | Evidência |
|---|---|---|
| Campos pessoais/sensíveis identificados | 2 | Inventário com classificação por grupo + [DATASET.md](DATASET.md#campos-pessoais-ou-sensiveis-lgpd) |
| Tratamento e proteção de PII descritos | 4 | Tabela de medidas com implementação correspondente |
| Base legal considerada para decisões de crédito | 3 | Art. 7º, X e IX; Art. 20; teste de balanceamento |
| Plano de retenção/descarte documentado | 6 | Tabela com prazos, locais e método de descarte |
| Privacy by Design e by Default considerados | 5 | Princípios mapeados para medidas concretas |

## 9. Pendências

- Confirmar a licença de uso do dataset na página do Kaggle ([DATASET.md](DATASET.md)).
- Revisar prazos de retenção com a equipe antes da entrega final.
- Detalhar a revisão humana das decisões automatizadas no processo de negócio
  (fora do escopo de código deste pipeline).
- Avaliar vieses e medidas de mitigação na issue F4-02 e integrá-los a este plano.

## Referências

- Lei nº 13.709/2018 — Lei Geral de Proteção de Dados (LGPD), arts. 5º, 7º, 18, 20 e 46.
- Lei nº 12.414/2011 — Cadastro Positivo.
- [DATASET.md](DATASET.md) — origem, campos pessoais/sensíveis e limitações dos dados.
- [ARQUITETURA.md](ARQUITETURA.md) — contratos, persistência e observabilidade.

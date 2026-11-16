"""ENT-01: demonstracao sintetica integrada, sem rede ou dados reais.

Execute na raiz do repositorio: python -m monitoramento.integracao.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from monitoramento import lote_invalido, preparacao, treinamento, validacao
from monitoramento.drift.analise import servico as analise
from monitoramento.drift.deteccao import servico as deteccao
from monitoramento.infraestrutura.arquivos import escrever_json_atomico, sha256_arquivo
from monitoramento.observabilidade import alertas, dashboard, logs, metricas_saude
from monitoramento.predicao import servico as predicao
from monitoramento.simulacao.data_drift import servico as simulacao

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
DATASET_ID = "ent01-sintetico"


def ler_json(caminho: Path) -> dict:
    return json.loads(caminho.read_text(encoding="utf-8"))


def exigir(condicao: bool, mensagem: str) -> None:
    if not condicao:
        raise RuntimeError(mensagem)


def gerar_origem(caminho: Path, seed: int) -> None:
    """Gera 6.000 exemplos artificiais que exercitam o contrato existente.

    Distribuicoes e rotulos servem exclusivamente ao teste de integracao;
    nao representam clientes nem estimativas do Lending Club.
    """
    rng = np.random.default_rng(seed)
    n = 6000
    renda = rng.uniform(60000, 80000, n)
    dti = rng.uniform(10, 18, n)
    fico = rng.integers(705, 736, n)
    base = pd.DataFrame(
        {
            "record_id": [f"LC{i + 1:07d}" for i in range(n)],
            "issue_d": np.repeat(
                pd.date_range("2010-01-01", periods=60, freq="MS").strftime("%b-%Y"), 100
            ),
            "loan_amnt": rng.integers(1000, 35000, n),
            "annual_inc": renda,
            "dti": dti,
            "fico_range_low": fico,
            "fico_range_high": fico + 4,
            "revol_util": rng.uniform(35, 45, n),
            "int_rate": rng.uniform(5, 25, n),
            "installment": rng.uniform(100, 1000, n),
            "grade": rng.choice(["A", "B", "C"], n),
            "sub_grade": "B1",
            "term": rng.choice(["36 months", "60 months"], n),
            "emp_length": rng.choice(["2 years", "10+ years"], n),
            "home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE"], n),
            "verification_status": rng.choice(["Verified", "Not Verified"], n),
            "purpose": rng.choice(["credit_card", "debt_consolidation"], n),
            "addr_state": rng.choice(["CA", "NY", "TX"], n),
            "initial_list_status": rng.choice(["w", "f"], n),
            "application_type": rng.choice(["Individual", "Joint App"], n),
            "earliest_cr_line": "Jan-2000",
        }
    )
    for coluna in ("delinq_2yrs", "inq_last_6mths", "pub_rec", "mort_acc", "pub_rec_bankruptcies"):
        base[coluna] = rng.integers(0, 4, n)
    base["open_acc"] = rng.integers(5, 20, n)
    base["total_acc"] = rng.integers(20, 50, n)
    base["revol_bal"] = rng.integers(1000, 30000, n)
    sinal = (dti - 14) / 3 - (renda - 70000) / 10000 + rng.normal(0, 0.5, n)
    base["loan_status"] = np.where(sinal > 0, "Charged Off", "Fully Paid")
    base.to_csv(caminho, index=False, encoding="utf-8")


def artefato(manifesto: dict, tipo: str) -> Path:
    return Path(next(a["path"] for a in manifesto["outputs"] if a["artifact_type"] == tipo))


class Demonstracao:
    """Encadeia servicos reais e mantem evidencias isoladas por execucao."""

    def __init__(self, destino: Path):
        self.destino = destino
        self.runs = destino / "runs"
        self.etapas: list[dict] = []

    def config(self, nome: str, **alteracoes) -> tuple[Path, dict]:
        config = ler_json(CONFIGS / f"{nome}.json")
        config["dataset_id"] = DATASET_ID
        config.update(alteracoes)
        caminho = self.destino / "configs" / f"{nome}-{len(self.etapas)}.json"
        escrever_json_atomico(caminho, config)
        return caminho, config

    def etapa(
        self,
        nome: str,
        servico,
        cfg: tuple[Path, dict],
        entradas=None,
        esperado: int = 0,
        legado: bool = False,
    ) -> dict:
        destino = self.runs / nome
        if legado:
            destino.mkdir(parents=True, exist_ok=False)
        args = (*cfg, *([entradas] if entradas is not None else []), nome, destino)
        codigo = servico(*args)
        return self.registrar(nome, codigo, esperado)

    def registrar(self, nome: str, codigo: int, esperado: int = 0) -> dict:
        self.etapas.append({"run_id": nome, "codigo": codigo, "esperado": esperado})
        exigir(codigo == esperado, f"Etapa {nome}: codigo {codigo}, esperado {esperado}.")
        manifesto = ler_json(self.runs / nome / "manifest.json")
        status = {0: "success", 1: "failed", 2: "rejected"}[esperado]
        exigir(manifesto["status"] == status, f"Estado inesperado em {nome}.")
        return manifesto

    def simular(self, nome: str, origem: Path, cfg: tuple[Path, dict]) -> dict:
        codigo = simulacao.gerar(
            config_path=cfg[0],
            source_path=origem,
            analysis_dir=self.destino / "analysis",
            batch_id=nome,
            run_id=nome,
            output_dir=self.runs / nome,
            seed=None,
        )
        return self.registrar(nome, codigo)

    def executar(self, seed: int) -> dict:
        origem = self.destino / "origem-sintetica.csv"
        gerar_origem(origem, seed)
        cfg = self.config(
            "preparacao",
            preparation_id="preparacao-sintetica",
            source={
                "artifact_type": "origem",
                "path": origem.as_posix(),
                "source_url": None,
                "license": "Dados sinteticos de demonstracao.",
            },
            outputs={
                "reference_dir": (self.destino / "reference").as_posix(),
                "analysis_dir": (self.destino / "analysis").as_posix(),
            },
        )
        prep = self.etapa("preparacao", preparacao.processar, cfg, legado=True)
        particoes = [artefato(prep, p) for p in ("train", "validation", "test", "simulation_base")]
        conjuntos = [set(pd.read_csv(p)["record_id"]) for p in particoes]
        exigir(
            sum(map(len, conjuntos)) == len(set.union(*conjuntos)) == 6000,
            "Particoes nao sao disjuntas ou perderam registros.",
        )
        val_cfg = self.config("validacao")
        aprovado = self.etapa(
            "validacao-reference", validacao.processar, val_cfg, prep, legado=True
        )
        treino_cfg = self.config(
            "treinamento",
            outputs={
                "models_dir": (self.destino / "models").as_posix(),
                "reference_dir": (self.destino / "reference").as_posix(),
            },
        )
        modelo = self.etapa("treinamento", treinamento.processar, treino_cfg, aprovado, legado=True)
        protegidos = [origem, *particoes, *(Path(a["path"]) for a in modelo["outputs"])]
        hashes = {p: sha256_arquivo(p) for p in protegidos}
        sim_cfg = self.config("simulacao")
        lote = self.simular("simulacao", particoes[3], sim_cfg)
        prod_cfg = self.config("validacao", profile="production", artifacts=["features"])
        qualidade = self.etapa(
            "validacao-production", validacao.processar, prod_cfg, lote, legado=True
        )
        pred_cfg = self.config(
            "predicao",
            outputs={
                "analysis_dir": (self.destino / "analysis").as_posix(),
            },
        )
        previsto = self.etapa("predicao", predicao.processar, pred_cfg, [modelo, lote, qualidade])
        self.etapa("deteccao", deteccao.processar, self.config("deteccao"), [modelo, lote])
        self.etapa("analise", analise.processar, self.config("analise_drift"), [modelo, lote])

        # Exerce o gerador F1-04 e seu contrato de referencia sem altera-lo.
        codigo = lote_invalido.gerar(
            config_path=val_cfg[0],
            source_path=particoes[3],
            analysis_dir=self.destino / "invalidos",
            batch_id="invalido-reference",
            run_id="lote-invalido",
            output_dir=self.runs / "lote-invalido",
            total=600,
            seed=42,
        )
        invalido_ref = self.registrar("lote-invalido", codigo)
        rejeitado_ref = self.etapa(
            "rejeicao-reference",
            validacao.processar,
            val_cfg,
            invalido_ref,
            esperado=2,
            legado=True,
        )

        # Novo produtor real com uma violacao de faixa em producao.
        base_invalida = self.destino / "base-invalida-sintetica.csv"
        base = pd.read_csv(particoes[3])
        base.loc[0, "loan_amnt"] = 50000
        base.to_csv(base_invalida, index=False)
        invalido = self.simular("simulacao-invalida", base_invalida, sim_cfg)
        rejeitado = self.etapa(
            "rejeicao-production", validacao.processar, prod_cfg, invalido, esperado=2, legado=True
        )
        bloqueado = self.etapa(
            "predicao-bloqueada",
            predicao.processar,
            pred_cfg,
            [modelo, invalido, rejeitado],
            esperado=1,
        )
        exigir(not bloqueado["outputs"], "Lote reprovado publicou predições.")
        exigir(
            not (artefato(invalido, "features").parent / modelo["model_id"]).exists(),
            "Lote reprovado criou diretorio de inferencia.",
        )
        exigir(
            rejeitado["error_code"] == rejeitado_ref["error_code"] == "QUALIDADE_REPROVADA",
            "Rejeicoes sem codigo de qualidade.",
        )

        resumo_logs = logs.gerar(self.runs, self.destino / "logs")
        metricas_saude.gerar(self.runs, self.destino / "saude", CONFIGS / "metricas_saude.json")
        integrado = alertas.gerar(
            self.destino / "saude/saude.json",
            self.destino / "logs/resumo.json",
            CONFIGS / "alertas.json",
            self.destino / "alertas",
        )
        painel = dashboard.carregar_painel(
            self.runs,
            self.destino / "logs/resumo.json",
            self.destino / "saude/saude.json",
            self.destino / "alertas/alertas.json",
        )
        estados = {"success": 10, "rejected": 2, "failed": 1}
        exigir(resumo_logs["execucoes"]["por_status"] == estados, "Logs inconsistentes.")
        exigir(painel["resumo_execucoes"]["por_status"] == estados, "Painel inconsistente.")
        exigir(all(painel["disponibilidade"].values()), "Fontes indisponiveis no painel.")
        exigir(
            any(f["error_code"] == "QUALIDADE_REPROVADA" for f in resumo_logs["falhas"]),
            "Logs sem rejeicao de qualidade.",
        )
        exigir(
            any(
                a["origem"] == "rejeicao-production" and a["severidade"] == "critical"
                for a in integrado["alertas"]
            ),
            "Rejeicao sem alerta critico.",
        )
        exigir(
            any(a["origem"] == "analise" for a in integrado["alertas"]),
            "Analise estatistica sem alerta.",
        )
        drift = ler_json(self.runs / "deteccao/drift.json")
        estatistica = ler_json(self.runs / "analise/analise.json")
        alteradas = set(sim_cfg[1]["drift"]["variables"])
        exigir(
            alteradas <= set(drift["summary"]["drifted_features"]),
            "Deteccao nao evidenciou as variaveis alteradas.",
        )
        exigir(
            alteradas <= set(estatistica["changed_variables"]["confirmed"]),
            "Analise nao confirmou as variaveis alteradas.",
        )
        exigir((self.runs / "deteccao/drift.html").stat().st_size > 0, "HTML ausente.")
        pred_path = artefato(previsto, "predictions")
        preds = pd.read_csv(pred_path)
        exigir(
            len(preds) == 600 and set(preds["record_id"]) == conjuntos[3],
            "Predicoes nao correspondem a producao reservada.",
        )
        exigir(
            list(preds.columns) == ["record_id", "prediction", "score"]
            and preds["score"].between(0, 1).all(),
            "Predicoes invalidas.",
        )
        exigir(hashes == {p: sha256_arquivo(p) for p in hashes}, "Entrada protegida alterada.")
        return {
            "status": "success",
            "demonstracao_sintetica": True,
            "seed": seed,
            "dataset_id": DATASET_ID,
            "etapas": self.etapas,
            "criterios": {
                "contrato_aprova": True,
                "lote_invalido_bloqueado": True,
                "baseline_prediz": True,
                "drift_estatistico": True,
                "observabilidade_coerente": True,
            },
            "reproducibilidade": {
                "origem_sha256": sha256_arquivo(origem),
                "features_sha256": sha256_arquivo(artefato(lote, "features")),
                "predictions_sha256": sha256_arquivo(pred_path),
                "deteccao": drift["features"],
                "analise": estatistica["features"],
            },
            "fontes_dashboard": painel["fontes"],
            "estados": estados,
            "predicoes": len(preds),
            "variaveis_confirmadas": estatistica["changed_variables"]["confirmed"],
            "limitacoes": [
                "Dados artificiais; nao valida resultados do Lending Club.",
                "Data Drift nao comprova Concept Drift nem queda de desempenho.",
                "Tempos, espaco livre e contagens de alertas de saude podem variar.",
                "Reproducibilidade deve ser comparada entre duas execucoes.",
            ],
        }


def executar(destino: Path, seed: int = 42, comparar_com: Path | None = None) -> dict:
    """Reserva um diretorio novo; falhas deixam diagnostico sem aprovar o fluxo."""
    if seed < 0:
        raise ValueError("Seed deve ser nao negativa.")
    destino.mkdir(parents=True, exist_ok=False)
    demo = Demonstracao(destino)
    try:
        (destino / "configs").mkdir()
        resultado = demo.executar(seed)
        resultado["criterios"]["reproducibilidade_confirmada"] = None
        if comparar_com is not None:
            anterior = ler_json(comparar_com)
            exigir(
                anterior.get("status") == "success"
                and anterior.get("demonstracao_sintetica") is True,
                "Comparacao exige demonstracao sintetica concluida.",
            )
            exigir(
                resultado["reproducibilidade"] == anterior["reproducibilidade"],
                "Resultados deterministas divergiram da execucao anterior.",
            )
            resultado["criterios"]["reproducibilidade_confirmada"] = True
            resultado["comparado_com"] = comparar_com.as_posix()
    except Exception as erro:
        escrever_json_atomico(
            destino / "resultado.json",
            {
                "status": "failed",
                "demonstracao_sintetica": True,
                "etapas": demo.etapas,
                "error_type": type(erro).__name__,
            },
        )
        raise
    escrever_json_atomico(destino / "resultado.json", resultado)
    return resultado


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/runs") / f"ent01-{uuid4().hex[:12]}"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compare-to", type=Path, help="resultado.json de outra demonstracao.")
    args = parser.parse_args()
    try:
        resultado = executar(args.output_dir, args.seed, args.compare_to)
    except Exception as erro:
        parser.exit(
            1, f"Integracao falhou ({type(erro).__name__}). Consulte o diretorio de saida.\n"
        )
    print(f"Demonstracao sintetica: {resultado['status']}. Evidencias: {args.output_dir}")


if __name__ == "__main__":
    main()

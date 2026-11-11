"""Relatorio HTML derivado das mesmas decisoes do JSON e dos alertas."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any


def salvar_html(documento: dict[str, Any], caminho: Path) -> None:
    """Renderiza resultados agregados, sem dados individuais ou scripts externos."""
    resumo = documento["summary"]
    linhas = []
    rotulos = {True: ("drift", "Drift"), False: ("ok", "Sem drift"), None: ("na", "Nao avaliavel")}
    for item in documento["features"]:
        classe, rotulo = rotulos[item["drift_detected"]]
        celulas = [item["feature"], item["method"], item["statistic"], item["threshold"]]
        valores = "".join(f"<td>{escape(str(v)) if v is not None else '—'}</td>" for v in celulas)
        linhas.append(
            f'<tr class="{classe}">{valores}<td>{rotulo}</td>'
            f"<td>{escape(item.get('reason', ''))}</td></tr>"
        )
    dados = json.dumps(documento, ensure_ascii=False, allow_nan=False)
    dados = dados.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    alerta = "Drift de conjunto detectado" if resumo["dataset_drift"] else "Sem alerta de conjunto"
    ref = documento["samples"]["reference"]["row_count"]
    atual = documento["samples"]["current"]["row_count"]
    nao_avaliadas = len(documento["features"]) - resumo["evaluated_columns"]
    html = f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Monitoramento · Data Drift</title><style>
body{{font:16px system-ui,sans-serif;background:#f3f6fb;color:#17283f;margin:0}}
main{{max-width:1200px;margin:auto;padding:40px 24px}}h1{{font-size:36px;margin:12px 0}}
.kicker{{color:#425f80;text-transform:uppercase;letter-spacing:.12em;font-size:12px}}
.cards{{display:flex;gap:16px;flex-wrap:wrap;margin:28px 0}}
.card{{background:white;border:1px solid #d6dfeb;border-radius:12px;padding:20px;flex:1}}
.card strong{{display:block;font-size:30px}}.card span{{color:#425f80}}
.table{{overflow-x:auto;background:white;border:1px solid #d6dfeb;border-radius:12px}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:14px;text-align:left;
border-bottom:1px solid #e5eaf1;vertical-align:top}}th{{background:#e8eef6}}
tr.drift td:first-child{{border-left:4px solid #b3431b}}tr.na{{background:#faf3df}}
p,footer{{line-height:1.6}}footer{{font-size:13px;color:#425f80;margin-top:22px}}
</style></head><body><main>
<div class="kicker">Confiabilidade do modelo ·
Evidently {escape(documento["tool"]["version"])}</div>
<h1>Deteccao de Data Drift</h1><p>{escape(alerta)} · Execucao {escape(documento["run_id"])}<br>
Cenario {escape(str(documento["scenario_id"]))} · Lote {escape(str(documento["batch_id"]))}</p>
<section class="cards">
<div class="card"><strong>{resumo["drifted_columns"]} / {resumo["evaluated_columns"]}</strong>
<span>Features com drift / avaliadas</span></div>
<div class="card"><strong>{resumo["drift_share"]:.2%}</strong><span>Proporcao com drift</span></div>
<div class="card"><strong>{nao_avaliadas}</strong><span>Features nao avaliaveis</span></div>
</section><p>Referencia: {ref:,} linhas · Producao: {atual:,} linhas.
Alerta quando mais de 50% das features avaliadas apresentam drift.</p>
<div class="table"><table><thead><tr><th>Feature</th><th>Metodo estatistico</th>
<th>Estatistica</th><th>Limiar</th><th>Decisao</th><th>Motivo</th></tr></thead>
<tbody>{"".join(linhas)}</tbody></table></div>
<footer>P-valor: drift abaixo do limiar. Distancia: drift acima do limiar.
Valores iguais ao limiar nao indicam drift. Features nao avaliaveis ficam fora do denominador.
JSON, HTML, metricas e alertas usam as mesmas decisoes. As estatisticas sao calculadas pelo
Evidently; os limiares e a apresentacao seguem a configuracao desta execucao.</footer>
<script type="application/json" id="drift-data">{dados}</script>
</main></body></html>"""
    caminho.write_text(html, encoding="utf-8")

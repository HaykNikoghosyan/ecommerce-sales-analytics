"""HTML report, CSV and JSON exports."""

from __future__ import annotations

import json
from datetime import datetime

from .config import Config, Report, log


HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sales Analytics Report</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0;
        background: #f6f7f9; color: #1d2330; }}
 main {{ max-width: 1100px; margin: auto; padding: 24px 16px; }}
 h1 {{ margin-bottom: 4px; }} .sub {{ color: #667; margin-top: 0; }}
 .kpis {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
         gap: 12px; margin: 20px 0; }}
 .kpi {{ background: #fff; border-radius: 10px; padding: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
 .kpi b {{ display: block; font-size: 20px; margin-top: 4px; }}
 .kpi span {{ font-size: 12px; color: #667; }}
 section {{ background: #fff; border-radius: 10px; padding: 16px 20px; margin: 16px 0;
           box-shadow: 0 1px 3px rgba(0,0,0,.08); overflow-x: auto; }}
 table {{ border-collapse: collapse; font-size: 13px; }}
 th, td {{ padding: 5px 10px; border-bottom: 1px solid #e5e7eb; text-align: right; }}
 th:first-child, td:first-child {{ text-align: left; }}
 img {{ max-width: 100%; border-radius: 6px; }}
 .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
           gap: 16px; }}
 li {{ margin: 6px 0; }} pre {{ background: #f3f4f6; padding: 10px; border-radius: 6px; }}
</style></head><body><main>
<h1>Sales Analytics Report</h1>
<p class="sub">Generated {generated}</p>
<div class="kpis">{kpis}</div>
<section><h2>Key insights</h2><ul>{insights}</ul></section>
<section><h2>Data cleaning</h2><pre>{cleaning}</pre></section>
<section><h2>Statistical tests</h2><pre>{tests}</pre></section>
<section><h2>Charts</h2><div class="charts">{charts}</div></section>
{tables}
</main></body></html>"""


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:,.2f}"
    return f"{v:,}" if isinstance(v, int) else str(v)


def write_report(report: Report, cfg: Config, frames: dict):
    out = cfg.out_dir
    log.info("Writing report to %s/", out)
    for name, frame in frames.items():
        if frame is not None:
            frame.to_csv(out / f"{name}.csv")

    kpis = "".join(f'<div class="kpi"><span>{k}</span><b>{_fmt(v)}</b></div>'
                   for k, v in report.kpis.items())
    insights = "".join(f"<li>{i}</li>" for i in report.insights)
    charts = "".join(f'<figure><img src="{p}" alt="{t}"></figure>'
                     for t, p in report.charts)
    tables = "".join(
        f"<section><h2>{name}</h2>{tbl.to_html(float_format=lambda x: f'{x:,.2f}')}"
        f"</section>" for name, tbl in report.tables.items())
    html = HTML_TEMPLATE.format(
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
        kpis=kpis, insights=insights, charts=charts, tables=tables,
        cleaning=json.dumps(report.cleaning, indent=2),
        tests=json.dumps(report.tests, indent=2, default=str))
    (out / "report.html").write_text(html, encoding="utf-8")

    summary = {"kpis": report.kpis, "cleaning": report.cleaning,
               "tests": report.tests, "insights": report.insights}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

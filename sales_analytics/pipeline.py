"""Pipeline orchestration and command-line interface."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from .cleaning import DataCleaner, add_features
from .cohort import cohort_analysis
from .config import Config, Report, log
from .database import DatabaseExporter, run_query
from .eda import Explorer
from .forecast import run_forecast
from .generator import DataGenerator
from .plotting import Plotter
from .reporting import write_report
from .segmentation import CustomerClustering, RFMAnalyzer
from .stats import statistical_tests

REQUIRED_COLUMNS = {"order_id", "order_date", "customer_id", "category",
                    "unit_price", "quantity", "discount", "channel",
                    "region", "age", "returned", "rating"}
OPTIONAL_DEFAULTS = {"gender": "Unknown", "payment": "Unknown"}


def load_data(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Return (transactions, customer master or None)."""
    if cfg.input_csv:
        log.info("Loading %s", cfg.input_csv)
        df = pd.read_csv(cfg.input_csv)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            sys.exit(f"Input CSV is missing columns: {sorted(missing)}")
        for col, default in OPTIONAL_DEFAULTS.items():
            if col not in df.columns:
                df[col] = default
        return df, None
    gen = DataGenerator(cfg)
    df = gen.generate()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.out_dir / "raw_transactions.csv", index=False)
    return df, gen.customers_


def run_pipeline(cfg: Config) -> Report:
    t0 = time.perf_counter()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    report = Report()
    plot = Plotter(cfg.out_dir, report)

    raw, customers = load_data(cfg)
    clean = DataCleaner(report).clean(raw)
    df = add_features(clean)

    monthly = Explorer(df, report, plot).run()
    rfm = RFMAnalyzer(df, report, plot).run()
    rfm = CustomerClustering(rfm, cfg, report, plot).run()
    retention = cohort_analysis(df, report, plot)
    statistical_tests(df, report, plot)
    forecast = run_forecast(monthly, cfg, report, plot)

    DatabaseExporter(cfg, report).export(
        df, rfm, monthly, retention, forecast, customers)

    write_report(report, cfg, {
        "clean_transactions": df.set_index("order_id"),
        "customer_segments": rfm,
        "monthly_summary": monthly,
        "cohort_retention": retention,
        "revenue_forecast": forecast,
    })
    log.info("Done in %.1fs. Report: %s | Database: %s",
             time.perf_counter() - t0, cfg.out_dir / "report.html", cfg.database)
    return report


def parse_args(argv=None) -> tuple[Config, argparse.Namespace]:
    p = argparse.ArgumentParser(
        prog="sales_analytics",
        description="E-commerce sales analytics pipeline with SQLite export")
    p.add_argument("--rows", type=int, default=20_000, help="synthetic rows")
    p.add_argument("--customers", type=int, default=2_500, help="synthetic customers")
    p.add_argument("--clusters", type=int, default=4, help="K-Means clusters")
    p.add_argument("--horizon", type=int, default=6, help="forecast months")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--input", type=Path, help="analyse your own CSV instead")
    p.add_argument("--out", type=Path, default=Path("output"), help="output folder")
    p.add_argument("--db", type=Path, help="SQLite path (default: <out>/sales.db)")
    p.add_argument("--sql", help="run a SQL query on an existing database and exit")
    a = p.parse_args(argv)
    cfg = Config(rows=a.rows, customers=a.customers, n_clusters=a.clusters,
                 forecast_months=a.horizon, seed=a.seed, input_csv=a.input,
                 out_dir=a.out, db_path=a.db)
    return cfg, a


def main(argv=None):
    cfg, args = parse_args(argv)
    if args.sql:
        with pd.option_context("display.width", 140, "display.max_columns", 20):
            print(run_query(cfg.database, args.sql).to_string(index=False))
        return
    report = run_pipeline(cfg)
    print("\nKey insights:")
    for line in report.insights:
        print("  -", line)

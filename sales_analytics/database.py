"""
SQLite export layer.

Writes every result of the pipeline into one normalized SQLite database so it
can be explored with SQL, DB Browser for SQLite, Power BI, Tableau, etc.

Tables
    customers          one row per customer (+ RFM scores, segment, cluster)
    categories         product category dimension
    orders             cleaned transactions (fact table)
    monthly_summary    revenue / orders per month
    cohort_retention   long-format cohort matrix
    forecast           revenue forecast with 95% interval
    kpis               headline metrics
    test_results       statistical test outputs
    run_metadata       when / how the database was produced

Views
    v_category_performance, v_monthly_channel, v_top_customers,
    v_segment_summary, v_region_category
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import Config, Report, log

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE categories (
    category_id   INTEGER PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE
);

CREATE TABLE customers (
    customer_id   TEXT PRIMARY KEY,
    age           INTEGER,
    gender        TEXT,
    region        TEXT,
    signup_date   TEXT,
    first_order   TEXT,
    last_order    TEXT,
    recency_days  INTEGER,
    orders        INTEGER,
    revenue       REAL,
    r_score       INTEGER,
    f_score       INTEGER,
    m_score       INTEGER,
    rfm_code      TEXT,
    segment       TEXT,
    cluster       TEXT
);

CREATE TABLE orders (
    order_id      TEXT PRIMARY KEY,
    order_date    TEXT NOT NULL,
    month         TEXT NOT NULL,
    customer_id   TEXT NOT NULL REFERENCES customers(customer_id),
    category_id   INTEGER NOT NULL REFERENCES categories(category_id),
    unit_price    REAL NOT NULL CHECK (unit_price > 0),
    quantity      INTEGER NOT NULL CHECK (quantity > 0),
    discount      REAL NOT NULL CHECK (discount BETWEEN 0 AND 1),
    gross         REAL NOT NULL,
    revenue       REAL NOT NULL,
    channel       TEXT,
    payment       TEXT,
    rating        INTEGER CHECK (rating IS NULL OR rating BETWEEN 1 AND 5),
    returned      INTEGER NOT NULL CHECK (returned IN (0, 1)),
    region        TEXT,
    weekday       TEXT,
    hour          INTEGER
);

CREATE TABLE monthly_summary (
    month         TEXT PRIMARY KEY,
    revenue       REAL,
    orders        INTEGER
);

CREATE TABLE cohort_retention (
    cohort        TEXT,
    period        INTEGER,
    retention_pct REAL,
    PRIMARY KEY (cohort, period)
);

CREATE TABLE forecast (
    month         TEXT PRIMARY KEY,
    forecast      REAL,
    lower_95      REAL,
    upper_95      REAL
);

CREATE TABLE kpis (
    name          TEXT PRIMARY KEY,
    value         REAL
);

CREATE TABLE test_results (
    test          TEXT PRIMARY KEY,
    result_json   TEXT
);

CREATE TABLE run_metadata (
    key           TEXT PRIMARY KEY,
    value         TEXT
);

CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_month    ON orders(month);
CREATE INDEX idx_orders_category ON orders(category_id);
CREATE INDEX idx_customers_seg   ON customers(segment);

CREATE VIEW v_category_performance AS
SELECT c.name                               AS category,
       COUNT(*)                             AS orders,
       ROUND(SUM(o.revenue), 2)             AS revenue,
       ROUND(AVG(o.unit_price), 2)          AS avg_price,
       ROUND(100.0 * AVG(o.returned), 2)    AS return_rate_pct,
       ROUND(AVG(o.rating), 2)              AS avg_rating
FROM orders o JOIN categories c USING (category_id)
GROUP BY c.name
ORDER BY revenue DESC;

CREATE VIEW v_monthly_channel AS
SELECT month, channel, ROUND(SUM(revenue), 2) AS revenue, COUNT(*) AS orders
FROM orders
GROUP BY month, channel
ORDER BY month, channel;

CREATE VIEW v_top_customers AS
SELECT customer_id, region, segment, cluster, orders,
       ROUND(revenue, 2) AS revenue, recency_days
FROM customers
ORDER BY revenue DESC
LIMIT 100;

CREATE VIEW v_segment_summary AS
SELECT segment,
       COUNT(*)                     AS customers,
       ROUND(AVG(recency_days), 1)  AS avg_recency,
       ROUND(AVG(orders), 2)        AS avg_orders,
       ROUND(SUM(revenue), 2)       AS revenue
FROM customers
GROUP BY segment
ORDER BY revenue DESC;

CREATE VIEW v_region_category AS
SELECT o.region, c.name AS category, ROUND(SUM(o.revenue), 2) AS revenue
FROM orders o JOIN categories c USING (category_id)
GROUP BY o.region, c.name;
"""


class DatabaseExporter:
    def __init__(self, cfg: Config, report: Report):
        self.path: Path = cfg.database
        self.cfg = cfg
        self.report = report

    # ---------------------------------------------------------------- build
    def _customers_table(self, df: pd.DataFrame, rfm: pd.DataFrame,
                         customers: pd.DataFrame | None) -> pd.DataFrame:
        agg = df.groupby("customer_id").agg(
            first_order=("order_date", "min"), last_order=("order_date", "max"),
            age=("age", "first"), gender=("gender", "first"),
            region=("region", "first"))
        if customers is not None:  # synthetic run: use the real customer master
            master = customers.set_index("customer_id")[["signup_date"]]
            agg = agg.join(master, how="left")
        else:
            agg["signup_date"] = agg["first_order"]
        out = agg.join(rfm[["recency", "frequency", "monetary", "R", "F", "M",
                            "RFM", "segment", "cluster_label"]], how="left")
        out = out.rename(columns={
            "recency": "recency_days", "frequency": "orders", "monetary": "revenue",
            "R": "r_score", "F": "f_score", "M": "m_score", "RFM": "rfm_code",
            "cluster_label": "cluster"})
        for col in ("first_order", "last_order", "signup_date"):
            out[col] = pd.to_datetime(out[col]).dt.strftime("%Y-%m-%d")
        return out.reset_index()

    def _orders_table(self, df: pd.DataFrame, cat_ids: dict) -> pd.DataFrame:
        o = df[["order_id", "order_date", "month", "customer_id", "category",
                "unit_price", "quantity", "discount", "gross", "revenue", "channel",
                "payment", "rating", "returned", "region", "weekday", "hour"]].copy()
        o = o.drop_duplicates("order_id")
        o["order_date"] = o["order_date"].dt.strftime("%Y-%m-%d %H:%M:%S")
        o["month"] = o["month"].astype(str)
        o["category_id"] = o.pop("category").map(cat_ids)
        o["returned"] = o["returned"].astype(int)
        o["rating"] = o["rating"].astype("Int64")
        return o

    def export(self, df: pd.DataFrame, rfm: pd.DataFrame, monthly: pd.DataFrame,
               retention: pd.DataFrame, forecast: pd.DataFrame | None,
               customers: pd.DataFrame | None = None) -> Path:
        log.info("Writing SQLite database to %s", self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()

        categories = sorted(df["category"].unique())
        cat_ids = {name: i + 1 for i, name in enumerate(categories)}

        with closing(sqlite3.connect(self.path)) as con:
            con.executescript(SCHEMA)
            con.executemany("INSERT INTO categories VALUES (?, ?)",
                            [(i, n) for n, i in cat_ids.items()])

            self._customers_table(df, rfm, customers).to_sql(
                "customers", con, if_exists="append", index=False)
            self._orders_table(df, cat_ids).to_sql(
                "orders", con, if_exists="append", index=False, chunksize=5000)

            m = monthly.copy()
            m.index = m.index.strftime("%Y-%m")
            m.rename_axis("month").reset_index().to_sql(
                "monthly_summary", con, if_exists="append", index=False)

            r = retention.copy()
            r.index = r.index.astype(str)
            long = (r.rename_axis(index="cohort", columns="period").stack()
                     .rename("retention_pct").reset_index())
            long.to_sql("cohort_retention", con, if_exists="append", index=False)

            if forecast is not None:
                f = forecast.copy()
                f.index = f.index.strftime("%Y-%m")
                f.rename_axis("month").reset_index().to_sql(
                    "forecast", con, if_exists="append", index=False)

            con.executemany("INSERT INTO kpis VALUES (?, ?)",
                            list(self.report.kpis.items()))
            con.executemany("INSERT INTO test_results VALUES (?, ?)",
                            [(k, json.dumps(v, default=str))
                             for k, v in self.report.tests.items()])
            meta = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "source": str(self.cfg.input_csv or "synthetic"),
                "seed": str(self.cfg.seed),
                "rows": str(len(df)),
                "date_from": df["order_date"].min().strftime("%Y-%m-%d"),
                "date_to": df["order_date"].max().strftime("%Y-%m-%d"),
            }
            con.executemany("INSERT INTO run_metadata VALUES (?, ?)", meta.items())
            con.commit()
            con.execute("VACUUM")

        self.verify()
        return self.path

    # --------------------------------------------------------------- verify
    def verify(self):
        """Sanity checks: row counts, FK integrity and revenue reconciliation."""
        with closing(sqlite3.connect(self.path)) as con:
            counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                      for t in ("customers", "orders", "categories",
                                "monthly_summary", "cohort_retention", "forecast")}
            fk_errors = con.execute("PRAGMA foreign_key_check").fetchall()
            db_rev = con.execute("SELECT SUM(revenue) FROM orders").fetchone()[0]
        kpi_rev = self.report.kpis.get("Total revenue", db_rev)
        ok = not fk_errors and abs(db_rev - kpi_rev) < 1.0
        self.report.tests["Database checks"] = {
            "row_counts": counts, "foreign_key_errors": len(fk_errors),
            "revenue_matches_kpi": bool(abs(db_rev - kpi_rev) < 1.0)}
        log.info("DB tables: %s", ", ".join(f"{k}={v}" for k, v in counts.items()))
        if not ok:
            log.warning("Database verification found issues: %s",
                        self.report.tests["Database checks"])


def run_query(db: Path, sql: str, limit: int = 50) -> pd.DataFrame:
    """Run an ad-hoc SQL query against the exported database."""
    if not db.exists():
        raise FileNotFoundError(f"{db} not found - run the pipeline first.")
    with closing(sqlite3.connect(db)) as con:
        return pd.read_sql_query(sql, con).head(limit)

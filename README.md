# E-Commerce Sales Analytics

An end-to-end data analytics project in Python. It takes raw transactions and
turns them into charts, an HTML report, CSV files and a SQLite database.

Built with pandas, NumPy and matplotlib only. K-Means, the statistical tests
and the forecasting model are implemented from scratch.

## What it does

| Step | Module | Output |
|------|--------|--------|
| Generate realistic data with dirty records | `generator.py` | `raw_transactions.csv` |
| Clean: duplicates, missing values, outliers, bad values | `cleaning.py` | `clean_transactions.csv` |
| Exploratory analysis: KPIs, trends, breakdowns, Pareto | `eda.py` | charts, tables |
| RFM segmentation | `segmentation.py` | `customer_segments.csv` |
| K-Means clustering (k-means++, elbow, silhouette) | `segmentation.py` | cluster chart |
| Cohort retention | `cohort.py` | retention heatmap |
| Welch t-test, chi-square, correlations | `stats.py` | test results |
| Revenue forecast with back-test | `forecast.py` | `revenue_forecast.csv` |
| SQLite export with schema, indexes and views | `database.py` | `sales.db` |
| HTML / JSON report | `reporting.py` | `report.html`, `summary.json` |

## Project structure

```
sales_project/
├── sales_analytics/
│   ├── __init__.py
│   ├── __main__.py      # python -m sales_analytics
│   ├── config.py        # constants, Config, Report
│   ├── generator.py     # synthetic data
│   ├── cleaning.py      # cleaning + feature engineering
│   ├── plotting.py      # chart helpers
│   ├── eda.py           # exploratory analysis
│   ├── segmentation.py  # RFM + K-Means
│   ├── cohort.py        # retention
│   ├── stats.py         # statistical tests
│   ├── forecast.py      # forecasting
│   ├── database.py      # SQLite export + queries
│   ├── reporting.py     # HTML / CSV / JSON
│   └── pipeline.py      # orchestration + CLI
├── queries.sql          # 10 example SQL queries
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Default run: 20,000 synthetic orders
python -m sales_analytics

# More data, 5 clusters, 12-month forecast
python -m sales_analytics --rows 50000 --clusters 5 --horizon 12

# Analyse your own CSV
python -m sales_analytics --input my_orders.csv

# Custom output folder and database path
python -m sales_analytics --out results --db data/shop.db

# Query the database
python -m sales_analytics --sql "SELECT * FROM v_category_performance"
```

Your own CSV needs these columns: `order_id, order_date, customer_id, category,
unit_price, quantity, discount, channel, region, age, returned, rating`.
`gender` and `payment` are optional.

## The database (`output/sales.db`)

**Tables**

| Table | Contents |
|-------|----------|
| `orders` | cleaned transactions (fact table) |
| `customers` | one row per customer with RFM scores, segment, cluster |
| `categories` | category dimension |
| `monthly_summary` | revenue and orders per month |
| `cohort_retention` | cohort, period, retention % |
| `forecast` | forecast with 95% interval |
| `kpis` | headline metrics |
| `test_results` | statistical test results as JSON |
| `run_metadata` | creation time, source, date range |

**Views:** `v_category_performance`, `v_monthly_channel`, `v_top_customers`,
`v_segment_summary`, `v_region_category`.

The schema uses primary and foreign keys, CHECK constraints and indexes. After
export the pipeline checks foreign-key integrity and that total revenue in the
database matches the computed KPI.

Open it with [DB Browser for SQLite](https://sqlitebrowser.org/), DBeaver,
Power BI or Tableau, or see `queries.sql` for examples using window functions,
subqueries, CASE and UNION.

## Sample results (default run)

- Total revenue ~4.7M, 19.9k orders, 2.2k customers, 22.9% YoY growth
- Electronics brings ~57% of revenue and has the highest return rate
- Top 20% of customers generate ~74% of revenue
- Forecast back-test MAPE ~7% vs ~18% for a seasonal-naive baseline

## Methods

- **RFM:** quintile scores for recency, frequency and monetary value, mapped to
  rule-based segments (Champions, Loyal, At Risk, ...).
- **K-Means:** log-transformed, standardised RFM features, k-means++ init,
  best of several restarts; k chosen by elbow and silhouette.
- **Statistics:** Welch's t-test and chi-square test of independence, with
  p-values computed from the incomplete beta and gamma functions.
- **Forecast:** least-squares fit of log revenue on a linear trend plus
  monthly seasonal dummies; evaluated on a 6-month holdout.

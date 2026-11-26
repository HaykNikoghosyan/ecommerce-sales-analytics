"""Constants, configuration and the shared Report container."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("analytics")

CATEGORIES = {
    "Electronics": (80, 900),
    "Clothing": (10, 150),
    "Home & Garden": (15, 400),
    "Books": (5, 60),
    "Sports": (20, 300),
    "Beauty": (8, 120),
    "Toys": (10, 150),
}
REGIONS = ["Yerevan", "Gyumri", "Vanadzor", "Vagharshapat", "Abovyan", "Kapan"]
REGION_WEIGHTS = [0.45, 0.15, 0.12, 0.10, 0.10, 0.08]
CHANNELS = ["Website", "Mobile App", "Marketplace"]
CHANNEL_WEIGHTS = [0.45, 0.40, 0.15]
PAYMENT_METHODS = ["Card", "Cash on Delivery", "Bank Transfer", "Wallet"]

PALETTE = ["#2F6FEB", "#E8710A", "#1A9E6B", "#C93C7A", "#7A5AF8", "#8C8C8C", "#D4A20B"]


@dataclass
class Config:
    rows: int = 20_000
    customers: int = 2_500
    start: str = "2024-01-01"
    end: str = "2025-12-31"
    seed: int = 42
    n_clusters: int = 4
    forecast_months: int = 6
    out_dir: Path = Path("output")
    input_csv: Path | None = None
    db_path: Path | None = None  # defaults to <out_dir>/sales.db

    @property
    def database(self) -> Path:
        return self.db_path or self.out_dir / "sales.db"


@dataclass
class Report:
    """Collects every result so it can be written to HTML / JSON at the end."""
    kpis: dict = field(default_factory=dict)
    cleaning: dict = field(default_factory=dict)
    tables: dict = field(default_factory=dict)
    tests: dict = field(default_factory=dict)
    charts: list = field(default_factory=list)
    insights: list = field(default_factory=list)

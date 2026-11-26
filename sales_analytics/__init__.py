"""E-commerce sales analytics package."""

from .config import Config, Report
from .pipeline import main, run_pipeline

__all__ = ["Config", "Report", "main", "run_pipeline"]
__version__ = "1.0.0"

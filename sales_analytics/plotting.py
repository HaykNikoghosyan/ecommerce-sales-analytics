"""Matplotlib helpers that save charts and register them in the report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

from .config import PALETTE, Report


class Plotter:
    def __init__(self, out_dir: Path, report: Report):
        self.dir = out_dir / "charts"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.report = report
        plt.rcParams.update({
            "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "grid.alpha": 0.25, "font.size": 10,
            "axes.titleweight": "bold", "axes.titlesize": 12,
        })

    def save(self, fig, name: str, title: str):
        path = self.dir / f"{name}.png"
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        self.report.charts.append((title, f"charts/{name}.png"))

    def bar(self, series: pd.Series, name, title, xlabel="", horizontal=False, fmt="{:,.0f}"):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(series))]
        if horizontal:
            s = series.sort_values()
            ax.barh(s.index.astype(str), s.values, color=PALETTE[0])
            for i, v in enumerate(s.values):
                ax.text(v, i, " " + fmt.format(v), va="center", fontsize=8)
        else:
            ax.bar(series.index.astype(str), series.values, color=colors)
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        self.save(fig, name, title)

    def line(self, frame: pd.DataFrame, name, title, ylabel=""):
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for i, col in enumerate(frame.columns):
            ax.plot(frame.index, frame[col], marker="o", ms=3, lw=1.8,
                    color=PALETTE[i % len(PALETTE)], label=col)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if len(frame.columns) > 1:
            ax.legend(frameon=False)
        self.save(fig, name, title)

    def heatmap(self, matrix: pd.DataFrame, name, title, fmt="{:.0f}", cmap="Blues"):
        fig, ax = plt.subplots(figsize=(max(6, 0.55 * matrix.shape[1] + 2),
                                        max(4, 0.35 * matrix.shape[0] + 1.5)))
        data = matrix.values.astype(float)
        im = ax.imshow(data, cmap=cmap, aspect="auto")
        ax.set_xticks(range(matrix.shape[1]), [str(c) for c in matrix.columns],
                      rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(matrix.shape[0]), [str(r) for r in matrix.index], fontsize=8)
        ax.grid(False)
        vmax = np.nanmax(data) if np.isfinite(data).any() else 1
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if not np.isnan(data[i, j]):
                    ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center",
                            fontsize=6.5, color="white" if data[i, j] > vmax * 0.6 else "black")
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title)
        self.save(fig, name, title)

"""RFM segmentation and K-Means clustering (implemented from scratch)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import PALETTE, Config, Report, log
from .plotting import Plotter


class RFMAnalyzer:
    SEGMENTS = [
        ("Champions", lambda r, f, m: r >= 4 and f >= 4),
        ("Loyal", lambda r, f, m: r >= 3 and f >= 4),
        ("Big Spenders", lambda r, f, m: m >= 4 and r >= 3),
        ("Promising", lambda r, f, m: r >= 4 and f <= 2),
        ("Need Attention", lambda r, f, m: r == 3),
        ("At Risk", lambda r, f, m: r <= 2 and f >= 3),
        ("Hibernating", lambda r, f, m: r <= 2),
    ]

    def __init__(self, df: pd.DataFrame, report: Report, plot: Plotter):
        self.df, self.report, self.plot = df, report, plot

    @staticmethod
    def _score(s: pd.Series, reverse=False) -> pd.Series:
        ranked = s.rank(method="first")
        scores = pd.qcut(ranked, 5, labels=[1, 2, 3, 4, 5]).astype(int)
        return 6 - scores if reverse else scores

    def _segment(self, row) -> str:
        for name, rule in self.SEGMENTS:
            if rule(row.R, row.F, row.M):
                return name
        return "Others"

    def run(self) -> pd.DataFrame:
        log.info("Computing RFM segments...")
        snapshot = self.df["order_date"].max() + pd.Timedelta(days=1)
        rfm = self.df.groupby("customer_id").agg(
            recency=("order_date", lambda d: (snapshot - d.max()).days),
            frequency=("order_id", "nunique"),
            monetary=("revenue", "sum"),
        )
        rfm["R"] = self._score(rfm["recency"], reverse=True)
        rfm["F"] = self._score(rfm["frequency"])
        rfm["M"] = self._score(rfm["monetary"])
        rfm["RFM"] = rfm["R"].astype(str) + rfm["F"].astype(str) + rfm["M"].astype(str)
        rfm["segment"] = rfm.apply(self._segment, axis=1)

        summary = rfm.groupby("segment").agg(
            customers=("R", "size"), avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"), revenue=("monetary", "sum"))
        summary["revenue_%"] = 100 * summary["revenue"] / summary["revenue"].sum()
        summary = summary.sort_values("revenue", ascending=False)
        self.report.tables["RFM segments"] = summary.round(2)
        self.plot.bar(summary["customers"], "rfm_segments", "Customers per RFM segment")

        if "At Risk" in summary.index:
            ar = summary.loc["At Risk"]
            self.report.insights.append(
                f"{int(ar.customers)} 'At Risk' customers once spent "
                f"{ar.revenue:,.0f} - a win-back campaign target.")
        return rfm


class KMeans:
    """Minimal K-Means with k-means++ initialisation and multiple restarts."""

    def __init__(self, k: int, n_init: int = 8, max_iter: int = 300,
                 tol: float = 1e-6, seed: int = 0):
        self.k, self.n_init, self.max_iter, self.tol = k, n_init, max_iter, tol
        self.rng = np.random.default_rng(seed)
        self.centers_ = None
        self.labels_ = None
        self.inertia_ = np.inf

    def _init_centers(self, X: np.ndarray) -> np.ndarray:
        centers = [X[self.rng.integers(len(X))]]
        for _ in range(1, self.k):
            d2 = np.min(((X[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
            centers.append(X[self.rng.choice(len(X), p=d2 / d2.sum())])
        return np.array(centers)

    def _single_run(self, X):
        centers = self._init_centers(X)
        for _ in range(self.max_iter):
            dist = ((X[:, None, :] - centers[None]) ** 2).sum(-1)
            labels = dist.argmin(1)
            new = np.array([X[labels == j].mean(0) if np.any(labels == j) else centers[j]
                            for j in range(self.k)])
            shift = np.abs(new - centers).max()
            centers = new
            if shift < self.tol:
                break
        inertia = ((X - centers[labels]) ** 2).sum()
        return centers, labels, inertia

    def fit(self, X: np.ndarray) -> "KMeans":
        for _ in range(self.n_init):
            c, l, inert = self._single_run(X)
            if inert < self.inertia_:
                self.centers_, self.labels_, self.inertia_ = c, l, inert
        return self


def silhouette(X: np.ndarray, labels: np.ndarray, sample: int = 1500, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(sample, len(X)), replace=False)
    Xs, ls = X[idx], labels[idx]
    D = np.sqrt(((Xs[:, None] - Xs[None]) ** 2).sum(-1))
    scores = []
    for i in range(len(Xs)):
        same = ls == ls[i]
        if same.sum() <= 1:
            continue
        a = D[i, same].sum() / (same.sum() - 1)
        b = min(D[i, ls == c].mean() for c in np.unique(ls) if c != ls[i])
        scores.append((b - a) / max(a, b))
    return float(np.mean(scores))


class CustomerClustering:
    def __init__(self, rfm: pd.DataFrame, cfg: Config, report: Report, plot: Plotter):
        self.rfm, self.cfg, self.report, self.plot = rfm, cfg, report, plot

    def run(self) -> pd.DataFrame:
        log.info("Clustering customers with K-Means...")
        feats = np.log1p(self.rfm[["recency", "frequency", "monetary"]].values)
        X = (feats - feats.mean(0)) / feats.std(0)

        ks, inertias, sils = range(2, 9), [], []
        for k in ks:
            km = KMeans(k, n_init=4, seed=self.cfg.seed).fit(X)
            inertias.append(km.inertia_)
            sils.append(silhouette(X, km.labels_, seed=self.cfg.seed))
        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        ax1.plot(list(ks), inertias, "o-", color=PALETTE[0], label="Inertia")
        ax1.set_xlabel("k")
        ax1.set_ylabel("Inertia")
        ax2 = ax1.twinx()
        ax2.plot(list(ks), sils, "s--", color=PALETTE[1], label="Silhouette")
        ax2.set_ylabel("Silhouette")
        ax1.set_title("Choosing k: elbow and silhouette")
        self.plot.save(fig, "kmeans_elbow", "Choosing k: elbow and silhouette")

        km = KMeans(self.cfg.n_clusters, seed=self.cfg.seed).fit(X)
        rfm = self.rfm.copy()
        rfm["cluster"] = km.labels_
        prof = rfm.groupby("cluster").agg(
            customers=("recency", "size"), recency=("recency", "mean"),
            frequency=("frequency", "mean"), monetary=("monetary", "mean"))
        # Give clusters human-readable names by value
        order = prof["monetary"].rank(ascending=False).astype(int)
        names = {1: "VIP", 2: "Regular", 3: "Occasional", 4: "Low value"}
        prof["label"] = [names.get(o, f"Group {o}") for o in order]
        rfm["cluster_label"] = rfm["cluster"].map(prof["label"])
        self.report.tables["K-Means clusters"] = prof.round(1)
        self.report.tests["Silhouette (chosen k)"] = round(
            silhouette(X, km.labels_, seed=self.cfg.seed), 3)

        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (c, g) in enumerate(rfm.groupby("cluster_label")):
            ax.scatter(g["frequency"], g["monetary"], s=12, alpha=0.6,
                       color=PALETTE[i % len(PALETTE)], label=c)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=10)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Frequency (orders)")
        ax.set_ylabel("Monetary (revenue)")
        ax.set_title("Customer clusters")
        ax.legend(frameon=False)
        self.plot.save(fig, "clusters_scatter", "Customer clusters")
        return rfm

from dataclasses import dataclass

# ========== 1. Modellparameter ========== 

@dataclass
class CISimConfig:
    n: int
    p_true: float
    gamma: float = 0.95

    m: int = 100
    seed: int = 42

    x_min: float = 0.0
    x_max: float = 1.0

# ========================================

# ========== 1. Stilparameter ============ 
@dataclass
class CISimStyle:
    """
    Darstellungseinstellungen für die Simulation
    vieler Konfidenzintervalle.
    """

    color_cover: str = "blue"
    color_miss: str = "red"

    figsize: tuple = (4.2, 6.2)

    show_stats: bool = True

# ========================================

# ========= Rechenkern ===================

from math import sqrt
from statistics import NormalDist
import numpy as np

def z_value(gamma: float) -> float:
    alpha = 1.0 - gamma
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)

def wilson_ci(k: int, n: int, z: float):
    h = k / n
    denom = 1.0 + (z**2) / n
    center = (h + (z**2) / (2.0 * n)) / denom
    half = z * sqrt((h * (1.0 - h) / n) + (z**2) / (4.0 * n**2)) / denom
    return center - half, center + half

# ===========================================


# ========= Plot-Funktion ===================

import matplotlib.pyplot as plt

def plot_ci_simulation(cfg, style=None, save=None):
    if style is None:
        style = CISimStyle()

    z = z_value(cfg.gamma)
    rng = np.random.default_rng(cfg.seed)
    X = rng.binomial(cfg.n, cfg.p_true, size=cfg.m)

    intervals = np.empty((cfg.m, 2))
    cover = np.empty(cfg.m, dtype=bool)

    for i, k in enumerate(X):
        L, R = wilson_ci(int(k), cfg.n, z)
        intervals[i] = (L, R)
        cover[i] = (L <= cfg.p_true <= R)

    coverage_rate = cover.mean()

    fig, ax = plt.subplots(figsize=style.figsize)
    y = np.arange(1, cfg.m + 1)

    for i, (L, R) in enumerate(intervals):
        color = style.color_cover if cover[i] else style.color_miss
        ax.hlines(y[i], L, R, linewidth=1.2, color=color)

    ax.axvline(cfg.p_true, linewidth=1.0, color="gray")

    ax.set_ylim(0, cfg.m + 1)
    ax.set_xlim(cfg.x_min, cfg.x_max)
    ax.set_xlabel(r"$p$",fontsize=12)
    ax.set_ylabel("Realisierung",fontsize=12)

    if style.show_stats:
        ax.set_title(
            rf"{cfg.m} Intervalle (Wilson), $n={cfg.n}, \gamma={cfg.gamma}$" "\n"
            rf"Trefferquote $\approx {coverage_rate:.2f}$ | Seed {cfg.seed}"
        )
    else:
        ax.set_title(rf"$n={cfg.n}, \gamma={cfg.gamma}$")

    fig.tight_layout()

    if save is not None:
        fig.savefig(save, bbox_inches="tight")

    plt.show()
    plt.close(fig)

    return float(coverage_rate)

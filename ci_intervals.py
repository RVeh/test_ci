from dataclasses import dataclass

# ========== 1. Modell- Styleparameter ========== 
from typing import Tuple

@dataclass(frozen=True)
class CIStyle:
    curve_upper: str = "black"
    curve_lower: str = "green"
    ci_bar: str = "blue"
    interval_bar: str = "tab:blue"
    helper_lines: str = "gray"

    area_color: str = "lightgray"
    area_alpha: float = 0.4   # 0.0 → aus
    
    grid: bool = True
    ticks: str = "normal"   # "normal" | "fine"

    figsize: Tuple[int, int] = (4, 4)
    
    show_prediction_overlay: bool = False
    prediction_steps: int = 10
    prediction_alpha: float = 0.8


# Typografie
@dataclass(frozen=True)
class TypographyStyle:
    title_fontsize: int = 14
    subtitle_fontsize: int = 12    
    label_fontsize: int = 12
    tick_fontsize: int = 11
    legend_fontsize: int = 10


@dataclass(frozen=True)
class PIStyle:
    curve_upper: str = "black"
    curve_lower: str = "green"
    
    interval_bar: str = "blue"
    helper_lines: str = "gray"

    area_color: str = "lightgray"
    area_alpha: float = 0.0   # standardmäßig aus bei PI

    grid: bool = True
    ticks: str = "normal"   # "normal" | "fine"

    figsize: Tuple[int, int] = (4, 4)


@dataclass(frozen=True)
class CISimStyle:
    """
    Style settings for CI simulation plots.
    """
    color_cover: str = "blue"
    color_miss: str = "red"

    figsize: Tuple[float, float] = (4.2, 6.2)

    show_stats: bool = True

@dataclass
class PIConfig:
    p: float
    n: int
    gamma: float = 0.95

    p_min: float = 0.0
    p_max: float = 1.0
    figsize: tuple = (4, 4)

@dataclass
class CIConfig:
    n: int
    gamma: float = 0.95

    p_min: float = 0.0
    p_max: float = 1.0
    figsize: tuple = (4, 4)

@dataclass
class CIRealConfig:
    h: float
    n: int
    gamma: float = 0.95


@dataclass
class CISimConfig:
    n: int
    p_true: float
    gamma: float = 0.95

    m: int = 100
    seed: int = 42

    x_min: float = 0.0
    x_max: float = 1.0

@dataclass(frozen=True)
class CIModelConfig:
    """
    Model parameters for confidence intervals.
    """
    h: float
    n: int
    gamma: float

@dataclass(frozen=True)
class CIGeometryConfig:
    """
    Geometric parameters for CI plots.
    """
    p_min: float = 0.0
    p_max: float = 0.5
    points: int = 3000


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

#==========
@dataclass(frozen=True)
class PIModelConfig:
    """
    Model parameters for prediction intervals.
    """
    p: float
    n: int
    gamma: float = 0.95


@dataclass(frozen=True)
class PIGeometryConfig:
    """
    Geometric parameters for PI plots.
    """
    p_min: float = 0.0
    p_max: float = 1.0

# ======== 2. Intervall-Methoden ========= 
from math import sqrt
from statistics import NormalDist
from scipy.stats import beta

def z_value(gamma: float) -> float:
    alpha = 1.0 - gamma
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)

def wald_ci(h, n, gamma):
    z = z_value(gamma)
    half = z * sqrt(h * (1 - h) / n)
    return h - half, h + half

def wilson_ci(h, n, gamma):
    z = z_value(gamma)
    denom = 1 + z**2 / n
    center = (h + z**2 / (2*n)) / denom
    radius = z * sqrt((h * (1 - h) + z**2 / (4*n)) / n) / denom
    return center - radius, center + radius

def clopper_pearson_ci(h, n, gamma):
    k = round(h * n)
    alpha = 1.0 - gamma
    L = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    R = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return L, R

# ======== 3. Ausgabe-Funktion =========== 
def ci_realisations(cfg: CIRealConfig):
    return {
        "Wald-KI": wald_ci(cfg.h, cfg.n, cfg.gamma),
        "Wilson_KI": wilson_ci(cfg.h, cfg.n, cfg.gamma),
        "Clopper-Pearson-KI": clopper_pearson_ci(cfg.h, cfg.n, cfg.gamma)
    }
# ============= Plot-CI =================
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_ci(
    model: CIModelConfig,
    geometry: CIGeometryConfig,
    style: CIStyle,
    *,
    save: str | None = None,
    show_info: bool = False,
):
    # --- Geometrie ---
    p = np.linspace(geometry.p_min, geometry.p_max, geometry.points)

    # --- Modell ---
    z = norm.ppf((1 + model.gamma) / 2)

    def f(x):
        return x + z * np.sqrt(x * (1 - x) / model.n)

    def g(x):
        return x - z * np.sqrt(x * (1 - x) / model.n)

    # Wilson-Intervall
    p_L, p_R = wilson_ci(model.h, model.n, model.gamma)

    # Wald-Intervall
    p_L_Wald, p_R_Wald = wald_ci(model.h,model.n,model.gamma)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=style.figsize)

    ax.plot(p, f(p), color=style.curve_lower)
    ax.plot(p, g(p), color=style.curve_upper)

    ax.fill_between(
    p,
    g(p),
    f(p),
    color=style.area_color,
    alpha=style.area_alpha,
    zorder=0,
)
   #ax.hlines(0, p_L, p_R, linewidth=5, color=style.ci_bar)
    ax.hlines(geometry.p_min, p_L, p_R, linewidth=7, color=style.ci_bar)

    ax.hlines(model.h, 0, 1, linestyle=":", color=style.helper_lines)
    ax.hlines(model.h, p_L, p_R, linewidth=2, color=style.ci_bar)
    ax.vlines([p_L, p_R], 0, model.h, linestyle=":", color=style.helper_lines)

    if style.show_prediction_overlay:
        p_vals = np.linspace(p_L, p_R, style.prediction_steps)

        for p in p_vals:
            li_p, re_p = prediction_interval(p, model.n, model.gamma)
            ax.vlines(
                p,
                li_p,
                re_p,
                color=style.helper_lines,
                linewidth=1.2,
                alpha=style.prediction_alpha,
                zorder=3,
            )
    
    ax.set_xlim(geometry.p_min, geometry.p_max)
    ax.set_ylim(geometry.p_min, geometry.p_max)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(r"$p$", fontsize=11)
    ax.set_ylabel(r"$h$", fontsize=11)

    if style.grid:
        ax.grid(True, alpha=0.8)

    if style.ticks == "fine":
        ax.minorticks_on()
        ax.tick_params(which="major", length=6)

    ax.text(
            0.5,
            1.09,
            "WILSON-Konfidenzintervall",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=12,
        )

    ax.text(
            0.5,
            1.02,
            rf"Setzung: $n={model.n},\; \gamma={model.gamma}$ | Stichprobe: $h={model.h}$",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
      )
    
    if show_info:
        ax.text(
            0.5,
            -0.16,
            rf"WILSON-KI $\approx$[{p_L:.3f};{p_R:.3f}] | (WALD-KI $\approx$[{p_L_Wald:.3f};{p_R_Wald:.3f}])" ,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10
            )
        
    if save is not None:
        fig.savefig(save, bbox_inches="tight")

    plt.show()
    return p_L, p_R

# ==========================================

import numpy as np
from scipy.stats import norm

def prediction_interval(p, n, gamma):
    z = norm.ppf((1 + gamma) / 2)
    radius = z * np.sqrt(p * (1 - p) / n)
    return p - radius, p + radius

def prediction_interval_abs(p, n, gamma):
    z = norm.ppf((1 + gamma) / 2)
    mu = n * p
    sigma = np.sqrt(n * p * (1 - p))
    return mu - z * sigma, mu + z * sigma

# ========= Plot_PI_Realtiv =============

import matplotlib.pyplot as plt
import numpy as np

def plot_pi(cfg, style=None, save=None, show_info=True):
    if style is None:
        style = PIStyle()

    x = np.linspace(cfg.p_min, cfg.p_max, 3000)
    z = norm.ppf((1 + cfg.gamma) / 2)

    def f(x): return x + z * np.sqrt(x * (1 - x) / cfg.n)
    def g(x): return x - z * np.sqrt(x * (1 - x) / cfg.n)

    li, re = prediction_interval(cfg.p, cfg.n, cfg.gamma)

    fig, ax = plt.subplots(figsize=cfg.figsize)

    # Grenzkurven
    ax.plot(x, f(x), color=style.curve_upper)
    ax.plot(x, g(x), color=style.curve_lower)

    # Prognoseintervall
    ax.vlines(cfg.p_min, li, re, linewidth=7, color=style.interval_bar)
    ax.vlines(cfg.p, li, re, linewidth=2, color=style.interval_bar)

    # Hilfslinien
    ax.hlines([li, re], 0, cfg.p, linestyle=":", color=style.helper_lines)
    ax.vlines(cfg.p, 0, re, linestyle=":", color=style.helper_lines)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(cfg.p_min, cfg.p_max)
    ax.set_ylim(cfg.p_min, cfg.p_max)

    ax.set_xlabel(r"$p$",fontsize=11)
    ax.set_ylabel(r"$h$",fontsize=11)
    
    if style.grid:
        ax.grid(True, alpha=0.8)

    if style.ticks == "fine":
        ax.minorticks_on()
        ax.tick_params(which="major", length=6)

    if style.ticks == "fine":
            ax.minorticks_on()
            ax.tick_params(which="major", length=6)

    ax.text(
            0.5,
            1.08,
            rf"relatives Prognoseintervall für $H=X/n$",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=12,
        )

    ax.text(
            0.5,
            1.01,
            rf"$X\sim$Bin(n,p):  "
            rf"$n={cfg.n},\; p={cfg.p},\; \gamma={cfg.gamma}$",
            #rf"$H_0: p={p0_txt},\; H_1: p\neq {p0_txt},\; n={n},\; \alpha={alpha}$",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
      )

    li_int = math.ceil(li*cfg.n)
    re_int = math.floor(re*cfg.n)   
    
    if show_info:
        ax.text(
            0.5,
            -0.16,
            rf"{cfg.gamma*100:.0f}%-PI$\approx$[{li:.3f} ; {re:.3f}] | Äquivalent: $\frac{{{li_int}}}{{{cfg.n}}} \leq H \leq \frac{{{re_int}}}{{{cfg.n}}}$",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10
            )

    if save is not None:
        fig.savefig(save, bbox_inches="tight")

    plt.show()
    return li, re

# ========= Plot_CI_Absolut =============

import matplotlib.pyplot as plt
import numpy as np
import math

def plot_pi_abs(cfg, style=None, save=None, show_info=True):
    if style is None:
        style = CIStyle()

    x = np.linspace(cfg.p_min, cfg.p_max, 3000)
    z = norm.ppf((1 + cfg.gamma) / 2)

    def f(x): return cfg.n * x + z * np.sqrt(cfg.n * x * (1 - x))
    def g(x): return cfg.n * x - z * np.sqrt(cfg.n * x * (1 - x))

    li, re = prediction_interval_abs(cfg.p, cfg.n, cfg.gamma)

    fig, ax = plt.subplots(figsize=cfg.figsize)

    # Grenzkurven
    ax.plot(x, f(x), color=style.curve_upper)
    ax.plot(x, g(x), color=style.curve_lower)

    # Prognoseintervall
    ax.vlines(0, li, re, linewidth=7, color=style.interval_bar)
    #ax.vlines(cfg.p, li, re, linewidth=2, color=style.helper_lines)
    ax.vlines(cfg.p, li, re, linewidth=2, color=style.interval_bar)

    # Hilfslinien
    ax.hlines([li, re], 0, cfg.p, linestyle=":", color=style.helper_lines)
    ax.vlines(cfg.p, 0, re, linestyle=":", color=style.helper_lines)

    ax.set_aspect("auto")
    ax.set_xlim(cfg.p_min, cfg.p_max)
    ax.set_ylim(0, cfg.n)
    ax.set_yticks(range(0, cfg.n + 1, max(1, cfg.n // 10)))
    ax.set_xlabel(r"$p$",fontsize=11)
    ax.set_ylabel(r"$k$",fontsize=11)
    
    if style.grid:
        ax.grid(True, alpha=0.8)

    if style.ticks == "fine":
        ax.minorticks_on()
        ax.tick_params(which="major", length=6)

    ax.text(
            0.5,
            1.08,
            rf"absolutes Prognoseintervall für $X$",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=12,
        )

    ax.text(
            0.5,
            1.01,
            rf"$X\sim$Bin(n,p):  "
            rf"$n={cfg.n},\; p={cfg.p},\; \gamma={cfg.gamma}$",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
      )

    li_int = math.ceil(li)
    re_int = math.floor(re)   
    
    if show_info:
        ax.text(
            0.5,
            -0.16,
            rf"{cfg.gamma*100:.0f}%-PI$\approx$[{li:.3f} ; {re:.3f}] | Äquivalent: {li_int} $\leq$ X $\leq$ {re_int}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10
            )
      
    if save is not None:
        fig.savefig(save, bbox_inches="tight")

    plt.show()
    return li, re

# ========= Rechenkern ===================

from math import sqrt
from statistics import NormalDist
import numpy as np

def z_value(gamma: float) -> float:
    alpha = 1.0 - gamma
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)

def wilson_k_ci(k: int, n: int, z: float):
    h = k / n
    denom = 1.0 + (z**2) / n
    center = (h + (z**2) / (2.0 * n)) / denom
    half = z * sqrt((h * (1.0 - h) / n) + (z**2) / (4.0 * n**2)) / denom
    return center - half, center + half#

# ========= PlotSim-Funktion ===================

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
        L, R = wilson_k_ci(int(k), cfg.n, z)
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
    ax.set_xlabel(r"$p$",fontsize=11)
    ax.set_ylabel("Realisierung",fontsize=12)

    ax.text(
        0.5,
        1.07,
        "Simulation: WILSON-Konfidenzintervalle",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        )

    ax.text(
        0.5,
        1.01,
        rf"$n={cfg.n},\; \gamma={cfg.gamma}$",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        )

    if style.show_stats:
        ax.text(
            0.5,
            -0.12,
            rf"{cfg.m} Intervalle |  Überdeckungsrate $\approx {coverage_rate*100:.2f}\%$ | Seed = {cfg.seed}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10
            )
         
    else:
      ax.text(
            0.5,
            -0.12,
            rf"{cfg.m} Intervalle |  Seed {cfg.seed}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10
            )

    fig.tight_layout()

    if save is not None:
        fig.savefig(save, bbox_inches="tight")

    plt.show()
    plt.close(fig)

    return float(coverage_rate)


# =========== plot ci_ellipse =========================
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_ci_ellipse(
    model: CIModelConfig,
    geometry: CIGeometryConfig,
    style: CIStyle,
    *,
    save: str | None = None,
):
    p = np.linspace(geometry.p_min, geometry.p_max, 3000)
    z = norm.ppf((1 + model.gamma) / 2)

    def f(p):
        return p + z * np.sqrt(p * (1 - p) / model.n)

    def g(p):
        return p - z * np.sqrt(p * (1 - p) / model.n)

    fig, ax = plt.subplots(figsize=style.figsize)

    # Grenzkurven
    ax.plot(p, f(p), color=style.curve_lower)
    ax.plot(p, g(p), color=style.curve_upper)

    ax.fill_between(
    p,
    g(p),
    f(p),
    color=style.area_color,
    alpha=style.area_alpha,
    zorder=0,
)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(geometry.p_min, geometry.p_max)
    ax.set_ylim(geometry.p_min, geometry.p_max)

    ax.set_xlabel(r"$p$",fontsize=12)
    ax.set_ylabel(r"$h$",fontsize=12)

    if style.grid:
        ax.grid(True, alpha=0.8)

    if style.ticks == "fine":
        ax.minorticks_on()
        ax.tick_params(which="major", length=6)

    ax.set_title(
        rf"$n={model.n},\; \gamma={model.gamma}$",
        y=1.02
    )

    if save is not None:
        fig.savefig(save, bbox_inches="tight")

    plt.show()

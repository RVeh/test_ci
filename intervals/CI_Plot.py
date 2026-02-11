from dataclasses import dataclass

@dataclass
class CIConfig:
    h: float
    n: int
    gamma: float

    p_min: float = 0.0
    p_max: float = 0.3
    points: int = 3000

    figsize: tuple = (4, 4)


import numpy as np
from scipy.stats import norm

def wilson_ci(h, n, gamma):
    z = norm.ppf((1 + gamma) / 2)
    denom = 1 + z**2 / n
    center = (h + z**2 / (2*n)) / denom
    radius = z * np.sqrt(
        (h * (1 - h) + z**2 / (4*n)) / n
    ) / denom
    return center - radius, center + radius


@dataclass
class CIStyle:
    curve_upper: str = "black"
    curve_lower: str = "green"
    ci_bar: str = "blue"
    helper_lines: str = "gray"

    grid: bool = True
    ticks: str = "normal"   # "normal" | "fine"

import matplotlib.pyplot as plt

def plot_ci(cfg, style=None, save=None, show_info=True):
    if style is None:
        style = CIStyle()
   
    z = norm.ppf((1 + cfg.gamma) / 2)
    def f(x): return x + z * np.sqrt(x * (1 - x) / cfg.n)
    def g(x): return x - z * np.sqrt(x * (1 - x) / cfg.n)

    p = np.linspace(cfg.p_min, cfg.p_max, cfg.points)

    p_L, p_R = wilson_ci(cfg.h, cfg.n, cfg.gamma)


    fig, ax = plt.subplots(figsize=cfg.figsize)

    ax.plot(p, f(p), color=style.curve_upper)
    ax.plot(p, g(p), color=style.curve_lower)
    ax.hlines(0, p_L, p_R, linewidth=5, color=style.ci_bar)

    ax.hlines(cfg.h, 0, 1, linestyle=":", color="gray")
    ax.hlines(cfg.h, p_L, p_R, linewidth=2, color=style.helper_lines)
    ax.hlines(0, p_L, p_R, linewidth=5, color="blue")
    ax.vlines([p_L, p_R], 0, cfg.h, linestyle=":", color=style.helper_lines)

    ax.set_xlim(cfg.p_min, cfg.p_max)
    ax.set_ylim(cfg.p_min, cfg.p_max)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(r"$p$",fontsize=12)
    ax.set_ylabel(r"$h$",fontsize=12)

    ax.set_title(rf"$n={cfg.n}$; $h={cfg.h}; \gamma={cfg.gamma}$")

    if style.grid:
        ax.grid(True, alpha=0.8)

    if style.ticks == "fine":
        ax.minorticks_on()
        ax.tick_params(which="major", length=6)
       
    if show_info:
        ax.text(
            cfg.p_min + 0.01,
            0.92 * cfg.p_max,
            rf"{cfg.gamma*100:.0f}%-KI$\approx$[{p_L:.3f};{p_R:.3f}]"
            )

    if save is not None:
        fig.savefig(save, bbox_inches="tight")
        
    plt.show()

    return p_L, p_R



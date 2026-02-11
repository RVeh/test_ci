from dataclasses import dataclass

@dataclass
class CIConfig:
    n: int
    gamma: float = 0.95

    p_min: float = 0.0
    p_max: float = 1.0
    figsize: tuple = (4, 4)


import numpy as np
from scipy.stats import norm


@dataclass
class CIStyle:
    curve_upper: str = "black"
    curve_lower: str = "green"
    
    grid: bool = True
    ticks: str = "normal"   # "normal" | "fine"

import matplotlib.pyplot as plt
import numpy as np

def plot_ci(cfg, style=None, save=None):
    if style is None:
        style = CIStyle()

    x = np.linspace(cfg.p_min, cfg.p_max, 3000)
    z = norm.ppf((1 + cfg.gamma) / 2)

    def f(x): return x + z * np.sqrt(x * (1 - x) / cfg.n)
    def g(x): return x - z * np.sqrt(x * (1 - x) / cfg.n)


    fig, ax = plt.subplots(figsize=cfg.figsize)

    # Grenzkurven
    ax.plot(x, f(x), color=style.curve_upper)
    ax.plot(x, g(x), color=style.curve_lower)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(cfg.p_min, cfg.p_max)
    ax.set_ylim(cfg.p_min, cfg.p_max)

    ax.set_xlabel(r"$p$")
    ax.set_ylabel(r"$h$")
    
    if style.grid:
        ax.grid(True, alpha=0.8)

    if style.ticks == "fine":
        ax.minorticks_on()
        ax.tick_params(which="major", length=6)

    ax.set_title(
        rf"$n={cfg.n},\; \gamma={cfg.gamma}$",
        y=1.02
    )   

    if save is not None:
        fig.savefig(save, bbox_inches="tight")

    plt.show()
    return 

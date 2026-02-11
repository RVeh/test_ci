# tests/plots/plot_model.py
from __future__ import annotations

from dataclasses import dataclass
import math
import matplotlib.pyplot as plt

from tests.model.binomial import BinomialModel


# ------------------------------------------------------------------
# Plot-Konfiguration (bewusst explizit)
# ------------------------------------------------------------------
from typing import Tuple

@dataclass(frozen=True)
class ModelPlotStyle:
    # Balken
    bar_color: str = "black"
    bar_width: float = 1.2

    # Erwartungswert
    mean_color: str = "gray"
    mean_style: str = "--"
    mean_width: float = 1.2

    # Modellbereich
    sigma_range: float = 5.0   # ± k·σ

    # Referenzlinien / Marker (Power)
    power_ref_color: str = "gray"
    power_ref_width: float = 1.0
    power_ref_style: str = "--"
    power_point_size: float = 40.0

    power_p0_color: str = "tab:blue"
    power_pstar_color: str = "tab:blue"

    # Typografie
    title_fontsize: int = 14
    subtitle_fontsize: int = 12    
    label_fontsize: int = 12
    tick_fontsize: int = 11
    legend_fontsize: int = 10

    # Abbildung
    figsize: Tuple[float, float] = (8.0, 4.0)

# --- NEU: Farbe der Modellverteilung ---
    model_color: str = "tab:blue"
    reject_color: str = "tab:red"
    


# ------------------------------------------------------------------
# Grafik 1: Stichprobenverteilung
# ------------------------------------------------------------------

def plot_binomial_model(
    model: BinomialModel,
    style: ModelPlotStyle = ModelPlotStyle(),
    ax: plt.Axes | None = None,
    *,
    save: str | None = None,
):
    """
    Grafik 1 – Fundament:
    Stichprobenverteilung unter H0.

    - Definitionsmenge: ± k·σ um μ
    - diskrete Darstellung
    - Erwartungswert als geometrisches Zentrum
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=style.figsize)

    # Modellkennzahlen
    mu = model.n * model.p
    sigma = math.sqrt(model.n * model.p * (1 - model.p))

    # sinnvolle Definitionsmenge (ganzzahlig!)
    k_min = max(0, int(math.floor(mu - style.sigma_range * sigma)))
    k_max = min(model.n, int(math.ceil(mu + style.sigma_range * sigma)))

    k_vals = [k for k in model.support if k_min <= k <= k_max]
    p_vals = [model.pmf(k) for k in k_vals]

    # diskrete Wahrscheinlichkeitsfunktion
    ax.vlines(
        k_vals,
        [0],
        p_vals,
        linewidth=style.bar_width,
        color=style.bar_color,
    )

    # Erwartungswert
    ax.axvline(
        mu,
        linestyle=style.mean_style,
        linewidth=style.mean_width,
        color=style.mean_color,
        label=rf"Erwartungswert $E(X)={mu:.0f}$",
    )

    # Achsen
    ax.set_xlim(k_min, k_max)
    ax.set_ylim(0, max(p_vals) * 1.1)


    ax.set_xlabel(
        r"Anzahl der Erfolge $k$",
        fontsize=style.label_fontsize,
    )
    ax.set_ylabel(
        r"$P(X = k)$",
        fontsize=style.label_fontsize,
    )
    
    ax.tick_params(
        axis="both",
        labelsize=style.tick_fontsize,
    )

    
    # --------------------------------------------------------------
    # Titel + Subtitel als EIN Block (gedankliches Rechteck)
    # --------------------------------------------------------------

    ax.text(
    0.5,
    1.11,
    "Stichprobenverteilung",
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=style.title_fontsize,
    #fontweight="bold",
    )

    ax.text(
        0.5,
        1.02,
        rf"Binomialmodell: $n={model.n},\; p_0={model.p}$",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=style.subtitle_fontsize,
    )


    # Legende (nur semantisch relevant)
    ax.legend(
        frameon=False,
        fontsize=style.legend_fontsize,
)

    if save is not None:
        ax.figure.savefig(save, bbox_inches="tight")
    
    return ax

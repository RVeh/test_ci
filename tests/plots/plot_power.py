from __future__ import annotations

import matplotlib.pyplot as plt
from typing import Callable

from tests.power.power_function import power_curve, power_at
from tests.plots.plot_model import ModelPlotStyle

def plot_power_curve(
    model_factory: Callable[[float], object],
    rejection_region,
    *,
    p0: float,
    n: int,
    alpha: float,
    p_star: float | None = None,
    p_min: float = 0.0,
    p_max: float = 1.0,
    p_step: float = 0.01,
    style: ModelPlotStyle = ModelPlotStyle(),
    save: str | None = None,
):
    """
    Referenzgrafik: Powerfunktion des Tests.

    Darstellung:
    - Power g_n(p) = P_p(X ∈ K)
    - festes Testverfahren (K_alpha)
    - variierender wahrer Parameter p
    """

    fig, ax = plt.subplots(figsize=style.figsize)

    # --- p-Gitter ---
    p_values = []
    p = p_min
    while p <= p_max + 1e-12:
        p_values.append(round(p, 6))
        p += p_step

    curve = power_curve(
        model_factory=model_factory,
        p_values=p_values,
        rejection_region=rejection_region,
    )

    p_vals = [p for p, _ in curve]
    g_vals = [g for _, g in curve]

    # --- Powerkurve ---
    ax.plot(p_vals, g_vals, linewidth=2.0)

    # --- Referenzpunkt p0 ---
    g_p0 = power_at(model_factory(p0), rejection_region)

    ax.vlines(
        p0,
        ymin=0,
        ymax=g_p0,
        color=style.power_ref_color,
        linewidth=style.power_ref_width,
        linestyle=style.power_ref_style,
    )
    ax.scatter(
        [p0],
        [g_p0],
        s=style.power_point_size,
        color=style.power_p0_color,
        zorder=3,
    )

    # --- Optionaler Referenzpunkt p* ---
    g_ps = None
    if p_star is not None:
        g_ps = power_at(model_factory(p_star), rejection_region)

        ax.vlines(
            p_star,
            ymin=0,
            ymax=g_ps,
            color=style.power_ref_color,
            linewidth=style.power_ref_width,
            linestyle=":",
        )
        ax.scatter(
            [p_star],
            [g_ps],
            s=style.power_point_size,
            color=style.power_pstar_color,
            zorder=3,
        )

    # --- Achsen ---
    ax.set_xlim(p_min, p_max)
    ax.set_ylim(0, 1.05)

    ax.set_xlabel(
        r"Wahrer Parameter $p$",
        fontsize=style.label_fontsize,
    )
    ax.set_ylabel(
        r"Power $g_n(p)$",
        fontsize=style.label_fontsize,
    )

    ax.tick_params(axis="both", labelsize=style.tick_fontsize)

    # --- Titel & Subtitel (exakt dein Standard) ---
    ax.text(
        0.5,
        1.11,
        "Powerfunktion des Tests",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=style.title_fontsize,
    )

    p0_txt = f"{p0:g}"
    
    ax.text(
        0.5,
        1.01,
        rf"Zweiseitiger Binomialtest: "
        rf"$H_0: p={p0_txt},\; H_1: p\neq {p0_txt},\; n={n},\; \alpha={alpha}$",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=style.subtitle_fontsize,
    )

    # --- Text unter der Grafik (semantisch, nicht grafisch) ---
    from tests.utils.format_K import format_rejection_region_intervals
    

    K_text = format_rejection_region_intervals(rejection_region.K, n)

    text_parts = [K_text]
    
    if p_star is not None and g_ps is not None:
        text_parts.append(
            rf"Beispiel: $p={p_star}\Rightarrow g_{{{n}}}({p_star})={g_ps:.3f}$"
        )

    ax.text(
    0.5,
    -0.20,
    "   |   ".join(text_parts),
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=style.tick_fontsize,
    )

    # --- Speichern ---
    if save is not None:
        fig.savefig(save, bbox_inches="tight", pad_inches=0.3)

    return ax

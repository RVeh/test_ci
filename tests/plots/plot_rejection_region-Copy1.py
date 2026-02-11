# tests/plots/plot_rejection_region.py
from __future__ import annotations

import matplotlib.pyplot as plt

from tests.model.binomial import BinomialModel
from tests.geometry.construct_rejection_region import two_sided_equal_tails
from tests.plots.plot_model import plot_binomial_model, ModelPlotStyle


def plot_binomial_model_with_rejection_region(
    model: BinomialModel,
    alpha: float,
    style: ModelPlotStyle = ModelPlotStyle(),
    *,
    save: str | None = None,
):
    """
    Grafik 2:
    Stichprobenverteilung + Ablehnungsbereich
    (zweiseitig, equal tails).
    """

    fig, ax = plt.subplots(figsize=style.figsize)


    # 1. Basisgrafik (Referenz!)
    plot_binomial_model(
        model=model,
       # alpha=alpha,
        style=style,
        ax=ax,
    )

    # 2. Ablehnungsbereich (Setzung!)
    R = two_sided_equal_tails(model, alpha)

    k_reject = sorted(R.K)
    p_reject = [model.pmf(k) for k in k_reject]

    ax.vlines(
        k_reject,
        [0],
        p_reject,
        linewidth=style.bar_width,
        color="tab:red",
        label=rf"Ablehnungsbereich ($\alpha={alpha}$)",
    )

    # 3. Legende erweitern (nur Bedeutung!)
    ax.legend(frameon=False, fontsize=style.legend_fontsize)

    # --- Text unter der Grafik (semantisch, nicht grafisch) ---
    from tests.utils.format_K import format_rejection_region_intervals
    from tests.decision.p_value import p_value_equal_tails
    
    K_text = format_rejection_region_intervals(rejection_region.K, model.n)
    
    text_parts = [K_text]
    
    if k_obs is not None:
        p_val = p_value_equal_tails(k_obs, model)
        text_parts.append(
            rf"$k_{{obs}}={k_obs},\; p\text{{-Wert}}={p_val:.3f}$"
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


    
    # 4. Optional speichern
    if save is not None:
        fig.savefig(save, bbox_inches="tight")

    return ax

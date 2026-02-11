# tests/plots/plot_p_value.py
from __future__ import annotations

import math
import matplotlib.pyplot as plt

from tests.model.binomial import BinomialModel
from tests.decision.p_value import p_value_two_sided_equal_tails
from tests.plots.plot_model import plot_binomial_model, ModelPlotStyle


def plot_binomial_model_with_p_value(
    model: BinomialModel,
    x_obs: int,
    style: ModelPlotStyle = ModelPlotStyle(),
    *,
    save: str | None = None,
):
    """
    Grafik 3:
    Stichprobenverteilung + Beobachtung + p-Wert
    (zweiseitig, equal tails).

    Keine Entscheidung, kein alpha.
    """

    fig, ax = plt.subplots(figsize=style.figsize)

    # 1. Basisgrafik (Referenz!)
    plot_binomial_model(
        model=model,
        style=style,
        ax=ax,
    )

    # 2. Beobachtung
    ax.axvline(
        x_obs,
        color="tab:gray",
        linewidth=1.0,
        label=rf"Beobachtung $x={x_obs}$",
    )

    # 3. p-Wert-Bereiche (Randmassen)
    mu = model.n * model.p
    mirror = int(round(2 * mu - x_obs))

    k_vals = list(model.support)

    # linke Randmasse
    k_left = [k for k in k_vals if k <= min(x_obs, mirror)]
    p_left = [model.pmf(k) for k in k_left]

    # rechte Randmasse
    k_right = [k for k in k_vals if k >= max(x_obs, mirror)]
    p_right = [model.pmf(k) for k in k_right]

    ax.vlines(
        k_left,
        [0],
        p_left,
        color="tab:orange",
        alpha=0.6,
        linewidth=style.bar_width,
        label="p-Wert (Randmasse)",
    )

    ax.vlines(
        k_right,
        [0],
        p_right,
        color="tab:orange",
        alpha=0.6,
        linewidth=style.bar_width,
    )

    # 4. p-Wert berechnen (nur für Beschriftung!)
    p_val = p_value_two_sided_equal_tails(model, x_obs)

    ax.legend(
        frameon=False,
        fontsize=style.legend_fontsize,
        title=rf"$p = {p_val:.4f}$",
        title_fontsize=style.legend_fontsize,
    )

    # 5. Optional speichern
    if save is not None:
        fig.savefig(save, bbox_inches="tight")

    return ax

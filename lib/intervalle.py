"""
intervalle.py

Referenzprogramme für Konfidenz- und Prognoseintervalle
(Sek II / Lehrkräftefortbildung)

Ziel:
- stabile, fertige Werkzeuge für den Unterricht
- eine Funktion = eine Grafik
- Modell, Geometrie und Darstellung sind strikt getrennt
- geeignet für Binder (ressourcenschonend, reproduzierbar)

Hinweis:
Diese Datei ist zum BENUTZEN gedacht, nicht zum Umbauen.
"""

# ============================================================
# 0. Imports (einmal, zentral)
# ============================================================

from dataclasses import dataclass
from typing import Optional
from typing import Tuple
from math import sqrt
from statistics import NormalDist

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, norm
from matplotlib.ticker import MaxNLocator

from lib.rng_tools import get_rng
from lib.plot_style import (
    set_title_block,
    set_info_line,
    apply_grid,
    finalize_figure,
)

# ============================================================
# 1. KONFIGURATIONEN – das dürfen LuL verändern
# ============================================================

# ---------- Modelle (Statistik) ----------

@dataclass(frozen=True)
class CIConfig:
    """
    Modell für Konfidenzintervalle
    h      beobachteter Anteil, falls None: keine konkrete Stichprobe
    n      Stichprobengröße
    gamma  Sicherheitsniveau
    """
   
    n: int
    gamma: float = 0.95
    h: Optional[float]=None

@dataclass(frozen=True)
class PIModel:
    """
    Modell für Prognoseintervalle
    p      bekannte Trefferwahrscheinlichkeit
    n      Stichprobengröße
    gamma  Sicherheitsniveau
    """
    p: float
    n: int
    gamma: float = 0.95


# ---------- Geometrie (Darstellungsraum, keine Statistik!) ----------

@dataclass(frozen=True)
class IntervalGeometry:
    """
    Zeichenraum der Grafik
    """
    x_min: float = 0.0
    x_max: float = 1.0
    points: int = 1200


# ---------- Darstellung (rein optisch) ----------

@dataclass(frozen=True)
class CIStyle:
    curve_upper: str = "black"
    curve_lower: str = "green"

    area_color: str = "lightgray"
    area_alpha: float = 0.4

    interval_color: str = "tab:blue"
    helper_lines: str = "gray"

    grid: bool = True
    figsize: Tuple[int, int] = (4, 4)

# -------- Referenzklassen – Simulation Überdeckungsrate ------

# ------------ 1 Modell & Setzung (fachlich) ------------------

@dataclass(frozen=True)
class CISimConfig:
    """
    Fachliche Parameter der Simulation.

    Legt das statistische Modell und die Wiederholung fest.
    """
    n: int
    p_true: float
    gamma: float = 0.95

    m: int = 100
    seed: int | None = 42

    # Darstellungsfenster (Modellentscheidung!)
    x_min: float = 0.0
    x_max: float = 1.0

# --------- 2 Darstellung / didaktischer Modus ---------------
@dataclass(frozen=True)
class CISimStyle:
    """
    Didaktische Darstellung der Simulation.
    """

    # Farben
    color_cover: str = "tab:blue"
    color_miss: str = "tab:red"

    # Didaktischer Modus
    color_mode: str = "uniform"   # "uniform" | "coverage"
    show_stats: bool = False
    show_grid: bool = False

# ------------ Typografie --------------
class TypographyStyle:
    """
    Einheitliche Typografie für Referenzgrafiken.
    """
    title_fontsize: int = 13
    subtitle_fontsize: int = 12    
    label_fontsize: int = 12
    tick_fontsize: int = 11
    info_fontsize: int = 10

# ------------------------------------------------------------
# Hilfsfunktion: figsize aus Anzahl der Intervalle ableiten
# ------------------------------------------------------------

def figsize_from_m(m: int) -> tuple[float, float]:
    """
    Höhe der Grafik ist proportional zur Anzahl der simulierten Intervalle.
    """
    width = 4.2
    height_per_interval = 0.055
    height = max(4.0, m * height_per_interval)
    return width, height

 
# ============================================================
# 2. RECHENKERN – Mathematik (Finger weg)
# ============================================================
"""
Berechnung von Konfidenzintervallen für Binomialparameter.

Enthält:
- Wald-Intervall
- Wilson-Intervall
- Clopper–Pearson-Intervall
"""

def z_value(gamma: float) -> float:
    alpha = 1.0 - gamma
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)


def wald_ci(h: float, n: int, gamma: float) -> tuple[float, float]:
    z = z_value(gamma)
    half = z * sqrt(h * (1 - h) / n)
    return h - half, h + half


def wilson_ci(h: float, n: int, gamma: float) -> tuple[float, float]:
    z = z_value(gamma)
    denom = 1 + z**2 / n
    center = (h + z**2 / (2 * n)) / denom
    radius = z * sqrt((h * (1 - h) + z**2 / (4 * n)) / n) / denom
    return center - radius, center + radius


def clopper_pearson_ci(h: float, n: int, gamma: float) -> tuple[float, float]:
    k = round(h * n)
    alpha = 1.0 - gamma
    L = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    R = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return L, R


def prediction_interval(p: float, n: int, gamma: float) -> tuple[float, float]:
    z = z_value(gamma)
    radius = z * sqrt(p * (1 - p) / n)
    return p - radius, p + radius


def prediction_interval_abs(p: float, n: int, gamma: float) -> tuple[float, float]:
    z = z_value(gamma)
    mu = n * p
    sigma = sqrt(n * p * (1 - p))
    return mu - z * sigma, mu + z * sigma

# ============================================================
# 3. PLOT-FUNKTIONEN 
# ============================================================

# ================ Ellipse – Prognoseintervalle =================

def plot_pi(
    model: PIModel,
    geometry: IntervalGeometry,
    style: CIStyle,
    *,
    absolute: bool = False,
    save: str | None = None,
):
    """
    Geometrische Darstellung von Prognoseintervallen.

    - Modellparameter p ist fixiert (unter der Modellannahme)
    - Zufälligkeit liegt im Stichprobenergebnis H = X/n
    - Prognoseintervalle entstehen durch vertikale Schnitte
      der gleichen Geometrie wie bei Konfidenzintervallen
    """

    # --------------------------------------------------
    # Geometrie
    # --------------------------------------------------
    p = np.linspace(geometry.x_min, geometry.x_max, geometry.points)

    # --------------------------------------------------
    # Modell
    # --------------------------------------------------
    z = norm.ppf((1 + model.gamma) / 2)

    def f(x):
        h = x + z * np.sqrt(x * (1 - x) / model.n)
        return model.n * h if absolute else h

    def g(x):
        h = x - z * np.sqrt(x * (1 - x) / model.n)
        return model.n * h if absolute else h
    
    #def f(x):
     #   return x + z * np.sqrt(x * (1 - x) / model.n)
        
   # def g(x):
       # return x - z * np.sqrt(x * (1 - x) / model.n)

    # Prognoseintervall (relativ)
    h_L, h_R = prediction_interval(model.p, model.n, model.gamma)

    # Äquivalentes absolutes Intervall
    k_L = model.n * h_L
    k_R = model.n * h_R

    k_L_int = int(np.ceil(k_L))
    k_R_int = int(np.floor(k_R))

    if absolute:
        y_L, y_R = k_L, k_R
    else:
        y_L, y_R = h_L, h_R

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=style.figsize)

    # --- Ellipse (geometrischer Raum, leise) ---
    ax.fill_between(
        p,
        g(p),
        f(p),
        color=style.area_color,
        alpha=style.area_alpha,
        zorder=0,
    )

    # --- Randkurven (rechnerische Grundlage) ---
    ax.plot(p, f(p), color=style.curve_upper, linewidth=1.0)
    ax.plot(p, g(p), color=style.curve_lower, linewidth=1.0)

    # --- Prognoseintervall (vertikaler Schnitt) ---
    if absolute:
        y_L, y_R = k_L, k_R
    else:
        y_L, y_R = h_L, h_R
        
    ax.vlines(
        model.p,
        y_L,
        y_R,
        color=style.interval_color,
        linewidth=2.5,
        zorder=3,
    )

    # --- Projektion des Intervalls ---
    ax.vlines(
        geometry.x_min,
        y_L,
        y_R,
        color=style.interval_color,
        linewidth=6,
        alpha=0.7,
        zorder=2,
    )

    # --- Hilfslinien (gedacht) ---
    ax.vlines(
        model.p,
        geometry.x_min,
        geometry.x_max,
        linestyle=":",
        color=style.helper_lines,
        linewidth=1.2,
        alpha=0.8,
    )
    
    ax.hlines(
        [y_L, y_R],
        geometry.x_min,
        model.p,
        linestyle=":",
        color=style.helper_lines,
        linewidth=1.2,
        alpha=0.8,
    )

    if absolute:
        ax.vlines(
            model.p,
            0,
            model.n,
            linestyle=":",
            color=style.helper_lines,
            linewidth=1.2,
            alpha=0.8,
            zorder=1,
        )
    # --------------------------------------------------
    # Achsen & Layout
    # --------------------------------------------------
    ax.set_xlim(geometry.x_min, geometry.x_max)

    if absolute:
        ax.set_ylim(0, model.n)
    else:
        ax.set_ylim(geometry.x_min, geometry.x_max)
        
    if not absolute:
        ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(r"$p$", fontsize=12)
    
    if absolute:
        ax.set_ylabel(r"$k$", fontsize=12)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        ax.set_ylabel(r"$h$", fontsize=12)

    if style.grid:
        ax.grid(True, alpha=0.6)

    ax.minorticks_on()
    ax.tick_params(which="major", length=6)

    # --------------------------------------------------
    # Titel
    # --------------------------------------------------
    if absolute:
        title = "Absolutes Prognoseintervall"
        subtitle = rf"Modell: $X \sim \mathrm{{Bin}}({model.n},{model.p}),\; \gamma={model.gamma}$"
    else:
        title = "Relatives Prognoseintervall"
        subtitle = rf"Modell: $H=X/n,\; p={model.p},\; n={model.n},\; \gamma={model.gamma}$"

    ax.text(
        0.5,
        1.09,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
    )

    ax.text(
        0.5,
        1.02,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
    )

    # --------------------------------------------------
    # Ergebniszeile (Äquivalenz explizit!)
    # --------------------------------------------------
    if absolute:
        result_text = (
            rf"{model.gamma*100}%-PI: "
            rf"[{k_L:.2f}; {k_R:.2f}]"
            rf" | äquivalent: "
            rf"[{k_L_int}; {k_R_int}]"
        )
    else:
        result_text = (
            rf"{model.gamma*100}%-PI: "
            rf"$[{h_L:.3f}; {h_R:.3f}]$"
            rf" | äquivalent: "
            rf"{k_L_int}/{model.n} ≤ X/{model.n} ≤ {k_R_int}/{model.n}"
        )

    ax.text(
        0.5,
        -0.18,
        result_text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )

    # --------------------------------------------------
    # Ausgabe
    # --------------------------------------------------
    if save is not None:
        fig.savefig(save, bbox_inches="tight")

    plt.show()
    plt.close(fig)
    return h_L, h_R

# ================ Konfidenzellipse - Wilscon_CI ============

def plot_ci(
    model: CIConfig,
    geometry: IntervalGeometry,
    style: CIStyle,
    *,
    show_interval: bool=True,
    prediction_steps: int = 0,
    save: str | None = None,
    show_info: bool = False,
):
    """
    Geometrische Darstellung von Konfidenzintervallen
    über die Konfidenzellipse (Randkurven f und g).

    - Randkurven: rechnerische Grundlage
    - Ellipse: geometrischer Zusammenhang (leise angedeutet)
    - Schnitte: Entstehung von Intervallen
    """

    p_L = None
    p_R = None
    
    # --------------------------------------------------
    # Geometrie
    # --------------------------------------------------
    p = np.linspace(geometry.x_min, geometry.x_max, geometry.points)

    # --------------------------------------------------
    # Modell (fixiert unter der Modellannahme)
    # --------------------------------------------------
    z = norm.ppf((1 + model.gamma) / 2)

    def f(x):
        return x + z * np.sqrt(x * (1 - x) / model.n)

    def g(x):
        return x - z * np.sqrt(x * (1 - x) / model.n)

    # Konfidenzintervalle
    if show_interval and model.h is not None:
        p_L, p_R = wilson_ci(model.h, model.n, model.gamma)
        p_L_Wald, p_R_Wald = wald_ci(model.h, model.n, model.gamma)

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=style.figsize)

    # --- Ellipse (geometrischer Raum, leise) ---
    ax.fill_between(
        p,
        g(p),
        f(p),
        color=style.area_color,
        alpha=style.area_alpha,
        zorder=0,
    )

    # --- Randkurven (rechnerisch relevant) ---
    ax.plot(p, f(p), color=style.curve_upper, linewidth=1.0)
    ax.plot(p, g(p), color=style.curve_lower, linewidth=1.0)

    # --- Schnitte: Konfidenzintervall ---
    #if show_interval and model.h is not None:
    if p_L is not None:
        ax.hlines(
            model.h,
            p_L,
            p_R,
            linewidth=2.5,
            color=style.interval_color,
            zorder=3,
        )
    
        # --- Projektion des Intervalls ---
        ax.hlines(
            geometry.x_min,
            p_L,
            p_R,
            linewidth=4,
            color=style.interval_color,
            alpha=0.7,
            zorder=2,
        )
    
        # --- Hilfslinien (gedacht, nicht dominant) ---
        ax.hlines(
            model.h,
            geometry.x_min,
            geometry.x_max,
            linestyle=":",
            color="gray",
            linewidth=1.2,
            alpha=0.9,
        )
        ax.vlines(
            [p_L, p_R],
            geometry.x_min,
            model.h,
            linestyle=":",
            color="gray",
            linewidth=1.2,
            alpha=0.9,
        )

    # --- Prognoseintervalle (Schnitte parallel zur y-Achse) ---
    if prediction_steps > 0:
        p_vals = np.linspace(p_L, p_R, prediction_steps)
        for p0 in p_vals:
            li_p, re_p = prediction_interval(p0, model.n, model.gamma)
            ax.vlines(
                p0,
                li_p,
                re_p,
                color=style.helper_lines,
                linewidth=1.2,
                linestyle="--",
                alpha=0.7,
                zorder=2,
            )

    # --------------------------------------------------
    # Achsen & Layout
    # --------------------------------------------------
    ax.set_xlim(geometry.x_min, geometry.x_max)
    ax.set_ylim(geometry.x_min, geometry.x_max)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(r"$p$", fontsize=12)
    ax.set_ylabel(r"$h$", fontsize=12)

    if style.grid:
        ax.grid(True, alpha=0.6)

    ax.minorticks_on()
    ax.tick_params(which="major", length=6)

    # --------------------------------------------------
    # Titel
    # --------------------------------------------------
    ax.text(
        0.5,
        1.09,
        "WILSON-Konfidenzintervall",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
    )

    if p_L is not None:
        subtitle = rf"Setzung: $n={model.n},\; \gamma={model.gamma}$ | Stichprobe: $h={model.h}$"
    else:
        subtitle = rf"Setzung: $n={model.n},\; \gamma={model.gamma}$"

    ax.text(
        0.5,
        1.02,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
    )
      
    # --------------------------------------------------
    # Zusatzinfo (bewusst optional)
    # --------------------------------------------------
    if show_info:
        ax.text(
            0.5,
            -0.18,
            rf"WILSON-KI ≈ [{p_L:.3f}; {p_R:.3f}]  |  Vergleich: WALD-KI ≈ [{p_L_Wald:.3f}; {p_R_Wald:.3f}]",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
        )

    # --------------------------------------------------
    # Ausgabe
    # --------------------------------------------------
    if save is not None:
        fig.savefig(save, bbox_inches="tight")

    plt.show()
    plt.close(fig)
    return p_L, p_R
    
# ============================================================
# 4. Simulation – Meta-Ebene
# ============================================================

# ------------------------------------------------------------
# Simulation: Überdeckungsrate von Konfidenzintervallen
# ------------------------------------------------------------

def plot_ci_simulation(
    cfg: CISimConfig,
    style: CISimStyle,
    *,
    save: str | None = None,
    show: bool = True,
):
    """
    Simulation der Überdeckungsrate von Wilson-Konfidenzintervallen.

    Didaktische Modi
    ----------------
    - color_mode = "uniform"
        → alle Intervalle gleichfarbig
        → geeignet für Zählaufgaben (SuS)

    - color_mode = "coverage"
        → überdeckende / nicht überdeckende Intervalle farblich getrennt

    - show_stats = True / False
        → Ein-/Ausblenden der Überdeckungsrate

    Reproduzierbarkeit
    ------------------
    seed = 42
        → reproduzierbare Referenzsimulation

    seed = None
        → echte Zufallsrealisierung
    """

    # ---------- RNG ----------
    rng = get_rng(cfg.seed)

    # ---------- Simulation ----------
    X = rng.binomial(cfg.n, cfg.p_true, size=cfg.m)
    h = X / cfg.n

    intervals = [wilson_ci(hi, cfg.n, cfg.gamma) for hi in h]
    covered = [(L <= cfg.p_true <= R) for (L, R) in intervals]
    coverage = sum(covered) / cfg.m

    # ---------- Figure ----------
    fig, ax = plt.subplots(figsize=figsize_from_m(cfg.m))
    AXIS_LABEL_FONTSIZE = 12

    ax.set_xlabel(r"$p$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Realisierung", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xlim(cfg.x_min, cfg.x_max)
    ax.set_ylim(-1, cfg.m)

    # ---------- Farben ----------
    if style.color_mode == "uniform":
        colors = [style.color_cover] * cfg.m
    elif style.color_mode == "coverage":
        colors = [
            style.color_cover if c else style.color_miss
            for c in covered
        ]
    else:
        raise ValueError("color_mode must be 'uniform' or 'coverage'")

    # ---------- Intervalle ----------
    for i, ((L, R), col) in enumerate(zip(intervals, colors)):
        ax.hlines(i, L, R, color=col, linewidth=1.2)

    # ---------- Referenzlinie ----------
    ax.axvline(cfg.p_true, color="gray", linestyle="--", linewidth=0.8)

    # ---------- Titel ----------
    set_title_block(
        ax,
        "Simulation: WILSON-Konfidenzintervalle",
        subtitle=rf"$n={cfg.n},\ \gamma={cfg.gamma}$",
    )

    # ---------- Statistik ----------
    if style.show_stats:
        info = (
            rf"{cfg.m} Intervalle | "
            rf"Überdeckungsrate ≈ {coverage*100:.1f}% | "
            rf"Seed = {cfg.seed}"
        )
        set_info_line(ax, info)

    apply_grid(ax, enabled=style.show_grid)

    # ---------- Abschluss ----------
    finalize_figure(fig, save=save, show=show)

    return coverage
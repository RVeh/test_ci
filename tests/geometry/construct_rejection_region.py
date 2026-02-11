# tests/geometry/construct_rejection_region.py
from __future__ import annotations

from typing import Iterable, Set, Protocol

from .rejection_region import RejectionRegion


class DiscreteModel(Protocol):
    """
    Minimales Protokoll für diskrete Modelle,
    wie sie zur Konstruktion von Ablehnungsbereichen benötigt werden.
    """

    @property
    def support(self) -> Iterable[int]:
        ...

    def pmf(self, x: int) -> float:
        ...


# ---------------------------------------------------------------------
# Einseitige Setzungen
# ---------------------------------------------------------------------

def left_tail(model: DiscreteModel, alpha: float) -> RejectionRegion:
    """
    Einseitige Setzung (links):
    K = {0, 1, 2, ...} mit P(X ∈ K) <= alpha.
    """
    prob = 0.0
    K: Set[int] = set()

    for x in sorted(model.support):
        p = model.pmf(x)
        if prob + p > alpha:
            break
        K.add(x)
        prob += p

    return RejectionRegion(model=model, K=K)


def right_tail(model: DiscreteModel, alpha: float) -> RejectionRegion:
    """
    Einseitige Setzung (rechts):
    K = {..., k_max-1, k_max} mit P(X ∈ K) <= alpha.
    """
    prob = 0.0
    K: Set[int] = set()

    for x in sorted(model.support, reverse=True):
        p = model.pmf(x)
        if prob + p > alpha:
            break
        K.add(x)
        prob += p

    return RejectionRegion(model=model, K=K)


# ---------------------------------------------------------------------
# Zweiseitige Setzungen
# ---------------------------------------------------------------------

def two_sided_equal_tails(model: DiscreteModel, alpha: float) -> RejectionRegion:
    """
    Schul-Setzung (equal tails):

    Konstruktion eines zweiseitigen Ablehnungsbereichs mit
    - linker Randmasse <= alpha/2
    - rechter Randmasse <= alpha/2

    Diskretheit wird explizit akzeptiert:
    Die Gesamtmasse ist i. A. < alpha.
    """
    alpha_half = alpha / 2

    # linker Rand
    prob_left = 0.0
    K_left: Set[int] = set()

    for x in sorted(model.support):
        p = model.pmf(x)
        if prob_left + p > alpha_half:
            break
        K_left.add(x)
        prob_left += p

    # rechter Rand
    prob_right = 0.0
    K_right: Set[int] = set()

    for x in sorted(model.support, reverse=True):
        p = model.pmf(x)
        if prob_right + p > alpha_half:
            break
        K_right.add(x)
        prob_right += p

    K = K_left | K_right
    return RejectionRegion(model=model, K=K)


def two_sided_symmetric(model: DiscreteModel, alpha: float) -> RejectionRegion:
    """
    Alternative Setzung (symmetrisch von außen):

    Zweiseitiger Ablehnungsbereich, der abwechselnd
    von links und rechts Masse aufsammelt,
    bis alpha ausgeschöpft ist (<= alpha).

    Diese Setzung ist:
    - symmetrisch gedacht,
    - aber NICHT identisch mit equal tails.
    """
    left = sorted(model.support)
    right = list(reversed(left))

    prob = 0.0
    K: Set[int] = set()

    i = 0
    while i < len(left):
        # linker Rand
        p_left = model.pmf(left[i])
        if prob + p_left <= alpha:
            K.add(left[i])
            prob += p_left
        else:
            break

        # rechter Rand
        p_right = model.pmf(right[i])
        if prob + p_right <= alpha:
            K.add(right[i])
            prob += p_right
        else:
            break

        i += 1

    return RejectionRegion(model=model, K=K)

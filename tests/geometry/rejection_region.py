# tests/geometry/rejection_region.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Set, Protocol


class DiscreteModel(Protocol):
    """
    Minimales Protokoll für diskrete Modelle in der Test-Geometrie.
    """

    @property
    def support(self) -> Iterable[int]:
        ...

    def pmf(self, x: int) -> float:
        ...


@dataclass(frozen=True)
class RejectionRegion:
    """
    Ablehnungsbereich K ⊂ support.
    """
    model: DiscreteModel
    K: Set[int]

    def probability(self) -> float:
        return sum(self.model.pmf(x) for x in self.K)

    def contains(self, x: int) -> bool:
        return x in self.K

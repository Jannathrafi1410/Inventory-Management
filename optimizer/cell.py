"""Decision cell economics for inventory pre-buy optimization.

Assumption: `cap` is the maximum purchasable/committable units for the cell (supplier cap when
provided, else physical capacity). Monetary values are in Rs-equivalent simulator units.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Cell:
    """A pre-buy decision cell with contract and demand-fallback economics."""

    id: str
    p: float
    c_d: float
    c_b: float
    k: float
    r: float
    a: float
    g: float
    cap: int
    contract_type: str
    group: str

    def delta_m(self) -> float:
        """Margin advantage over the dynamic fallback."""
        return (self.p - self.c_b) - self.a * (self.p - self.c_d) + (1.0 - self.a) * self.g

    def c_o(self) -> float:
        """Cash destroyed per unsold unit."""
        return self.k * (1.0 - self.r)

    def carry(self, lock_days: int = 120, r_annual: float = 0.12) -> float:
        """Carry cost per unit on locked cash."""
        return self.k * r_annual * lock_days / 365.0


__all__ = ["Cell"]

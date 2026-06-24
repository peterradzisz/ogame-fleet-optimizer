"""Player research-tech levels (Weapon / Shield / Armor).

In OGame combat, each research level multiplies the corresponding base stat
by ``(10 + level) / 10``.  Level 10 therefore doubles a unit's damage,
shield, or hull — the canonical "I just researched one tier" payoff.

This dataclass has **no upper cap**.  Live-game universes stop at level
100, but the optimizer should not impose its own ceiling (custom / test
universes, future-proofing).  Python ints are unbounded anyway.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_TECH_FIELDS = ("weapon", "shield", "armor")


def _validate_tech_value(name: str, value: Any) -> None:
    """Reject anything that isn't a non-negative ``int`` (no floats, no bools)."""
    # ``bool`` is a subclass of ``int`` in Python; reject explicitly so
    # ``TechLevels(weapon=True)`` fails loud instead of silently becoming 1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{name} must be a non-negative int, got {type(value).__name__}"
        )
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


@dataclass
class TechLevels:
    """Player research levels for the three combat technologies.

    All three default to ``0`` — the case for a brand-new account with no
    research.  Mutation after construction is also validated, so callers
    cannot accidentally bypass the bound check by re-assigning fields.
    """

    weapon: int = 0
    shield: int = 0
    armor: int = 0

    def __post_init__(self) -> None:
        for name in _TECH_FIELDS:
            _validate_tech_value(name, getattr(self, name))

    # ``__setattr__`` catches post-init tampering (e.g. ``t.weapon = -5``)
    # that ``__post_init__`` alone would miss.
    def __setattr__(self, name: str, value: Any) -> None:
        if name in _TECH_FIELDS:
            _validate_tech_value(name, value)
        super().__setattr__(name, value)

    # ----- OGame combat multiplier helpers -----

    def weapon_multiplier(self) -> float:
        """Return ``(10 + weapon) / 10`` — applied to base attack power."""
        return (10 + self.weapon) / 10.0

    def shield_multiplier(self) -> float:
        """Return ``(10 + shield) / 10`` — applied to base shield power."""
        return (10 + self.shield) / 10.0

    def armor_multiplier(self) -> float:
        """Return ``(10 + armor) / 10`` — applied to base hull / armor."""
        return (10 + self.armor) / 10.0


__all__ = ["TechLevels"]
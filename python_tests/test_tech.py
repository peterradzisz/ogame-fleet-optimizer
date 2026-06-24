"""TDD tests for the player/research-tech module.

These tests cover ``TechLevels``, the player's research levels (Weapon /
Shield / Armor).  In OGame combat, each level multiplies the corresponding
base stat by ``(10 + level) / 10``.

The Python optimizer consumes these levels so it can size counter-fleets
that actually defeat the enemy — there is intentionally no upper cap.
"""
from __future__ import annotations

import pytest

from ogame_optimizer.core.tech import TechLevels


# ---------------------------------------------------------------------------
# Default construction
# ---------------------------------------------------------------------------


def test_default_zero() -> None:
    # A freshly constructed TechLevels has every field at zero — matches a
    # brand-new account with no research.
    t = TechLevels()
    assert t.weapon == 0
    assert t.shield == 0
    assert t.armor == 0


def test_explicit_values_stored() -> None:
    t = TechLevels(weapon=5, shield=7, armor=3)
    assert t.weapon == 5
    assert t.shield == 7
    assert t.armor == 3


def test_equality_is_value_based() -> None:
    # Two TechLevels with the same values should compare equal (dataclass Eq).
    assert TechLevels(weapon=2, shield=3, armor=4) == TechLevels(weapon=2, shield=3, armor=4)


# ---------------------------------------------------------------------------
# Validation — non-negative integers
# ---------------------------------------------------------------------------


def test_negative_rejected() -> None:
    # Plan QA scenario: negative level is meaningless and must raise.
    with pytest.raises(ValueError):
        TechLevels(weapon=-1)
    with pytest.raises(ValueError):
        TechLevels(shield=-1)
    with pytest.raises(ValueError):
        TechLevels(armor=-1)


def test_negative_via_post_init() -> None:
    # Field is constructed then tampered with — __post_init__ must catch it
    # even when set after construction (defensive design).
    t = TechLevels()
    with pytest.raises(ValueError):
        t.weapon = -5  # type: ignore[misc]


def test_zero_accepted() -> None:
    # Boundary: zero is the minimum legal level.
    t = TechLevels(weapon=0, shield=0, armor=0)
    assert (t.weapon, t.shield, t.armor) == (0, 0, 0)


# ---------------------------------------------------------------------------
# No upper cap — late-game players have very high tech levels
# ---------------------------------------------------------------------------


def test_max_values_allowed() -> None:
    # OGame caps research at level 100 in the live game, but the optimizer
    # should not impose its own ceiling (custom / test universes go higher).
    t = TechLevels(weapon=100, shield=100, armor=100)
    assert (t.weapon, t.shield, t.armor) == (100, 100, 100)


def test_very_large_values_allowed() -> None:
    # No upper bound whatsoever — Python ints are unbounded anyway.
    t = TechLevels(weapon=10_000, shield=10_000, armor=10_000)
    assert t.weapon == 10_000


# ---------------------------------------------------------------------------
# Defence uses these multipliers: (10 + level) / 10.  Spot-check the math
# lives on the dataclass as a method so the optimizer can call it directly.
# ---------------------------------------------------------------------------


def test_multiplier_at_zero_is_one() -> None:
    t = TechLevels()
    assert t.weapon_multiplier() == pytest.approx(1.0)
    assert t.shield_multiplier() == pytest.approx(1.0)
    assert t.armor_multiplier() == pytest.approx(1.0)


def test_multiplier_at_level_ten_is_two() -> None:
    # OGame's canonical "double-damage" level is 10.
    t = TechLevels(weapon=10, shield=10, armor=10)
    assert t.weapon_multiplier() == pytest.approx(2.0)
    assert t.shield_multiplier() == pytest.approx(2.0)
    assert t.armor_multiplier() == pytest.approx(2.0)
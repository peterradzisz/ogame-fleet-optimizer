"""TDD tests for the fleet/budget module.

These tests cover the Python-side representation of OGame fleet and defense
compositions, budget calculation against the enemy, and multiplier validation.

Test keys are lowercase snake_case (e.g. ``light_fighter``), matching the
public Python API surface. The Rust core uses PascalCase identifiers
(``LightFighter``); bridging is handled by Task 6 (PyO3).

Cost constants are intentionally hard-coded in the assertions below so that
any drift between ``fleet.py`` and ``src/ships.rs`` is caught immediately.
"""
from __future__ import annotations

import pytest

from ogame_optimizer.core.fleet import (
    DEFENSES_COST,
    SHIPS_COST,
    Fleet,
    compute_budget,
    fleet_value,
    validate_fleet_in_budget,
    validate_multiplier,
)


# ---------------------------------------------------------------------------
# Known-cost sanity: the SHIPS_COST / DEFENSES_COST dicts must match
# src/ships.rs line for line.  If these break, the optimizer's budget math
# has drifted from the Rust combat simulator.
# ---------------------------------------------------------------------------


def test_ships_cost_matches_rust_source_of_truth() -> None:
    assert SHIPS_COST["small_cargo"] == (2_000, 2_000, 0)
    assert SHIPS_COST["large_cargo"] == (6_000, 6_000, 0)
    assert SHIPS_COST["light_fighter"] == (3_000, 1_000, 0)
    assert SHIPS_COST["heavy_fighter"] == (6_000, 4_000, 0)
    assert SHIPS_COST["cruiser"] == (20_000, 7_000, 2_000)
    assert SHIPS_COST["battleship"] == (45_000, 15_000, 0)
    assert SHIPS_COST["battlecruiser"] == (30_000, 40_000, 15_000)
    assert SHIPS_COST["bomber"] == (50_000, 25_000, 15_000)
    assert SHIPS_COST["destroyer"] == (60_000, 50_000, 15_000)
    assert SHIPS_COST["deathstar"] == (5_000_000, 4_000_000, 1_000_000)
    assert SHIPS_COST["espionage_probe"] == (0, 1_000, 0)


def test_defenses_cost_matches_rust_source_of_truth() -> None:
    assert DEFENSES_COST["rocket_launcher"] == (2_000, 0, 0)
    assert DEFENSES_COST["light_laser"] == (1_500, 500, 0)
    assert DEFENSES_COST["heavy_laser"] == (6_000, 2_000, 0)
    assert DEFENSES_COST["gauss_cannon"] == (20_000, 15_000, 2_000)
    assert DEFENSES_COST["ion_cannon"] == (5_000, 3_000, 0)
    assert DEFENSES_COST["plasma_turret"] == (50_000, 50_000, 30_000)
    assert DEFENSES_COST["small_shield_dome"] == (10_000, 10_000, 0)
    assert DEFENSES_COST["large_shield_dome"] == (50_000, 50_000, 0)


# ---------------------------------------------------------------------------
# Fleet dataclass
# ---------------------------------------------------------------------------


def test_fleet_dataclass_defaults_empty() -> None:
    fleet = Fleet()
    assert fleet.ships == {}
    assert fleet.total_cost() == 0


def test_fleet_dataclass_holds_ship_counts() -> None:
    fleet = Fleet(ships={"light_fighter": 100, "battleship": 5})
    assert fleet.ships["light_fighter"] == 100
    assert fleet.ships["battleship"] == 5


def test_fleet_total_cost_sums_metal_crystal_deuterium() -> None:
    # LF: 4,000 × 100 = 400,000
    # BS: 60,000 × 5 = 300,000
    fleet = Fleet(ships={"light_fighter": 100, "battleship": 5})
    assert fleet.total_cost() == 700_000


def test_fleet_rejects_negative_ship_counts() -> None:
    with pytest.raises(ValueError):
        Fleet(ships={"light_fighter": -1})


def test_fleet_rejects_unknown_ship_type() -> None:
    with pytest.raises(ValueError):
        Fleet(ships={"atlantis": 1})


# ---------------------------------------------------------------------------
# fleet_value()
# ---------------------------------------------------------------------------


def test_fleet_value_empty_zero() -> None:
    assert fleet_value({}) == 0
    assert fleet_value({}, {}) == 0


def test_fleet_value_ships_only() -> None:
    # 100 LF × 4,000 = 400,000
    assert fleet_value({"light_fighter": 100}) == 400_000


def test_fleet_value_includes_defenses_when_provided() -> None:
    # 100 LF × 4,000 = 400,000
    #  50 RL × 2,000 = 100,000
    assert fleet_value({"light_fighter": 100}, {"rocket_launcher": 50}) == 500_000


def test_fleet_value_unknown_ship_raises() -> None:
    with pytest.raises(ValueError):
        fleet_value({"made_up_ship": 10})


def test_fleet_value_unknown_defense_raises() -> None:
    with pytest.raises(ValueError):
        fleet_value({}, {"made_up_defense": 10})


# ---------------------------------------------------------------------------
# compute_budget() — the headline feature
# ---------------------------------------------------------------------------


def test_budget_with_defenses() -> None:
    # Plan QA scenario (lines 530-539):
    # enemy_fleet={"light_fighter": 100}, enemy_defenses={"rocket_launcher": 50},
    # multiplier=1.5 → (100×4000 + 50×2000) × 1.5 = 750_000
    budget = compute_budget(
        enemy_fleet={"light_fighter": 100},
        enemy_defenses={"rocket_launcher": 50},
        multiplier=1.5,
    )
    assert budget == 750_000


def test_zero_enemy_budget_zero() -> None:
    # No enemy ships and no defenses → zero raw value → zero budget regardless of multiplier.
    assert compute_budget({}, {}, 1.0) == 0
    assert compute_budget({}, {}, 2.5) == 0


def test_budget_with_only_defenses() -> None:
    # 100 RL × 2,000 × 1.0 = 200_000
    assert compute_budget({}, {"rocket_launcher": 100}, 1.0) == 200_000


def test_budget_with_only_fleet() -> None:
    # 10 BS × 60,000 × 1.0 = 600_000
    assert compute_budget({"battleship": 10}, None, 1.0) == 600_000


def test_budget_default_multiplier_is_one() -> None:
    # (100 LF × 4000) × 1.0 = 400_000
    assert compute_budget({"light_fighter": 100}) == 400_000


# ---------------------------------------------------------------------------
# validate_multiplier() — 0.1-step grid
# ---------------------------------------------------------------------------


def test_multiplier_steps_accepts_grid_values() -> None:
    for m in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0):
        # No raise.
        validate_multiplier(m)


def test_multiplier_steps_rejects_off_grid_values() -> None:
    for m in (0.37, 0.55, 1.25, 2.15, 3.77):
        with pytest.raises(ValueError):
            validate_multiplier(m)


def test_negative_multiplier_rejected() -> None:
    with pytest.raises(ValueError):
        validate_multiplier(-0.5)
    with pytest.raises(ValueError):
        validate_multiplier(-1.0)


def test_zero_multiplier_rejected() -> None:
    with pytest.raises(ValueError):
        validate_multiplier(0.0)


def test_multiplier_grid_handles_float_precision() -> None:
    # 0.1 + 0.1 + 0.1 + 0.1 + 0.1 == 0.5 in IEEE 754 but the value the user typed
    # may end up as 0.4999999... — validate_multiplier must tolerate this.
    accumulated = 0.1 + 0.1 + 0.1 + 0.1 + 0.1  # 0.5 with rounding noise
    validate_multiplier(accumulated)  # no raise


# ---------------------------------------------------------------------------
# validate_fleet_in_budget()
# ---------------------------------------------------------------------------


def test_validate_fleet_in_budget_returns_true_when_under() -> None:
    fleet = {"light_fighter": 10}  # 40,000
    assert validate_fleet_in_budget(fleet, 100_000) is True


def test_validate_fleet_in_budget_returns_true_when_equal() -> None:
    fleet = {"light_fighter": 10}  # 40,000
    assert validate_fleet_in_budget(fleet, 40_000) is True


def test_validate_fleet_in_budget_returns_false_when_over() -> None:
    fleet = {"light_fighter": 100}  # 400,000
    assert validate_fleet_in_budget(fleet, 100_000) is False


# ---------------------------------------------------------------------------
# RIP / Deathstar large-number arithmetic
# ---------------------------------------------------------------------------


def test_rip_value() -> None:
    # Plan QA scenario (lines 553-562):
    # {"deathstar": 100} → 100 × 10,000,000 = 1,000,000,000
    assert fleet_value({"deathstar": 100}) == 1_000_000_000


def test_rip_budget() -> None:
    # 100 RIP × 10,000,000 × 1.5 = 1,500,000,000
    assert compute_budget({"deathstar": 100}, None, 1.5) == 1_500_000_000


def test_rip_does_not_overflow_python_int() -> None:
    # 100,000 RIP × 10,000,000 = 1e12 — fits trivially in Python int.
    assert fleet_value({"deathstar": 100_000}) == 1_000_000_000_000
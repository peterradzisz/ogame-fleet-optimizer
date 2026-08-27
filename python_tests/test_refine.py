"""Regression tests for the Refine feature (seed_fleet re-optimization).

Covers:
- Fix 1: base_fleet mode must not double-count the base fleet when the
  refine seed is the previously recommended MERGED fleet (base+additions).
- Fix 2: an over-budget refine seed must be scaled down at adoption time
  (Phase A), so its 200-sim validated loss cannot benchmark budget-
  constrained GA candidates against an infeasible fleet.
"""
from __future__ import annotations
import pytest

import ogame_optimizer.optimizer.orchestration as orch
from ogame_optimizer.core.fleet import compute_budget, fleet_value
from ogame_optimizer.optimizer.orchestration import optimize


@pytest.fixture
def captured_fleets(monkeypatch):
    """Record every attacker fleet orchestration sends into simulate_batch."""
    fleets = []
    real = orch.simulate_batch

    def wrapper(attacker, **kwargs):
        fleets.append(dict(attacker))
        return real(attacker=attacker, **kwargs)

    monkeypatch.setattr(orch, "simulate_batch", wrapper)
    return fleets


# ga_time_budget below the 1.5s Phase-C threshold and with explore/refine
# rounds too short to run (<0.5s each) -> the Phase-A seed passes through
# validation and the final result almost deterministically.
_QUICK = dict(ga_time_budget=0.4, final_sims=50)


def test_refine_base_mode_does_not_double_count_base(captured_fleets):
    """base_fleet + merged seed: base counts must be stripped from the seed
    so the base is not merged a second time (100 LFs, never 200)."""
    result = optimize(
        enemy_fleet={"light_fighter": 500, "cruiser": 50},
        enemy_defenses={},
        enemy_tech=(0, 0, 0),
        attacker_tech=(0, 0, 0),
        budget_multiplier=1.0,
        mode="attack",
        base_seed=42,
        base_fleet={"light_fighter": 100},
        seed_fleet={"light_fighter": 100, "cruiser": 50},  # UI sends merged fleet
        **_QUICK,
    )
    # Greedy baseline validation (2nd sim call; the base-already-wins check
    # runs first) must see base + seed-additions: exactly 100 LFs.
    assert captured_fleets[1] == {"light_fighter": 100, "cruiser": 50}
    for fleet in captured_fleets:
        assert fleet.get("light_fighter", 0) <= 100, f"base double-counted: {fleet}"
    # Result reports the base once and only the seed's additions.
    assert result.recommended_fleet.get("light_fighter") == 100
    assert result.recommended_additions.get("cruiser") == 50
    assert result.recommended_additions.get("light_fighter", 0) == 0


def test_refine_seed_over_budget_scaled_at_adoption(captured_fleets, caplog):
    """Over-budget seed: scaled to budget in Phase A, BEFORE validation, so
    every combat evaluation (including the greedy benchmark that gates GA
    candidates) is budget-feasible."""
    enemy = {"light_fighter": 100}  # budget = 400_000
    budget = compute_budget(enemy, {}, 1.0)
    with caplog.at_level("WARNING", logger="ogame.optimizer.orchestration"):
        result = optimize(
            enemy_fleet=enemy,
            enemy_defenses={},
            enemy_tech=(0, 0, 0),
            attacker_tech=(0, 0, 0),
            budget_multiplier=1.0,
            mode="attack",
            base_seed=42,
            seed_fleet={"cruiser": 100},  # 2_900_000 >> budget
            **_QUICK,
        )
    assert any("refine seed over budget" in rec.getMessage() for rec in caplog.records)
    # No fleet evaluated (incl. the greedy benchmark) exceeds the budget.
    for fleet in captured_fleets:
        assert fleet_value(fleet) <= budget, f"over-budget fleet evaluated: {fleet}"
    # int()-truncated proportional scale: int(100 * 400k / 2.9M) == 13 cruisers.
    assert result.recommended_fleet.get("cruiser") == 13


def test_refine_base_mode_seed_additions_scaled_to_budget(captured_fleets):
    """base mode: only the seed's ADDITIONS are scaled to the additions
    budget; the base fleet is preserved untouched."""
    enemy = {"battleship": 20}  # additions budget = 1_200_000
    result = optimize(
        enemy_fleet=enemy,
        enemy_defenses={},
        enemy_tech=(0, 0, 0),
        attacker_tech=(0, 0, 0),
        budget_multiplier=1.0,
        mode="attack",
        base_seed=42,
        base_fleet={"light_fighter": 100},
        seed_fleet={"light_fighter": 100, "cruiser": 100},  # additions 2.9M > budget
        **_QUICK,
    )
    budget = compute_budget(enemy, {}, 1.0)
    base_cost = fleet_value({"light_fighter": 100})
    # alternatives (Option B/C) simulate extra base-merged fleets after the primary; base preservation only holds for primary-path captures
    for fleet in captured_fleets:
        assert fleet.get("light_fighter", 0) == 100  # base preserved exactly
        assert fleet_value(fleet) - base_cost <= budget, f"additions over budget: {fleet}"
        if fleet == result.recommended_fleet:
            break  # primary final fleet simulated: everything after is alternatives
    # Additions scaled down: int(100 * 1.2M / 2.9M) == 41 cruisers max.
    assert result.recommended_fleet.get("light_fighter") == 100
    assert result.recommended_additions.get("cruiser", 0) <= 41


def test_refine_seed_equal_to_base_falls_back_gracefully(captured_fleets, caplog):
    """Seed == base (nothing to add): normalises to an empty additions seed
    and falls back to a fresh greedy optimisation instead of adopting an
    empty (or double-counted) fleet."""
    with caplog.at_level("WARNING", logger="ogame.optimizer.orchestration"):
        result = optimize(
            enemy_fleet={"light_fighter": 500, "cruiser": 50},
            enemy_defenses={},
            enemy_tech=(0, 0, 0),
            attacker_tech=(0, 0, 0),
            budget_multiplier=1.0,
            mode="attack",
            base_seed=42,
            base_fleet={"light_fighter": 100},
            seed_fleet={"light_fighter": 100},  # == base -> additions empty
            **_QUICK,
        )
    assert any("refine seed empty after normalisation" in rec.getMessage()
               for rec in caplog.records)
    # Fresh greedy ran and produced a usable fleet; base never doubled.
    assert sum(result.recommended_fleet.values()) > 0
    for fleet in captured_fleets:
        # Base (100) + greedy additions may add LFs, but never 200 for free.
        assert fleet.get("light_fighter", 0) != 200 or fleet_value(fleet) > fleet_value({"light_fighter": 200})

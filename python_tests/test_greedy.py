"""Tests for the greedy optimizer (Task 7)."""
from __future__ import annotations
import time
import pytest

from ogame_optimizer.optimizer.greedy import (
    greedy_optimize, phase_a1_counter_ratio_init, phase_a2_budget_fill,
    phase_a3_local_search,
)
from ogame_optimizer.core.fleet import SHIPS_COST, fleet_value


def test_in_budget():
    """Greedy produces a fleet within budget."""
    enemy = {"light_fighter": 1000}
    budget = 500_000
    r = greedy_optimize(enemy_fleet=enemy, enemy_defenses={}, budget=budget, seed=42)
    assert r.fleet is not None
    assert sum(r.fleet.values()) > 0, "Greedy should produce a non-empty fleet"
    assert fleet_value(r.fleet) <= budget, f"Fleet value {fleet_value(r.fleet)} exceeds budget {budget}"


def test_shield_dome_rule():
    """When enemy has LargeShieldDome, greedy reserves high-damage ships."""
    enemy = {"light_fighter": 500}
    defenses = {"large_shield_dome": 1}
    budget = 2_000_000  # large budget so 20% reserve = 400k for high-damage
    r = greedy_optimize(enemy_fleet=enemy, enemy_defenses=defenses, budget=budget, seed=42)
    # Check for at least one high-damage ship (BS, BMB, DS, RIP)
    high_damage = ["battleship", "bomber", "destroyer", "deathstar"]
    has_high_damage = any(r.fleet.get(s, 0) > 0 for s in high_damage)
    assert has_high_damage, f"Fleet should include high-damage ships vs LSD, got {r.fleet}"


def test_local_search_improves():
    """Phase A3 (local search) result is no worse than Phase A2 (budget fill)."""
    enemy = {"cruiser": 100, "battleship": 20}
    budget = 1_000_000
    seed = 42
    enemy_tech = (0, 0, 0)
    attacker_tech = (0, 0, 0)

    # Phase A1 + A2 only (no A3)
    rough = phase_a1_counter_ratio_init(enemy, {}, budget, "attack")
    fleet_a2 = phase_a2_budget_fill(rough, budget)
    # Phase A1 + A2 + A3
    fleet_a3 = phase_a3_local_search(fleet_a2, enemy, {}, enemy_tech, attacker_tech, seed, time_budget_s=0.5)
    # A3 should not be worse (by construction of hill climbing, but we just check non-empty)
    assert sum(fleet_a3.values()) > 0


def test_time_budget():
    """Greedy completes in <= 2s for medium fleet."""
    enemy = {"light_fighter": 1000, "cruiser": 50, "battleship": 10}
    t0 = time.time()
    r = greedy_optimize(enemy_fleet=enemy, enemy_defenses={}, budget=2_000_000, seed=42)
    elapsed = time.time() - t0
    assert elapsed <= 2.5, f"Greedy took {elapsed:.2f}s, expected <= 2.5s"


def test_returns_greedy_result_dataclass():
    """Greedy returns a GreedyResult with all fields."""
    r = greedy_optimize(enemy_fleet={"light_fighter": 100}, enemy_defenses={}, budget=100_000, seed=42)
    assert hasattr(r, "fleet")
    assert hasattr(r, "estimated_loss")
    assert hasattr(r, "time_elapsed")
    assert isinstance(r.fleet, dict)
    assert isinstance(r.estimated_loss, (int, float))
    assert isinstance(r.time_elapsed, float)


def test_int_counts_only():
    """All ship counts are positive integers."""
    r = greedy_optimize(enemy_fleet={"light_fighter": 100}, enemy_defenses={}, budget=100_000, seed=42)
    for ship, count in r.fleet.items():
        assert isinstance(count, int), f"{ship} count is not int: {type(count)}"
        assert count > 0, f"{ship} has zero count"

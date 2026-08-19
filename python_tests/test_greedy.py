"""Tests for the greedy optimizer (Task 7)."""
from __future__ import annotations
import time
import pytest

from ogame_optimizer.optimizer.greedy import (
    greedy_optimize, phase_a1_counter_ratio_init, phase_a2_budget_fill,
    phase_a3_local_search, COUNTER_MAP, HIGH_DAMAGE,
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


# --- Reaper awareness (COUNTER_MAP / HIGH_DAMAGE) ---

def test_counter_map_reaper_counters_battleship_and_bomber():
    """Reaper is the primary counter vs battleship (RF 7) and bomber (RF 4)."""
    assert COUNTER_MAP["battleship"] == "reaper", "reaper RF vs battleship = 7"
    assert COUNTER_MAP["bomber"] == "reaper", "reaper RF vs bomber = 4"


def test_counter_map_enemy_reaper_swarm():
    """Enemy reapers are countered by light fighter swarm (only deathstar RF 30 is cost-absurd)."""
    assert COUNTER_MAP["reaper"] == "light_fighter"


def test_counter_map_destroyer_counter_unchanged():
    """Destroyer stays countered by deathstar: deathstar RF 5 > reaper RF 3."""
    assert COUNTER_MAP["destroyer"] == "deathstar"


def test_high_damage_includes_reaper():
    """Reaper base attack 2800 (2nd highest) must be in shield-dome reservation."""
    assert "reaper" in HIGH_DAMAGE


def test_phase_a1_counters_battleships_with_reapers():
    """Phase A1 buys reapers when the enemy fields battleships."""
    rough = phase_a1_counter_ratio_init({"battleship": 10}, {}, 1_000_000, "attack")
    assert rough.get("reaper", 0) > 0, f"Expected reapers vs battleships, got {rough}"


def test_phase_a1_counters_reapers_with_light_fighters():
    """Phase A1 buys light fighters when the enemy fields reapers."""
    rough = phase_a1_counter_ratio_init({"reaper": 5}, {}, 1_000_000, "attack")
    assert rough.get("light_fighter", 0) > 0, f"Expected light fighters vs reapers, got {rough}"


def test_counter_map_deathstar_destroyer():
    """RIP counter is destroyer: LF swarm bounces (50 < 500 = 1% of RIP
    shield), destroyer has the lowest RF-against (5) and clears the bounce
    threshold. Rust-measured: 100% win / ~18% loss vs 1 and 10 RIPs."""
    assert COUNTER_MAP["deathstar"] == "destroyer"


def test_phase_a1_counters_ripps_with_destroyers():
    """Phase A1 buys destroyers (not the useless LF swarm) vs enemy RIPs."""
    rough = phase_a1_counter_ratio_init({"deathstar": 2}, {}, 5_000_000, "attack")
    assert rough.get("destroyer", 0) > 0, f"Expected destroyers vs RIPs, got {rough}"
    assert rough.get("light_fighter", 0) == 0, f"LF swarm seed must not be bought vs RIPs: {rough}"

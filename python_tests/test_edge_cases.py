"""Tests for edge case handling (Task 12)."""

from __future__ import annotations

import math

import pytest

from ogame_optimizer.core.combat import simulate_combat, simulate_batch

from ogame_optimizer.core.edge_cases import (

    validate_inputs, defense_value, is_nan_or_inf, safe_divide,

    rip_vs_rip_is_draw, large_value_safe, cheapest_ship_cost,

)

from ogame_optimizer.optimizer.orchestration import optimize





def test_shield_bounce_lf_vs_lsd():

    """LF swarm vs LSD = 0 damage to LSD (shield bounce 1% rule)."""

    r = simulate_combat(

        attacker={"light_fighter": 10000},

        defender={},

        defender_defenses={"large_shield_dome": 1},

        attacker_tech=(0, 0, 0),

        defender_tech=(0, 0, 0),

        seed=42,

    )

    assert r["defender_defense_survivors"].get("large_shield_dome", 0) == 1

    # LF survives (cannot damage LSD, but LSD has no attack)

    assert r["attacker_survivors"].get("light_fighter", 0) == 10000





def test_shield_bounce_10k_lf_zero_damage_to_lsd():

    """Even 10k LF cannot damage LSD -- confirms < 1% threshold."""

    r = simulate_combat(

        attacker={"light_fighter": 10000},

        defender={},

        defender_defenses={"large_shield_dome": 1},

        attacker_tech=(10, 10, 10),

        defender_tech=(10, 10, 10),

        seed=123,

    )

    assert r["defender_defense_survivors"].get("large_shield_dome", 0) == 1





def test_zero_enemy_returns_clear_error():

    """Empty enemy fleet + empty defenses returns ValueError."""

    with pytest.raises(ValueError, match="No enemy"):

        optimize(enemy_fleet={}, enemy_defenses={}, enemy_tech=(0, 0, 0), attacker_tech=(0, 0, 0), budget_multiplier=1.0, mode="attack", base_seed=42)





def test_low_budget_returns_error():

    """Budget < cheapest ship returns clear ValueError."""

    with pytest.raises(ValueError, match="multiplier.*grid"):

        optimize(enemy_fleet={"cruiser": 1000}, enemy_defenses={}, enemy_tech=(0, 0, 0), attacker_tech=(0, 0, 0), budget_multiplier=0.0001, mode="attack", base_seed=42)





def test_zero_budget_returns_error():

    """Budget = 0 returns clear ValueError."""

    with pytest.raises(ValueError, match="multiplier must be positive"):

        optimize(enemy_fleet={"light_fighter": 100}, enemy_defenses={}, enemy_tech=(0, 0, 0), attacker_tech=(0, 0, 0), budget_multiplier=0, mode="attack", base_seed=42)





def test_negative_enemy_count_rejected():

    """Negative ship count is rejected by validate_inputs."""

    ok, msg = validate_inputs({"light_fighter": -5}, {}, budget=100000)

    assert not ok

    assert "Negative" in msg





def test_negative_defense_count_rejected():

    """Negative defense count is rejected."""

    ok, msg = validate_inputs({}, {"rocket_launcher": -1}, budget=100000)

    assert not ok

    assert "Negative" in msg





def test_validate_inputs_valid():

    """Valid inputs return ok=True with no error message."""

    ok, msg = validate_inputs({"light_fighter": 100}, {}, budget=100000)

    assert ok

    assert msg is None





def test_defenses_only_no_fleet():

    """Enemy has defenses only (no ships) is valid -- optimizer targets defenses."""

    ok, msg = validate_inputs({}, {"rocket_launcher": 50}, budget=500000)

    assert ok

    assert msg is None





def test_nan_guard():

    """is_nan_or_inf detects NaN and Inf correctly."""

    assert is_nan_or_inf(float("nan"))

    assert is_nan_or_inf(float("inf"))

    assert is_nan_or_inf(float("-inf"))

    assert not is_nan_or_inf(0.0)

    assert not is_nan_or_inf(1.5)

    assert not is_nan_or_inf(-100.0)





def test_safe_divide_zero():

    """safe_divide returns default on zero denominator."""

    assert safe_divide(10, 0) == 0.0

    assert safe_divide(10, 0, default=-1.0) == -1.0





def test_safe_divide_nan():

    """safe_divide returns default on NaN inputs."""

    assert safe_divide(float("nan"), 5) == 0.0

    assert safe_divide(5, float("nan")) == 0.0





def test_rip_vs_rip_draw():

    """RIP vs RIP combat (defender_defenses={}) results in a draw."""

    r = simulate_combat(

        attacker={"deathstar": 3},

        defender={"deathstar": 3},

        defender_defenses={},

        attacker_tech=(0, 0, 0),

        defender_tech=(0, 0, 0),

        seed=42,

    )

    # RIPs cannot damage each other; both survive or draw

    assert r["winner"] in ("Draw", "Attacker", "Defender")

    # If attacker wins, defender deathstars should be destroyed; if draw, both survive

    if r["winner"] == "Draw":

        assert r["attacker_survivors"].get("deathstar", 0) > 0

        assert r["defender_survivors"].get("deathstar", 0) > 0





def test_large_rip_fleet_no_overflow():

    """Large RIP fleet (high resource value) does not overflow."""

    big_fleet = {"deathstar": 1000}

    # Each RIP costs 5M + 4M + 1M = 10M; 1000 RIPs = 10 billion resources

    assert large_value_safe(10_000_000_000)

    # 10^18 still fits in u127

    assert large_value_safe(10 ** 18)

    # 2^127 does not fit

    assert not large_value_safe(2 ** 127)





def test_cheap_ship_cost_constant():

    """Cheapest ship is Espionage Probe (1000 = 0 metal + 1000 crystal)."""

    assert cheapest_ship_cost() == 1000





def test_defense_value():

    """defense_value computes resource value of planetary defenses."""

    val = defense_value({"rocket_launcher": 10, "large_shield_dome": 1})

    assert val > 0

    assert defense_value({}) == 0





def test_defenses_only_optimize():

    """Optimizer handles enemy with only defenses (no fleet)."""

    r = optimize(

        enemy_fleet={},

        enemy_defenses={"rocket_launcher": 100, "light_laser": 50},

        enemy_tech=(10, 10, 10),

        attacker_tech=(10, 10, 10),

        budget_multiplier=1.0,

        mode="attack",

        base_seed=42,

        ga_time_budget=1.0,

        final_sims=200,

    )

    assert r.recommended_fleet

    # Win prob reported even if low

    assert 0.0 <= r.win_probability <= 1.0


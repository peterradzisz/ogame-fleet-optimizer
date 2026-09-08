"""Tests for the optimizer orchestration (Task 10)."""
from __future__ import annotations
import pytest
from ogame_optimizer.optimizer.orchestration import optimize, OptimizationResult


def test_end_to_end_attack():
    r = optimize(enemy_fleet={"light_fighter": 200, "cruiser": 30}, enemy_defenses={}, enemy_tech=(10, 10, 10), attacker_tech=(10, 10, 10), budget_multiplier=1.5, mode="attack", base_seed=42, ga_time_budget=1.0, final_sims=300)
    assert isinstance(r, OptimizationResult)
    assert isinstance(r.recommended_fleet, dict)
    assert len(r.recommended_fleet) > 0
    assert all(isinstance(v, int) and v > 0 for v in r.recommended_fleet.values())
    assert r.expected_loss_mean >= 0
    assert 0.0 <= r.win_probability <= 1.0
    assert len(r.confidence_interval_95) == 2
    assert r.confidence_interval_95[0] <= r.expected_loss_mean <= r.confidence_interval_95[1]
    assert r.sims_run_final >= 300
    assert r.mode == "attack"
    assert r.total_time > 0


def test_end_to_end_defend():
    r = optimize(enemy_fleet={"light_fighter": 100}, enemy_defenses={"rocket_launcher": 50}, enemy_tech=(10, 10, 10), attacker_tech=(10, 10, 10), budget_multiplier=1.0, mode="defend", base_seed=42, ga_time_budget=1.0, final_sims=300)
    assert isinstance(r, OptimizationResult)
    assert r.mode == "defend"
    assert r.recommended_fleet


def test_dual_mode_differs():
    enemy = {"light_fighter": 200, "cruiser": 30}
    tech = (10, 10, 10)
    r_attack = optimize(enemy_fleet=enemy, enemy_defenses={}, enemy_tech=tech, attacker_tech=tech, budget_multiplier=1.0, mode="attack", base_seed=42, ga_time_budget=1.0, final_sims=300)
    r_defend = optimize(enemy_fleet=enemy, enemy_defenses={}, enemy_tech=tech, attacker_tech=tech, budget_multiplier=1.0, mode="defend", base_seed=42, ga_time_budget=1.0, final_sims=300)
    assert r_attack.mode != r_defend.mode
    assert isinstance(r_attack.recommended_fleet, dict)
    assert isinstance(r_defend.recommended_fleet, dict)


def test_invalid_multiplier():
    with pytest.raises(ValueError):
        optimize(enemy_fleet={"light_fighter": 100}, enemy_defenses={}, enemy_tech=(0, 0, 0), attacker_tech=(0, 0, 0), budget_multiplier=-1.0, mode="attack", base_seed=42)


def test_empty_enemy():
    with pytest.raises(ValueError, match="No enemy"):
        optimize(enemy_fleet={}, enemy_defenses={}, enemy_tech=(0, 0, 0), attacker_tech=(0, 0, 0), budget_multiplier=1.0, mode="attack", base_seed=42)


def test_result_is_serializable():
    r = optimize(enemy_fleet={"light_fighter": 100}, enemy_defenses={}, enemy_tech=(0, 0, 0), attacker_tech=(0, 0, 0), budget_multiplier=1.0, mode="attack", base_seed=42, ga_time_budget=0.5, final_sims=200)
    d = r.__dict__
    assert "recommended_fleet" in d
    assert "win_probability" in d
    assert "confidence_interval_95" in d


# ---------------------------------------------------------------------------
# Regression: impact_pct must compare effective-vs-effective (Session 5 fix).
# Before the fix, variants were effective (raw * loss_scale) but the base was
# raw, so EVERY ship read as a constant (loss_scale - 1) * 100 = -80%% in
# profit mode (loss_scale=0.2) regardless of composition.
# ---------------------------------------------------------------------------


def test_sensitivity_impact_not_constant_in_profit_mode():
    from ogame_optimizer.optimizer.orchestration import _sensitivity_analysis
    from ogame_optimizer.core.combat import simulate_batch

    enemy = {"light_fighter": 2000, "cruiser": 300, "battleship": 50}
    fleet = {"cruiser": 400, "battleship": 60, "light_fighter": 1500}
    tech_a, tech_d = (22, 21, 21), (19, 18, 18)

    batch = simulate_batch(fleet, enemy, {}, tech_a, tech_d, n_sims=40,
                           base_seed=7, debris_pct=0.8, deuterium_in_debris=True)
    sens = _sensitivity_analysis(
        fleet=fleet, enemy_fleet=enemy, enemy_defenses={},
        attacker_tech=tech_a, enemy_tech=tech_d,
        base_loss=float(batch["mean_attacker_loss"]),
        debris_pct=0.8, deuterium_in_debris=True,
        base_seed=7, n_sims=40, loss_scale=0.20,
    )
    vals = [info["impact_pct"] for info in sens.values()]
    assert len(vals) >= 3
    assert not all(abs(v + 80.0) < 1.0 for v in vals), (
        "impact_pct collapsed to the constant -80%% artifact again "
        f"(values: {vals})"
    )
    # Different ship types must genuinely differ in this scenario.
    assert max(vals) - min(vals) > 10.0


def test_sensitivity_sim_divisor_scales_down():
    """sim_divisor=10 runs the same analysis on 1/10 fleets without error
    and still returns per-ship entries for every analyzed type."""
    from ogame_optimizer.optimizer.orchestration import _sensitivity_analysis

    enemy = {"light_fighter": 20000, "cruiser": 3000, "battleship": 500}
    fleet = {"cruiser": 4000, "battleship": 600, "light_fighter": 15000}

    sens = _sensitivity_analysis(
        fleet=fleet, enemy_fleet=enemy, enemy_defenses={},
        attacker_tech=(22, 21, 21), enemy_tech=(19, 18, 18),
        base_loss=1_000_000.0,
        debris_pct=0.8, deuterium_in_debris=True,
        base_seed=7, n_sims=20, loss_scale=1.0,
        sim_divisor=10,
    )
    assert set(sens.keys()) == {"cruiser", "battleship", "light_fighter"}
    for info in sens.values():
        assert "impact_pct" in info and "tag" in info

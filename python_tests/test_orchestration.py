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

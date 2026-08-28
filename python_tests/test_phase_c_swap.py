"""Phase C greedy individual-swap acceptance (promised-swap refinement).

Covers the replacement of the old all-at-once prune validation:
  (a) unit tests of the acceptance gate, the swap-variant builder, the
      budget enforcement and the greedy pass ordering/accept logic
      (deterministic: combat sims are monkeypatched with scripted results),
  (b) an integration smoke test of optimize() with include_alternatives
      False (small, fast scenario; Phase C itself needs ga >= 1.5s and is
      exercised directly by (c) instead - the full 13-type repro is too
      slow for CI),
  (c) a direct test of the Phase C swap-validation helper on a BS-heavy
      fleet vs a battlecruiser wall at reduced sims: when the sensitivity
      deltas are decisive, the destroyer/reaper swap must be chosen over
      keeping the battleships.
"""
from __future__ import annotations

import pytest

import ogame_optimizer.optimizer.orchestration as orch
from ogame_optimizer.core.fleet import SHIPS_COST, fleet_value
from ogame_optimizer.optimizer.orchestration import (
    _enforce_swap_budget,
    _greedy_swap_refine,
    _swap_accepted,
    _swap_variant,
    optimize,
)

TECH0 = (0, 0, 0)


# ---------------------------------------------------------------------------
# (a1) acceptance gate: > 5% relative improvement AND win not worse
# ---------------------------------------------------------------------------

def test_swap_accepted_truth_table():
    # The >5% gate is strict: exactly -5.0% is NOT enough.
    assert _swap_accepted(-5.0, 1.0, 1.0, "attack") is False
    assert _swap_accepted(-4.9, 1.0, 1.0, "attack") is False
    assert _swap_accepted(-5.001, 1.0, 1.0, "attack") is True
    assert _swap_accepted(-50.0, 1.0, 1.0, "attack") is True
    # No improvement / worse: reject.
    assert _swap_accepted(0.0, 1.0, 1.0, "attack") is False
    assert _swap_accepted(+30.0, 1.0, 1.0, "attack") is False
    # Never trade a win for a loss: win probability drop vetoes.
    assert _swap_accepted(-50.0, 0.99, 1.0, "attack") is False
    assert _swap_accepted(-50.0, 0.5, 0.9, "attack") is False
    # Equal win probability is fine.
    assert _swap_accepted(-50.0, 1.0, 1.0, "attack") is True
    assert _swap_accepted(-50.0, 0.90, 0.90, "attack") is True
    # Defend mode: LOWER attacker win probability is better.
    assert _swap_accepted(-50.0, 0.50, 0.90, "defend") is True
    assert _swap_accepted(-50.0, 0.95, 0.90, "defend") is False


# ---------------------------------------------------------------------------
# (a2) swap-variant builder: budget-neutral redistribution math
# ---------------------------------------------------------------------------

def test_swap_variant_redistributes_whole_budget():
    fleet = {"battleship": 300, "reaper": 10}
    variant = _swap_variant(fleet, "battleship", "reaper")
    assert variant is not None
    assert "battleship" not in variant
    bs_cost = sum(SHIPS_COST["battleship"])
    reaper_cost = sum(SHIPS_COST["reaper"])
    expected = 10 + (300 * bs_cost) // reaper_cost
    assert variant["reaper"] == expected
    # Cost-neutral (floor division only loses).
    assert fleet_value(variant) <= fleet_value(fleet)


def test_swap_variant_base_mode_keeps_locked_base():
    base = {"battleship": 40}
    fleet = {"battleship": 100, "reaper": 10}
    variant = _swap_variant(fleet, "battleship", "reaper", base)
    assert variant is not None
    assert variant["battleship"] == 40  # locked base preserved
    bs_cost = sum(SHIPS_COST["battleship"])
    reaper_cost = sum(SHIPS_COST["reaper"])
    expected = 10 + (60 * bs_cost) // reaper_cost  # only additions freed
    assert variant["reaper"] == expected


def test_swap_variant_nothing_to_move():
    assert _swap_variant({"reaper": 5}, "battleship", "reaper") is None
    # Base-locked type with no additions: nothing to swap out.
    assert _swap_variant({"battleship": 40}, "battleship", "reaper",
                         {"battleship": 40}) is None


def test_enforce_swap_budget_scales_over_budget_fleet():
    bs_cost = sum(SHIPS_COST["battleship"])
    fleet = {"battleship": 100}
    budget = bs_cost * 50
    out = _enforce_swap_budget(fleet, None, budget)
    assert fleet_value(out) <= budget
    assert out["battleship"] == 50
    # Within budget: returned unchanged (same object).
    assert _enforce_swap_budget(fleet, None, bs_cost * 100) is fleet
    # Base mode: only additions scaled, base preserved.
    base = {"battleship": 10}
    out = _enforce_swap_budget({"battleship": 60}, base, bs_cost * 30)
    assert out["battleship"] == 40  # 10 base + 30 additions
    assert fleet_value({k: v - base.get(k, 0) for k, v in out.items()}) <= bs_cost * 30


# ---------------------------------------------------------------------------
# (a3) greedy pass: promise filter, ordering, chained acceptance vs the
#      UPDATED fleet, and the nothing-accepted == unchanged contract.
#      (simulate_batch monkeypatched -> fully deterministic)
# ---------------------------------------------------------------------------

class _Recorder:
    """Scripted simulate_batch: pops one result per call, records fleets."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def __call__(self, attacker=None, **kwargs):
        self.calls.append(dict(attacker))
        res = self.results.pop(0)
        assert res is not None, "unexpected extra simulate_batch call"
        return res


def _run_pass(monkeypatch, results, sensitivity, incumbent_loss=100.0,
              incumbent_wp=1.0, fleet=None):
    fleet = fleet if fleet is not None else {
        "battleship": 100, "cruiser": 50, "reaper": 10, "destroyer": 5,
    }
    rec = _Recorder(results)
    monkeypatch.setattr(orch, "simulate_batch", rec)
    out_fleet, out_loss, out_wp, n_acc = _greedy_swap_refine(
        fleet=fleet, incumbent_loss=incumbent_loss, incumbent_wp=incumbent_wp,
        sensitivity=sensitivity, enemy_fleet={"battlecruiser": 300},
        enemy_defenses={}, attacker_tech=TECH0, enemy_tech=TECH0,
        mode="attack", base_fleet=None, budget=fleet_value(fleet),
        base_seed=42, n_sims=60, debris_pct=0.30, deuterium_in_debris=False,
        loss_scale=1.0, resource_weights=(1.0, 1.0, 1.0), preference_beta=0.0,
    )
    return out_fleet, out_loss, out_wp, n_acc, rec


def test_greedy_pass_filters_by_promise_and_orders_largest_first(monkeypatch):
    sensitivity = {
        "battleship": {"impact_pct": -40.0, "redistributed_to": "reaper"},
        "cruiser": {"impact_pct": -8.0, "redistributed_to": "reaper"},   # < 15%: skipped
        "destroyer": {"impact_pct": -10.0, "redistributed_to": "reaper"},  # < 15%: skipped
        "reaper": {"impact_pct": +50.0, "redistributed_to": "destroyer"},  # positive: skipped
    }
    out_fleet, out_loss, out_wp, n_acc, rec = _run_pass(
        monkeypatch,
        results=[{"mean_attacker_loss": 200.0, "win_probability": 1.0}],  # rejected
        sensitivity=sensitivity,
    )
    # Only the >= 15% promise candidate earned a validation sim.
    assert len(rec.calls) == 1
    assert "battleship" not in rec.calls[0]
    assert n_acc == 0
    # Nothing accepted -> fleet and incumbent loss returned unchanged.
    assert out_loss == 100.0
    assert out_wp == 1.0
    assert out_fleet == {"battleship": 100, "cruiser": 50, "reaper": 10, "destroyer": 5}


def test_greedy_pass_accepts_then_chains_against_updated_fleet(monkeypatch):
    sensitivity = {
        "battleship": {"impact_pct": -40.0, "redistributed_to": "reaper"},
        "cruiser": {"impact_pct": -20.0, "redistributed_to": "reaper"},
    }
    # Call 1: BS swap validates -50%  -> ACCEPTED.
    # Call 2: cruiser swap vs the UPDATED fleet validates -4% -> rejected
    # by the >5% gate (loss 48 vs the accepted 50).
    out_fleet, out_loss, out_wp, n_acc, rec = _run_pass(
        monkeypatch,
        results=[
            {"mean_attacker_loss": 50.0, "win_probability": 1.0},
            {"mean_attacker_loss": 48.0, "win_probability": 1.0},
        ],
        sensitivity=sensitivity,
    )
    assert n_acc == 1
    # Candidates ran largest-promise-first: battleship first.
    assert "battleship" not in rec.calls[0]
    # The second candidate was built from the UPDATED fleet: battleship
    # is already gone there.
    assert "battleship" not in rec.calls[1]
    assert "cruiser" not in rec.calls[1]
    assert len(rec.calls) == 2
    # Final fleet = updated fleet after the accepted swap (cruiser kept).
    assert "battleship" not in out_fleet
    assert out_fleet["cruiser"] == 50
    assert out_loss == 50.0


def test_greedy_pass_win_drop_vetoes_swap(monkeypatch):
    sensitivity = {
        "battleship": {"impact_pct": -40.0, "redistributed_to": "reaper"},
    }
    out_fleet, out_loss, out_wp, n_acc, rec = _run_pass(
        monkeypatch,
        results=[{"mean_attacker_loss": 40.0, "win_probability": 0.70}],
        sensitivity=sensitivity,
    )
    # -60% loss but win 1.0 -> 0.7: never trade a win for a loss.
    assert n_acc == 0
    assert out_fleet["battleship"] == 100
    assert out_loss == 100.0


def test_greedy_pass_no_candidates_no_sims(monkeypatch):
    sensitivity = {
        "cruiser": {"impact_pct": -8.0, "redistributed_to": "reaper"},
        "reaper": {"impact_pct": +2.0, "redistributed_to": "destroyer"},
    }
    called = []
    monkeypatch.setattr(orch, "simulate_batch",
                        lambda **kw: called.append(kw) or {})
    out_fleet, out_loss, out_wp, n_acc = _greedy_swap_refine(
        fleet={"cruiser": 50, "reaper": 10}, incumbent_loss=100.0,
        incumbent_wp=1.0, sensitivity=sensitivity,
        enemy_fleet={"battlecruiser": 300}, enemy_defenses={},
        attacker_tech=TECH0, enemy_tech=TECH0, mode="attack",
        base_fleet=None, budget=10**9, base_seed=42, n_sims=60,
        debris_pct=0.30, deuterium_in_debris=False, loss_scale=1.0,
        resource_weights=(1.0, 1.0, 1.0), preference_beta=0.0,
    )
    assert called == []  # no candidate >= 15% promise: zero sims spent
    assert n_acc == 0
    assert out_fleet == {"cruiser": 50, "reaper": 10}


# ---------------------------------------------------------------------------
# (b) integration smoke: optimize() sanity with include_alternatives=False
# ---------------------------------------------------------------------------

def test_optimize_smoke_no_alternatives():
    r = optimize(
        enemy_fleet={"light_fighter": 1000}, enemy_defenses={},
        enemy_tech=TECH0, attacker_tech=TECH0, budget_multiplier=1.0,
        mode="attack", base_seed=42, ga_time_budget=0.3, final_sims=50,
        include_alternatives=False,
    )
    assert sum(r.recommended_fleet.values()) > 0
    assert 0.0 <= r.win_probability <= 1.0
    assert r.win_probability >= 0.95  # 1.0x vs a pure LF wall is winnable
    assert r.win_threshold_met is True
    assert r.alternatives == []
    assert r.expected_loss_mean >= 0.0


# ---------------------------------------------------------------------------
# (c) direct Phase C helper on a BS-heavy fleet vs a battlecruiser wall:
#     with decisive sensitivity deltas the destroyer/reaper swap must be
#     chosen over keeping the battleships (reduced sims, fixed seeds).
# ---------------------------------------------------------------------------

def test_phase_c_picks_destroyer_swap_over_battleships():
    enemy = {"battlecruiser": 300}
    fleet = {"battleship": 300, "light_fighter": 500, "cruiser": 200}
    sensitivity = {
        # Decisive promise for the battleship (mirrors the real defect
        # scenario's -32%..-59% flags): swap its budget into destroyers.
        "battleship": {"impact_pct": -30.0, "redistributed_to": "destroyer"},
        # Small promises: below the 15% bar, must not be tested.
        "light_fighter": {"impact_pct": -8.0, "redistributed_to": "reaper"},
        "cruiser": {"impact_pct": -6.0, "redistributed_to": "reaper"},
    }
    incumbent = orch.simulate_batch(
        attacker=dict(fleet), defender=enemy, defender_defenses={},
        attacker_tech=TECH0, defender_tech=TECH0, n_sims=200,
        base_seed=42 + 7777, debris_pct=0.30, deuterium_in_debris=False,
    )
    inc_loss = float(incumbent.get("mean_attacker_loss", 0))
    inc_wp = float(incumbent.get("win_probability", 0))

    out_fleet, out_loss, out_wp, n_acc = _greedy_swap_refine(
        fleet=fleet, incumbent_loss=inc_loss, incumbent_wp=inc_wp,
        sensitivity=sensitivity, enemy_fleet=enemy, enemy_defenses={},
        attacker_tech=TECH0, enemy_tech=TECH0, mode="attack",
        base_fleet=None, budget=fleet_value(fleet), base_seed=42,
        n_sims=60, debris_pct=0.30, deuterium_in_debris=False,
        loss_scale=1.0, resource_weights=(1.0, 1.0, 1.0),
        preference_beta=0.05,
    )
    assert n_acc == 1
    # The battleships were swapped out, destroyers came in their place.
    assert "battleship" not in out_fleet
    assert out_fleet.get("destroyer", 0) > 0
    # Untouched types stay (only the promised candidate was tested).
    assert out_fleet["light_fighter"] == 500
    assert out_fleet["cruiser"] == 200
    # Materially better and no win traded away.
    assert out_loss < 0.5 * inc_loss
    assert out_wp >= inc_wp


def test_phase_c_keeps_fleet_when_delta_below_gate():
    # Same shape but a mild real delta: destroyer swap validates ~-4%
    # (measured), below the >5% acceptance gate -> fleet kept unchanged.
    enemy = {"battlecruiser": 150, "cruiser": 100}
    fleet = {"battleship": 400, "light_fighter": 1000, "cruiser": 150}
    sensitivity = {
        "battleship": {"impact_pct": -30.0, "redistributed_to": "destroyer"},
    }
    incumbent = orch.simulate_batch(
        attacker=dict(fleet), defender=enemy, defender_defenses={},
        attacker_tech=TECH0, defender_tech=TECH0, n_sims=200,
        base_seed=42 + 7777, debris_pct=0.30, deuterium_in_debris=False,
    )
    inc_loss = float(incumbent.get("mean_attacker_loss", 0))
    inc_wp = float(incumbent.get("win_probability", 0))
    out_fleet, out_loss, out_wp, n_acc = _greedy_swap_refine(
        fleet=fleet, incumbent_loss=inc_loss, incumbent_wp=inc_wp,
        sensitivity=sensitivity, enemy_fleet=enemy, enemy_defenses={},
        attacker_tech=TECH0, enemy_tech=TECH0, mode="attack",
        base_fleet=None, budget=fleet_value(fleet), base_seed=42,
        n_sims=60, debris_pct=0.30, deuterium_in_debris=False,
        loss_scale=1.0, resource_weights=(1.0, 1.0, 1.0),
        preference_beta=0.05,
    )
    assert n_acc == 0
    assert out_fleet == fleet
    assert out_loss == inc_loss
    assert out_wp == inc_wp

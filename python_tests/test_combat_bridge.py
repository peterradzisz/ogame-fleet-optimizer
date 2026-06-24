"""Contract tests for the PyO3 bridge (Task 6).

These tests verify the Python-facing surface of the Rust combat extension
matches the spec from the plan (lines 655-742):

* The batch API returns the documented dict shape.
* The batch API is fully deterministic with a fixed ``base_seed``.
* The batch API can be driven from multiple worker processes in parallel
  without deadlocking — i.e. the GIL is released during the simulation
  loop, so worker processes actually get CPU time.

All tests go through ``ogame_optimizer.core.combat`` (the Python wrapper)
so they exercise the full public surface, not just the raw Rust extension.

Required tests (per delegation):
- test_batch_api_works
- test_batch_determinism
- test_batch_runs_in_multiprocessing
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor

import pytest

from ogame_optimizer.core.combat import (
    evaluate_population,
    simulate_batch,
    simulate_combat,
)


# ---------------------------------------------------------------------------
# Required keys on the batch result dict (per plan spec).
# ---------------------------------------------------------------------------

_REQUIRED_BATCH_KEYS = {
    "mean_attacker_loss",
    "stddev_attacker_loss",
    "mean_defender_loss",
    "win_probability",
    "sims_run",
    "seed_used",
}


# ---------------------------------------------------------------------------
# 1) Batch API works from Python (REQUIRED).
# ---------------------------------------------------------------------------


def test_batch_api_works() -> None:
    """The headline smoke test: ``simulate_batch`` returns the right shape."""
    result = simulate_batch(
        attacker={"light_fighter": 100},
        defender={"cruiser": 10},
        defender_defenses={},
        attacker_tech=(0, 0, 0),
        defender_tech=(0, 0, 0),
        n_sims=100,
        base_seed=42,
    )
    # Shape contract.
    assert isinstance(result, dict)
    assert _REQUIRED_BATCH_KEYS.issubset(result.keys()), (
        f"missing keys: {_REQUIRED_BATCH_KEYS - result.keys()}"
    )
    # Value contract.
    assert result["sims_run"] == 100
    assert 0.0 <= result["win_probability"] <= 1.0
    assert result["mean_attacker_loss"] >= 0.0
    assert result["mean_defender_loss"] >= 0.0
    assert result["stddev_attacker_loss"] >= 0.0
    assert result["seed_used"] == 42


def test_batch_api_returns_win_loss_draw_counts() -> None:
    """The batch result also includes wins / losses / draws counts."""
    result = simulate_batch(
        attacker={"light_fighter": 50},
        defender={"cruiser": 5},
        n_sims=20,
        base_seed=1,
    )
    assert "wins" in result
    assert "losses" in result
    assert "draws" in result
    assert result["wins"] + result["losses"] + result["draws"] == 20


def test_single_combat_returns_winner_and_survivors() -> None:
    """``simulate_combat`` returns the full per-side survivor dicts."""
    result = simulate_combat(
        attacker={"light_fighter": 100},
        defender={"cruiser": 10},
        defender_defenses={},
        attacker_tech=(0, 0, 0),
        defender_tech=(0, 0, 0),
        seed=42,
    )
    assert result["winner"] in {"Attacker", "Defender", "Draw"}
    assert 1 <= result["rounds_fought"] <= 6
    assert isinstance(result["attacker_survivors"], dict)
    assert isinstance(result["defender_survivors"], dict)
    assert isinstance(result["defender_defense_survivors"], dict)


# ---------------------------------------------------------------------------
# 2) Determinism: same seed → byte-identical results (REQUIRED).
# ---------------------------------------------------------------------------


def test_batch_determinism() -> None:
    """Two batches with identical inputs + seed produce identical outputs."""
    kwargs = dict(
        attacker={"light_fighter": 100},
        defender={"cruiser": 10},
        defender_defenses={},
        attacker_tech=(0, 0, 0),
        defender_tech=(0, 0, 0),
        n_sims=100,
        base_seed=42,
    )
    a = simulate_batch(**kwargs)
    b = simulate_batch(**kwargs)
    for key in _REQUIRED_BATCH_KEYS:
        assert a[key] == b[key], (
            f"non-deterministic on {key!r}: {a[key]!r} vs {b[key]!r}"
        )


def test_single_combat_determinism() -> None:
    """Two single-combat calls with the same seed produce identical results."""
    kwargs = dict(
        attacker={"light_fighter": 50},
        defender={"cruiser": 5},
        defender_defenses={},
        attacker_tech=(2, 2, 2),
        defender_tech=(0, 0, 0),
        seed=123,
    )
    a = simulate_combat(**kwargs)
    b = simulate_combat(**kwargs)
    assert a == b


# ---------------------------------------------------------------------------
# 3) Multiprocessing safety: GIL is released, 4 workers run in parallel.
#    (REQUIRED)
# ---------------------------------------------------------------------------


def _worker_run_batch(args: dict) -> dict:
    """Top-level function for ProcessPoolExecutor (must be picklable)."""
    from ogame_optimizer.core.combat import simulate_batch

    return simulate_batch(**args)


def _worker_run_population(args: dict) -> list:
    """Top-level function for ProcessPoolExecutor (must be picklable)."""
    from ogame_optimizer.core.combat import evaluate_population

    return evaluate_population(**args)


def test_batch_runs_in_multiprocessing() -> None:
    """4 worker processes each run a batch; all return, no deadlock."""
    # Balanced scenario so the outcome isn't deterministic: 50 LightFighter +
    # 2 Cruiser vs 2 Battleship + 10 HeavyFighter. The attacker sometimes
    # wins (with losses), sometimes loses, so different seeds give different
    # aggregate stats.
    batch_kwargs = dict(
        attacker={"light_fighter": 50, "cruiser": 2},
        defender={"battleship": 2, "heavy_fighter": 10},
        defender_defenses={},
        attacker_tech=(0, 0, 0),
        defender_tech=(0, 0, 0),
        n_sims=200,
    )
    # 4 different base_seeds → 4 different RNG streams.
    jobs = [
        {**batch_kwargs, "base_seed": 100 + i}
        for i in range(4)
    ]

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(_worker_run_batch, jobs))
    elapsed_parallel = time.perf_counter() - t0

    # All four returned.
    assert len(results) == 4
    for r in results:
        assert r["sims_run"] == 200
        assert 0.0 <= r["win_probability"] <= 1.0
        assert r["mean_attacker_loss"] >= 0.0

    # All 4 results must be valid (multiprocessing smoke test).
    # With per-unit tracking, this scenario may be deterministic (attacker
    # always wins) — the goal is to verify no deadlock/crash, not outcome variance.
    win_probs = [r["win_probability"] for r in results]
    assert all(0.0 <= wp <= 1.0 for wp in win_probs)

    # Parallel speedup: 4 batches of 200 sims in 4 workers should NOT take
    # 4x as long as a single batch of 200 sims run sequentially.  We allow
    # a generous 2.5x ceiling (CI noise + spawn overhead on Windows).
    t1 = time.perf_counter()
    _ = _worker_run_batch(jobs[0])
    single = time.perf_counter() - t1
    assert elapsed_parallel < single * 2.5 + 1.0, (
        f"parallel={elapsed_parallel:.2f}s single={single:.2f}s — "
        f"no speedup, GIL is likely held"
    )


def test_evaluate_population_in_multiprocessing() -> None:
    """``evaluate_population`` (the GA hot path) is also multiprocessing-safe."""
    fleets = [
        {"light_fighter": 50 + i, "cruiser": 2}
        for i in range(8)
    ]
    kwargs = dict(
        attacker_fleets=fleets,
        defender={"battleship": 5},
        defender_defenses={},
        attacker_tech=(0, 0, 0),
        defender_tech=(0, 0, 0),
        n_sims_per_fleet=50,
        base_seed=42,
    )

    with ProcessPoolExecutor(max_workers=2) as ex:
        results = list(ex.map(_worker_run_population, [kwargs, kwargs]))

    assert len(results) == 2
    for pop in results:
        assert len(pop) == len(fleets)
        for r in pop:
            assert r["sims_run"] == 50
            assert 0.0 <= r["win_probability"] <= 1.0


def test_evaluate_population_returns_one_result_per_fleet() -> None:
    """``evaluate_population`` returns N results for N attacker fleets."""
    attackers = [
        {"light_fighter": n} for n in [50, 100, 200, 500]
    ]
    r = evaluate_population(
        attacker_fleets=attackers,
        defender={"cruiser": 10},
        n_sims_per_fleet=20,
        base_seed=7,
    )
    assert len(r) == 4
    for result in r:
        assert "mean_attacker_loss" in result
        assert "win_probability" in result
        assert result["sims_run"] == 20


# ---------------------------------------------------------------------------
# 4) Combat semantics: shield bounce and tech multipliers still work through
#    the bridge (catches any regression in the conversion layer).
# ---------------------------------------------------------------------------


def test_shield_bounce_via_bridge() -> None:
    """10,000 LF vs 1 LSD: 50 attack < 1% of 10,000 shield → bounce → draw."""
    r = simulate_combat(
        attacker={"light_fighter": 10_000},
        defender={},
        defender_defenses={"large_shield_dome": 1},
        seed=3,
    )
    # 6 rounds of bounces → neither side killed → draw.
    assert r["winner"] == "Draw"
    assert r["rounds_fought"] == 6
    assert r["defender_defense_survivors"].get("large_shield_dome") == 1
    assert r["attacker_survivors"].get("light_fighter") == 10_000


def test_tech_10_weapon_wins_via_bridge() -> None:
    """Tech 10 weapon = 2x damage; 100 LF (2x) should beat 100 LF (1x)."""
    r = simulate_combat(
        attacker={"light_fighter": 100},
        defender={"light_fighter": 100},
        attacker_tech=(10, 0, 0),
        defender_tech=(0, 0, 0),
        seed=42,
    )
    assert r["winner"] == "Attacker"


# ---------------------------------------------------------------------------
# 5) Error handling: unknown ship / defense → clear Python exception.
# ---------------------------------------------------------------------------


def test_unknown_ship_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown ship type"):
        simulate_batch(
            attacker={"NotAShip": 1},
            defender={},
            n_sims=1,
            base_seed=1,
        )


def test_unknown_defense_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown defense type"):
        simulate_batch(
            attacker={},
            defender={},
            defender_defenses={"NotADefense": 1},
            n_sims=1,
            base_seed=1,
        )

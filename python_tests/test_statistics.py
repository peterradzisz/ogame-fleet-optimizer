"""Tests for CRN manager, variance reporting, and dual-mode objective (Task 9).

TDD spec from the plan (lines 972-1018, 1028-1065):

* CRN: deterministic ``base_seed + gen`` formula; same generation → same seed.
* Hard constraint: ``compute_fitness`` returns ``-inf`` when the relevant
  probability is below 0.95, finite otherwise.
* VarianceReport: mean/stddev/min/max/p05/p95/confidence_interval_95; the CI
  must contain the mean.
* evaluate_fleet: end-to-end through ``simulate_batch`` returns a
  VarianceReport.
"""
from __future__ import annotations

import math
from typing import Any

import pytest

from ogame_optimizer.optimizer.objective import (
    ObjectiveMode,
    evaluate_fleet,
)
from ogame_optimizer.optimizer.statistics import (
    CRNManager,
    VarianceReport,
    compute_fitness,
)


# ---------------------------------------------------------------------------
# CRN determinism.
# ---------------------------------------------------------------------------


class TestCRNDeterministic:
    """Scenario: CRN produces deterministic per-generation seeds."""

    def test_crn_deterministic(self) -> None:
        """base_seed=42 → gen 0 is 42, gen 1 is 43 (documented formula)."""
        mgr = CRNManager(base_seed=42)
        assert mgr.seed_for_generation(0) == 42
        assert mgr.seed_for_generation(1) == 43

    def test_crn_same_gen_same_seed(self) -> None:
        """Same manager + same gen → identical seed every call."""
        mgr = CRNManager(base_seed=7)
        first = mgr.seed_for_generation(5)
        second = mgr.seed_for_generation(5)
        assert first == second == 12

    def test_crn_distinct_base_distinct_seed(self) -> None:
        """Different base seeds give different per-gen seeds."""
        a = CRNManager(base_seed=0).seed_for_generation(3)
        b = CRNManager(base_seed=1000).seed_for_generation(3)
        assert a == 3
        assert b == 1003
        assert a != b

    def test_crn_monotonic(self) -> None:
        """Monotonically increasing generations yield increasing seeds."""
        mgr = CRNManager(base_seed=99)
        seeds = [mgr.seed_for_generation(g) for g in range(10)]
        for prev, nxt in zip(seeds, seeds[1:]):
            assert nxt == prev + 1


# ---------------------------------------------------------------------------
# Hard constraint: win_prob < 0.95 → -inf.
# ---------------------------------------------------------------------------


def _mock_batch(win_prob: float, mean_loss: float = 1000.0,
                mean_def_loss: float = 2000.0) -> dict[str, Any]:
    """Build a minimal batch_result dict matching simulate_batch's shape."""
    return {
        "mean_attacker_loss": mean_loss,
        "stddev_attacker_loss": 50.0,
        "mean_defender_loss": mean_def_loss,
        "win_probability": win_prob,
        "sims_run": 100,
        "seed_used": 0,
    }


class TestHardConstraint:
    """Scenario: Hard constraint enforced (win prob < 95% = -inf fitness)."""

    def test_hard_constraint_attack_below_threshold(self) -> None:
        """win_prob=0.80 in ATTACK mode → -inf."""
        batch = _mock_batch(win_prob=0.80)
        fit = compute_fitness(batch, mode="attack", budget=5000)
        assert fit == float("-inf")

    def test_hard_constraint_defend_below_threshold(self) -> None:
        """win_prob=0.99 in DEFEND mode → attacker won, so survive=0.01<0.95 → -inf."""
        batch = _mock_batch(win_prob=0.99)  # attacker wins 99%, defender survives 1%
        fit = compute_fitness(batch, mode="defend", budget=5000)
        assert fit == float("-inf")

    def test_hard_constraint_attack_pass(self) -> None:
        """win_prob=0.99 in ATTACK mode → finite negative value."""
        batch = _mock_batch(win_prob=0.99, mean_loss=1000.0)
        fit = compute_fitness(batch, mode="attack", budget=5000)
        assert math.isfinite(fit)
        assert fit == pytest.approx(-(1000.0 / 5000.0))

    def test_hard_constraint_defend_pass(self) -> None:
        """win_prob=0.01 in DEFEND mode → attacker wins 1%, survive 99% → finite."""
        batch = _mock_batch(win_prob=0.01, mean_def_loss=2000.0)
        fit = compute_fitness(batch, mode="defend", budget=5000)
        assert math.isfinite(fit)
        assert fit == pytest.approx(-(2000.0 / 5000.0))

    def test_hard_constraint_boundary_exact_threshold(self) -> None:
        """win_prob=0.95 exactly → finite (>= 0.95 passes)."""
        batch = _mock_batch(win_prob=0.95, mean_loss=1000.0)
        fit = compute_fitness(batch, mode="attack", budget=5000)
        assert math.isfinite(fit)

    def test_hard_constraint_just_below_threshold(self) -> None:
        """win_prob=0.9499 → -inf (strictly below)."""
        batch = _mock_batch(win_prob=0.9499)
        fit = compute_fitness(batch, mode="attack", budget=5000)
        assert fit == float("-inf")


# ---------------------------------------------------------------------------
# VarianceReport / confidence interval.
# ---------------------------------------------------------------------------


class TestVarianceReport:
    """Scenario: Confidence interval calculation."""

    def test_confidence_interval_contains_mean(self) -> None:
        """Samples [100,110,90,105,95] mean=100, CI must contain 100."""
        samples = [100.0, 110.0, 90.0, 105.0, 95.0]
        rpt = VarianceReport(samples)
        lo, hi = rpt.confidence_interval_95
        assert lo < 100.0 < hi
        # Width must be reasonable (nonzero, not absurdly wide).
        assert hi - lo > 0.0
        assert hi - lo < 50.0  # for n=5, ~7.9 stddev → CI width ~14

    def test_variance_report_mean_stddev(self) -> None:
        samples = [100.0, 110.0, 90.0, 105.0, 95.0]
        rpt = VarianceReport(samples)
        assert rpt.mean == pytest.approx(100.0)
        # sample stddev ≈ 7.9056
        assert rpt.stddev == pytest.approx(7.90569, rel=1e-3)

    def test_variance_report_min_max(self) -> None:
        samples = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
        rpt = VarianceReport(samples)
        assert rpt.min == 1.0
        assert rpt.max == 9.0

    def test_variance_report_percentiles(self) -> None:
        samples = [float(i) for i in range(0, 100)]  # 0..99
        rpt = VarianceReport(samples)
        # p05 at index int(0.05*100)=5 → 5.0; p95 at index int(0.95*100)=95 → 95.0
        assert rpt.p05 == 5.0
        assert rpt.p95 == 95.0

    def test_variance_report_sample_count(self) -> None:
        samples = [1.0, 2.0, 3.0]
        rpt = VarianceReport(samples)
        assert rpt.sample_count == 3

    def test_variance_report_from_batch(self) -> None:
        """from_batch pulls mean_attacker_loss and sims_run from batch dict."""
        batch = _mock_batch(win_prob=0.99, mean_loss=1234.0)
        rpt = VarianceReport.from_batch(batch)
        # from_batch builds a report around the attacker-loss mean/stddev.
        assert rpt.mean == pytest.approx(1234.0)
        assert rpt.sample_count == 100

    def test_variance_report_to_dict_roundtrip(self) -> None:
        samples = [10.0, 20.0, 30.0]
        rpt = VarianceReport(samples)
        d = rpt.to_dict()
        for key in ("mean", "stddev", "min", "max", "p05", "p95",
                    "sample_count", "confidence_interval_95"):
            assert key in d
        lo, hi = d["confidence_interval_95"]
        assert lo <= d["mean"] <= hi


# ---------------------------------------------------------------------------
# evaluate_fleet end-to-end.
# ---------------------------------------------------------------------------


class TestEvaluateFleet:
    """Scenario: evaluate_fleet runs a real batch and returns a VarianceReport."""

    def test_evaluate_fleet_attack_returns_report(self) -> None:
        """A small battle returns a VarianceReport with finite stats."""
        # 100 light fighters vs 10 light fighters — attacker should dominate.
        # combat.py expects PascalCase ship names (no snake_case normalisation).
        fleet = {"LightFighter": 100}
        enemy = {"LightFighter": 10}
        rpt = evaluate_fleet(
            fleet=fleet,
            enemy=enemy,
            enemy_defenses={},
            enemy_tech=(0, 0, 0),
            own_tech=(10, 10, 10),
            budget=100000,
            mode=ObjectiveMode.ATTACK,
            n_sims=50,
            seed=42,
        )
        assert isinstance(rpt, VarianceReport)
        assert rpt.sample_count == 50
        assert rpt.mean >= 0.0

    def test_evaluate_fleet_defend_returns_report(self) -> None:
        """Defend mode also returns a VarianceReport."""
        fleet = {"LightFighter": 10}
        enemy = {"LightFighter": 100}
        rpt = evaluate_fleet(
            fleet=fleet,
            enemy=enemy,
            enemy_defenses={},
            enemy_tech=(10, 10, 10),
            own_tech=(0, 0, 0),
            budget=100000,
            mode=ObjectiveMode.DEFEND,
            n_sims=50,
            seed=42,
        )
        assert isinstance(rpt, VarianceReport)
        assert rpt.sample_count == 50

    def test_evaluate_fleet_deterministic_with_seed(self) -> None:
        """Same seed → same report mean (CRN-style reproducibility)."""
        fleet = {"LightFighter": 50}
        enemy = {"LightFighter": 50}
        r1 = evaluate_fleet(
            fleet=fleet, enemy=enemy, enemy_defenses={},
            enemy_tech=(0, 0, 0), own_tech=(0, 0, 0),
            budget=100000, mode=ObjectiveMode.ATTACK,
            n_sims=50, seed=123,
        )
        r2 = evaluate_fleet(
            fleet=fleet, enemy=enemy, enemy_defenses={},
            enemy_tech=(0, 0, 0), own_tech=(0, 0, 0),
            budget=100000, mode=ObjectiveMode.ATTACK,
            n_sims=50, seed=123,
        )
        assert r1.mean == pytest.approx(r2.mean)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

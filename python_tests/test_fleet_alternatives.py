"""Feature tests: fleet alternatives ("Option B/C") - engine + API + UI wiring.

Covers the pure share-math helpers (_budget_shares / _share_distance), the
API alternatives contract (11 frozen keys per entry, quality/diversity
gates, opt-out flag, unwinnable skip), the pairwise diversity invariant,
the client-side wiring strings (app.js / index.html), and the dataclass
defaults.

Determinism strategy: the ga=5s fixture is time-budget stochastic (fleet
composition legitimately varies run to run), so stochastic assertions are
ONLY on invariants (shapes, gates, thresholds) - never exact fleets. The
mixed-tech scenario (33 battlecruisers + 4 reapers) produced exactly one
alternative in independent probes; a soft canary retries seed 43 once
before failing on "no alternatives".
"""
from __future__ import annotations

import json
from dataclasses import fields

import pytest
from fastapi.testclient import TestClient

from ogame_optimizer.api.app import app
from ogame_optimizer.core.fleet import SHIPS_COST
from ogame_optimizer.optimizer.orchestration import (
    AlternativeResult,
    OptimizationResult,
    _budget_shares,
    _share_distance,
)

# The frozen 11-key alternatives entry contract (schemas.py / routes.py).
ALT_KEYS = (
    "label",
    "fleet",
    "win_probability",
    "expected_loss_mean",
    "expected_loss_stddev",
    "confidence_interval_95",
    "fleet_cost_metal",
    "fleet_cost_crystal",
    "fleet_cost_deuterium",
    "kill_estimates",
    "difference_vs_primary",
)


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def _alt_payload(seed: int, **overrides):
    """Mixed-tech winnable scenario: 33 BC + 4 reaper, 1.0x budget, 5s GA.

    Mirrors test_ui_costs_kills._optimize_payload (nested ships/defenses
    wrappers, techs 0, attack mode) with the alternatives-relevant knobs.
    """
    payload = {
        "enemy_fleet": {"ships": {"battlecruiser": 33, "reaper": 4}},
        "enemy_defenses": {"defenses": {}},
        "attacker_tech": {"weapon": 0, "shield": 0, "armor": 0},
        "defender_tech": {"weapon": 0, "shield": 0, "armor": 0},
        "budget_multiplier": 1.0,
        "mode": "attack",
        "seed": seed,
        "ga_time_budget": 5,
        "final_sims": 500,
    }
    payload.update(overrides)
    return payload


@pytest.fixture(scope="module")
def alt_result(client):
    """ONE 5s-GA optimize with alternatives (module-scoped, ~13-15s).

    Soft canary: seed 42 produced exactly 1 alternative in independent
    probes; if it ever comes back empty, retry once with seed 43 before
    failing (ga=5s output is time-budget stochastic by design).
    """
    attempts = []
    for seed in (42, 43):
        r = client.post("/api/optimize", json=_alt_payload(seed))
        assert r.status_code == 200, r.text
        data = r.json()
        attempts.append((seed, len(data["alternatives"])))
        if data["alternatives"]:
            return data
    pytest.fail("no alternatives produced for seeds 42/43: %r" % (attempts,))


# ---------------------------------------------------------------------------
# 1. Pure-function unit tests (deterministic, no API)
# ---------------------------------------------------------------------------

def test_share_distance_identity_is_zero():
    f = {"cruiser": 30, "light_fighter": 70, "reaper": 3}
    assert _share_distance(f, f) == pytest.approx(0.0, abs=1e-9)


def test_share_distance_disjoint_single_type_fleets_is_one():
    # Fully disjoint ship types -> total variation 1.0, any counts.
    pairs = [
        ({"light_fighter": 120}, {"cruiser": 37}),
        ({"reaper": 4}, {"battlecruiser": 33}),
        ({"light_fighter": 1}, {"cruiser": 100000}),
    ]
    for a, b in pairs:
        assert _share_distance(a, b) == pytest.approx(1.0, abs=1e-9), (a, b)


def test_share_distance_partial_case_matches_budget_shares():
    """Hand-computed partial mix: shares are COST shares, so the expected
    distance is computed from _budget_shares output itself (never from
    count-share intuition)."""
    f1 = {"cruiser": 30, "light_fighter": 70}
    f2 = {"cruiser": 20, "light_fighter": 80}
    s1, s2 = _budget_shares(f1), _budget_shares(f2)
    # Sanity: non-empty fleets -> share vectors that partition 1.0.
    assert s1 and s2
    assert sum(s1.values()) == pytest.approx(1.0, abs=1e-9)
    assert sum(s2.values()) == pytest.approx(1.0, abs=1e-9)
    expected = 0.5 * sum(
        abs(s1.get(t, 0.0) - s2.get(t, 0.0)) for t in set(s1) | set(s2)
    )
    assert _share_distance(f1, f2) == pytest.approx(expected, abs=1e-9)


def test_share_distance_equal_cost_fleets_hand_computed():
    """Equal per-unit total cost -> cost shares equal count shares, so the
    textbook hand computation (0.5/0.5 vs 0.4/0.6 -> TV 0.1) is exact."""
    lf_total = sum(SHIPS_COST["light_fighter"])
    sc_total = sum(SHIPS_COST["small_cargo"])
    assert lf_total == sc_total, (lf_total, sc_total)  # guards the hand math
    f1 = {"light_fighter": 50, "small_cargo": 50}  # 0.5 / 0.5
    f2 = {"light_fighter": 40, "small_cargo": 60}  # 0.4 / 0.6
    assert _share_distance(f1, f2) == pytest.approx(0.1, abs=1e-9)


def test_budget_shares_empty_and_disjoint_edges():
    assert _budget_shares({}) == {}
    assert _budget_shares({"light_fighter": 0, "cruiser": 0}) == {}
    assert _budget_shares({"cruiser": 5}) == {"cruiser": 1.0}


# ---------------------------------------------------------------------------
# 2. API alternatives present & well-formed (module fixture)
# ---------------------------------------------------------------------------

def _cost_split(fleet: dict) -> tuple:
    m = c = d = 0
    for ship, count in fleet.items():
        sm, sc, sd = SHIPS_COST[ship]
        m += sm * count
        c += sc * count
        d += sd * count
    return m, c, d


def test_alternatives_present_and_well_formed(alt_result):
    data = alt_result
    alts = data["alternatives"]
    assert isinstance(alts, list)
    # 0..2 by contract; >=1 is the soft canary (fixture retried seed 43).
    assert 0 <= len(alts) <= 2, len(alts)
    assert len(alts) >= 1, len(alts)
    # Alternatives exist only when the primary met the win threshold.
    assert data["win_threshold_met"] is True
    labels = [a["label"] for a in alts]
    for entry in alts:
        assert set(entry) == set(ALT_KEYS), set(entry) ^ set(ALT_KEYS)
        assert entry["label"] in ("Option B", "Option C"), entry["label"]
        fleet = entry["fleet"]
        assert isinstance(fleet, dict) and fleet, entry["label"]
        for ship, count in fleet.items():
            assert ship in SHIPS_COST, ship
            assert isinstance(count, int) and count > 0, (ship, count)
        # Diversity gate: share distance to the primary, pre-rounded 4dp.
        assert entry["difference_vs_primary"] >= 0.10, entry["label"]
        # Quality gate: loss within 1.10x of the primary's frozen loss.
        assert (
            entry["expected_loss_mean"]
            <= 1.10 * data["expected_loss_mean"] + 1e-6
        ), entry["label"]
        # Win-threshold status equal to a winning primary -> alt wins too.
        assert entry["win_probability"] >= 0.95, entry["label"]
        # Cost split must be exactly the SHIPS_COST split of the fleet,
        # and the three components must sum to the fleet's total cost.
        m, c, d = _cost_split(fleet)
        assert (
            entry["fleet_cost_metal"],
            entry["fleet_cost_crystal"],
            entry["fleet_cost_deuterium"],
        ) == (m, c, d), entry["label"]
        assert (
            entry["fleet_cost_metal"]
            + entry["fleet_cost_crystal"]
            + entry["fleet_cost_deuterium"]
            == m + c + d
        )
        ci = entry["confidence_interval_95"]
        assert isinstance(ci, list) and len(ci) == 2, ci
        assert ci[0] <= ci[1], ci
        assert isinstance(entry["kill_estimates"], dict)
        # Strict-JSON safety: no NaN/Infinity anywhere in the entry.
        json.dumps(entry, allow_nan=False)
    if len(alts) == 2:
        assert set(labels) == {"Option B", "Option C"}, labels


# ---------------------------------------------------------------------------
# 3. Opt-out
# ---------------------------------------------------------------------------

def test_alternatives_opt_out_returns_empty_list(client):
    r = client.post(
        "/api/optimize", json=_alt_payload(42, include_alternatives=False)
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["alternatives"] == []
    assert "win_probability" in data
    assert isinstance(data["win_probability"], (int, float))


# ---------------------------------------------------------------------------
# 4. Unwinnable skip
# ---------------------------------------------------------------------------

def test_unwinnable_scenario_skips_alternatives(client):
    payload = _alt_payload(42).copy()
    payload.update(
        {
            "enemy_fleet": {"ships": {"battleship": 100}},
            "budget_multiplier": 0.1,
            "ga_time_budget": 0.3,
            "final_sims": 50,
        }
    )
    r = client.post("/api/optimize", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["win_threshold_met"] is False
    assert data["alternatives"] == []


# ---------------------------------------------------------------------------
# 5. Pairwise diversity invariant (recomputed via the imported helper)
# ---------------------------------------------------------------------------

def test_alternatives_pairwise_diversity(alt_result):
    data = alt_result
    alts = data["alternatives"]
    fleets = [a["fleet"] for a in alts]
    # Alt-vs-alt: every pair must clear the 0.10 share-distance gate.
    # (With a single alternative this loop body never runs - trivial pass.)
    for i in range(len(fleets)):
        for j in range(i + 1, len(fleets)):
            dist = _share_distance(fleets[i], fleets[j])
            assert dist >= 0.10, (i, j, dist)
    # Alt-vs-primary: recompute and cross-check against the reported
    # (pre-rounded to 4dp) difference_vs_primary.
    primary = data["recommended_fleet"]
    for entry in alts:
        dist = _share_distance(primary, entry["fleet"])
        assert dist >= 0.10, entry["label"]
        assert dist == pytest.approx(
            entry["difference_vs_primary"], abs=1e-4
        ), entry["label"]


# ---------------------------------------------------------------------------
# 6. UI wiring strings
# ---------------------------------------------------------------------------

def test_ui_wiring_strings_alternatives(client):
    index = client.get("/").text
    for needle in ("alt-pills", "include_alternatives", "v=20260830a"):
        assert needle in index, needle
    js = client.get("/static/app.js").text
    for needle in (
        "buildAltPills",
        "selectAlt",
        "renderResultsCore",
        "difference_vs_primary",
        "alternatives: data.alternatives",
    ):
        assert needle in js, needle


# ---------------------------------------------------------------------------
# 7. Dataclass defaults
# ---------------------------------------------------------------------------

def test_dataclass_defaults_alternatives_contract():
    result = OptimizationResult(
        recommended_fleet={},
        expected_loss_mean=0.0,
        expected_loss_stddev=0.0,
        win_probability=0.0,
        confidence_interval_95=[0.0, 0.0],
        sims_run_final=0,
        greedy_baseline_loss=0.0,
        ga_improvement_pct=0.0,
        time_elapsed_greedy=0.0,
        time_elapsed_ga=0.0,
        total_time=0.0,
        seed_used=0,
    )
    assert result.alternatives == []
    # AlternativeResult fields ARE the 11-key contract, in order.
    assert tuple(f.name for f in fields(AlternativeResult)) == ALT_KEYS
    dummy = AlternativeResult(
        label="Option B",
        fleet={"cruiser": 1},
        win_probability=1.0,
        expected_loss_mean=0.0,
        expected_loss_stddev=0.0,
        confidence_interval_95=(0.0, 0.0),
        fleet_cost_metal=0,
        fleet_cost_crystal=0,
        fleet_cost_deuterium=0,
        kill_estimates={},
        difference_vs_primary=0.5,
    )
    assert dummy.label == "Option B"
    assert dummy.fleet == {"cruiser": 1}

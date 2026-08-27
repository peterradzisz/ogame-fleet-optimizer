"""Feature tests: win_threshold_met end-to-end wiring (win banner).

The engine (orchestration.py OptimizationResult.win_threshold_met) has long
set this flag; these tests lock the API passthrough (schemas.py / routes.py)
and the web UI wiring (index.html banner container, app.js guard, cache-bust
bump) so an unwinnable-at-budget scenario surfaces a warning banner instead
of a silently losing "least-bad" fleet.

Deterministic by construction: fixed seed 42, ga_time_budget 0.3 (skips
most GA), final_sims 50; assertions on flag values / inequalities, not on
GA-dependent fleet compositions.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ogame_optimizer.api.app import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def _optimize_payload(enemy: dict, budget_multiplier: float = 1.0, **overrides):
    """Mirror of test_ui_costs_kills._optimize_payload (seed=42, ga 0.3s, 50 sims)."""
    payload = {
        "enemy_fleet": {"ships": enemy},
        "enemy_defenses": {"defenses": {}},
        "attacker_tech": {"weapon": 0, "shield": 0, "armor": 0},
        "defender_tech": {"weapon": 0, "shield": 0, "armor": 0},
        "budget_multiplier": budget_multiplier,
        "mode": "attack",
        "seed": 42,
        "ga_time_budget": 0.3,
        "final_sims": 50,
    }
    payload.update(overrides)
    return payload


@pytest.fixture(scope="module")
def unwinnable_result(client):
    """100 battleships vs 0.1x budget: hopeless (probe-ish fleet, ~0% wins)."""
    r = client.post("/api/optimize", json=_optimize_payload({"battleship": 100}, 0.1))
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture(scope="module")
def winnable_result(client):
    """1000 light fighters vs 1.0x budget: cruisers counter fodder, > 50% wins."""
    r = client.post("/api/optimize", json=_optimize_payload({"light_fighter": 1000}, 1.0))
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------
# 1. Unwinnable scenario: flag False, low win probability
# ---------------------------------------------------------------------------

def test_unwinnable_flag_false(unwinnable_result):
    """100 BS @ 0.1x budget: win_threshold_met False + win_probability < 0.5."""
    data = unwinnable_result
    assert data["win_threshold_met"] is False, data["win_probability"]
    assert data["win_probability"] < 0.5, data["win_probability"]


# ---------------------------------------------------------------------------
# 2. Winnable scenario: flag True
# ---------------------------------------------------------------------------

def test_winnable_flag_true(winnable_result):
    """1000 LF @ 1.0x budget: win_threshold_met True."""
    data = winnable_result
    assert data["win_threshold_met"] is True, data["win_probability"]
    assert data["win_probability"] >= 0.5, data["win_probability"]


# ---------------------------------------------------------------------------
# 3. UI wiring: banner container, strict guard, cache-bust
# ---------------------------------------------------------------------------

def test_ui_wiring_win_banner(client):
    """index.html carries the banner div + bumped cache-bust; app.js carries
    the strict === false guard (legacy snapshots without the field stay
    banner-hidden) and the .win-banner class hook."""
    index = client.get("/").text
    assert 'id="win-threshold-banner"' in index
    assert 'class="win-banner hidden"' in index
    assert "v=20260828a" in index  # cache-bust bumped (now: LF-share banner)

    js = client.get("/static/app.js").text
    assert "data.win_threshold_met === false" in js  # strict guard: only === false shows
    assert "win-threshold-banner" in js
    assert "win-banner" in js
    # snapshot keeps the field so re-rendered history restores the banner
    assert "win_threshold_met: data.win_threshold_met" in js

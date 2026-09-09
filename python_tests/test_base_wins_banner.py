"""Feature tests: base_already_wins end-to-end wiring (base-wins banner).

When base_fleet mode is active and the base fleet alone already wins
(>=95% of sims), orchestration.optimize() deliberately skips optimization
and returns the base fleet unchanged (commit 7db3901: "no useless
dead-weight adds"). These tests lock the API passthrough (flag set,
additions empty, fleet = base) and the web UI wiring (banner container in
index.html, guard in app.js, cache-bust pin) so users get an explanatory
banner instead of silent no-changes.

Deterministic by construction: fixed seed 42; the winning scenario never
reaches the GA (early-exit after a 200-sim base check).
"""
from __future__ import annotations

import pathlib

import pytest
from fastapi.testclient import TestClient

from ogame_optimizer.api.app import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def _payload(enemy, mult=1.0, **overrides):
    payload = {
        "enemy_fleet": {"ships": enemy},
        "enemy_defenses": {"defenses": {}},
        "attacker_tech": {"weapon": 0, "shield": 0, "armor": 0},
        "defender_tech": {"weapon": 0, "shield": 0, "armor": 0},
        "budget_multiplier": mult,
        "mode": "attack",
        "seed": 42,
        "ga_time_budget": 0.3,
        "final_sims": 50,
        "base_fleet": {"battlecruiser": 50000},
    }
    payload.update(overrides)
    return payload


@pytest.fixture(scope="module")
def wins_result(client):
    """50k BCs (~4.25bn) vs 100 Battleships (~6M): trivial >=95% win -> early exit."""
    r = client.post("/api/optimize", json=_payload({"battleship": 100}))
    assert r.status_code == 200, r.text
    return r.json()


def test_base_already_wins_flag_set(wins_result):
    assert wins_result["base_already_wins"] is True
    assert wins_result["win_probability"] >= 0.95
    # No additions proposed; recommended fleet IS the base fleet
    assert wins_result["recommended_fleet"] == {"battlecruiser": 50000}
    adds = wins_result.get("recommended_additions") or {}
    assert sum(adds.values()) == 0


def test_budget_multiplier_echoed(wins_result):
    """The sim-only banner keys off budget_multiplier in the response."""
    assert wins_result["budget_multiplier"] == 1.0


def test_flag_false_without_base_fleet(client):
    """Same tiny enemy, no base fleet -> normal optimize path, flag False."""
    r = client.post("/api/optimize", json=_payload({"battleship": 100}, base_fleet=None))
    assert r.status_code == 200, r.text
    assert r.json()["base_already_wins"] is False


def test_ui_wiring():
    root = pathlib.Path(__file__).resolve().parents[1]
    idx = (root / "python/ogame_optimizer/web/templates/index.html").read_text(encoding="utf-8")
    js = (root / "python/ogame_optimizer/web/static/app.js").read_text(encoding="utf-8")
    assert 'id="base-wins-banner"' in idx, "banner container missing"
    assert "base_already_wins === true" in js, "app.js guard missing"
    assert "v=20260908e" in idx, "cache-bust pin not bumped"
    for f in ("test_fleet_alternatives.py", "test_lf_share.py", "test_ui_costs_kills.py", "test_win_banner.py"):
        assert "v=20260908e" in (root / "python_tests" / f).read_text(encoding="utf-8"), f"{f}: pin stale"

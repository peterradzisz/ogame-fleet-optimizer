"""Feature tests: lf_share field + LF-dominance banner wiring.

lf_share is the cost-proportional light-fighter share of the recommended
fleet (percent 0-100), computed server-side in routes.py with the default
resource weights (M:2.0, C:1.0, D:1.0). Because LF are cheap, count-share
and cost-share differ; cost-share is the "resources invested" measure.

Deterministic by construction:
- API-level tests use fixed seed 42, ga_time_budget 0.3, final_sims 50 and
  assert only on field presence/type/range (not GA-dependent compositions).
- Computation tests pin the pure helper _lf_cost_share directly (no GA).
- UI wiring tests grep served HTML/JS for the banner hooks.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ogame_optimizer.api.app import app
from ogame_optimizer.api.routes import _lf_cost_share


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def _optimize_payload(enemy: dict, budget_multiplier: float = 1.0, **overrides):
    """Mirror of test_win_banner._optimize_payload (seed=42, ga 0.3s, 50 sims)."""
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


# ---------------------------------------------------------------------------
# 1. API field present + type correct on every response
# ---------------------------------------------------------------------------

def test_api_field_present_typed_unwinnable(client):
    """Unwinnable scenario (100 BS @ 0.1x): lf_share exists, float, 0-100."""
    r = client.post("/api/optimize", json=_optimize_payload({"battleship": 100}, 0.1))
    assert r.status_code == 200, r.text
    data = r.json()
    assert "lf_share" in data
    assert isinstance(data["lf_share"], float)
    assert 0.0 <= data["lf_share"] <= 100.0


def test_api_field_present_typed_winnable(client):
    """Winnable scenario (1000 LF @ 1.0x): lf_share exists, float, 0-100."""
    r = client.post("/api/optimize", json=_optimize_payload({"light_fighter": 1000}, 1.0))
    assert r.status_code == 200, r.text
    data = r.json()
    assert "lf_share" in data
    assert isinstance(data["lf_share"], float)
    assert 0.0 <= data["lf_share"] <= 100.0


# ---------------------------------------------------------------------------
# 2. Computation semantics (pure helper, deterministic)
# ---------------------------------------------------------------------------

def test_pure_lf_fleet_share_100():
    """A fleet of only light fighters is 100% LF by cost."""
    assert _lf_cost_share({"light_fighter": 500}) == 100.0


def test_cruiser_only_fleet_share_0():
    """A 100% cruiser fleet has zero LF cost share."""
    assert _lf_cost_share({"cruiser": 100}) == 0.0


def test_mixed_60pct_lf_by_count_below_100():
    """60% LF by count: cost-share is well below the count-share because
    cruisers are far more expensive per hull (weighted 49k vs 7k per unit).

    Exact expectation: 60*7000 / (60*7000 + 40*49000) * 100 = 17.6%.
    """
    fleet = {"light_fighter": 60, "cruiser": 40}
    share = _lf_cost_share(fleet)
    assert share < 100.0
    assert share == pytest.approx(17.6, abs=0.1)


def test_empty_fleet_share_0():
    """Empty fleet -> zero total cost -> lf_share 0.0 (guard, no div-by-zero)."""
    assert _lf_cost_share({}) == 0.0


# ---------------------------------------------------------------------------
# 3. UI wiring: banner container, cache-bust, JS hooks
# ---------------------------------------------------------------------------

def test_ui_wiring_lf_banner(client):
    """index.html carries the LF banner div + bumped cache-bust; app.js
    carries the lf_share read, banner id, .lf-banner class hook and the
    winnable-only strict guard."""
    index = client.get("/").text
    assert 'id="lf-share-banner"' in index
    assert 'class="lf-banner hidden"' in index
    assert "v=20260828a" in index  # cache-bust bumped with LF-share banner

    js = client.get("/static/app.js").text
    assert "lf_share" in js
    assert "lf-share-banner" in js
    assert "lf-banner" in js
    # banner only stacks on a WINNING result (=== true) at >= 60% cost share
    assert "data.win_threshold_met === true" in js
    assert "lf >= 60.0" in js
    # snapshot keeps the field so history restores re-render the banner
    assert "lf_share: data.lf_share" in js

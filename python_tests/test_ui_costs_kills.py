"""Feature tests: UI Costs & Kills Transparency (Wave 2, Task 5).

Covers the API cost-split fields (fleet_cost_* / additions_cost_*), the
kill_estimates block (engine-matrix fast path + heuristic Rust-path
fallback), defender destroyed_count consistency, the client-side UI
strings (app.js / index.html cache-bust), and the mcd <-> SHIPS_COST
consistency lock between app.js SHIP_META and fleet.py.

Deterministic by construction: fixed seed 42, ga_time_budget 0.3 (skips
most GA), and assertions on shapes / inequalities rather than exact
GA-dependent fleet compositions.
"""
from __future__ import annotations

import json
import re

import pytest
from fastapi.testclient import TestClient

from ogame_optimizer.api.app import app
from ogame_optimizer.core.fast_combat import simulate_batch_fast
from ogame_optimizer.core.fleet import SHIPS_COST


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def _optimize_payload(enemy: dict, budget_multiplier: float = 1.0, **overrides):
    """Mirror of test_api._seed_fleet_payload style (seed=42, ga 0.3s, 50 sims)."""
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
def big_result(client):
    """Winnable big scenario: 1000 LF enemy, 1.0x budget (fast path, > 500 units).

    Cruisers counter pure fodder, so the optimizer wins and every defender
    LF dies -- the attribution matrix path produces non-zero kill rows.
    """
    r = client.post("/api/optimize", json=_optimize_payload({"light_fighter": 1000}))
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------
# 1. API cost split
# ---------------------------------------------------------------------------

def test_fleet_cost_split_sums_to_fleet_value(big_result):
    """fleet_cost_metal + crystal + deuterium must equal fleet_value exactly."""
    data = big_result
    for field in ("fleet_cost_metal", "fleet_cost_crystal", "fleet_cost_deuterium"):
        val = data[field]
        assert isinstance(val, int) and val >= 0, (field, val)
    total = data["fleet_cost_metal"] + data["fleet_cost_crystal"] + data["fleet_cost_deuterium"]
    assert total == data["fleet_value"], (total, data["fleet_value"])
    # And the split itself must match the per-unit SHIPS_COST table.
    exp_m = exp_c = exp_d = 0
    for ship, count in data["recommended_fleet"].items():
        m, c, d = SHIPS_COST[ship]
        exp_m += m * count
        exp_c += c * count
        exp_d += d * count
    assert (data["fleet_cost_metal"], data["fleet_cost_crystal"], data["fleet_cost_deuterium"]) == (exp_m, exp_c, exp_d)


# ---------------------------------------------------------------------------
# 2. destroyed_count consistency
# ---------------------------------------------------------------------------

def test_defender_destroyed_count_consistency(big_result):
    """destroyed_count == count - surviving_count for every defender entry."""
    analysis = big_result["defender_fleet_analysis"]
    assert analysis, "defender_fleet_analysis unexpectedly empty"
    for ship, entry in analysis.items():
        assert entry["destroyed_count"] == entry["count"] - entry["surviving_count"], ship
        assert 0 <= entry["destroyed_count"] <= entry["count"], ship


# ---------------------------------------------------------------------------
# 3. kill_estimates shape
# ---------------------------------------------------------------------------

def _assert_kill_entry_shape(entry: dict, ship: str) -> None:
    assert "kills_est" in entry and "cost_per_kill" in entry and "damage_share" in entry, ship
    kills = entry["kills_est"]
    assert isinstance(kills, (int, float)) and not isinstance(kills, bool) and kills >= 0, (ship, kills)
    cpk = entry["cost_per_kill"]
    assert cpk is None or (isinstance(cpk, (int, float)) and cpk > 0), (ship, cpk)
    share = entry["damage_share"]
    assert isinstance(share, (int, float)) and 0.0 <= share <= 1.0, (ship, share)


def test_kill_estimates_shape_and_share_normalization(big_result):
    """Non-empty vs non-empty fleet; per-entry shape; shares sum to ~1."""
    data = big_result
    assert data["recommended_fleet"], "expected a non-empty recommended fleet"
    ke = data["kill_estimates"]
    assert ke, "kill_estimates empty despite non-empty recommended_fleet"
    for ship, entry in ke.items():
        _assert_kill_entry_shape(entry, ship)
    # When any damage potential exists the shares must partition ~1.0.
    share_sum = sum(e["damage_share"] for e in ke.values())
    if any(e["damage_share"] > 0 for e in ke.values()):
        assert abs(share_sum - 1.0) <= 0.05, share_sum
    # Consistency: zero kills must pair with a null cost_per_kill.
    for ship, entry in ke.items():
        if entry["kills_est"] == 0:
            assert entry["cost_per_kill"] is None, ship


# ---------------------------------------------------------------------------
# 4. Rust-path fallback (totals < 500 units -> no engine attribution matrix)
# ---------------------------------------------------------------------------

def test_rust_path_heuristic_fallback_kill_estimates(client):
    """Small scenario stays on the Rust path; heuristic fallback still fills
    kill_estimates with valid entries."""
    data = client.post(
        "/api/optimize", json=_optimize_payload({"light_fighter": 50}, 0.5)
    ).json()
    total_units = sum(data["recommended_fleet"].values()) + 50
    assert total_units < 500, total_units  # sanity: fast-path threshold not crossed
    ke = data["kill_estimates"]
    assert ke, "heuristic fallback must still produce kill_estimates"
    for ship, entry in ke.items():
        _assert_kill_entry_shape(entry, ship)
    # Cruisers/hf fire real damage vs LF: shares partition ~1.
    share_sum = sum(e["damage_share"] for e in ke.values())
    assert abs(share_sum - 1.0) <= 0.05, share_sum


# ---------------------------------------------------------------------------
# 5. Fast-path attribution (engine matrix)
# ---------------------------------------------------------------------------

def test_fast_path_attribution_matrix_kills(big_result):
    """1000 LF enemy (> 500 total units) rides the fast path; the engine
    attribution matrix yields at least one strictly-positive kills_est."""
    data = big_result
    assert sum(data["recommended_fleet"].values()) + 1000 > 500  # fast path
    ke = data["kill_estimates"]
    assert ke
    # Optimizer wins vs pure fodder (cruiser counter): kills must show up.
    assert any(e["kills_est"] > 0 for e in ke.values()), ke
    # Winning vs 1000 LF means the whole defender fleet died: sum of kills
    # attributed across attacker types equals the destroyed LF count.
    destroyed = data["defender_fleet_analysis"]["light_fighter"]["destroyed_count"]
    assert destroyed == 1000
    assert abs(sum(e["kills_est"] for e in ke.values()) - destroyed) <= 1e-6


# ---------------------------------------------------------------------------
# 6. Zero-kill safety
# ---------------------------------------------------------------------------

def test_zero_kill_safety_unwinnable_scenario(client):
    """Unwinnable fight must not produce NaN/Inf or bogus cost_per_kill.

    SPEC DEVIATION: budget_multiplier 0.05 is rejected by the API's
    0.1-step validator (422), so the nearest valid step 0.1 is used --
    600k budget vs 100 battleships (6M) is equally unwinnable.
    """
    r = client.post(
        "/api/optimize", json=_optimize_payload({"battleship": 100}, 0.1)
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["win_probability"] < 0.5  # sanity: genuinely losing fight
    ke = data["kill_estimates"]
    assert ke
    for ship, entry in ke.items():
        _assert_kill_entry_shape(entry, ship)
    assert any(
        e["cost_per_kill"] is None or e["kills_est"] == 0 for e in ke.values()
    ), "unwinnable fight should carry null/zero kill entries"
    # Strict-JSON safety: no NaN/Infinity anywhere in the response.
    json.dumps(data, allow_nan=False)


# ---------------------------------------------------------------------------
# 7. Base-mode additions cost split
# ---------------------------------------------------------------------------

def _cost_split(fleet: dict) -> tuple:
    m = c = d = 0
    for ship, count in fleet.items():
        sm, sc, sd = SHIPS_COST[ship]
        m += sm * count
        c += sc * count
        d += sd * count
    return m, c, d


def test_base_mode_additions_and_fleet_cost_split(client):
    """additions_cost_* = split of recommended_additions;
    fleet_cost_* = split of recommended_fleet (base + additions)."""
    r = client.post(
        "/api/optimize",
        json=_optimize_payload(
            {"cruiser": 30, "light_fighter": 100}, 1.0, base_fleet={"cruiser": 10}
        ),
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["base_fleet"] == {"cruiser": 10}
    additions, fleet = data["recommended_additions"], data["recommended_fleet"]
    assert additions, "expected non-empty additions for this scenario"
    got_add = (data["additions_cost_metal"], data["additions_cost_crystal"], data["additions_cost_deuterium"])
    got_fleet = (data["fleet_cost_metal"], data["fleet_cost_crystal"], data["fleet_cost_deuterium"])
    assert got_add == _cost_split(additions), (got_add, _cost_split(additions))
    assert got_fleet == _cost_split(fleet), (got_fleet, _cost_split(fleet))
    # fleet = base + additions per ship type.
    for ship in fleet:
        assert fleet[ship] == data["base_fleet"].get(ship, 0) + additions.get(ship, 0), ship
    # And the fleet split still sums to fleet_value.
    assert sum(got_fleet) == data["fleet_value"]


# ---------------------------------------------------------------------------
# 8. UI strings (served statics)
# ---------------------------------------------------------------------------

def test_ui_strings_costs_kills_transparency(client):
    """app.js and index.html carry all Costs & Kills transparency markers."""
    js = client.get("/static/app.js").text
    assert "function fmtM" in js
    assert js.count("mcd: [") >= 14, js.count("mcd: [")
    assert "kill-estimate-table" in js
    assert "destroyed_count" in js
    assert "colspan=6" in js
    assert "Fleet Cost (M/C/D)" in js
    assert "Additions Cost (M/C/D)" in js
    index = client.get("/").text
    assert "v=20260826c" in index  # cache-bust bumped with this feature


# ---------------------------------------------------------------------------
# 9. mcd consistency lock (app.js SHIP_META vs fleet.py SHIPS_COST)
# ---------------------------------------------------------------------------

# Hardcoded SHIP_META cost totals (m+c+d) straight from app.js -- a change
# to either side without the other fails this lock. pathfinder/reaper carry
# the Fandom-corrected values fixed this session.
_META_COST = {
    "light_fighter": 4000,
    "heavy_fighter": 10000,
    "cruiser": 29000,
    "battleship": 60000,
    "battlecruiser": 85000,
    "bomber": 90000,
    "destroyer": 125000,
    "deathstar": 10000000,
    "small_cargo": 4000,
    "large_cargo": 12000,
    "espionage_probe": 1000,
    "pathfinder": 31000,
    "recycler": 18000,
    "reaper": 160000,
}


def test_app_js_mcd_tuples_match_fleet_cost(client):
    """All 14 mcd tuples in app.js must equal fleet.py SHIPS_COST and sum to
    the SHIP_META cost totals."""
    src = client.get("/static/app.js").text
    tuples = re.findall(
        r'"([a-z_]+)":\s*\{[^}]*?cost:\s*(\d+),\s*mcd:\s*\[(\d+),\s*(\d+),\s*(\d+)\]',
        src,
    )
    assert len(tuples) == 14, f"expected 14 mcd tuples, parsed {len(tuples)}"
    parsed = {}
    for ship, cost, m, c, d in tuples:
        parsed[ship] = (int(m), int(c), int(d), int(cost))
    assert set(parsed) == set(SHIPS_COST), set(parsed) ^ set(SHIPS_COST)
    for ship, (m, c, d, cost) in parsed.items():
        assert (m, c, d) == SHIPS_COST[ship], (ship, (m, c, d), SHIPS_COST[ship])
        assert m + c + d == cost == _META_COST[ship], (ship, m + c + d, cost)


# ---------------------------------------------------------------------------
# 10. Attribution magnitude sanity (RNG-invariance duel anchors live in
#     test_fast_combat_overkill.py -- not duplicated here).
# ---------------------------------------------------------------------------

def test_attribution_magnitude_cruiser_vs_lf():
    """Cruisers (RF 6 vs LF) must attribute positive damage and actually
    thin a 3000-LF wall over 20 sims."""
    r = simulate_batch_fast(
        {"cruiser": 300}, {"light_fighter": 3000}, {},
        (0, 0, 0), (0, 0, 0), n_sims=20, base_seed=42, want_attribution=True,
    )
    am = r["attribution_mean"]
    assert am["cruiser"]["light_fighter"] > 0.0
    # Extinct defender types are dropped from survivors_mean, so .get(..., 0)
    # covers both "thinned" and "wiped out entirely" (observed: full wipe).
    assert r["defender_survivors_mean"].get("light_fighter", 0.0) < 3000.0

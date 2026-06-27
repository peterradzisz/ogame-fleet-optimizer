"""Tests for the FastAPI surface (Task 11)."""
from __future__ import annotations
import pytest
from fastapi.testclient import TestClient

from ogame_optimizer.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_root_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "OGame Fleet Auto-Optimizer" in r.text


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200


def test_list_ships(client):
    r = client.get("/api/ships")
    assert r.status_code == 200
    ships = r.json()["ships"]
    assert len(ships) == 16  # 14 original + Solar Satellite + Crawler (civil ships)
    keys = {s["key"] for s in ships}
    assert "light_fighter" in keys
    assert "deathstar" in keys


def test_list_defenses(client):
    r = client.get("/api/defenses")
    assert r.status_code == 200
    defenses = r.json()["defenses"]
    assert len(defenses) == 8
    keys = {d["key"] for d in defenses}
    assert "rocket_launcher" in keys
    assert "large_shield_dome" in keys


def test_combat_endpoint(client):
    r = client.post("/api/combat", json={
        "attacker": {"ships": {"light_fighter": 100}},
        "defender": {"ships": {"cruiser": 10}},
        "defender_defenses": {"defenses": {}},
        "attacker_tech": {"weapon": 0, "shield": 0, "armor": 0},
        "defender_tech": {"weapon": 0, "shield": 0, "armor": 0},
        "n_sims": 30,
        "seed": 42,
    })
    assert r.status_code == 200
    body = r.json()
    assert "mean_attacker_loss" in body
    assert "win_probability" in body
    assert body["sims_run"] == 30
    assert 0.0 <= body["win_probability"] <= 1.0


def test_combat_unknown_ship_rejected(client):
    r = client.post("/api/combat", json={
        "attacker": {"ships": {"unknown_ship": 100}},
        "defender": {"ships": {"cruiser": 10}},
        "defender_defenses": {"defenses": {}},
        "attacker_tech": {"weapon": 0, "shield": 0, "armor": 0},
        "defender_tech": {"weapon": 0, "shield": 0, "armor": 0},
        "n_sims": 10,
        "seed": 42,
    })
    assert r.status_code == 422


def test_optimize_invalid_multiplier(client):
    r = client.post("/api/optimize", json={
        "enemy_fleet": {"ships": {"light_fighter": 100}},
        "enemy_defenses": {"defenses": {}},
        "attacker_tech": {"weapon": 0, "shield": 0, "armor": 0},
        "defender_tech": {"weapon": 0, "shield": 0, "armor": 0},
        "budget_multiplier": 0.37,
        "mode": "attack",
        "seed": 42,
    })
    assert r.status_code == 422


def test_optimize_invalid_mode(client):
    r = client.post("/api/optimize", json={
        "enemy_fleet": {"ships": {"light_fighter": 100}},
        "enemy_defenses": {"defenses": {}},
        "attacker_tech": {"weapon": 0, "shield": 0, "armor": 0},
        "defender_tech": {"weapon": 0, "shield": 0, "armor": 0},
        "budget_multiplier": 1.0,
        "mode": "invalid",
        "seed": 42,
    })
    assert r.status_code == 422


def test_optimize_negative_counts(client):
    r = client.post("/api/optimize", json={
        "enemy_fleet": {"ships": {"light_fighter": -1}},
        "enemy_defenses": {"defenses": {}},
        "attacker_tech": {"weapon": 0, "shield": 0, "armor": 0},
        "defender_tech": {"weapon": 0, "shield": 0, "armor": 0},
        "budget_multiplier": 1.0,
        "mode": "attack",
        "seed": 42,
    })
    assert r.status_code == 422


def test_optimize_empty_enemy_returns_400(client):
    r = client.post("/api/optimize", json={
        "enemy_fleet": {"ships": {}},
        "enemy_defenses": {"defenses": {}},
        "attacker_tech": {"weapon": 0, "shield": 0, "armor": 0},
        "defender_tech": {"weapon": 0, "shield": 0, "armor": 0},
        "budget_multiplier": 1.0,
        "mode": "attack",
        "seed": 42,
    })
    assert r.status_code == 400
    assert "No enemy" in r.json()["detail"]


def test_combat_n_sims_out_of_range(client):
    r = client.post("/api/combat", json={
        "attacker": {"ships": {"light_fighter": 10}},
        "defender": {"ships": {"cruiser": 1}},
        "defender_defenses": {"defenses": {}},
        "attacker_tech": {"weapon": 0, "shield": 0, "armor": 0},
        "defender_tech": {"weapon": 0, "shield": 0, "armor": 0},
        "n_sims": 999999,
        "seed": 42,
    })
    assert r.status_code == 422



def test_optimize_min_gain_pct_accepted(client):
    """Verify min_gain_pct is accepted by the API and echoed back in response."""
    payload = {
        "enemy_fleet": {"ships": {"light_fighter": 100, "cruiser": 20}},
        "enemy_defenses": {"defenses": {}},
        "attacker_tech": {"weapon": 5, "shield": 5, "armor": 5},
        "defender_tech": {"weapon": 5, "shield": 5, "armor": 5},
        "budget_multiplier": 1.0,
        "mode": "attack",
        "base_seed": 42,
        "ga_time_budget": 0.3,
        "final_sims": 100,
        "debris_pct": 0.30,
        "min_gain_pct": 20.0,
    }
    r = client.post("/api/optimize", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    # min_gain_pct should be echoed back
    assert data["min_gain_required"] == 20.0
    # min_gain_met should be a boolean
    assert isinstance(data["min_gain_met"], bool)
    # actual_roi_pct should be a number
    assert isinstance(data["actual_roi_pct"], (int, float))


def test_optimize_min_gain_pct_rejects_out_of_range(client):
    """Verify min_gain_pct outside [0, 100] is rejected."""
    payload = {
        "enemy_fleet": {"ships": {"light_fighter": 100}},
        "enemy_defenses": {"defenses": {}},
        "attacker_tech": {"weapon": 5, "shield": 5, "armor": 5},
        "defender_tech": {"weapon": 5, "shield": 5, "armor": 5},
        "budget_multiplier": 1.0,
        "mode": "attack",
        "min_gain_pct": 150.0,  # out of range
    }
    r = client.post("/api/optimize", json=payload)
    assert r.status_code == 422  # Pydantic validation error


def test_optimize_min_gain_pct_default_zero(client):
    """Verify min_gain_pct defaults to 0 (no constraint) and met=True."""
    payload = {
        "enemy_fleet": {"ships": {"light_fighter": 100}},
        "enemy_defenses": {"defenses": {}},
        "attacker_tech": {"weapon": 5, "shield": 5, "armor": 5},
        "defender_tech": {"weapon": 5, "shield": 5, "armor": 5},
        "budget_multiplier": 1.0,
        "mode": "attack",
        "base_seed": 42,
        "ga_time_budget": 0.2,
        "final_sims": 50,
    }
    r = client.post("/api/optimize", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["min_gain_required"] == 0.0
    assert data["min_gain_met"] is True

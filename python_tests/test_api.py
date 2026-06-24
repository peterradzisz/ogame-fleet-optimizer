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
    assert len(ships) == 14  # 13 original + Reaper (post-v0.84)
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

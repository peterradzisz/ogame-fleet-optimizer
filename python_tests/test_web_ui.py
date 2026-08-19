"""Tests for the web UI (Task 13)."""
from __future__ import annotations
import pytest
from fastapi.testclient import TestClient

from ogame_optimizer.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_index_loads(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    body = r.text
    assert "OGame Fleet Auto-Optimizer" in body
    assert 'name="mode"' in body
    assert 'value="attack"' in body
    assert 'value="defend"' in body


def test_index_has_fleet_inputs(client):
    r = client.get("/")
    body = r.text
    for ship in ["light_fighter", "heavy_fighter", "cruiser", "battleship",
                 "battlecruiser", "bomber", "destroyer", "deathstar",
                 "small_cargo", "large_cargo", "espionage_probe"]:
        assert f'name="{ship}"' in body, f"Missing input for {ship}"


def test_index_has_defense_inputs(client):
    r = client.get("/")
    body = r.text
    for defense in ["rocket_launcher", "light_laser", "heavy_laser", "gauss_cannon",
                    "ion_cannon", "plasma_turret", "small_shield_dome", "large_shield_dome"]:
        assert f'name="{defense}"' in body, f"Missing input for {defense}"


def test_index_has_tech_inputs(client):
    r = client.get("/")
    body = r.text
    for tech in ["attacker_weapon", "attacker_shield", "attacker_armor",
                 "defender_weapon", "defender_shield", "defender_armor"]:
        assert f'name="{tech}"' in body, f"Missing input for {tech}"


def test_index_has_budget_multiplier_default_1x(client):
    r = client.get("/")
    body = r.text
    assert 'value="1.0" selected' in body
    for opt in ["0.5", "1.0", "1.5", "2.0", "2.5", "3.0", "4.0", "5.0"]:
        assert f'value="{opt}"' in body, f"Missing budget option {opt}"


def test_index_has_optimize_button(client):
    r = client.get("/")
    body = r.text
    assert 'id="optimize-btn"' in body
    assert "Optimize" in body


def test_static_css_served(client):
    r = client.get("/static/style.css")
    assert r.status_code == 200
    assert "text/css" in r.headers["content-type"]
    assert ".btn-primary" in r.text


def test_static_js_served(client):
    r = client.get("/static/app.js")
    assert r.status_code == 200
    assert "javascript" in r.headers["content-type"]
    assert "fetch" in r.text or "fetch(" in r.text


def test_index_includes_app_js(client):
    r = client.get("/")
    assert "/static/app.js" in r.text


def test_results_section_hidden_initially(client):
    r = client.get("/")
    body = r.text
    assert 'id="results"' in body
    assert "hidden" in body


def test_healthz_still_works(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_app_js_refine_seeds_from_additions_in_base_mode(client):
    """Refine must seed from recommended_additions (base mode), not the
    merged recommended_fleet, to avoid double-counting the base fleet."""
    body = client.get("/static/app.js").text
    assert "function getRefineSeed()" in body
    assert "recommended_additions" in body
    assert "seed_fleet: (useSeedFleet && lastResult) ? getRefineSeed() : null" in body


def test_app_js_has_reset_refine_state_wired_to_scenario_inputs(client):
    """Enemy/defense/tech/mode edits must reset the refine seed via a
    shared resetRefineState() helper."""
    body = client.get("/static/app.js").text
    assert "function resetRefineState(" in body
    assert "resetRefineState('Budget changed')" in body
    assert "resetRefineState('Tab changed')" in body
    assert "resetRefineState('Inputs changed')" in body
    # Wired to enemy + defense inputs, tech inputs, and mode radios.
    assert "SHIP_KEYS.concat(DEFENSE_KEYS).forEach" in body
    assert 'input[name="attacker_weapon"]' in body or '"attacker_weapon"' in body
    assert "input[name=\"mode\"]" in body


def test_app_js_history_load_disables_refine_for_old_entries(client):
    """Loading an old history snapshot must disable Refine (only the
    latest result may seed a refinement)."""
    body = client.get("/static/app.js").text
    assert "refineBtn.disabled = (idx !== 0);" in body

"""Tables in tests/fixtures/ogame_tables.json must match the Python source.

If this test fails, re-run tools/gen_ogame_tables.py to regenerate the
fixture (likely a table was edited in fast_combat.py without re-exporting).
"""
import json
from pathlib import Path
import pytest

from ogame_optimizer.core.fast_combat import (
    SHIP_STATS, DEFENSE_STATS, RAPIDFIRE, SHIP_COSTS_MCD, DEFENSE_COSTS_MCD,
)

S2P = {
    "light_fighter": "LightFighter", "heavy_fighter": "HeavyFighter",
    "cruiser": "Cruiser", "battleship": "Battleship",
    "battlecruiser": "Battlecruiser", "bomber": "Bomber",
    "destroyer": "Destroyer", "deathstar": "Deathstar",
    "small_cargo": "SmallCargo", "large_cargo": "LargeCargo",
    "recycler": "Recycler", "espionage_probe": "EspionageProbe",
    "pathfinder": "Pathfinder", "reaper": "Reaper",
    "solar_satellite": "SolarSatellite", "crawler": "Crawler",
    "rocket_launcher": "RocketLauncher", "light_laser": "LightLaser",
    "heavy_laser": "HeavyLaser", "gauss_cannon": "GaussCannon",
    "ion_cannon": "IonCannon", "plasma_turret": "PlasmaTurret",
    "small_shield_dome": "SmallShieldDome", "large_shield_dome": "LargeShieldDome",
}

FIXTURE = Path(__file__).parent.parent / "tests" / "fixtures" / "ogame_tables.json"

@pytest.fixture(scope="module")
def fixture():
    if not FIXTURE.exists():
        pytest.skip(f"Fixture not found: {FIXTURE} (run tools/gen_ogame_tables.py)")
    return json.loads(FIXTURE.read_text())


def test_ship_stats_drift(fixture):
    expected = {
        S2P[k]: {"atk": v["atk"], "shield": v["shield"], "hull": v["hull"]}
        for k, v in SHIP_STATS.items()
    }
    actual = fixture["ship_stats_pascal"]
    assert actual == expected, (
        f"ship_stats drift. Python-only: {set(expected) - set(actual)}. "
        f"Fixture-only: {set(actual) - set(expected)}. "
        "Run tools/gen_ogame_tables.py to regenerate."
    )


def test_defense_stats_drift(fixture):
    expected = {
        S2P[k]: {"atk": v["atk"], "shield": v["shield"], "hull": v["hull"]}
        for k, v in DEFENSE_STATS.items()
    }
    actual = fixture["defense_stats_pascal"]
    assert actual == expected, (
        f"defense_stats drift. Run tools/gen_ogame_tables.py."
    )


def test_rapidfire_drift(fixture):
    expected = sorted(
        [[S2P[s], S2P[t], rf] for (s, t), rf in RAPIDFIRE.items()]
    )
    actual = sorted([list(entry) for entry in fixture["rapidfire_pascal"]])
    assert actual == expected, (
        f"rapidfire drift. {len(expected)} Python vs {len(actual)} fixture. "
        "Run tools/gen_ogame_tables.py."
    )


def test_ship_costs_drift(fixture):
    expected = {S2P[k]: list(v) for k, v in SHIP_COSTS_MCD.items()}
    actual = fixture["ship_costs_mcd_pascal"]
    assert actual == expected, "ship_costs_mcd drift"


def test_defense_costs_drift(fixture):
    expected = {S2P[k]: list(v) for k, v in DEFENSE_COSTS_MCD.items()}
    actual = fixture["defense_costs_mcd_pascal"]
    assert actual == expected, "defense_costs_mcd drift"


def test_fixture_version(fixture):
    assert fixture.get("version") == 1, (
        f"fixture version {fixture.get('version')} unsupported; "
        "bump the version and update the Rust parser."
    )

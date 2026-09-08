"""Emit tests/fixtures/ogame_tables.json from the Python source of truth.

Single source of truth: Python's SHIP_STATS / DEFENSE_STATS / RAPIDFIRE /
SHIP_COSTS_MCD / DEFENSE_COSTS_MCD. The Rust analytical resolver reads the
generated JSON at compile time (build.rs -> include_bytes!) and a parity
test verifies Rust's compiled tables match the JSON at every test run.

Run: python tools/gen_ogame_tables.py
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from ogame_optimizer.core.fast_combat import (
    SHIP_STATS, DEFENSE_STATS, RAPIDFIRE, SHIP_COSTS_MCD, DEFENSE_COSTS_MCD,
)

def main():
    snake_to_pascal = {
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

    out = {
        "version": 1,
        "ship_stats_pascal": {
            snake_to_pascal[k]: {"atk": v["atk"], "shield": v["shield"], "hull": v["hull"]}
            for k, v in SHIP_STATS.items()
        },
        "defense_stats_pascal": {
            snake_to_pascal[k]: {"atk": v["atk"], "shield": v["shield"], "hull": v["hull"]}
            for k, v in DEFENSE_STATS.items()
        },
        "rapidfire_pascal": [
            [snake_to_pascal[s], snake_to_pascal[t], rf] for (s, t), rf in RAPIDFIRE.items()
        ],
        "ship_costs_mcd_pascal": {
            snake_to_pascal[k]: list(v) for k, v in SHIP_COSTS_MCD.items()
        },
        "defense_costs_mcd_pascal": {
            snake_to_pascal[k]: list(v) for k, v in DEFENSE_COSTS_MCD.items()
        },
    }

    out_path = os.path.join(
        os.path.dirname(__file__), "..", "tests", "fixtures", "ogame_tables.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"wrote {out_path}")
    print(f"  ship_stats: {len(out['ship_stats_pascal'])} entries")
    print(f"  defense_stats: {len(out['defense_stats_pascal'])} entries")
    print(f"  rapidfire: {len(out['rapidfire_pascal'])} entries")

if __name__ == "__main__":
    main()

//! Static OGame tables loaded at compile time from the JSON fixture.
//!
//! The JSON fixture is the single source of truth (Python exports it via
//! tools/gen_ogame_tables.py). A drift test (test_ogame_tables_drift.py)
//! catches any divergence between Python and the Rust-compiled view.
//!
//! All fields are PascalCase to match the existing UnitType / ShipType /
//! DefenseType enums in ships.rs - keeps the parser layer thin.

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::rapidfire::UnitType;
use crate::ships::{DefenseType, ShipType};

/// Per-ship combat stats (weapon, shield, hull).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShipStats {
    pub atk: u32,
    pub shield: u32,
    pub hull: u32,
}

/// Per-defense combat stats.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DefenseStats {
    pub atk: u32,
    pub shield: u32,
    pub hull: u32,
}

/// Metal/Crystal/Deuterium cost per unit (for debris).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostMCD {
    pub metal: u32,
    pub crystal: u32,
    pub deuterium: u32,
}

/// Build a ShipType from the PascalCase string used in the JSON fixture.
pub fn parse_ship_pascal(name: &str) -> Option<ShipType> {
    Some(match name {
        "LightFighter" => ShipType::LightFighter,
        "HeavyFighter" => ShipType::HeavyFighter,
        "Cruiser" => ShipType::Cruiser,
        "Battleship" => ShipType::Battleship,
        "Battlecruiser" => ShipType::Battlecruiser,
        "Bomber" => ShipType::Bomber,
        "Destroyer" => ShipType::Destroyer,
        "Deathstar" => ShipType::Deathstar,
        "SmallCargo" => ShipType::SmallCargo,
        "LargeCargo" => ShipType::LargeCargo,
        "EspionageProbe" => ShipType::EspionageProbe,
        "Reaper" => ShipType::Reaper,
        "Pathfinder" => ShipType::Pathfinder,
        "SolarSatellite" => ShipType::SolarSatellite,
        "Crawler" => ShipType::Crawler,
        "Recycler" => ShipType::Recycler,
        _ => return None,
    })
}

pub fn parse_defense_pascal(name: &str) -> Option<DefenseType> {
    Some(match name {
        "RocketLauncher" => DefenseType::RocketLauncher,
        "LightLaser" => DefenseType::LightLaser,
        "HeavyLaser" => DefenseType::HeavyLaser,
        "GaussCannon" => DefenseType::GaussCannon,
        "IonCannon" => DefenseType::IonCannon,
        "PlasmaTurret" => DefenseType::PlasmaTurret,
        "SmallShieldDome" => DefenseType::SmallShieldDome,
        "LargeShieldDome" => DefenseType::LargeShieldDome,
        _ => return None,
    })
}

/// Loaded analytical tables. Cheap to clone (just references + small vecs).
#[derive(Debug, Clone)]
pub struct AnalyticalTables {
    pub ship_stats: HashMap<UnitType, ShipStats>,
    pub defense_stats: HashMap<DefenseType, DefenseStats>,
    /// Rapidfire: shooter -> target -> shots multiplier (e.g. 3 means 3 shots/round).
    pub rapidfire: HashMap<UnitType, HashMap<UnitType, u32>>,
    pub ship_costs: HashMap<UnitType, CostMCD>,
    pub defense_costs: HashMap<DefenseType, CostMCD>,
}

static TABLES: OnceLock<AnalyticalTables> = OnceLock::new();

/// Public entry: get the parsed tables (parses lazily on first call).
pub fn tables() -> &'static AnalyticalTables {
    TABLES.get_or_init(load)
}

/// Parse the embedded JSON fixture into structured tables.
fn load() -> AnalyticalTables {
    let json = include_str!("../tests/fixtures/ogame_tables.json");
    let parsed: serde_json::Value =
        serde_json::from_str(json).expect("embedded ogame_tables.json must parse");

    let mut ship_stats = HashMap::new();
    if let Some(obj) = parsed.get("ship_stats_pascal").and_then(|v| v.as_object()) {
        for (k, v) in obj {
            let Some(ship) = parse_ship_pascal(k) else {
                // Any future Python-only civil ships have no Rust
                // ShipType. Skip silently.
                continue;
            };
            ship_stats.insert(
                UnitType::Ship(ship),
                ShipStats {
                    atk: v["atk"].as_u64().unwrap() as u32,
                    shield: v["shield"].as_u64().unwrap() as u32,
                    hull: v["hull"].as_u64().unwrap() as u32,
                },
            );
        }
    }

    let mut defense_stats = HashMap::new();
    if let Some(obj) = parsed.get("defense_stats_pascal").and_then(|v| v.as_object()) {
        for (k, v) in obj {
            let Some(def) = parse_defense_pascal(k) else {
                panic!("unknown defense in fixture: {}", k)
            };
            defense_stats.insert(
                def,
                DefenseStats {
                    atk: v["atk"].as_u64().unwrap() as u32,
                    shield: v["shield"].as_u64().unwrap() as u32,
                    hull: v["hull"].as_u64().unwrap() as u32,
                },
            );
        }
    }

    let mut rapidfire: HashMap<UnitType, HashMap<UnitType, u32>> = HashMap::new();
    if let Some(arr) = parsed.get("rapidfire_pascal").and_then(|v| v.as_array()) {
        for entry in arr {
            let a = entry[0].as_str().expect("RF entry[0] must be string");
            let b = entry[1].as_str().expect("RF entry[1] must be string");
            let rf = entry[2].as_u64().expect("RF entry[2] must be int") as u32;
            let Some(shooter_ship) = parse_ship_pascal(a) else {
                panic!("unknown RF shooter: {}", a)
            };
            // RF targets can be ships or defenses (DS vs shield dome, etc.)
            let shooter = UnitType::Ship(shooter_ship);
            let target = if let Some(s) = parse_ship_pascal(b) {
                UnitType::Ship(s)
            } else if let Some(d) = parse_defense_pascal(b) {
                UnitType::Defense(d)
            } else {
                // Any future Python-only civil ships have no Rust
                // UnitType; skip their RF entries.
                continue;
            };
            rapidfire.entry(shooter).or_default().insert(target, rf);
        }
    }

    let mut ship_costs = HashMap::new();
    if let Some(obj) = parsed.get("ship_costs_mcd_pascal").and_then(|v| v.as_object()) {
        for (k, v) in obj {
            let Some(ship) = parse_ship_pascal(k) else {
                continue;  // unknown ship - skip
            };
            let arr = v.as_array().expect("cost must be array");
            ship_costs.insert(
                UnitType::Ship(ship),
                CostMCD {
                    metal: arr[0].as_u64().unwrap() as u32,
                    crystal: arr[1].as_u64().unwrap() as u32,
                    deuterium: arr[2].as_u64().unwrap() as u32,
                },
            );
        }
    }

    let mut defense_costs = HashMap::new();
    if let Some(obj) = parsed.get("defense_costs_mcd_pascal").and_then(|v| v.as_object()) {
        for (k, v) in obj {
            let Some(def) = parse_defense_pascal(k) else {
                panic!("unknown defense in costs: {}", k)
            };
            let arr = v.as_array().expect("cost must be array");
            defense_costs.insert(
                def,
                CostMCD {
                    metal: arr[0].as_u64().unwrap() as u32,
                    crystal: arr[1].as_u64().unwrap() as u32,
                    deuterium: arr[2].as_u64().unwrap() as u32,
                },
            );
        }
    }

    AnalyticalTables {
        ship_stats,
        defense_stats,
        rapidfire,
        ship_costs,
        defense_costs,
    }
}

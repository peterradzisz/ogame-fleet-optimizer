pub mod combat;
pub mod rapidfire;
pub mod ships;

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::combat::{CombatInput, CombatResult, Side, TechLevels};
use crate::rapidfire::UnitType;
use crate::ships::{defense_stats, ship_stats, DefenseType, ShipType};

/// Parse a ship name string into a ShipType variant.
fn parse_ship(name: &str) -> PyResult<ShipType> {
    match name {
        "LightFighter" => Ok(ShipType::LightFighter),
        "HeavyFighter" => Ok(ShipType::HeavyFighter),
        "Cruiser" => Ok(ShipType::Cruiser),
        "Battleship" => Ok(ShipType::Battleship),
        "Battlecruiser" => Ok(ShipType::Battlecruiser),
        "Bomber" => Ok(ShipType::Bomber),
        "Destroyer" => Ok(ShipType::Destroyer),
        "Deathstar" => Ok(ShipType::Deathstar),
        "SmallCargo" => Ok(ShipType::SmallCargo),
        "LargeCargo" => Ok(ShipType::LargeCargo),
        "EspionageProbe" => Ok(ShipType::EspionageProbe),
        "Reaper" => Ok(ShipType::Reaper),
        "Pathfinder" => Ok(ShipType::Pathfinder),
        "SolarSatellite" => Ok(ShipType::SolarSatellite),
        "Crawler" => Ok(ShipType::Crawler),
        other => Err(pyo3::exceptions::PyValueError::new_err(
            format!("Unknown ship type: {}", other)
        )),
    }
}

/// Parse a defense name string into a DefenseType variant.
fn parse_defense(name: &str) -> PyResult<DefenseType> {
    match name {
        "RocketLauncher" => Ok(DefenseType::RocketLauncher),
        "LightLaser" => Ok(DefenseType::LightLaser),
        "HeavyLaser" => Ok(DefenseType::HeavyLaser),
        "GaussCannon" => Ok(DefenseType::GaussCannon),
        "IonCannon" => Ok(DefenseType::IonCannon),
        "PlasmaTurret" => Ok(DefenseType::PlasmaTurret),
        "SmallShieldDome" => Ok(DefenseType::SmallShieldDome),
        "LargeShieldDome" => Ok(DefenseType::LargeShieldDome),
        other => Err(pyo3::exceptions::PyValueError::new_err(
            format!("Unknown defense type: {}", other)
        )),
    }
}

/// Convert a Python dict {ship_name: count} to a Fleet HashMap.
fn py_to_fleet(py_dict: &Bound<'_, PyDict>) -> PyResult<HashMap<UnitType, u64>> {
    let mut fleet = HashMap::new();
    for (key, value) in py_dict.iter() {
        let name: String = key.extract()?;
        let count: u64 = value.extract()?;
        let ship = parse_ship(&name)?;
        fleet.insert(UnitType::Ship(ship), count);
    }
    Ok(fleet)
}

/// Convert a Python dict {defense_name: count} to a defender_defenses HashMap.
fn py_to_defenses(py_dict: &Bound<'_, PyDict>) -> PyResult<HashMap<DefenseType, u64>> {
    let mut defenses = HashMap::new();
    for (key, value) in py_dict.iter() {
        let name: String = key.extract()?;
        let count: u64 = value.extract()?;
        let def = parse_defense(&name)?;
        defenses.insert(def, count);
    }
    Ok(defenses)
}

/// Convert a Python tuple (weapon, shield, armor) to TechLevels.
fn py_to_tech(py_tuple: &Bound<'_, pyo3::PyAny>) -> PyResult<TechLevels> {
    let w: u8 = py_tuple.get_item(0)?.extract()?;
    let s: u8 = py_tuple.get_item(1)?.extract()?;
    let a: u8 = py_tuple.get_item(2)?.extract()?;
    Ok(TechLevels { weapon: w, shield: s, armor: a })
}

/// Convert a Fleet HashMap to a Python dict.
fn fleet_to_py(py: Python, fleet: &HashMap<UnitType, u64>) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    for (unit, count) in fleet {
        let name = match unit {
            UnitType::Ship(s) => ship_to_str(*s),
            UnitType::Defense(_) => continue, // defenses go in their own dict
        };
        dict.set_item(name, count)?;
    }
    Ok(dict.into())
}

fn ship_to_str(s: ShipType) -> &'static str {
    match s {
        ShipType::LightFighter => "LightFighter",
        ShipType::HeavyFighter => "HeavyFighter",
        ShipType::Cruiser => "Cruiser",
        ShipType::Battleship => "Battleship",
        ShipType::Battlecruiser => "Battlecruiser",
        ShipType::Bomber => "Bomber",
        ShipType::Destroyer => "Destroyer",
        ShipType::Deathstar => "Deathstar",
        ShipType::SmallCargo => "SmallCargo",
        ShipType::LargeCargo => "LargeCargo",
        ShipType::EspionageProbe => "EspionageProbe",
        ShipType::Reaper => "Reaper",
        ShipType::Pathfinder => "Pathfinder",
        ShipType::SolarSatellite => "SolarSatellite",
        ShipType::Crawler => "Crawler",
    }
}

fn defense_to_str(d: DefenseType) -> &'static str {
    match d {
        DefenseType::RocketLauncher => "RocketLauncher",
        DefenseType::LightLaser => "LightLaser",
        DefenseType::HeavyLaser => "HeavyLaser",
        DefenseType::GaussCannon => "GaussCannon",
        DefenseType::IonCannon => "IonCannon",
        DefenseType::PlasmaTurret => "PlasmaTurret",
        DefenseType::SmallShieldDome => "SmallShieldDome",
        DefenseType::LargeShieldDome => "LargeShieldDome",
    }
}

fn defenses_to_py(py: Python, defenses: &HashMap<DefenseType, u64>) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    for (def, count) in defenses {
        dict.set_item(defense_to_str(*def), count)?;
    }
    Ok(dict.into())
}

fn side_to_str(side: Side) -> &'static str {
    match side {
        Side::Attacker => "Attacker",
        Side::Defender => "Defender",
        Side::Draw => "Draw",
    }
}

/// Compute approximate loss in resources for a fleet (metal+crystal+deuterium of survivors).
fn fleet_value(fleet: &HashMap<UnitType, u64>) -> f64 {
    let mut total: f64 = 0.0;
    for (unit, count) in fleet {
        let stats = match unit {
            UnitType::Ship(s) => ship_stats(*s),
            UnitType::Defense(d) => defense_stats(*d),
        };
        total += (stats.cost_metal + stats.cost_crystal + stats.cost_deuterium) as f64 * *count as f64;
    }
    total
}

/// Compute approximate loss in resources for a defense map.
fn defense_value(defenses: &HashMap<DefenseType, u64>) -> f64 {
    let mut total: f64 = 0.0;
    for (def, count) in defenses {
        let stats = defense_stats(*def);
        total += (stats.cost_metal + stats.cost_crystal + stats.cost_deuterium) as f64 * *count as f64;
    }
    total
}

/// Simulate a single combat. Returns a dict with combat results.
#[pyfunction]
fn simulate_combat_py<'py>(
    py: Python<'py>,
    attacker: &Bound<'_, PyDict>,
    defender: &Bound<'_, PyDict>,
    defender_defenses: &Bound<'_, PyDict>,
    attacker_tech: &Bound<'_, pyo3::PyAny>,
    defender_tech: &Bound<'_, pyo3::PyAny>,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let a = py_to_fleet(attacker)?;
    let d = py_to_fleet(defender)?;
    let ds = py_to_defenses(defender_defenses)?;
    let at = py_to_tech(attacker_tech)?;
    let dt = py_to_tech(defender_tech)?;
    let input = CombatInput { attacker: a, defender: d, defender_defenses: ds, attacker_tech: at, defender_tech: dt, seed };

    // Run inside allow_threads to release GIL
    let result: CombatResult = py.allow_threads(|| crate::combat::simulate_combat(&input));

    let dict = PyDict::new_bound(py);
    dict.set_item("winner", side_to_str(result.winner))?;
    dict.set_item("rounds_fought", result.rounds_fought)?;
    dict.set_item("attacker_survivors", fleet_to_py(py, &result.attacker_survivors)?)?;
    dict.set_item("defender_survivors", fleet_to_py(py, &result.defender_survivors)?)?;
    dict.set_item("defender_defense_survivors", defenses_to_py(py, &result.defender_defense_survivors)?)?;
    dict.set_item("debris_metal", result.debris_metal)?;
    dict.set_item("debris_crystal", result.debris_crystal)?;

    // Compute approximate resource losses (initial_value - survivor_value)
    let initial_att_val = fleet_value(&input.attacker) + fleet_value(&input.defender_defenses.iter()
        .map(|(d, _)| UnitType::Defense(*d))
        .zip(input.defender_defenses.values())
        .map(|(u, c)| (u, *c))
        .collect::<HashMap<_, _>>());
    let _ = initial_att_val; // not exposed yet
    Ok(dict)
}

/// Run N simulations and return aggregate stats (mean attacker loss, stddev, win probability).
/// `attacker_initial_value` and `defender_initial_value` are the resource values of the
/// initial forces; losses are computed as initial - survivor values.
#[pyfunction]
fn simulate_batch_py<'py>(
    py: Python<'py>,
    attacker: &Bound<'_, PyDict>,
    defender: &Bound<'_, PyDict>,
    defender_defenses: &Bound<'_, PyDict>,
    attacker_tech: &Bound<'_, pyo3::PyAny>,
    defender_tech: &Bound<'_, pyo3::PyAny>,
    n_sims: u32,
    base_seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let a = py_to_fleet(attacker)?;
    let d = py_to_fleet(defender)?;
    let ds = py_to_defenses(defender_defenses)?;
    let at = py_to_tech(attacker_tech)?;
    let dt = py_to_tech(defender_tech)?;
    let input = CombatInput { attacker: a, defender: d, defender_defenses: ds, attacker_tech: at, defender_tech: dt, seed: 0 };

    // Initial values
    let initial_att_val = fleet_value(&input.attacker);
    let initial_def_val = fleet_value(&input.defender) + defense_value(&input.defender_defenses);

    // Run all sims in Rust (no GIL, fast)
    let results: Vec<CombatResult> = py.allow_threads(|| {
        (0..n_sims).map(|i| {
            let mut input_i = input.clone();
            input_i.seed = base_seed.wrapping_add(i as u64);
            crate::combat::simulate_combat(&input_i)
        }).collect()
    });

    // Aggregate
    let mut attacker_losses: Vec<f64> = Vec::with_capacity(n_sims as usize);
    let mut defender_losses: Vec<f64> = Vec::with_capacity(n_sims as usize);
    let mut wins: u32 = 0;
    let mut losses: u32 = 0;
    let mut draws: u32 = 0;
    for r in &results {
        let att_loss = initial_att_val - fleet_value(&r.attacker_survivors);
        let def_loss = initial_def_val - fleet_value(&r.defender_survivors) - defense_value(&r.defender_defense_survivors);
        attacker_losses.push(att_loss);
        defender_losses.push(def_loss);
        match r.winner {
            Side::Attacker => wins += 1,
            Side::Defender => losses += 1,
            Side::Draw => draws += 1,
        }
    }

    let mean_al = mean(&attacker_losses);
    let stddev_al = stddev(&attacker_losses, mean_al);
    let mean_dl = mean(&defender_losses);
    let win_prob = wins as f64 / n_sims as f64;

    let dict = PyDict::new_bound(py);
    dict.set_item("mean_attacker_loss", mean_al)?;
    dict.set_item("stddev_attacker_loss", stddev_al)?;
    dict.set_item("mean_defender_loss", mean_dl)?;
    dict.set_item("win_probability", win_prob)?;
    dict.set_item("wins", wins)?;
    dict.set_item("losses", losses)?;
    dict.set_item("draws", draws)?;
    dict.set_item("sims_run", n_sims)?;
    dict.set_item("seed_used", base_seed)?;
    Ok(dict)
}

/// Evaluate a population of attacker fleets against the same defender. Returns a list
/// of batch results (one per attacker fleet). GA batch API: avoids per-sim Python->Rust
/// calls by running the whole population in Rust.
#[pyfunction]
fn evaluate_population_py<'py>(
    py: Python<'py>,
    attacker_fleets: &Bound<'_, PyList>,
    defender: &Bound<'_, PyDict>,
    defender_defenses: &Bound<'_, PyDict>,
    attacker_tech: &Bound<'_, pyo3::PyAny>,
    defender_tech: &Bound<'_, pyo3::PyAny>,
    n_sims_per_fleet: u32,
    base_seed: u64,
) -> PyResult<Bound<'py, PyList>> {
    let d = py_to_fleet(defender)?;
    let ds = py_to_defenses(defender_defenses)?;
    let at = py_to_tech(attacker_tech)?;
    let dt = py_to_tech(defender_tech)?;

    // Parse all attacker fleets
    let mut attackers = Vec::with_capacity(attacker_fleets.len());
    for item in attacker_fleets.iter() {
        let dict: &Bound<'_, PyDict> = item.downcast()?;
        attackers.push(py_to_fleet(dict)?);
    }

    let _initial_def_val = fleet_value(&d) + defense_value(&ds);

    // Run all attackers x all sims in Rust
    let results_per_fleet: Vec<Vec<CombatResult>> = py.allow_threads(|| {
        attackers.iter().enumerate().map(|(i, a)| {
            (0..n_sims_per_fleet).map(|s| {
                let input = CombatInput {
                    attacker: a.clone(),
                    defender: d.clone(),
                    defender_defenses: ds.clone(),
                    attacker_tech: at,
                    defender_tech: dt,
                    seed: base_seed.wrapping_add((i * n_sims_per_fleet as usize + s as usize) as u64),
                };
                crate::combat::simulate_combat(&input)
            }).collect()
        }).collect()
    });

    // Aggregate per fleet
    let list = PyList::empty_bound(py);
    for (i, results) in results_per_fleet.iter().enumerate() {
        let initial_att_val = fleet_value(&attackers[i]);
        let mut attacker_losses: Vec<f64> = Vec::with_capacity(n_sims_per_fleet as usize);
        let mut wins: u32 = 0;
        for r in results.iter() {
            let att_loss = initial_att_val - fleet_value(&r.attacker_survivors);
            attacker_losses.push(att_loss);
            match r.winner {
                Side::Attacker => wins += 1,
                _ => {}
            }
        }
        let mean_al = mean(&attacker_losses);
        let stddev_al = stddev(&attacker_losses, mean_al);
        let dict = PyDict::new_bound(py);
        dict.set_item("mean_attacker_loss", mean_al)?;
        dict.set_item("stddev_attacker_loss", stddev_al)?;
        dict.set_item("win_probability", wins as f64 / n_sims_per_fleet as f64)?;
        dict.set_item("sims_run", n_sims_per_fleet)?;
        list.append(dict)?;
    }
    Ok(list)
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() { return 0.0; }
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn stddev(xs: &[f64], mean: f64) -> f64 {
    if xs.len() < 2 { return 0.0; }
    let variance = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (xs.len() - 1) as f64;
    variance.sqrt()
}

/// PyO3 module entrypoint.
#[pymodule]
fn _ogame_combat(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(simulate_combat_py, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_batch_py, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_population_py, m)?)?;
    Ok(())
}

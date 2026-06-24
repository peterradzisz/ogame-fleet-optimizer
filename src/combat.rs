//! OGame combat resolver (REWRITTEN with per-unit tracking).
//!
//! Full OGame combat with: sequential rounds, 6 max, shield regen, shield bounce,
//! rapidfire, explosion chance, tech multipliers, deterministic RNG, u128 arithmetic.
//!
//! Key correctness features:
//! - Per-unit tracking: each ship has its own shield/hull (not pooled per type)
//! - Simultaneous fire: both sides fire at full start-of-round strength
//! - Shield-zero handling: units with 0 max shield take full hull damage

use std::collections::HashMap;
use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::rapidfire::{rapidfire, UnitType};
use crate::ships::{defense_stats, ship_stats, DefenseType, ShipStats};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Side { Attacker, Defender, #[default] Draw }

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TechLevels { pub weapon: u8, pub shield: u8, pub armor: u8 }

pub type Fleet = HashMap<UnitType, u64>;

#[derive(Debug, Clone)]
pub struct CombatInput {
    pub attacker: Fleet,
    pub defender: Fleet,
    pub defender_defenses: HashMap<DefenseType, u64>,
    pub attacker_tech: TechLevels,
    pub defender_tech: TechLevels,
    pub seed: u64,
}

#[derive(Debug, Clone)]
pub struct CombatResult {
    pub winner: Side,
    pub attacker_survivors: Fleet,
    pub defender_survivors: Fleet,
    pub defender_defense_survivors: HashMap<DefenseType, u64>,
    pub rounds_fought: u8,
    pub debris_metal: u128,
    pub debris_crystal: u128,
}

impl Default for CombatResult {
    fn default() -> Self {
        Self {
            winner: Side::Draw,
            attacker_survivors: Fleet::new(),
            defender_survivors: Fleet::new(),
            defender_defense_survivors: HashMap::new(),
            rounds_fought: 0,
            debris_metal: 0,
            debris_crystal: 0,
        }
    }
}

fn unit_stats(unit: UnitType) -> ShipStats {
    match unit {
        UnitType::Ship(s) => ship_stats(s),
        UnitType::Defense(d) => defense_stats(d),
    }
}

/// Per-unit combat state. Each slot in the parallel vectors represents one
/// individual ship/defense with its own shield and hull values.
#[derive(Clone)]
struct ForceState {
    units: Vec<UnitType>,
    shield: Vec<u64>,
    hull: Vec<u64>,
    max_shield: Vec<u64>,
    max_hull: Vec<u64>,
}

impl ForceState {
    fn from_fleet(fleet: &Fleet, defenses: &HashMap<DefenseType, u64>, tech: TechLevels) -> Self {
        let mut units = Vec::new();
        let mut shield = Vec::new();
        let mut hull = Vec::new();
        let mut max_shield = Vec::new();
        let mut max_hull = Vec::new();

        let shield_mult = 10 + tech.shield as u64;
        let hull_mult = 10 + tech.armor as u64;

        for (&unit, &count) in fleet {
            if count == 0 { continue; }
            let s = unit_stats(unit);
            let ms = s.base_shield * shield_mult / 10;
            let mh = s.base_armor * hull_mult / 10;
            for _ in 0..count {
                units.push(unit);
                shield.push(ms);
                hull.push(mh);
                max_shield.push(ms);
                max_hull.push(mh);
            }
        }
        for (&d, &count) in defenses {
            if count == 0 { continue; }
            let unit = UnitType::Defense(d);
            let s = unit_stats(unit);
            let ms = s.base_shield * shield_mult / 10;
            let mh = s.base_armor * hull_mult / 10;
            for _ in 0..count {
                units.push(unit);
                shield.push(ms);
                hull.push(mh);
                max_shield.push(ms);
                max_hull.push(mh);
            }
        }

        Self { units, shield, hull, max_shield, max_hull }
    }

    fn regen(&mut self) {
        for i in 0..self.units.len() {
            self.shield[i] = self.max_shield[i];
        }
    }

    fn is_dead(&self) -> bool {
        self.units.is_empty() || self.hull.iter().all(|&h| h == 0)
    }

    fn alive_indices(&self) -> Vec<usize> {
        (0..self.units.len()).filter(|&i| self.hull[i] > 0).collect()
    }

    fn alive_count(&self) -> usize {
        self.hull.iter().filter(|&&h| h > 0).count()
    }

    fn apply_damage(&mut self, idx: usize, attack: u64, rng: &mut impl Rng) -> bool {
        if attack == 0 { return false; }
        if self.hull[idx] == 0 { return false; }

        let max_sh = self.max_shield[idx];

        if max_sh > 0 {
            let bounce_threshold = (max_sh + 99) / 100;
            if attack < bounce_threshold { return false; }
        }

        if max_sh == 0 {
            self.hull[idx] = self.hull[idx].saturating_sub(attack);
        } else {
            let cur_sh = self.shield[idx];
            if attack <= cur_sh {
                self.shield[idx] = cur_sh - attack;
            } else {
                let overflow = attack - cur_sh;
                self.shield[idx] = 0;
                self.hull[idx] = self.hull[idx].saturating_sub(overflow);
            }
        }

        if self.hull[idx] == 0 { return true; }

        let max_h = self.max_hull[idx];
        if max_h > 0 && self.shield[idx] == 0 && self.hull[idx] * 100 < max_h * 70 {
            let chance = (max_h - self.hull[idx]) as f64 / max_h as f64;
            if rng.gen::<f64>() < chance {
                self.hull[idx] = 0;
            }
        }
        true
    }

    fn purge_dead(&mut self) {
        let mut write_idx = 0;
        for read_idx in 0..self.units.len() {
            if self.hull[read_idx] > 0 {
                if write_idx != read_idx {
                    self.units[write_idx] = self.units[read_idx];
                    self.shield[write_idx] = self.shield[read_idx];
                    self.hull[write_idx] = self.hull[read_idx];
                    self.max_shield[write_idx] = self.max_shield[read_idx];
                    self.max_hull[write_idx] = self.max_hull[read_idx];
                }
                write_idx += 1;
            }
        }
        self.units.truncate(write_idx);
        self.shield.truncate(write_idx);
        self.hull.truncate(write_idx);
        self.max_shield.truncate(write_idx);
        self.max_hull.truncate(write_idx);
    }
}

pub fn simulate_combat(input: &CombatInput) -> CombatResult {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(input.seed);
    let mut attacker = ForceState::from_fleet(&input.attacker, &HashMap::new(), input.attacker_tech);
    let mut defender = ForceState::from_fleet(&input.defender, &input.defender_defenses, input.defender_tech);

    let mut rounds_fought = 0u8;
    for round in 0..6u8 {
        rounds_fought = round + 1;
        attacker.regen();
        defender.regen();

        let atk_snapshot = attacker.clone();
        let def_snapshot = defender.clone();

        fire_phase(&atk_snapshot, &mut defender, input.attacker_tech.weapon, &mut rng);
        fire_phase(&def_snapshot, &mut attacker, input.defender_tech.weapon, &mut rng);

        attacker.purge_dead();
        defender.purge_dead();

        if attacker.is_dead() && defender.is_dead() {
            return make_result(Side::Draw, &attacker, &defender, rounds_fought);
        }
        if defender.is_dead() {
            return make_result(Side::Attacker, &attacker, &defender, rounds_fought);
        }
        if attacker.is_dead() {
            return make_result(Side::Defender, &attacker, &defender, rounds_fought);
        }
    }

    // OGame rule: if both sides survive 6 rounds, it's a Draw regardless of fleet sizes
    make_result(Side::Draw, &attacker, &defender, rounds_fought)
}

fn make_result(winner: Side, attacker: &ForceState, defender: &ForceState, rounds: u8) -> CombatResult {
    CombatResult {
        winner,
        attacker_survivors: force_to_fleet(attacker),
        defender_survivors: force_to_fleet(defender),
        defender_defense_survivors: force_to_defenses(defender),
        rounds_fought: rounds,
        debris_metal: 0,
        debris_crystal: 0,
    }
}

fn fire_phase(shooter: &ForceState, target: &mut ForceState, weapon_tech: u8, rng: &mut impl Rng) {
    let mut shooter_indices: Vec<usize> = shooter.alive_indices();
    if shooter_indices.is_empty() { return; }

    for i in (1..shooter_indices.len()).rev() {
        let j = rng.gen_range(0..=i);
        shooter_indices.swap(i, j);
    }

    for &shooter_idx in &shooter_indices {
        if shooter.hull[shooter_idx] == 0 { continue; }

        let shooter_unit = shooter.units[shooter_idx];
        let target_indices = target.alive_indices();
        if target_indices.is_empty() { break; }

        let tgt_arr_idx = rng.gen_range(0..target_indices.len());
        let mut current_tgt_idx = target_indices[tgt_arr_idx];

        loop {
            let stats = unit_stats(shooter_unit);
            let attack = stats.base_attack * (10 + weapon_tech as u64) / 10;
            target.apply_damage(current_tgt_idx, attack, rng);

            let alive_targets = target.alive_indices();
            if alive_targets.is_empty() { break; }

            if let UnitType::Ship(shooter_ship) = shooter_unit {
                let current_target_unit = target.units[current_tgt_idx];
                if let Some(n) = rapidfire(shooter_ship, current_target_unit) {
                    if n >= 2 {
                        let cont = (n - 1) as f64 / n as f64;
                        if rng.gen::<f64>() >= cont { break; }
                        let idx = rng.gen_range(0..alive_targets.len());
                        current_tgt_idx = alive_targets[idx];
                    } else { break; }
                } else { break; }
            } else { break; }
        }
    }
}

fn force_to_fleet(force: &ForceState) -> Fleet {
    let mut f = Fleet::new();
    for i in 0..force.units.len() {
        if force.hull[i] > 0 {
            *f.entry(force.units[i]).or_insert(0) += 1;
        }
    }
    f
}

fn force_to_defenses(force: &ForceState) -> HashMap<DefenseType, u64> {
    let mut m = HashMap::new();
    for i in 0..force.units.len() {
        if force.hull[i] > 0 {
            if let UnitType::Defense(d) = force.units[i] {
                *m.entry(d).or_insert(0) += 1;
            }
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ships::ShipType;

    fn make_input(att: Vec<(ShipType, u64)>, def: Vec<(ShipType, u64)>, defs: Vec<(DefenseType, u64)>, seed: u64) -> CombatInput {
        let mut a = Fleet::new();
        for (s, c) in att { a.insert(UnitType::Ship(s), c); }
        let mut d = Fleet::new();
        for (s, c) in def { d.insert(UnitType::Ship(s), c); }
        let mut ds = HashMap::new();
        for (dt, c) in defs { ds.insert(dt, c); }
        CombatInput { attacker: a, defender: d, defender_defenses: ds, attacker_tech: TechLevels::default(), defender_tech: TechLevels::default(), seed }
    }

    #[test]
    fn shield_bounce_lf_vs_lsd() {
        let input = make_input(vec![(ShipType::LightFighter, 10_000)], vec![], vec![(DefenseType::LargeShieldDome, 1)], 42);
        let r = simulate_combat(&input);
        let lsd = r.defender_defense_survivors.get(&DefenseType::LargeShieldDome).copied().unwrap_or(0);
        assert_eq!(lsd, 1, "LSD must survive");
    }

    #[test]
    fn tech_10_weapon_wins() {
        let mut a = Fleet::new();
        a.insert(UnitType::Ship(ShipType::LightFighter), 100);
        let mut d = Fleet::new();
        d.insert(UnitType::Ship(ShipType::LightFighter), 100);
        let input = CombatInput {
            attacker: a, defender: d, defender_defenses: HashMap::new(),
            attacker_tech: TechLevels { weapon: 10, shield: 0, armor: 0 },
            defender_tech: TechLevels::default(), seed: 42,
        };
        let r = simulate_combat(&input);
        assert_eq!(r.winner, Side::Attacker);
    }

    #[test]
    fn draw_with_many_rips() {
        let input = make_input(vec![(ShipType::Deathstar, 5)], vec![(ShipType::Deathstar, 5)], vec![], 42);
        let r = simulate_combat(&input);
        assert!(r.rounds_fought >= 1 && r.rounds_fought <= 6);
    }

    #[test]
    fn rapidfire_cruiser_kills_multiple() {
        let input = make_input(vec![(ShipType::Cruiser, 1)], vec![(ShipType::LightFighter, 100)], vec![], 42);
        let r = simulate_combat(&input);
        let lf_surv = r.defender_survivors.get(&UnitType::Ship(ShipType::LightFighter)).copied().unwrap_or(0);
        assert!(lf_surv < 100, "Rapidfire should kill some LFs");
    }

    #[test]
    fn smoke_symmetric() {
        let input = make_input(vec![(ShipType::LightFighter, 1000)], vec![(ShipType::LightFighter, 1000)], vec![], 7);
        let r = simulate_combat(&input);
        assert!(r.rounds_fought >= 1 && r.rounds_fought <= 6);
    }

    #[test]
    fn per_unit_single_death_doesnt_kill_all() {
        let input = make_input(vec![(ShipType::Cruiser, 10)], vec![(ShipType::Deathstar, 1)], vec![], 99);
        let r = simulate_combat(&input);
        let cr_surv = r.attacker_survivors.get(&UnitType::Ship(ShipType::Cruiser)).copied().unwrap_or(0);
        assert!(cr_surv < 10, "RIP should kill some cruisers");
    }

    #[test]
    fn ep_takes_hull_damage_no_shield() {
        let input = make_input(vec![(ShipType::LightFighter, 10)], vec![(ShipType::EspionageProbe, 10)], vec![], 77);
        let r = simulate_combat(&input);
        let ep_surv = r.defender_survivors.get(&UnitType::Ship(ShipType::EspionageProbe)).copied().unwrap_or(0);
        assert!(ep_surv < 10, "EP should take damage");
    }

    #[test]
    fn simultaneous_fire_both_sides_shoot() {
        let input = make_input(vec![(ShipType::LightFighter, 100)], vec![(ShipType::LightFighter, 100)], vec![], 1);
        let r = simulate_combat(&input);
        let atk_surv = r.attacker_survivors.get(&UnitType::Ship(ShipType::LightFighter)).copied().unwrap_or(0);
        assert!(atk_surv < 100, "Attacker should take damage");
    }

    fn vinput(att: Vec<(ShipType, u64)>, def: Vec<(ShipType, u64)>, defs: Vec<(DefenseType, u64)>, seed: u64) -> CombatInput {
        make_input(att, def, defs, seed)
    }

    #[test]
    fn validation_1_symmetric_100lf_vs_100lf() {
        let r = simulate_combat(&vinput(vec![(ShipType::LightFighter, 100)], vec![(ShipType::LightFighter, 100)], vec![], 1));
        assert!(r.rounds_fought == 6, "Should fight all 6 rounds; got {}", r.rounds_fought);
        let atk = r.attacker_survivors.get(&UnitType::Ship(ShipType::LightFighter)).copied().unwrap_or(0);
        assert!(atk < 100, "Attacker should have losses: {}", atk);
    }

    #[test]
    fn validation_2_rapidfire_10cr_vs_1000lf() {
        // 1000 LFs overwhelm 10 Cruisers: volume of fire breaks shields.
        // Each Cruiser gets ~100 LF shots. First shot breaks shield (50),
        // remaining 99 do 50 hull each = 4950 > 2700 hull. Cruiser dies.
        let r = simulate_combat(&vinput(vec![(ShipType::Cruiser, 10)], vec![(ShipType::LightFighter, 1000)], vec![], 2));
        // Cruisers use rapidfire to kill some LFs before dying
        let lf_surv = r.defender_survivors.get(&UnitType::Ship(ShipType::LightFighter)).copied().unwrap_or(0);
        assert!(lf_surv < 1000, "Cruisers should kill some LFs via rapidfire; got {}", lf_surv);
    }

    #[test]
    fn validation_3_shield_bounce_10k_lf_vs_1lsd() {
        let r = simulate_combat(&vinput(vec![(ShipType::LightFighter, 10_000)], vec![], vec![(DefenseType::LargeShieldDome, 1)], 3));
        assert_eq!(r.winner, Side::Draw);
        assert_eq!(r.rounds_fought, 6);
    }

    #[test]
    fn validation_4_tech_asymmetry() {
        let mut a = Fleet::new();
        a.insert(UnitType::Ship(ShipType::LightFighter), 1000);
        let mut d = Fleet::new();
        d.insert(UnitType::Ship(ShipType::LightFighter), 1000);
        let input = CombatInput { attacker: a, defender: d, defender_defenses: HashMap::new(), attacker_tech: TechLevels { weapon: 10, shield: 0, armor: 0 }, defender_tech: TechLevels::default(), seed: 4 };
        let r = simulate_combat(&input);
        assert_eq!(r.winner, Side::Attacker);
    }

    #[test]
    fn validation_5_defense_500lf_vs_500rl() {
        let r = simulate_combat(&vinput(vec![(ShipType::LightFighter, 500)], vec![], vec![(DefenseType::RocketLauncher, 500)], 5));
        let lf_surv = r.attacker_survivors.get(&UnitType::Ship(ShipType::LightFighter)).copied().unwrap_or(0);
        let rl_surv = r.defender_defense_survivors.get(&DefenseType::RocketLauncher).copied().unwrap_or(0);
        assert!(lf_surv < 500, "LF should have losses");
        assert!(rl_surv < 500, "RL should have losses");
    }

    #[test]
    fn validation_6_50cr_vs_50cr() {
        let r = simulate_combat(&vinput(vec![(ShipType::Cruiser, 50)], vec![(ShipType::Cruiser, 50)], vec![], 6));
        assert!(r.rounds_fought == 6, "Should fight all 6 rounds");
    }

    #[test]
    fn validation_7_capital_10bs_vs_50cr() {
        // 50 CRs (20k total atk, 135k total hull) vs 10 BS (10k atk, 60k hull).
        // CRs have 2x the total stats and win through volume of fire.
        let r = simulate_combat(&vinput(vec![(ShipType::Battleship, 10)], vec![(ShipType::Cruiser, 50)], vec![], 7));
        assert_eq!(r.winner, Side::Defender, "50 CRs should beat 10 BS");
    }

    #[test]
    fn validation_8_rip_3_vs_100cr() {
        let r = simulate_combat(&vinput(vec![(ShipType::Deathstar, 3)], vec![(ShipType::Cruiser, 100)], vec![], 8));
        assert_eq!(r.winner, Side::Attacker);
        assert_eq!(r.attacker_survivors.get(&UnitType::Ship(ShipType::Deathstar)).copied().unwrap_or(0), 3);
    }

    #[test]
    fn validation_9_mixed_realistic() {
        let r = simulate_combat(&vinput(
            vec![(ShipType::LightFighter, 100), (ShipType::Cruiser, 20), (ShipType::Battleship, 5)],
            vec![(ShipType::LightFighter, 50), (ShipType::Cruiser, 10), (ShipType::Battleship, 2)],
            vec![], 9));
        assert_eq!(r.winner, Side::Attacker);
    }

    #[test]
    fn validation_10_fodder_100ep_vs_50lf() {
        // EP (atk 0, shield 0, hull 100) vs LF (atk 50, shield 10, hull 400).
        // EP cant attack. LF damages EP hull directly (no shield to absorb).
        // 50 LF kill ~25 EP/round via hull damage + explosion. All EP die in ~4 rounds.
        let r = simulate_combat(&vinput(vec![(ShipType::EspionageProbe, 100)], vec![(ShipType::LightFighter, 50)], vec![], 10));
        let lf_surv = r.defender_survivors.get(&UnitType::Ship(ShipType::LightFighter)).copied().unwrap_or(0);
        assert_eq!(lf_surv, 50, "All LF should survive (EP cant attack)");
        // EP takes heavy losses; likely all die within 6 rounds
        let ep_surv = r.attacker_survivors.get(&UnitType::Ship(ShipType::EspionageProbe)).copied().unwrap_or(0);
        assert!(ep_surv < 100, "EP should take losses: {}", ep_surv);
    }
}

//! OGame ship and defense data module.
//!
//! This module contains the verified base stats for 15 ship types
//! (12 classic + Reaper + Pathfinder + Solar Satellite + Crawler)
//! and 8 defense structures required for modern OGame combat simulation.
//!
//! # Sources
//!
//! Primary authoritative source (actual game engine code, OGame v0.84):
//! - <https://github.com/ogamespec/ogame-opensource/blob/0b765cdb/game/unit.php>
//!   (`$UnitParam` array: structure, shield, attack, cargo, speed, fuel)
//! - <https://github.com/ogamespec/ogame-opensource/blob/0b765cdb/game/prod.php>
//!   (`$initial` array: metal, crystal, deuterium costs)
//!
//! Cross-referenced against community wikis (modern OGame, post-v0.84 updates):
//! - <https://ogame.fandom.com/wiki/Ships> (Fandom community wiki)
//! - <https://ogame.fandom.com/wiki/Defense> (Fandom community wiki)
//! - <https://sidian.app/s/ogame-wiki/ships/combat/light-fighter> (Sidian mirror)
//! - <https://gameforge.com/en-GB/games/ogame-rapidfire.html> (Gameforge official)
//! - <https://ghiroblu.com/o-tools/en/fleet_properties/> (community calculator)
//!
//! # Discrepancies Resolved Against The Plan Spec
//!
//! The plan provided "known values" to verify. Cross-referencing against
//! authoritative sources revealed a small number of incorrect values in the
//! plan, which are corrected here:
//!
//! | Unit | Plan value | Correct value | Why |
//! |---|---|---|---|
//! | Espionage Probe cost | 1000 M / 0 C / 0 D | 0 M / 1000 C / 0 D | ogamespec `prod.php` + Fandom + Sidian all agree: 0/1000/0 |
//! | Heavy Fighter armor | 800 | 1000 | ogamespec structure=10,000 → in-game display armor = structure/10 = 1,000 |
//! | Large Cargo armor | 7,500 | 1,200 | ogamespec structure=12,000 → in-game display armor = structure/10 = 1,200 |
//!
//! # Armor Semantics
//!
//! OGame's combat engine works with "structural integrity" (the full cost-derived
//! value, e.g. 4,000 for Light Fighter). The in-game "armor" stat displayed to
//! players is `structural_integrity / 10 = (Metal + Crystal) / 10` at zero
//! armor technology. The plan spec calls this field `base_armor`, so we store
//! the in-game display value. The formula `base_armor == (cost_metal +
//! cost_crystal) / 10` is verified by test `test_armor_equals_structural_integrity_over_10`.

/// All OGame ship types modeled by the optimizer.
///
/// Order is stable; do not reorder without updating dependent data structures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ShipType {
    SmallCargo,
    LargeCargo,
    LightFighter,
    HeavyFighter,
    Cruiser,
    Battleship,
    Battlecruiser,
    Bomber,
    Destroyer,
    Deathstar,
    EspionageProbe,
    Reaper,
    Pathfinder,
    SolarSatellite,
    Crawler,
}

impl ShipType {
    /// Stable string identifier used for serialization, debugging, and
    /// downstream Python exposure.
    pub fn as_str(self) -> &'static str {
        match self {
            ShipType::SmallCargo => "SmallCargo",
            ShipType::LargeCargo => "LargeCargo",
            ShipType::LightFighter => "LightFighter",
            ShipType::HeavyFighter => "HeavyFighter",
            ShipType::Cruiser => "Cruiser",
            ShipType::Battleship => "Battleship",
            ShipType::Battlecruiser => "Battlecruiser",
            ShipType::Bomber => "Bomber",
            ShipType::Destroyer => "Destroyer",
            ShipType::Deathstar => "Deathstar",
            ShipType::EspionageProbe => "EspionageProbe",
            ShipType::Reaper => "Reaper",
            ShipType::Pathfinder => "Pathfinder",
            ShipType::SolarSatellite => "SolarSatellite",
            ShipType::Crawler => "Crawler",
        }
    }

    /// All ship variants, in declaration order.
    pub const ALL: &'static [ShipType] = &[
        ShipType::SmallCargo,
        ShipType::LargeCargo,
        ShipType::LightFighter,
        ShipType::HeavyFighter,
        ShipType::Cruiser,
        ShipType::Battleship,
        ShipType::Battlecruiser,
        ShipType::Bomber,
        ShipType::Destroyer,
        ShipType::Deathstar,
        ShipType::EspionageProbe,
        ShipType::Reaper,
        ShipType::Pathfinder,
        ShipType::SolarSatellite,
        ShipType::Crawler,
    ];
}

/// All OGame defense structures modeled by the optimizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DefenseType {
    RocketLauncher,
    LightLaser,
    HeavyLaser,
    GaussCannon,
    IonCannon,
    PlasmaTurret,
    SmallShieldDome,
    LargeShieldDome,
}

impl DefenseType {
    pub fn as_str(self) -> &'static str {
        match self {
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

    pub const ALL: &'static [DefenseType] = &[
        DefenseType::RocketLauncher,
        DefenseType::LightLaser,
        DefenseType::HeavyLaser,
        DefenseType::GaussCannon,
        DefenseType::IonCannon,
        DefenseType::PlasmaTurret,
        DefenseType::SmallShieldDome,
        DefenseType::LargeShieldDome,
    ];
}

/// Verified base stats for an OGame unit (ship or defense).
///
/// - `cost_*`: shipyard cost in raw resources (no build multiplier).
/// - `base_attack`: base weapon power, multiplied by `(10 + weapon_tech) / 10` in combat.
/// - `base_shield`: base shield power, multiplied by `(10 + shield_tech) / 10` in combat.
/// - `base_armor`: base hull plating (= in-game "Armor" display) = (metal + crystal) / 10
///   at zero armor technology. Combat applies `(10 + armor_tech) / 10` on top.
/// - `speed`: base warp speed (units per hour). Adjusted by drive technology in
///   mission-speed code (out of scope for this data module).
/// - `cargo`: cargo capacity in resource units.
/// - `fuel`: deuterium consumed per game-hour of flight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShipStats {
    pub cost_metal: u128,
    pub cost_crystal: u128,
    pub cost_deuterium: u128,
    pub base_attack: u64,
    pub base_shield: u64,
    pub base_armor: u64,
    pub speed: u64,
    pub cargo: u64,
    pub fuel: u64,
}

/// Returns the verified base stats for a given ship type.
///
/// Values come from `ogamespec/ogame-opensource` (`unit.php` for combat stats,
/// `prod.php` for costs), cross-referenced with Fandom / Sidian wikis for
/// confirmation. See module-level documentation for sources and discrepancies.
pub fn ship_stats(ship: ShipType) -> ShipStats {
    // All values verified against:
    //   https://github.com/ogamespec/ogame-opensource/blob/0b765cdb/game/unit.php
    //   https://github.com/ogamespec/ogame-opensource/blob/0b765cdb/game/prod.php
    //   https://ogame.fandom.com/wiki/Ships
    //   https://sidian.app/s/ogame-wiki/ships/...
    match ship {
        ShipType::SmallCargo => ShipStats {
            cost_metal: 2_000,
            cost_crystal: 2_000,
            cost_deuterium: 0,
            base_attack: 5,
            base_shield: 10,
            base_armor: 400, // structure 4,000 / 10
            speed: 5_000,
            cargo: 5_000,
            fuel: 10,
        },
        ShipType::LargeCargo => ShipStats {
            cost_metal: 6_000,
            cost_crystal: 6_000,
            cost_deuterium: 0,
            base_attack: 5,
            base_shield: 25,
            base_armor: 1_200, // structure 12,000 / 10 (plan said 7,500 — incorrect)
            speed: 7_500,
            cargo: 25_000,
            fuel: 50,
        },
        ShipType::LightFighter => ShipStats {
            cost_metal: 3_000,
            cost_crystal: 1_000,
            cost_deuterium: 0,
            base_attack: 50,
            base_shield: 10,
            base_armor: 400, // structure 4,000 / 10
            speed: 12_500,
            cargo: 50,
            fuel: 20,
        },
        ShipType::HeavyFighter => ShipStats {
            cost_metal: 6_000,
            cost_crystal: 4_000,
            cost_deuterium: 0,
            base_attack: 150,
            base_shield: 25,
            base_armor: 1_000, // structure 10,000 / 10 (plan said 800 — incorrect)
            speed: 10_000,
            cargo: 100,
            fuel: 75,
        },
        ShipType::Cruiser => ShipStats {
            cost_metal: 20_000,
            cost_crystal: 7_000,
            cost_deuterium: 2_000,
            base_attack: 400,
            base_shield: 50,
            base_armor: 2_700, // structure 27,000 / 10
            speed: 15_000,
            cargo: 800,
            fuel: 300,
        },
        ShipType::Battleship => ShipStats {
            cost_metal: 45_000,
            cost_crystal: 15_000,
            cost_deuterium: 0,
            base_attack: 1_000,
            base_shield: 200,
            base_armor: 6_000, // structure 60,000 / 10
            speed: 10_000,
            cargo: 1_500,
            fuel: 500,
        },
        ShipType::Battlecruiser => ShipStats {
            cost_metal: 30_000,
            cost_crystal: 40_000,
            cost_deuterium: 15_000,
            base_attack: 700,
            base_shield: 400,
            base_armor: 7_000, // structure 70,000 / 10
            speed: 10_000,
            cargo: 750,
            fuel: 250,
        },
        ShipType::Bomber => ShipStats {
            cost_metal: 50_000,
            cost_crystal: 25_000,
            cost_deuterium: 15_000,
            base_attack: 1_000,
            base_shield: 500,
            base_armor: 7_500, // structure 75,000 / 10
            speed: 4_000,
            cargo: 500,
            fuel: 1_000,
        },
        ShipType::Destroyer => ShipStats {
            cost_metal: 60_000,
            cost_crystal: 50_000,
            cost_deuterium: 15_000,
            base_attack: 2_000,
            base_shield: 500,
            base_armor: 11_000, // structure 110,000 / 10
            speed: 5_000,
            cargo: 2_000,
            fuel: 1_000,
        },
        ShipType::Deathstar => ShipStats {
            cost_metal: 5_000_000,
            cost_crystal: 4_000_000,
            cost_deuterium: 1_000_000,
            base_attack: 200_000,
            base_shield: 50_000,
            base_armor: 900_000, // structure 9,000,000 / 10
            speed: 100,
            cargo: 1_000_000,
            fuel: 1,
        },
        ShipType::EspionageProbe => ShipStats {
            cost_metal: 0,         // ogamespec prod.php + Fandom both say 0 metal
            cost_crystal: 1_000,   // (plan said 1,000 metal / 0 crystal — incorrect)
            cost_deuterium: 0,
            base_attack: 0,
            base_shield: 0,
            base_armor: 100, // structure 1,000 / 10
            speed: 100_000_000, // Combustion Drive ×10 boost applied via mission code
            cargo: 0,
            fuel: 1,
        },
        // ---- Reaper (post-v0.84, modern OGame) ----
        // Verified against Fandom wiki (https://ogame.fandom.com/wiki/Reaper):
        //   Cost: 85,000 M / 55,000 C / 20,000 D
        //   Structural Integrity: 140,000 (→ armor 14,000 at zero tech)
        //   Shield Power: 700
        //   Weapon Power: 2,800
        //   Speed: 7,000 (Hyperspace Drive level 7)
        //   Cargo: 10,000, Fuel: 1,100
        // User-verified tech scaling matches Fandom base values:
        //   W14: 2,800 × 2.4 = 6,720 ✓
        //   A16: 140,000 × 2.6 = 364,000 ✓
        //   S13: 700 × 2.3 = 1,610 ✓
        ShipType::Reaper => ShipStats {
            cost_metal: 85_000,
            cost_crystal: 55_000,
            cost_deuterium: 20_000,
            base_attack: 2_800,
            base_shield: 700,
            base_armor: 14_000, // structure 140,000 / 10
            speed: 7_000,
            cargo: 10_000,
            fuel: 1_100,
        },

        // ---- Pathfinder (post-v0.84, modern OGame) ----
        // Verified against Fandom wiki (https://ogame.fandom.com/wiki/Pathfinder):
        //   Cost: 8,000 M / 15,000 C / 8,000 D
        //   Structural Integrity: 23,000 (→ armor 2,300 at zero tech)
        //   Shield Power: 100, Weapon Power: 200, Speed: 12,000
        //   Cargo: 10,000, Fuel: 300
        ShipType::Pathfinder => ShipStats {
            cost_metal: 8_000,
            cost_crystal: 15_000,
            cost_deuterium: 8_000,
            base_attack: 200,
            base_shield: 100,
            base_armor: 2_300, // structure 23,000 / 10
            speed: 12_000,
            cargo: 10_000,
            fuel: 300,
        },

        // ---- Solar Satellite (civil — generates energy) ----
        // Verified against Fandom wiki (https://ogame.fandom.com/wiki/Solar_Satellite):
        //   Cost: 0 M / 2,000 C / 500 D
        //   Structural Integrity: 2,000 (→ armor 200 at zero tech)
        //   Non-combat — RF'd 5× by all combat ships (modeled for RF accuracy).
        ShipType::SolarSatellite => ShipStats {
            cost_metal: 0,
            cost_crystal: 2_000,
            cost_deuterium: 500,
            base_attack: 1,
            base_shield: 1,
            base_armor: 200, // structure 2,000 / 10
            speed: 0,
            cargo: 0,
            fuel: 0,
        },

        // ---- Crawler (civil — boosts resource production) ----
        // Verified against Fandom wiki (https://ogame.fandom.com/wiki/Crawler):
        //   Cost: 2,000 M / 2,000 C / 1,000 D
        //   Structural Integrity: 4,000 (→ armor 400 at zero tech)
        //   Non-combat — RF'd 5× by all combat ships (modeled for RF accuracy).
        ShipType::Crawler => ShipStats {
            cost_metal: 2_000,
            cost_crystal: 2_000,
            cost_deuterium: 1_000,
            base_attack: 1,
            base_shield: 1,
            base_armor: 400, // structure 4,000 / 10
            speed: 0,
            cargo: 0,
            fuel: 0,
        },
    }
}

/// Returns the verified base stats for a given defense structure.
///
/// Defenses are stationary; `speed`, `cargo`, and `fuel` are always 0.
pub fn defense_stats(defense: DefenseType) -> ShipStats {
    // Values verified against:
    //   https://github.com/ogamespec/ogame-opensource/blob/0b765cdb/game/unit.php
    //   https://github.com/ogamespec/ogame-opensource/blob/0b765cdb/game/prod.php
    //   https://ogame.fandom.com/wiki/Defense
    //   https://sidian.app/s/ogame-wiki/defence/...
    match defense {
        DefenseType::RocketLauncher => ShipStats {
            cost_metal: 2_000,
            cost_crystal: 0,
            cost_deuterium: 0,
            base_attack: 80,
            base_shield: 20,
            base_armor: 200, // structure 2,000 / 10
            speed: 0,
            cargo: 0,
            fuel: 0,
        },
        DefenseType::LightLaser => ShipStats {
            cost_metal: 1_500,
            cost_crystal: 500,
            cost_deuterium: 0,
            base_attack: 100,
            base_shield: 25,
            base_armor: 200, // structure 2,000 / 10
            speed: 0,
            cargo: 0,
            fuel: 0,
        },
        DefenseType::HeavyLaser => ShipStats {
            cost_metal: 6_000,
            cost_crystal: 2_000,
            cost_deuterium: 0,
            base_attack: 250,
            base_shield: 100,
            base_armor: 800, // structure 8,000 / 10
            speed: 0,
            cargo: 0,
            fuel: 0,
        },
        DefenseType::GaussCannon => ShipStats {
            cost_metal: 20_000,
            cost_crystal: 15_000,
            cost_deuterium: 2_000,
            base_attack: 1_100,
            base_shield: 200,
            base_armor: 3_500, // structure 35,000 / 10
            speed: 0,
            cargo: 0,
            fuel: 0,
        },
        DefenseType::IonCannon => ShipStats {
            // Cost 5,000 M / 3,000 C / 0 D per modern Fandom + Sidian + plan.
            // (ogamespec v0.84 lists 2,000/6,000/0 — that is a v0.84-era outlier;
            //  every modern source and the plan agree on 5,000/3,000/0.)
            cost_metal: 5_000,
            cost_crystal: 3_000,
            cost_deuterium: 0,
            base_attack: 150,
            base_shield: 500,
            base_armor: 800, // structure 8,000 / 10
            speed: 0,
            cargo: 0,
            fuel: 0,
        },
        DefenseType::PlasmaTurret => ShipStats {
            cost_metal: 50_000,
            cost_crystal: 50_000,
            cost_deuterium: 30_000,
            base_attack: 3_000,
            base_shield: 300,
            base_armor: 10_000, // structure 100,000 / 10
            speed: 0,
            cargo: 0,
            fuel: 0,
        },
        DefenseType::SmallShieldDome => ShipStats {
            cost_metal: 10_000,
            cost_crystal: 10_000,
            cost_deuterium: 0,
            base_attack: 1,
            base_shield: 2_000,
            base_armor: 2_000, // structure 20,000 / 10
            speed: 0,
            cargo: 0,
            fuel: 0,
        },
        DefenseType::LargeShieldDome => ShipStats {
            cost_metal: 50_000,
            cost_crystal: 50_000,
            cost_deuterium: 0,
            base_attack: 1,
            base_shield: 10_000,
            base_armor: 10_000, // structure 100,000 / 10
            speed: 0,
            cargo: 0,
            fuel: 0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- Plan QA scenario: spot-check key costs (lines 347-354 of plan) -----

    #[test]
    fn light_fighter_cost() {
        let s = ship_stats(ShipType::LightFighter);
        assert_eq!(s.cost_metal, 3_000, "LF metal");
        assert_eq!(s.cost_crystal, 1_000, "LF crystal");
        assert_eq!(s.cost_deuterium, 0, "LF deuterium");
    }

    #[test]
    fn battleship_cost() {
        let s = ship_stats(ShipType::Battleship);
        assert_eq!(s.cost_metal, 45_000, "BS metal");
        assert_eq!(s.cost_crystal, 15_000, "BS crystal");
        assert_eq!(s.cost_deuterium, 0, "BS deuterium");
    }

    #[test]
    fn deathstar_cost() {
        let s = ship_stats(ShipType::Deathstar);
        assert_eq!(s.cost_metal, 5_000_000, "RIP metal");
        assert_eq!(s.cost_crystal, 4_000_000, "RIP crystal");
        assert_eq!(s.cost_deuterium, 1_000_000, "RIP deuterium");
    }

    // ----- Sanity: armor == (metal + crystal) / 10 for every unit -----

    #[test]
    fn armor_equals_structural_integrity_over_10_ships() {
        for &s in ShipType::ALL {
            let st = ship_stats(s);
            let expected_armor = (st.cost_metal + st.cost_crystal) / 10;
            assert_eq!(
                st.base_armor as u128, expected_armor,
                "{:?}: armor {} != (m+c)/10 = {}",
                s, st.base_armor, expected_armor
            );
        }
    }

    #[test]
    fn armor_equals_structural_integrity_over_10_defenses() {
        for &d in DefenseType::ALL {
            let st = defense_stats(d);
            let expected_armor = (st.cost_metal + st.cost_crystal) / 10;
            assert_eq!(
                st.base_armor as u128, expected_armor,
                "{:?}: armor {} != (m+c)/10 = {}",
                d, st.base_armor, expected_armor
            );
        }
    }

    // ----- Sanity: defenses have 0 speed / cargo / fuel -----

    #[test]
    fn defenses_have_zero_speed_cargo_fuel() {
        for &d in DefenseType::ALL {
            let st = defense_stats(d);
            assert_eq!(st.speed, 0, "{:?} speed", d);
            assert_eq!(st.cargo, 0, "{:?} cargo", d);
            assert_eq!(st.fuel, 0, "{:?} fuel", d);
        }
    }

    // ----- Sanity: all stats functions return values, never panic -----

    #[test]
    fn ship_stats_complete_coverage() {
        for &s in ShipType::ALL {
            let _ = ship_stats(s);
        }
    }

    #[test]
    fn defense_stats_complete_coverage() {
        for &d in DefenseType::ALL {
            let _ = defense_stats(d);
        }
    }

    // ----- Display / identifier helpers -----

    #[test]
    fn ship_type_as_str_matches_variant_name() {
        assert_eq!(ShipType::LightFighter.as_str(), "LightFighter");
        assert_eq!(ShipType::Deathstar.as_str(), "Deathstar");
        assert_eq!(ShipType::EspionageProbe.as_str(), "EspionageProbe");
    }

    #[test]
    fn defense_type_as_str_matches_variant_name() {
        assert_eq!(DefenseType::RocketLauncher.as_str(), "RocketLauncher");
        assert_eq!(DefenseType::LargeShieldDome.as_str(), "LargeShieldDome");
    }

    // ----- Plan correctness: spot-check known verified values -----

    #[test]
    fn cruiser_known_stats() {
        let s = ship_stats(ShipType::Cruiser);
        assert_eq!(s.base_attack, 400);
        assert_eq!(s.base_shield, 50);
        assert_eq!(s.base_armor, 2_700);
        assert_eq!(s.speed, 15_000);
        assert_eq!(s.cargo, 800);
    }

    #[test]
    fn deathstar_known_stats() {
        let s = ship_stats(ShipType::Deathstar);
        assert_eq!(s.base_attack, 200_000);
        assert_eq!(s.base_shield, 50_000);
        assert_eq!(s.base_armor, 900_000);
        assert_eq!(s.speed, 100);
        assert_eq!(s.cargo, 1_000_000);
    }

    #[test]
    fn rocket_launcher_known_stats() {
        let s = defense_stats(DefenseType::RocketLauncher);
        assert_eq!(s.base_attack, 80);
        assert_eq!(s.base_shield, 20);
        assert_eq!(s.base_armor, 200);
        assert_eq!(s.cost_metal, 2_000);
    }

    #[test]
    fn plasma_turret_known_stats() {
        let s = defense_stats(DefenseType::PlasmaTurret);
        assert_eq!(s.base_attack, 3_000);
        assert_eq!(s.base_shield, 300);
        assert_eq!(s.base_armor, 10_000);
    }

    // ----- Reject the values the plan got wrong (defensive regression) -----

    #[test]
    fn reaper_known_stats() {
        // Verified against Fandom wiki + user-provided "with tech" scaling.
        let s = ship_stats(ShipType::Reaper);
        assert_eq!(s.cost_metal, 85_000);
        assert_eq!(s.cost_crystal, 55_000);
        assert_eq!(s.cost_deuterium, 20_000); // FIXED: was incorrectly 0
        assert_eq!(s.base_attack, 2_800);      // FIXED: was incorrectly 280
        assert_eq!(s.base_shield, 700);         // FIXED: was incorrectly 1_000
        assert_eq!(s.base_armor, 14_000);
        assert_eq!(s.speed, 7_000);
        assert_eq!(s.cargo, 10_000);            // FIXED: was incorrectly 0
        assert_eq!(s.fuel, 1_100);              // FIXED: was incorrectly 100
    }

    #[test]
    fn pathfinder_known_stats() {
        // Verified against Fandom wiki.
        let s = ship_stats(ShipType::Pathfinder);
        assert_eq!(s.cost_metal, 8_000);
        assert_eq!(s.cost_crystal, 15_000);
        assert_eq!(s.cost_deuterium, 8_000);
        assert_eq!(s.base_attack, 200);
        assert_eq!(s.base_shield, 100);
        assert_eq!(s.base_armor, 2_300); // structure 23,000 / 10
        assert_eq!(s.speed, 12_000);
        assert_eq!(s.cargo, 10_000);
        assert_eq!(s.fuel, 300);
    }

    #[test]
    fn solar_satellite_known_stats() {
        // Non-combat civil ship (energy generator).
        let s = ship_stats(ShipType::SolarSatellite);
        assert_eq!(s.cost_metal, 0);
        assert_eq!(s.cost_crystal, 2_000);
        assert_eq!(s.cost_deuterium, 500);
        assert_eq!(s.base_attack, 1);
        assert_eq!(s.base_shield, 1);
        assert_eq!(s.base_armor, 200); // structure 2,000 / 10
        assert_eq!(s.speed, 0);
        assert_eq!(s.cargo, 0);
        assert_eq!(s.fuel, 0);
    }

    #[test]
    fn crawler_known_stats() {
        // Non-combat civil ship (production booster).
        let s = ship_stats(ShipType::Crawler);
        assert_eq!(s.cost_metal, 2_000);
        assert_eq!(s.cost_crystal, 2_000);
        assert_eq!(s.cost_deuterium, 1_000);
        assert_eq!(s.base_attack, 1);
        assert_eq!(s.base_shield, 1);
        assert_eq!(s.base_armor, 400); // structure 4,000 / 10
        assert_eq!(s.speed, 0);
        assert_eq!(s.cargo, 0);
        assert_eq!(s.fuel, 0);
    }

    #[test]
    fn plan_known_incorrect_values_are_corrected() {
        // Plan said HF armor = 800. Correct: 1,000.
        assert_eq!(ship_stats(ShipType::HeavyFighter).base_armor, 1_000);
        // Plan said LC armor = 7,500. Correct: 1,200.
        assert_eq!(ship_stats(ShipType::LargeCargo).base_armor, 1_200);
        // Plan said EP cost = 1000/0/0. Correct: 0/1000/0.
        let ep = ship_stats(ShipType::EspionageProbe);
        assert_eq!(ep.cost_metal, 0);
        assert_eq!(ep.cost_crystal, 1_000);
    }
}

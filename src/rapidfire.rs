//! OGame rapid-fire (Schnellfeuer) table.
//!
//! Returns the expected number of additional shots a shooter fires per round
//! against a specific target. `Some(n)` means the shooter fires ~n shots
//! on average (the engine re-rolls `(n-1)/n` per extra shot). `None` means
//! no rapid fire: the shooter fires exactly once per round.
//!
//! # Sources
//!
//! Primary authoritative source — actual game engine code (OGame v0.84):
//! - <https://github.com/ogamespec/ogame-opensource/blob/0b765cdb/game/unit.php>
//!   (`$RapidFire` array)
//!
//! Cross-referenced against modern community sources for post-v0.84 updates:
//! - <https://sidian.app/s/ogame-wiki/guides/rapid-fire>
//! - <https://ogame.fandom.com/wiki/Ships> (per-ship pages)
//! - <https://gameforge.com/en-GB/games/ogame-rapidfire.html>
//! - <https://ghiroblu.com/o-tools/en/fleet_properties/>
//!
//! # Discrepancies Resolved Against The Plan Spec
//!
//! The plan provided "key rapidfire" values "to verify". Cross-referencing
//! against authoritative sources revealed several incorrect values in the
//! plan, which are corrected here. Documented inline below.
//!
//! # Scope
//!
//! Covers the rapid-fire relationships where the *shooter* is one of the 11
//! `ShipType` variants. Defenses never shoot with rapid fire (verified by
//! reading the `RapidFire()` PHP function: `if (IsDefense($atyp)) return 0;`).

use crate::ships::{DefenseType, ShipType};

/// Discriminated union of any unit the rapid-fire table can target.
///
/// Both ships and defenses can be targets. Wrapping both enums in a single
/// sum type keeps the `rapidfire()` signature uniform and prevents the
/// caller from accidentally mixing ship and defense lookups.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum UnitType {
    Ship(ShipType),
    Defense(DefenseType),
}

impl UnitType {
    pub fn as_str(self) -> &'static str {
        match self {
            UnitType::Ship(s) => s.as_str(),
            UnitType::Defense(d) => d.as_str(),
        }
    }
}

impl From<ShipType> for UnitType {
    fn from(s: ShipType) -> Self {
        UnitType::Ship(s)
    }
}

impl From<DefenseType> for UnitType {
    fn from(d: DefenseType) -> Self {
        UnitType::Defense(d)
    }
}

/// Returns the rapid-fire multiplier for `shooter` vs `target`.
///
/// `Some(n)` where `n >= 2` means the shooter fires on average `n` shots
/// per round against that target. `None` means no rapid fire (one shot).
/// The function is total: every (ShipType, UnitType) pair is handled and
/// never panics.
pub fn rapidfire(shooter: ShipType, target: UnitType) -> Option<u32> {
    // The table is structured by shooter to mirror the authoritative
    // ogamespec `$RapidFire` array. Each `match` arm lists every target
    // the shooter has rapid fire against; anything not listed returns None.
    //
    // Source: https://github.com/ogamespec/ogame-opensource/blob/0b765cdb/game/unit.php
    // Cross-referenced with https://ogame.fandom.com/wiki/* (modern OGame values
    // for Reaper, Pathfinder, and post-v0.84 ships).
    //
    // We use a flat approach: for each shooter, list every (target, value)
    // pair. This is slightly more verbose than a nested match but makes
    // the table trivially auditable against the wiki/gamespec sources.
    let value = match (shooter, target) {
        // ---- Light Fighter ----
        // Fandom: vs EP=5, SS=5, Crawler=5
        (ShipType::LightFighter, UnitType::Ship(ShipType::EspionageProbe)) => 5,
        (ShipType::LightFighter, UnitType::Ship(ShipType::SolarSatellite)) => 5,
        (ShipType::LightFighter, UnitType::Ship(ShipType::Crawler)) => 5,

        // ---- Heavy Fighter ----
        // Fandom: vs SC=3, EP=5, SS=5, Crawler=5
        (ShipType::HeavyFighter, UnitType::Ship(ShipType::SmallCargo)) => 3,
        (ShipType::HeavyFighter, UnitType::Ship(ShipType::EspionageProbe)) => 5,
        (ShipType::HeavyFighter, UnitType::Ship(ShipType::SolarSatellite)) => 5,
        (ShipType::HeavyFighter, UnitType::Ship(ShipType::Crawler)) => 5,

        // ---- Cruiser ----
        // Fandom: vs LF=6, EP=5, SS=5, Crawler=5, RL=10
        (ShipType::Cruiser, UnitType::Ship(ShipType::LightFighter)) => 6,
        (ShipType::Cruiser, UnitType::Ship(ShipType::EspionageProbe)) => 5,
        (ShipType::Cruiser, UnitType::Ship(ShipType::SolarSatellite)) => 5,
        (ShipType::Cruiser, UnitType::Ship(ShipType::Crawler)) => 5,
        (ShipType::Cruiser, UnitType::Defense(DefenseType::RocketLauncher)) => 10,

        // ---- Battleship ----
        // Fandom: vs EP=5, SS=5, Crawler=5, PF=5
        (ShipType::Battleship, UnitType::Ship(ShipType::EspionageProbe)) => 5,
        (ShipType::Battleship, UnitType::Ship(ShipType::SolarSatellite)) => 5,
        (ShipType::Battleship, UnitType::Ship(ShipType::Crawler)) => 5,
        (ShipType::Battleship, UnitType::Ship(ShipType::Pathfinder)) => 5,

        // ---- Battlecruiser ----
        // Fandom: vs EP=5, SS=5, Crawler=5, SC=3, LC=3, HF=4, CR=4, BS=7
        (ShipType::Battlecruiser, UnitType::Ship(ShipType::EspionageProbe)) => 5,
        (ShipType::Battlecruiser, UnitType::Ship(ShipType::SolarSatellite)) => 5,
        (ShipType::Battlecruiser, UnitType::Ship(ShipType::Crawler)) => 5,
        (ShipType::Battlecruiser, UnitType::Ship(ShipType::SmallCargo)) => 3,
        (ShipType::Battlecruiser, UnitType::Ship(ShipType::LargeCargo)) => 3,
        (ShipType::Battlecruiser, UnitType::Ship(ShipType::HeavyFighter)) => 4,
        (ShipType::Battlecruiser, UnitType::Ship(ShipType::Cruiser)) => 4,
        (ShipType::Battlecruiser, UnitType::Ship(ShipType::Battleship)) => 7,

        // ---- Bomber ----
        // Fandom: vs EP=5, SS=5, Crawler=5, RL=20, LL=20, HL=10, IC=10, GC=5, PT=5
        (ShipType::Bomber, UnitType::Ship(ShipType::EspionageProbe)) => 5,
        (ShipType::Bomber, UnitType::Ship(ShipType::SolarSatellite)) => 5,
        (ShipType::Bomber, UnitType::Ship(ShipType::Crawler)) => 5,
        (ShipType::Bomber, UnitType::Defense(DefenseType::RocketLauncher)) => 20,
        (ShipType::Bomber, UnitType::Defense(DefenseType::LightLaser)) => 20,
        (ShipType::Bomber, UnitType::Defense(DefenseType::HeavyLaser)) => 10,
        (ShipType::Bomber, UnitType::Defense(DefenseType::IonCannon)) => 10,
        (ShipType::Bomber, UnitType::Defense(DefenseType::GaussCannon)) => 5,
        (ShipType::Bomber, UnitType::Defense(DefenseType::PlasmaTurret)) => 5,

        // ---- Destroyer ----
        // Fandom: vs EP=5, SS=5, Crawler=5, BC=2, LL=10
        (ShipType::Destroyer, UnitType::Ship(ShipType::EspionageProbe)) => 5,
        (ShipType::Destroyer, UnitType::Ship(ShipType::SolarSatellite)) => 5,
        (ShipType::Destroyer, UnitType::Ship(ShipType::Crawler)) => 5,
        (ShipType::Destroyer, UnitType::Ship(ShipType::Battlecruiser)) => 2,
        (ShipType::Destroyer, UnitType::Defense(DefenseType::LightLaser)) => 10,

        // ---- Reaper (post-v0.84, modern OGame) ----
        // Fandom: vs EP=5, SS=5, Crawler=5, BS=7, Bo=4, De=3
        // FIXED: previous code had Reaper as anti-fighter (LF=3, HF=2)
        // which was a Pathfinder data copy-paste. Reaper is actually
        // anti-capital (kills Battleships, Bombers, Destroyers).
        (ShipType::Reaper, UnitType::Ship(ShipType::EspionageProbe)) => 5,
        (ShipType::Reaper, UnitType::Ship(ShipType::SolarSatellite)) => 5,
        (ShipType::Reaper, UnitType::Ship(ShipType::Crawler)) => 5,
        (ShipType::Reaper, UnitType::Ship(ShipType::Battleship)) => 7,
        (ShipType::Reaper, UnitType::Ship(ShipType::Bomber)) => 4,
        (ShipType::Reaper, UnitType::Ship(ShipType::Destroyer)) => 3,

        // ---- Pathfinder (post-v0.84, modern OGame) ----
        // Fandom: vs EP=5, SS=5, Crawler=5, LF=3, HF=2, CR=3
        (ShipType::Pathfinder, UnitType::Ship(ShipType::EspionageProbe)) => 5,
        (ShipType::Pathfinder, UnitType::Ship(ShipType::SolarSatellite)) => 5,
        (ShipType::Pathfinder, UnitType::Ship(ShipType::Crawler)) => 5,
        (ShipType::Pathfinder, UnitType::Ship(ShipType::LightFighter)) => 3,
        (ShipType::Pathfinder, UnitType::Ship(ShipType::HeavyFighter)) => 2,
        (ShipType::Pathfinder, UnitType::Ship(ShipType::Cruiser)) => 3,

        // ---- Solar Satellite (civil — passive, no RF) ----
        // ---- Crawler (civil — passive, no RF) ----
        (ShipType::SolarSatellite, _) => return None,
        (ShipType::Crawler, _) => return None,

        // ---- Deathstar ----
        // Fandom: vs EP=1250, SS=1250, Crawler=1250, LF=200, HF=100,
        //                  CR=33, BS=30, BC=15 (FIXED: was 250),
        //                  Bo=25, De=5, SC=250, LC=250, PF=30, Reaper=30
        // Note: BC was previously coded as 250 (assumed "modern") but
        // Fandom confirms it has remained at 15 in modern OGame. The 250
        // value was likely a balance change that was reverted.
        (ShipType::Deathstar, UnitType::Ship(ShipType::EspionageProbe)) => 1_250,
        (ShipType::Deathstar, UnitType::Ship(ShipType::SolarSatellite)) => 1_250,
        (ShipType::Deathstar, UnitType::Ship(ShipType::Crawler)) => 1_250,
        (ShipType::Deathstar, UnitType::Ship(ShipType::LightFighter)) => 200,
        (ShipType::Deathstar, UnitType::Ship(ShipType::HeavyFighter)) => 100,
        (ShipType::Deathstar, UnitType::Ship(ShipType::Cruiser)) => 33,
        (ShipType::Deathstar, UnitType::Ship(ShipType::Battleship)) => 30,
        (ShipType::Deathstar, UnitType::Ship(ShipType::Battlecruiser)) => 15, // FIXED: was 250
        (ShipType::Deathstar, UnitType::Ship(ShipType::Pathfinder)) => 30,    // NEW
        (ShipType::Deathstar, UnitType::Ship(ShipType::Reaper)) => 30,         // NEW
        (ShipType::Deathstar, UnitType::Ship(ShipType::Bomber)) => 25,
        (ShipType::Deathstar, UnitType::Ship(ShipType::Destroyer)) => 5,
        (ShipType::Deathstar, UnitType::Ship(ShipType::SmallCargo)) => 250,
        (ShipType::Deathstar, UnitType::Ship(ShipType::LargeCargo)) => 250,
        (ShipType::Deathstar, UnitType::Defense(DefenseType::RocketLauncher)) => 200,
        (ShipType::Deathstar, UnitType::Defense(DefenseType::LightLaser)) => 200,
        (ShipType::Deathstar, UnitType::Defense(DefenseType::HeavyLaser)) => 100,
        (ShipType::Deathstar, UnitType::Defense(DefenseType::IonCannon)) => 100,
        (ShipType::Deathstar, UnitType::Defense(DefenseType::GaussCannon)) => 50,

        // ---- Espionage Probe ----
        // Probe has no rapid fire against any target.
        (ShipType::EspionageProbe, _) => return None,

        // ---- Default: no rapid fire ----
        _ => return None,
    };
    Some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- Plan QA scenario: known rapid-fire values (lines 351-352) -----

    #[test]
    fn cruiser_vs_light_fighter_is_six() {
        assert_eq!(
            rapidfire(ShipType::Cruiser, UnitType::Ship(ShipType::LightFighter)),
            Some(6)
        );
    }

    #[test]
    fn cruiser_vs_espionage_probe_is_five() {
        assert_eq!(
            rapidfire(ShipType::Cruiser, UnitType::Ship(ShipType::EspionageProbe)),
            Some(5)
        );
    }

    // ----- Other plan-asserted values that survived verification -----

    #[test]
    fn battlecruiser_vs_espionage_probe_is_five() {
        assert_eq!(
            rapidfire(
                ShipType::Battlecruiser,
                UnitType::Ship(ShipType::EspionageProbe)
            ),
            Some(5)
        );
    }

    #[test]
    fn deathstar_vs_light_fighter_is_200() {
        assert_eq!(
            rapidfire(ShipType::Deathstar, UnitType::Ship(ShipType::LightFighter)),
            Some(200)
        );
    }

    #[test]
    fn deathstar_vs_cruiser_is_33() {
        assert_eq!(
            rapidfire(ShipType::Deathstar, UnitType::Ship(ShipType::Cruiser)),
            Some(33)
        );
    }

    // ----- Plan-asserted values that were WRONG — confirm we use the
    //       correct authoritative value instead -----

    #[test]
    fn heavy_fighter_vs_cruiser_is_none() {
        // Plan said 3. ogamespec + o-tools both say: no rapid fire.
        assert_eq!(
            rapidfire(ShipType::HeavyFighter, UnitType::Ship(ShipType::Cruiser)),
            None
        );
    }

    #[test]
    fn battleship_vs_battlecruiser_is_none() {
        // Plan said 4. ogamespec + o-tools + Sidian all say: no rapid fire.
        assert_eq!(
            rapidfire(
                ShipType::Battleship,
                UnitType::Ship(ShipType::Battlecruiser)
            ),
            None
        );
    }

    #[test]
    fn battlecruiser_vs_cruiser_is_four() {
        // Plan said 3. ogamespec + o-tools both say 4.
        assert_eq!(
            rapidfire(ShipType::Battlecruiser, UnitType::Ship(ShipType::Cruiser)),
            Some(4)
        );
    }

    #[test]
    fn destroyer_vs_light_fighter_is_none() {
        // Plan said 10. ogamespec doesn't list it; o-tools shows 0.
        assert_eq!(
            rapidfire(ShipType::Destroyer, UnitType::Ship(ShipType::LightFighter)),
            None
        );
    }

    #[test]
    fn destroyer_vs_battlecruiser_is_two() {
        // Plan said 3. ogamespec + o-tools + Sidian all say 2.
        assert_eq!(
            rapidfire(ShipType::Destroyer, UnitType::Ship(ShipType::Battlecruiser)),
            Some(2)
        );
    }

    // ----- Plan QA: rapidfire returns Some/None for ALL pairs, no panic -----

    #[test]
    fn no_panic_for_every_ship_vs_ship_pair() {
        for &shooter in ShipType::ALL {
            for &target in ShipType::ALL {
                let _: Option<u32> = rapidfire(shooter, UnitType::Ship(target));
            }
        }
    }

    #[test]
    fn no_panic_for_every_ship_vs_defense_pair() {
        for &shooter in ShipType::ALL {
            for &target in DefenseType::ALL {
                let _: Option<u32> = rapidfire(shooter, UnitType::Defense(target));
            }
        }
    }

    // ----- Symmetry sanity: rapidfire(A, B) == Some(n) ⇒ rapidfire(B, A) == None
    //       (rapid fire is not commutative). Exceptions are documented in the
    //       plan; in particular Cruiser→LightFighter=6 has no reverse. -----

    #[test]
    fn rapidfire_is_not_commutative_smoke() {
        // Spot-check a handful of known relationships and their inverses.
        assert_eq!(
            rapidfire(ShipType::Cruiser, UnitType::Ship(ShipType::LightFighter)),
            Some(6)
        );
        assert_eq!(
            rapidfire(ShipType::LightFighter, UnitType::Ship(ShipType::Cruiser)),
            None
        );

        assert_eq!(
            rapidfire(
                ShipType::Deathstar,
                UnitType::Ship(ShipType::EspionageProbe)
            ),
            Some(1_250)
        );
        assert_eq!(
            rapidfire(
                ShipType::EspionageProbe,
                UnitType::Ship(ShipType::Deathstar)
            ),
            None
        );
    }

    // ----- Specific values from authoritative sources -----

    #[test]
    fn light_fighter_rapidfire() {
        // Fandom: LF vs EP=5, SS=5, Crawler=5 (NOT vs SC — that was wrong)
        assert_eq!(
            rapidfire(ShipType::LightFighter, UnitType::Ship(ShipType::EspionageProbe)),
            Some(5)
        );
        assert_eq!(
            rapidfire(ShipType::LightFighter, UnitType::Ship(ShipType::SolarSatellite)),
            Some(5)
        );
        assert_eq!(
            rapidfire(ShipType::LightFighter, UnitType::Ship(ShipType::Crawler)),
            Some(5)
        );
        assert_eq!(
            rapidfire(ShipType::LightFighter, UnitType::Ship(ShipType::SmallCargo)),
            None  // No rapid fire vs SC per authoritative Fandom data
        );
    }

    #[test]
    fn heavy_fighter_rapidfire() {
        assert_eq!(
            rapidfire(ShipType::HeavyFighter, UnitType::Ship(ShipType::SmallCargo)),
            Some(3)
        );
        assert_eq!(
            rapidfire(ShipType::HeavyFighter, UnitType::Ship(ShipType::EspionageProbe)),
            Some(5)
        );
    }

    #[test]
    fn battlecruiser_rapidfire_full() {
        assert_eq!(
            rapidfire(ShipType::Battlecruiser, UnitType::Ship(ShipType::SmallCargo)),
            Some(3)
        );
        assert_eq!(
            rapidfire(ShipType::Battlecruiser, UnitType::Ship(ShipType::LargeCargo)),
            Some(3)
        );
        assert_eq!(
            rapidfire(ShipType::Battlecruiser, UnitType::Ship(ShipType::HeavyFighter)),
            Some(4)
        );
        assert_eq!(
            rapidfire(ShipType::Battlecruiser, UnitType::Ship(ShipType::Cruiser)),
            Some(4)
        );
        assert_eq!(
            rapidfire(ShipType::Battlecruiser, UnitType::Ship(ShipType::Battleship)),
            Some(7)
        );
    }

    #[test]
    fn bomber_rapidfire_against_defenses() {
        assert_eq!(
            rapidfire(ShipType::Bomber, UnitType::Defense(DefenseType::RocketLauncher)),
            Some(20)
        );
        assert_eq!(
            rapidfire(ShipType::Bomber, UnitType::Defense(DefenseType::LightLaser)),
            Some(20)
        );
        assert_eq!(
            rapidfire(ShipType::Bomber, UnitType::Defense(DefenseType::HeavyLaser)),
            Some(10)
        );
        assert_eq!(
            rapidfire(ShipType::Bomber, UnitType::Defense(DefenseType::IonCannon)),
            Some(10)
        );
        assert_eq!(
            rapidfire(ShipType::Bomber, UnitType::Defense(DefenseType::GaussCannon)),
            Some(5)
        );
        assert_eq!(
            rapidfire(ShipType::Bomber, UnitType::Defense(DefenseType::PlasmaTurret)),
            Some(5)
        );
    }

    #[test]
    fn destroyer_rapidfire_against_battlecruiser_and_ll() {
        assert_eq!(
            rapidfire(ShipType::Destroyer, UnitType::Ship(ShipType::Battlecruiser)),
            Some(2)
        );
        assert_eq!(
            rapidfire(ShipType::Destroyer, UnitType::Defense(DefenseType::LightLaser)),
            Some(10)
        );
    }

    #[test]
    fn reaper_rapidfire() {
        // Fandom: Reaper vs EP=5, SS=5, Crawler=5, BS=7, Bo=4, De=3
        // FIXED: previous code had anti-fighter RF (LF=3, HF=2) which was
        // actually Pathfinder data. Reaper is anti-capital.
        assert_eq!(
            rapidfire(ShipType::Reaper, UnitType::Ship(ShipType::EspionageProbe)),
            Some(5)
        );
        assert_eq!(
            rapidfire(ShipType::Reaper, UnitType::Ship(ShipType::SolarSatellite)),
            Some(5)
        );
        assert_eq!(
            rapidfire(ShipType::Reaper, UnitType::Ship(ShipType::Crawler)),
            Some(5)
        );
        assert_eq!(
            rapidfire(ShipType::Reaper, UnitType::Ship(ShipType::Battleship)),
            Some(7)
        );
        assert_eq!(
            rapidfire(ShipType::Reaper, UnitType::Ship(ShipType::Bomber)),
            Some(4)
        );
        assert_eq!(
            rapidfire(ShipType::Reaper, UnitType::Ship(ShipType::Destroyer)),
            Some(3)
        );
        // No RF against light/medium fighters (per Fandom)
        assert_eq!(
            rapidfire(ShipType::Reaper, UnitType::Ship(ShipType::LightFighter)),
            None
        );
        assert_eq!(
            rapidfire(ShipType::Reaper, UnitType::Ship(ShipType::HeavyFighter)),
            None
        );
    }

    #[test]
    fn deathstar_rapidfire_full() {
        // Fandom: vs ships
        // FIXED: BC was 250 (wrong), should be 15
        assert_eq!(
            rapidfire(ShipType::Deathstar, UnitType::Ship(ShipType::SmallCargo)),
            Some(250)
        );
        assert_eq!(
            rapidfire(ShipType::Deathstar, UnitType::Ship(ShipType::LargeCargo)),
            Some(250)
        );
        assert_eq!(
            rapidfire(ShipType::Deathstar, UnitType::Ship(ShipType::LightFighter)),
            Some(200)
        );
        assert_eq!(
            rapidfire(ShipType::Deathstar, UnitType::Ship(ShipType::HeavyFighter)),
            Some(100)
        );
        assert_eq!(
            rapidfire(ShipType::Deathstar, UnitType::Ship(ShipType::Cruiser)),
            Some(33)
        );
        assert_eq!(
            rapidfire(ShipType::Deathstar, UnitType::Ship(ShipType::Battleship)),
            Some(30)
        );
        assert_eq!(
            rapidfire(ShipType::Deathstar, UnitType::Ship(ShipType::Battlecruiser)),
            Some(15)  // FIXED: was 250, Fandom confirms 15
        );
        assert_eq!(
            rapidfire(ShipType::Deathstar, UnitType::Ship(ShipType::Pathfinder)),
            Some(30)  // NEW
        );
        assert_eq!(
            rapidfire(ShipType::Deathstar, UnitType::Ship(ShipType::Reaper)),
            Some(30)  // NEW
        );
        assert_eq!(
            rapidfire(ShipType::Deathstar, UnitType::Ship(ShipType::Bomber)),
            Some(25)
        );
        assert_eq!(
            rapidfire(ShipType::Deathstar, UnitType::Ship(ShipType::Destroyer)),
            Some(5)
        );
        assert_eq!(
            rapidfire(ShipType::Deathstar, UnitType::Ship(ShipType::EspionageProbe)),
            Some(1_250)
        );
        assert_eq!(
            rapidfire(ShipType::Deathstar, UnitType::Ship(ShipType::SolarSatellite)),
            Some(1_250)  // NEW
        );
        assert_eq!(
            rapidfire(ShipType::Deathstar, UnitType::Ship(ShipType::Crawler)),
            Some(1_250)  // NEW
        );
        // vs defenses
        assert_eq!(
            rapidfire(ShipType::Deathstar, UnitType::Defense(DefenseType::RocketLauncher)),
            Some(200)
        );
        assert_eq!(
            rapidfire(ShipType::Deathstar, UnitType::Defense(DefenseType::LightLaser)),
            Some(200)
        );
        assert_eq!(
            rapidfire(ShipType::Deathstar, UnitType::Defense(DefenseType::HeavyLaser)),
            Some(100)
        );
        assert_eq!(
            rapidfire(ShipType::Deathstar, UnitType::Defense(DefenseType::GaussCannon)),
            Some(50)
        );
        assert_eq!(
            rapidfire(ShipType::Deathstar, UnitType::Defense(DefenseType::IonCannon)),
            Some(100)
        );
    }

    #[test]
    fn espionage_probe_has_no_rapidfire() {
        for &target in ShipType::ALL {
            assert_eq!(
                rapidfire(ShipType::EspionageProbe, UnitType::Ship(target)),
                None,
                "Probe vs {:?}",
                target
            );
        }
        for &target in DefenseType::ALL {
            assert_eq!(
                rapidfire(ShipType::EspionageProbe, UnitType::Defense(target)),
                None,
                "Probe vs {:?}",
                target
            );
        }
    }

    #[test]
    fn unit_type_as_str_works_for_both_variants() {
        assert_eq!(UnitType::Ship(ShipType::Cruiser).as_str(), "Cruiser");
        assert_eq!(
            UnitType::Defense(DefenseType::PlasmaTurret).as_str(),
            "PlasmaTurret"
        );
    }

    #[test]
    fn unit_type_from_impls() {
        let _: UnitType = ShipType::Cruiser.into();
        let _: UnitType = DefenseType::RocketLauncher.into();
    }
}

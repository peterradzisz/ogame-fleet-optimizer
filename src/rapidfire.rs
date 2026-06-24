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
    // (plus modern OGame updates for entries added after v0.84).
    //
    // We use a flat approach: for each shooter, list every (target, value)
    // pair. This is slightly more verbose than a nested match but makes
    // the table trivially auditable against the wiki/gamespec sources.
    let value = match (shooter, target) {
        // ---- Light Fighter ----
        // ogamespec unit.php $RapidFire: LF → Probe=5, Sat=5
        // Modern OGame: also LF → SmallCargo=5
        (ShipType::LightFighter, UnitType::Ship(ShipType::EspionageProbe)) => 5,
        (ShipType::LightFighter, UnitType::Ship(ShipType::SmallCargo)) => 5,
        // LF vs any defense: no rapid fire. Falls through to the `_` arm below.

        // ---- Heavy Fighter ----
        // ogamespec unit.php $RapidFire: HF → SC=3, Probe=5, Sat=5
        // Plan said "HF vs Cruiser = 3" — this is incorrect. ogamespec and
        // o-tools (ghiroblu) both show HF has NO rapid fire against Cruiser.
        (ShipType::HeavyFighter, UnitType::Ship(ShipType::SmallCargo)) => 3,
        (ShipType::HeavyFighter, UnitType::Ship(ShipType::EspionageProbe)) => 5,

        // ---- Cruiser ----
        // ogamespec unit.php $RapidFire: CR → LF=6, Probe=5, Sat=5, RL=10
        (ShipType::Cruiser, UnitType::Ship(ShipType::LightFighter)) => 6,
        (ShipType::Cruiser, UnitType::Ship(ShipType::EspionageProbe)) => 5,
        (ShipType::Cruiser, UnitType::Defense(DefenseType::RocketLauncher)) => 10,

        // ---- Battleship ----
        // ogamespec unit.php $RapidFire: BS → Probe=5, Sat=5
        // Plan said "BS vs Battlecruiser = 4" — this is incorrect. ogamespec,
        // o-tools, and Sidian all show Battleship has NO rapid fire against
        // Battlecruiser.
        (ShipType::Battleship, UnitType::Ship(ShipType::EspionageProbe)) => 5,

        // ---- Battlecruiser ----
        // ogamespec unit.php $RapidFire: BC → SC=3, LC=3, HF=4, CR=4, BS=7, Probe=5, Sat=5
        // Plan said "BC vs Cruiser = 3" — this is incorrect. ogamespec and
        // o-tools both show BC vs CR = 4.
        (ShipType::Battlecruiser, UnitType::Ship(ShipType::SmallCargo)) => 3,
        (ShipType::Battlecruiser, UnitType::Ship(ShipType::LargeCargo)) => 3,
        (ShipType::Battlecruiser, UnitType::Ship(ShipType::HeavyFighter)) => 4,
        (ShipType::Battlecruiser, UnitType::Ship(ShipType::Cruiser)) => 4,
        (ShipType::Battlecruiser, UnitType::Ship(ShipType::Battleship)) => 7,
        (ShipType::Battlecruiser, UnitType::Ship(ShipType::EspionageProbe)) => 5,

        // ---- Bomber ----
        // ogamespec unit.php $RapidFire: Bomber → Probe=5, Sat=5, RL=20, LL=20, HL=10, IC=10
        // Modern OGame (Sidian) also adds: Bomber → GC=5, PT=5
        (ShipType::Bomber, UnitType::Ship(ShipType::EspionageProbe)) => 5,
        (ShipType::Bomber, UnitType::Defense(DefenseType::RocketLauncher)) => 20,
        (ShipType::Bomber, UnitType::Defense(DefenseType::LightLaser)) => 20,
        (ShipType::Bomber, UnitType::Defense(DefenseType::HeavyLaser)) => 10,
        (ShipType::Bomber, UnitType::Defense(DefenseType::IonCannon)) => 10,
        (ShipType::Bomber, UnitType::Defense(DefenseType::GaussCannon)) => 5,
        (ShipType::Bomber, UnitType::Defense(DefenseType::PlasmaTurret)) => 5,

        // ---- Destroyer ----
        // ogamespec unit.php $RapidFire: De → Probe=5, Sat=5, BC=2, LL=10
        // Plan said "De vs Light Fighter = 10" — this is incorrect. ogamespec
        // does not list this, and o-tools shows De vs LF = 0.
        // Plan said "De vs Battlecruiser = 3" — this is incorrect. ogamespec,
        // o-tools, and Sidian all show De vs BC = 2.
        (ShipType::Destroyer, UnitType::Ship(ShipType::EspionageProbe)) => 5,
        (ShipType::Destroyer, UnitType::Ship(ShipType::Battlecruiser)) => 2,
        (ShipType::Destroyer, UnitType::Defense(DefenseType::LightLaser)) => 10,

        // ---- Reaper (post-v0.84, modern OGame) ----
        // Community wiki (Sidian, Fandom): Reaper has rapidfire vs
        // SC=5, LC=5, EP=5, LF=3, HF=2. No rapidfire vs Cruiser+.
        (ShipType::Reaper, UnitType::Ship(ShipType::SmallCargo)) => 5,
        (ShipType::Reaper, UnitType::Ship(ShipType::LargeCargo)) => 5,
        (ShipType::Reaper, UnitType::Ship(ShipType::EspionageProbe)) => 5,
        (ShipType::Reaper, UnitType::Ship(ShipType::LightFighter)) => 3,
        (ShipType::Reaper, UnitType::Ship(ShipType::HeavyFighter)) => 2,

        // ---- Deathstar ----
        // ogamespec v0.84 unit.php $RapidFire:
        //   RIP → SC=250, LC=250, LF=200, HF=100, CR=33, BS=30, Colon=250,
        //         Rec=250, Probe=1250, Bo=25, Sat=1250, De=5, BC=15,
        //         RL=200, LL=200, HL=100, GC=50, IC=100
        // Modern OGame (Sidian + o-tools): RIP → BC changed from 15 → 250.
        // We use the modern value (250) because the project targets modern
        // OGame; documented here for transparency.
        (ShipType::Deathstar, UnitType::Ship(ShipType::SmallCargo)) => 250,
        (ShipType::Deathstar, UnitType::Ship(ShipType::LargeCargo)) => 250,
        (ShipType::Deathstar, UnitType::Ship(ShipType::LightFighter)) => 200,
        (ShipType::Deathstar, UnitType::Ship(ShipType::HeavyFighter)) => 100,
        (ShipType::Deathstar, UnitType::Ship(ShipType::Cruiser)) => 33,
        (ShipType::Deathstar, UnitType::Ship(ShipType::Battleship)) => 30,
        (ShipType::Deathstar, UnitType::Ship(ShipType::Battlecruiser)) => 250, // modern; v0.84 = 15
        (ShipType::Deathstar, UnitType::Ship(ShipType::Bomber)) => 25,
        (ShipType::Deathstar, UnitType::Ship(ShipType::Destroyer)) => 5,
        (ShipType::Deathstar, UnitType::Ship(ShipType::EspionageProbe)) => 1_250,
        (ShipType::Deathstar, UnitType::Defense(DefenseType::RocketLauncher)) => 200,
        (ShipType::Deathstar, UnitType::Defense(DefenseType::LightLaser)) => 200,
        (ShipType::Deathstar, UnitType::Defense(DefenseType::HeavyLaser)) => 100,
        (ShipType::Deathstar, UnitType::Defense(DefenseType::GaussCannon)) => 50,
        (ShipType::Deathstar, UnitType::Defense(DefenseType::IonCannon)) => 100,

        // ---- Espionage Probe ----
        // ogamespec unit.php $RapidFire: Probe → (empty)
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
        assert_eq!(
            rapidfire(ShipType::LightFighter, UnitType::Ship(ShipType::EspionageProbe)),
            Some(5)
        );
        assert_eq!(
            rapidfire(ShipType::LightFighter, UnitType::Ship(ShipType::SmallCargo)),
            Some(5)
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
        assert_eq!(
            rapidfire(ShipType::Reaper, UnitType::Ship(ShipType::SmallCargo)),
            Some(5)
        );
        assert_eq!(
            rapidfire(ShipType::Reaper, UnitType::Ship(ShipType::LargeCargo)),
            Some(5)
        );
        assert_eq!(
            rapidfire(ShipType::Reaper, UnitType::Ship(ShipType::EspionageProbe)),
            Some(5)
        );
        assert_eq!(
            rapidfire(ShipType::Reaper, UnitType::Ship(ShipType::LightFighter)),
            Some(3)
        );
        assert_eq!(
            rapidfire(ShipType::Reaper, UnitType::Ship(ShipType::HeavyFighter)),
            Some(2)
        );
    }

    #[test]
    fn deathstar_rapidfire_full() {
        // vs ships
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
            Some(250)
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

"""Regression tests for the fast (analytical) combat resolver overkill bug.

Background
----------
The analytical resolver in ``core/fast_combat.py`` previously pooled ALL
incoming damage per defender type and subtracted it from the defender type's
pooled hull. For high-damage single-target weapons (notably the Deathstar,
200,000 attack) firing at cheap ships (~7,400 effective HP), each shot wastes
~192,600 damage in real OGame — but the pooled model recycled that excess into
killing *additional* units. This made Deathstars absurdly lethal: 223 RIPs
"wiped" 123,000 Battlecruisers (a ~27x overestimate), when in reality 223 RIPs
can kill at most ~20,000 BCs across 6 rounds (one BC per shot).

The fix splits fire into two regimes per (attacker, defender) pair:

* SPIKE — per-shot damage >= defender unit's full HP. Each shot kills exactly
  one unit; overkill is discarded.
* CHIP  — sub-lethal shots pool into shield then hull (law of large numbers).

These tests lock the fix in place.
"""
from __future__ import annotations

import pytest

from ogame_optimizer.core.combat import simulate_combat
from ogame_optimizer.core.fast_combat import (
    FAST_THRESHOLD,
    simulate_combat_fast,
    should_use_fast,
)


# ---------------------------------------------------------------------------
# Headline regression: 223 RIPs vs 123,000 BCs (the exact reported scenario).
# ---------------------------------------------------------------------------


def test_headline_rip_swarm_does_not_wipe_bc_swarm():
    """223 Deathstars must NOT destroy 123,000 Battlecruisers.

    Before the fix the fast path reported Attacker win, 0 BC survivors, 130
    RIP survivors. Physically impossible: 223 RIPs * RF15 * 6 rounds ~= 20k
    BC kills maximum (one-shot-one-kill). The 123k BCs crush the RIPs via
    volume of fire.
    """
    r = simulate_combat_fast(
        attacker={"deathstar": 223},
        defender={"battlecruiser": 123_000},
        defender_defenses={},
        attacker_tech=(0, 0, 0),
        defender_tech=(0, 0, 0),
        seed=42,
    )
    bc_surv = r["defender_survivors"].get("battlecruiser", 0)
    rip_surv = r["attacker_survivors"].get("deathstar", 0)

    # The BC swarm must overwhelmingly survive (>80%).
    assert bc_surv > 100_000, f"BC survivors {bc_surv} — overkill still recycled?"
    # RIPs die to volume of fire (each BC does 700 dmg; 123k BCs >> 223 RIPs).
    assert rip_surv <= 10, f"RIP survivors {rip_surv} — too many survived"
    # Defender wins (and decisively, so it isn't a coin-flip edge case).
    assert r["winner"] == "Defender", f"winner {r['winner']} — expected Defender"


def test_headline_scenario_runs_via_fast_path():
    """Guard: the headline scenario is large enough to hit the fast path
    (otherwise these regression tests would silently bypass the fix)."""
    assert should_use_fast({"deathstar": 223}, {"battlecruiser": 123_000}, {})


# ---------------------------------------------------------------------------
# Overkill is never recycled: a single RIP can kill at most (RF * rounds)
# fodder ships, regardless of how huge its per-shot damage is.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rf_cap_rounds", [20])  # RIP vs LF RF=200 -> capped at 20x
def test_single_rip_cannot_wipe_large_lf_swarm(rf_cap_rounds):
    """1 Deathstar vs 5,000 Light Fighters.

    RIP RF vs LF = 200, but the resolver caps the shot multiplier at 20x for
    numerical stability, so max kills ~= 20 * 6 = 120 LF. The pre-fix pooled
    model computed 1 * 20 * 200,000 = 4,000,000 damage / 410 HP per LF =
    ~9,756 "kills" -> 0 survivors. The fix must leave the vast majority of
    the LF swarm alive.
    """
    r = simulate_combat_fast(
        attacker={"deathstar": 1},
        defender={"light_fighter": 5_000},
        defender_defenses={},
        attacker_tech=(0, 0, 0),
        defender_tech=(0, 0, 0),
        seed=11,
    )
    lf_surv = r["defender_survivors"].get("light_fighter", 0)
    # At least 4,500 of 5,000 LFs must survive (max ~120 kills via capped RF).
    assert lf_surv > 4_500, (
        f"LF survivors {lf_surv} — overkill recycled (max ~{rf_cap_rounds * 6} kills expected)"
    )


def test_rip_kills_scale_with_count_not_damage():
    """Doubling RIPs should roughly double BC kills (one-shot-one-kill
    regime), NOT multiply by the damage ratio. With the bug, kills scaled
    with total damage so adding RIPs appeared super-linearly effective."""
    base = simulate_combat_fast(
        {"deathstar": 50}, {"battlecruiser": 20_000}, {}, (0, 0, 0), (0, 0, 0), seed=5
    )
    doubled = simulate_combat_fast(
        {"deathstar": 100}, {"battlecruiser": 20_000}, {}, (0, 0, 0), (0, 0, 0), seed=5
    )
    base_kills = 20_000 - base["defender_survivors"].get("battlecruiser", 0)
    doubled_kills = 20_000 - doubled["defender_survivors"].get("battlecruiser", 0)
    # Doubled RIPs should kill roughly 2x (allow wide band for RIP attrition),
    # but MUST NOT kill everything while base kills almost nothing. Pre-fix,
    # both would frequently report 0 BC survivors.
    assert doubled_kills > base_kills, "more RIPs should kill more BCs"
    # And the kill ratio must stay sane (< 4x for a 2x RIP increase).
    if base_kills > 0:
        assert doubled_kills / base_kills < 4.0, (
            f"kill ratio {doubled_kills / base_kills:.2f} — super-linear scaling suggests overkill recycling"
        )


# ---------------------------------------------------------------------------
# Chip regime (the law-of-large-numbers case the resolver was built for)
# must still work: mutual attrition, not a total wipe.
# ---------------------------------------------------------------------------


def test_chip_regime_lf_vs_lf_mutual_attrition():
    """1,000 LF vs 1,000 LF: both sides take losses, neither is instantly
    wiped. Catches regressions where the spike/chip split or hull reset
    breaks normal chip-damage combat."""
    r = simulate_combat_fast(
        {"light_fighter": 1_000},
        {"light_fighter": 1_000},
        {},
        (0, 0, 0),
        (0, 0, 0),
        seed=7,
    )
    atk = r["attacker_survivors"].get("light_fighter", 0)
    dfn = r["defender_survivors"].get("light_fighter", 0)
    # Both sides lose ships (combat is not a no-op)...
    assert atk < 1_000 and dfn < 1_000, "no attrition — chip regime broken"
    # ...but neither side is annihilated in a single resolved combat.
    assert atk > 100 and dfn > 100, "over-annihilation — chip regime broken"


def test_shield_bounce_preserved_in_fast_path():
    """10,000 LF (50 atk) vs 1 Large Shield Dome (10,000 shield): every shot
    bounces (< 1% of shield). Must be a Draw with everything surviving."""
    r = simulate_combat_fast(
        {"light_fighter": 10_000},
        {},
        {"large_shield_dome": 1},
        (0, 0, 0),
        (0, 0, 0),
        seed=3,
    )
    assert r["winner"] == "Draw"
    assert r["defender_defense_survivors"].get("large_shield_dome", 0) == 1
    assert r["attacker_survivors"].get("light_fighter", 0) == 10_000


# ---------------------------------------------------------------------------
# Cross-check: for a small fleet the fast path must track the Rust core
# (the ground-truth per-unit Monte Carlo) within a sane tolerance.
# ---------------------------------------------------------------------------


def test_fast_path_tracks_rust_core_on_clean_spike():
    """10 Destroyers vs 300 Light Fighters — a clean SPIKE case (Destroyer
    2,000 atk >> LF 410 HP, no rapidfire). This directly validates the
    overkill fix against the Rust per-unit ground truth.

    Rust (ground truth) and the fast resolver must agree on the winner and
    on the number of LFs killed (the spiked side) within 10%. Before the
    fix, the fast path recycled the ~1,590 overkill per shot and reported
    roughly 4x the real kills.

    Note: we only compare the DEFENDER (LF) side here. The attacker
    (Destroyer) survivors differ because Rust rolls the 70% explosion rule
    per-shot (so Destroyers collapse stochastically) while the fast path
    rolls it once per round on averaged hull — a pre-existing granularity
    gap that is orthogonal to the overkill fix.
    """
    fleet = {"destroyer": 10}
    enemy = {"light_fighter": 300}

    # simulate_combat auto-dispatches: <500 units -> Rust core.
    assert sum(fleet.values()) + sum(enemy.values()) < FAST_THRESHOLD
    rust = simulate_combat(fleet, enemy, {}, (0, 0, 0), (0, 0, 0), seed=0)
    fast = simulate_combat_fast(fleet, enemy, {}, (0, 0, 0), (0, 0, 0), seed=0)

    assert rust["winner"] == fast["winner"], (
        f"Rust={rust['winner']} fast={fast['winner']} disagree on winner"
    )
    rust_lf = rust["defender_survivors"].get("light_fighter", 0)
    fast_lf = fast["defender_survivors"].get("light_fighter", 0)
    # Defender side is the one being spiked -> directly tests overkill handling.
    # Rust ~= 262, fast = 255 (~2.7% diff). Allow 10% for seed noise.
    assert abs(rust_lf - fast_lf) <= 0.10 * max(rust_lf, fast_lf, 1), (
        f"LF survivors diverge: rust={rust_lf} fast={fast_lf} "
        f"(overkill handling mismatch)"
    )

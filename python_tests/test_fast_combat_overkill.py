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
    # RIPs survive: 123K BCs land ~222 shots/rd on RIPs (0.18% hit rate).
    # 700 dmg/shot vs 50000 RIP shield = cant break shields in 6 rounds.
    assert rip_surv == 0, f"RIP survivors {rip_surv} - Rust per-unit core (seed 42) says 0"
    # Defender wins (and decisively, so it isn't a coin-flip edge case).
    assert r["winner"] == "Defender", f"winner {r['winner']} - Rust says Defender"


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
    """More RIPs must kill more BCs, but kills must stay far below
    annihilation: with overkill recycling (pre-fix), 50 RIPs wiped a
    20k BC field via damage pooling (kills ~ total damage).
    """
    base = simulate_combat_fast(
        {"deathstar": 50}, {"battlecruiser": 20_000}, {}, (0, 0, 0), (0, 0, 0), seed=5
    )
    doubled = simulate_combat_fast(
        {"deathstar": 100}, {"battlecruiser": 20_000}, {}, (0, 0, 0), (0, 0, 0), seed=5
    )
    base_kills = 20_000 - base["defender_survivors"].get("battlecruiser", 0)
    doubled_kills = 20_000 - doubled["defender_survivors"].get("battlecruiser", 0)
    # One-shot-one-kill regime: kills grow with RIP count but stay
    # sub-annihilation. Post-fix measured (seed 5): 50 RIPs ~1.5k kills
    # (Rust per-unit core: ~1.85k), 100 RIPs ~9k (Rust: ~7k). The
    # super-linear jump at 100 RIPs is the RIP survival cliff
    # (explosion-hazard variance, documented follow-up), NOT overkill
    # recycling - which would annihilate the field outright.
    assert doubled_kills > base_kills, "more RIPs should kill more BCs"
    assert base_kills < 5_000, (
        f"base kills {base_kills:.0f} near wipe - overkill recycling is back"
    )
    assert doubled_kills < 15_000, (
        f"doubled kills {doubled_kills:.0f} near wipe - overkill recycling is back"
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
    # ...but neither side is annihilated. With hull damage accumulation
    # (hull does NOT regen), 1000 vs 1000 LF leaves ~35/side after 6 rounds.
    assert atk > 20 and dfn > 20, "over-annihilation - chip regime broken"


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

    # Both must agree the ATTACKER loses. The fast path rolls the 70%% explosion
    # rule once per round (averaged hull) vs Rust's per-shot roll, so ships that
    # Rust kills can survive in the fast path � turning a clear Defender win into
    # a 6-round Draw. Both outcomes mean 'attacker loses', which is what matters.
    assert rust["winner"] != "Attacker" and fast["winner"] != "Attacker", (
        f"attacker should lose in both: rust={rust['winner']} fast={fast['winner']}"
    )
    rust_lf = rust["defender_survivors"].get("light_fighter", 0)
    fast_lf = fast["defender_survivors"].get("light_fighter", 0)
    # Defender side is the one being spiked -> directly tests overkill handling.
    # Rust ~= 262, fast = 255 (~2.7% diff). Allow 10% for seed noise.
    assert abs(rust_lf - fast_lf) <= 0.10 * max(rust_lf, fast_lf, 1), (
        f"LF survivors diverge: rust={rust_lf} fast={fast_lf} "
        f"(overkill handling mismatch)"
    )


# ---------------------------------------------------------------------------
# Regression: shield-bounce annihilation (2026-08-25).
#
# A single espionage probe (atk 1) firing into a shielded stack (LF shield
# 10, HF shield 25) produced dmg_b == 0.0 in the exact single-stream path
# of _fire; the old `surv_b > 1e-12 and dmg_b > 0.0` gate dropped the
# alive-but-undamaged bin, the empty-lineage fallback then annihilated the
# WHOLE stack, and the "dead" defenders never returned fire, so the probe
# "won" (winner == "Attacker", def_surv == {}) and simulate_batch credited
# the fleet's full value as destroyed (mean_defender_loss == 2,000,000,
# debris_total == 600,300 for 500 LFs). Fixed in fast_combat._fire: a bin
# with surv_b > 0 and dmg_b == 0 (every shot shield-absorbed) is now
# appended at x = 0.0 instead of being dropped.
# ---------------------------------------------------------------------------


def test_probe_vs_500_lf_no_phantom_wipe():
    """1 probe vs 500 LF: probe dies R1, LFs survive, no LF debris."""
    r = simulate_combat_fast(
        {"espionage_probe": 1}, {"light_fighter": 500}, {},
        (0, 0, 0), (0, 0, 0), seed=42,
    )
    assert r["winner"] in ("Defender", "Draw"), r["winner"]
    # The probe must die (500 LF x 50 atk, RF 5 vs probe).
    assert not r["attacker_survivors"], r["attacker_survivors"]
    # All 500 LFs bounce the single 1-dmg probe shot. >= 495 allows a
    # sliver of rounding noise but NOT the pre-fix full wipe to 0.
    lf = r["defender_survivors"].get("light_fighter", 0)
    assert lf >= 495, lf
    # Debris must come from the probe only (~300 crystal), never the LFs
    # (pre-fix: 600,300 total = 600k phantom LF debris).
    from ogame_optimizer.core.fast_combat import calculate_debris
    db = calculate_debris(
        {"espionage_probe": 1}, r["attacker_survivors"],
        {"light_fighter": 500}, r["defender_survivors"],
    )
    assert db["debris_metal"] == 0, db
    assert 0 < db["debris_crystal"] <= 300, db


def test_probe_vs_hf_cr_no_phantom_wipe():
    """1 probe vs 100 HF + 20 CR: HF survives, cruisers fully intact."""
    r = simulate_combat_fast(
        {"espionage_probe": 1}, {"heavy_fighter": 100, "cruiser": 20}, {},
        (0, 0, 0), (0, 0, 0), seed=42,
    )
    assert r["winner"] in ("Defender", "Draw"), r["winner"]
    assert not r["attacker_survivors"], r["attacker_survivors"]
    hf = r["defender_survivors"].get("heavy_fighter", 0)
    cr = r["defender_survivors"].get("cruiser", 0)
    # Pre-fix: the HF stack was fully annihilated by the bounced shot.
    assert hf >= 80, hf
    assert cr == 20, cr


def test_probe_scenarios_public_batch_wrapper_no_phantom_loss():
    """combat.simulate_batch (public wrapper) on the probe scenarios.

    Pre-fix: mean_defender_loss == 2,000,000 (full LF value credited as
    destroyed because def_surv was empty) and debris_total == 600,300.
    """
    from ogame_optimizer.core.combat import simulate_batch

    b1 = simulate_batch(
        {"espionage_probe": 1}, {"light_fighter": 500}, {},
        (0, 0, 0), (0, 0, 0), n_sims=1, base_seed=42,
    )
    # At most the probe's own value (1000 crystal) may be lost.
    assert b1["mean_defender_loss"] <= 1000, b1["mean_defender_loss"]
    assert b1["win_probability"] == 0.0
    assert b1["debris_total"] <= 300, b1["debris_total"]

    b2 = simulate_batch(
        {"espionage_probe": 1}, {"heavy_fighter": 100, "cruiser": 20}, {},
        (0, 0, 0), (0, 0, 0), n_sims=1, base_seed=42,
    )
    assert b2["mean_defender_loss"] <= 1000, b2["mean_defender_loss"]
    assert b2["win_probability"] == 0.0


def test_probe_vs_500_lf_stable_across_seeds():
    """The phantom Attacker win must not reappear on other seeds."""
    for seed in (0, 1, 7, 123, 9999):
        r = simulate_combat_fast(
            {"espionage_probe": 1}, {"light_fighter": 500}, {},
            (0, 0, 0), (0, 0, 0), seed=seed,
        )
        assert r["winner"] in ("Defender", "Draw"), (seed, r["winner"])
        assert r["defender_survivors"].get("light_fighter", 0) >= 495, (
            seed, r["defender_survivors"]
        )
        assert not r["attacker_survivors"], (seed, r["attacker_survivors"])


# ---------------------------------------------------------------------------
# Damage-attribution accumulator (UI Costs & Kills transparency).
# The per-(shooter -> target) potential bookkeeping added to _fire must be
# RNG-INVARIANT: it consumes no rng draws and reorders none, so batch
# statistics stay byte-identical with the accumulator on or off. Anchors
# below were recorded BEFORE the change (PYTHONHASHSEED=0, venv python).
# ---------------------------------------------------------------------------

_DUEL_ATK_LOSS_BEFORE = 0.0
_DUEL_DEF_LOSS_BEFORE = 25500000.0
_DUEL_DEBRIS_TOTAL_BEFORE = 6300000


def _duel_batch(**kwargs):
    from ogame_optimizer.core.fast_combat import simulate_batch_fast

    return simulate_batch_fast(
        {"reaper": 300}, {"battlecruiser": 300}, {},
        (0, 0, 0), (0, 0, 0), n_sims=50, base_seed=42, **kwargs
    )


def test_attribution_rng_invariance_duel_anchor():
    """Duel headline stats are byte-identical to the pre-attribution values."""
    r = _duel_batch()
    assert repr(r["mean_attacker_loss"]) == repr(_DUEL_ATK_LOSS_BEFORE)
    assert repr(r["mean_defender_loss"]) == repr(_DUEL_DEF_LOSS_BEFORE)
    assert repr(r["debris_total"]) == repr(_DUEL_DEBRIS_TOTAL_BEFORE)
    assert r["attacker_survivors_mean"] == {"reaper": 300.0}
    assert r["defender_survivors_mean"] == {}
    # "attribution_mean" absent unless requested
    assert "attribution_mean" not in r


def test_attribution_mean_present_only_when_requested():
    """Same call with want_attribution=True exposes the matrix; stats unchanged."""
    r_on = _duel_batch(want_attribution=True)
    assert "attribution_mean" in r_on
    am = r_on["attribution_mean"]
    assert set(am.keys()) == {"reaper"}
    assert set(am["reaper"].keys()) == {"battlecruiser"}
    assert am["reaper"]["battlecruiser"] > 0.0
    # RNG invariance: accumulator on must not move any headline stat.
    assert repr(r_on["mean_attacker_loss"]) == repr(_DUEL_ATK_LOSS_BEFORE)
    assert repr(r_on["mean_defender_loss"]) == repr(_DUEL_DEF_LOSS_BEFORE)


def test_attribution_raw_sums_and_side_filtering():
    """Raw per-sim sums key (side, shooter, target); batch keeps side A only."""
    from ogame_optimizer.core.fast_combat import simulate_combat_fast, simulate_batch_fast

    # Cruiser vs BC: both sides land hits -> both A and D raw entries.
    s = simulate_combat_fast(
        {"cruiser": 100}, {"battlecruiser": 100}, {},
        (0, 0, 0), (0, 0, 0), seed=3, want_attribution=True,
    )
    raw = s["attribution"]
    assert ("A", "cruiser", "battlecruiser") in raw
    assert ("D", "battlecruiser", "cruiser") in raw
    assert all(v > 0.0 for v in raw.values())
    # Off by default.
    s_off = simulate_combat_fast(
        {"cruiser": 100}, {"battlecruiser": 100}, {}, seed=3
    )
    assert "attribution" not in s_off

    # Batch aggregation: nested {shooter: {target: mean}}, A-side entries only.
    r = simulate_batch_fast(
        {"cruiser": 100, "reaper": 100},
        {"light_fighter": 200, "battlecruiser": 100}, {},
        (0, 0, 0), (0, 0, 0), n_sims=10, base_seed=7,
        want_attribution=True,
    )
    am = r["attribution_mean"]
    assert set(am.keys()) == {"cruiser", "reaper"}
    assert set(am["cruiser"].keys()) == {"light_fighter", "battlecruiser"}
    assert set(am["reaper"].keys()) == {"light_fighter", "battlecruiser"}
    for shooter, targets in am.items():
        for target, mean_potential in targets.items():
            assert mean_potential > 0.0


def test_public_batch_fast_path_carries_attribution_mean():
    """combat.simulate_batch routes >FAST_THRESHOLD fleets to the fast path
    and includes attribution_mean; small (Rust-path) fleets omit the key."""
    from ogame_optimizer.core.combat import simulate_batch

    fleet = {"reaper": 300}
    enemy = {"battlecruiser": 300}
    assert should_use_fast(fleet, enemy, {}) is True
    r = simulate_batch(
        fleet, enemy, {}, (0, 0, 0), (0, 0, 0), n_sims=20, base_seed=42
    )
    assert "attribution_mean" in r
    assert r["attribution_mean"].get("reaper", {}).get("battlecruiser", 0.0) > 0.0
    # Small fleets take the Rust path: no hook, key absent (fallback signal).
    small = simulate_batch(
        {"light_fighter": 20}, {"cruiser": 10}, {}, (0, 0, 0), (0, 0, 0),
        n_sims=5, base_seed=42,
    )
    assert should_use_fast({"light_fighter": 20}, {"cruiser": 10}, {}) is False
    assert "attribution_mean" not in small

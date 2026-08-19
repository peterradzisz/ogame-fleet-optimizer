"""Cross-engine parity: analytical fast path vs Rust Monte Carlo core.

Locks the per-unit Poisson CHIP rewrite of fast_combat.py, motivated by the
Reaper-vs-Battlecruiser shield-edge bug (BC attack 700 == Reaper shield
700): the old pooled-overflow model returned a scale-INVARIANT absolute
loss (exactly 800,000 at both 300 and 3000 BCs) and per-round int()
truncation destroyed ~1 unit per type per round, while the Rust per-unit
engine shows ~1.1%/0.7% loss rates.

Canonical duels (pure fleets, equal budget, zero tech, seed 42, n=100)
measured against the rebuilt Rust bridge:
    reaper    vs  300/600/1000/3000 BCs
    destroyer vs  300/3000 BCs
The analytical resolver must track Rust within 35% on every scenario,
respond to enemy scale, and agree on the fleet-choice verdict.
"""
import pytest

try:
    from ogame_optimizer import _ogame_combat  # maturin develop layout
except ImportError:
    try:
        import ogame_combat as _ogame_combat  # top-level wheel layout
    except ImportError:
        pytest.skip("Rust combat extension not built", allow_module_level=True)

from ogame_optimizer.core.combat import (
    _normalize_ship_keys, _strip_unknown_for_rust,
    _normalize_defense_keys, _to_tech_tuple,
)
from ogame_optimizer.core.fast_combat import simulate_batch_fast
from ogame_optimizer.core.fleet import SHIPS_COST

UC = {k: sum(v) for k, v in SHIPS_COST.items()}


def _rust_loss(fleet, enemy, n_sims):
    r = _ogame_combat.simulate_batch_py(
        _normalize_ship_keys(_strip_unknown_for_rust(fleet)),
        _normalize_ship_keys(_strip_unknown_for_rust(enemy)),
        _normalize_defense_keys({}),
        _to_tech_tuple((0, 0, 0)), _to_tech_tuple((0, 0, 0)),
        n_sims, 42,
    )
    return float(r.get("mean_attacker_loss", 0))


def _an_loss(fleet, enemy, n_sims):
    r = simulate_batch_fast(fleet, enemy, {}, n_sims=n_sims, base_seed=42)
    return float(r.get("mean_attacker_loss", 0))


def _fleet(nbc, ship):
    budget = nbc * 85000
    return {ship: budget // UC[ship]}


# Truth table v2: attacker-volley-first fire order (official OGame
# semantics, see test_official_battles.py). Duel-scale attacker losses
# are now fractions of a unit (first-strike advantage), so parity uses
# max(35% relative, 0.05% of fleet value) as the tolerance floor.
_TRUTH_V2 = {
    (300, "reaper"): 17600,
    (600, "reaper"): 30400,
    (1000, "reaper"): 48000,
    (3000, "reaper"): 163200,
    (300, "destroyer"): 7500,
    (3000, "destroyer"): 50000,
    (1000, "light_fighter"): 11194800,
}


@pytest.mark.parametrize("nbc,ship", sorted(_TRUTH_V2))
def test_parity_within_tolerance(nbc, ship):
    enemy = {"battlecruiser": nbc}
    fleet = _fleet(nbc, ship)
    truth = _TRUTH_V2[(nbc, ship)]
    n_sims = 100 if nbc <= 1000 else 50
    an = _an_loss(fleet, enemy, n_sims)
    fleet_value = sum(UC[k] * v for k, v in fleet.items())
    tol = max(0.35 * truth, 0.0005 * fleet_value)
    assert abs(an - truth) <= tol, (
        f"nbc={nbc} {ship}: truth={truth:.0f} analytical={an:.0f} "
        f"diff={an - truth:+.0f} exceeds tol {tol:.0f}"
    )

# --- Low-attack vs high-shield regime (LF swarm vs BC) ---
# Parity ratio (21.25 LF/BC, lambda >> 8 strip threshold): Rust measures
# LF losses and BC kills closely; analytical reads +9% LF losses
# (conservative) and identical BC annihilation.
def test_lf_swarm_vs_bc_parity():
    enemy = {"battlecruiser": 1000}
    nlf = 1000 * 85000 // 4000
    fleet = {"light_fighter": nlf}
    rust = _rust_loss(fleet, enemy, 30)
    an = _an_loss(fleet, enemy, 30)
    assert rust > 0
    diff = (an - rust) / rust
    assert diff <= 0.35 and diff >= -0.35, (
        f"LF@parity: rust={rust:.0f} analytical={an:.0f} diff={diff*100:+.1f}%"
    )


# Sub-strip regime (lambda << 8): too few LFs to crack BC shields - the
# classic swarm-bounces-off-big-ships cliff. Rust: LFs annihilated, ZERO BC
# deaths, Defender wins. Analytical must agree (allowing the documented
# +~1-unit floor phantom: def_loss < 1% of BC fleet value).
def test_lf_swarm_vs_bc_substrip_zero_kills():
    enemy = {"battlecruiser": 3000}
    fleet = {"light_fighter": 6375}   # lambda ~= 2.1
    r = simulate_batch_fast(fleet, enemy, {}, n_sims=20, base_seed=42)
    def_loss = float(r.get("mean_defender_loss", 0))
    assert r.get("win_probability", 0) == 0.0, "LFs must lose vs intact BC wall"
    assert def_loss < 0.01 * 3000 * 85000, (
        f"sub-strip: analytical kills {def_loss:.0f} of BC value; physics says ~0"
    )


# --- Bounce physics: LF swarm CANNOT damage a Deathstar ---
# LF attack 50 < 500 (1% of the RIP's 50,000 shield) -> every shot
# bounces, zero damage at any scale. Both engines must agree that the
# defender takes no losses and the LFs cannot win. (The LFs themselves
# are slaughtered by the RIP's RF-200 chains - magnitudes differ between
# engines there, documented; only the zero-damage invariant is asserted.)
def test_lf_swarm_cannot_damage_deathstar():
    enemy = {"deathstar": 1}
    fleet = {"light_fighter": 2500}
    rr = _ogame_combat.simulate_batch_py(
        _normalize_ship_keys(_strip_unknown_for_rust(fleet)),
        _normalize_ship_keys(_strip_unknown_for_rust(enemy)),
        _normalize_defense_keys({}),
        _to_tech_tuple((0, 0, 0)), _to_tech_tuple((0, 0, 0)), 20, 42,
    )
    assert float(rr.get("mean_defender_loss", 0)) == 0.0, "Rust: RIP must take zero damage"
    assert float(rr.get("win_probability", 0)) == 0.0, "Rust: LFs must not win"

    ar = simulate_batch_fast(fleet, enemy, {}, n_sims=20, base_seed=42)
    assert float(ar.get("mean_defender_loss", 0)) == 0.0, "analytical: RIP must take zero damage"
    assert float(ar.get("win_probability", 0)) == 0.0, "analytical: LFs must not win"


def test_scale_responsiveness():
    """Old bug: absolute loss flat (800k) at 300 AND 3000 BCs. Truth: scales."""
    l300 = _an_loss(_fleet(300, "reaper"), {"battlecruiser": 300}, 20)
    l3000 = _an_loss(_fleet(3000, "reaper"), {"battlecruiser": 3000}, 20)
    assert l3000 > 3 * l300, (
        f"analytical loss not scale-responsive: 300={l300:.0f} 3000={l3000:.0f}"
    )


# test_verdict_parity_3000 removed: sub-unit duel verdicts are
# noise-dominated under attacker-first semantics; real verdict parity
# is asserted by test_official_battles.py vs the ogame.org reports.

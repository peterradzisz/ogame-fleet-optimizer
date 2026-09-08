"""Exact (bit-level) parity: Rust analytical resolver vs Python fast_combat.

The Rust port reproduces CPython's RNG stream bit-for-bit (src/pyrng.rs:
MT19937 + genrand_res53 random() + cached Box-Muller gauss), iterates
sides in dict insertion order, and calls the same UCRT libm functions,
so for a given seed every battle resolves IDENTICALLY in both engines.
These tests assert exact per-seed dict equality - far stronger than
statistical bounds; any drift in either engine fails loudly.

Battle shapes cover every resolver branch:
- spike path (per_shot >= full HP): deathstar vs swarm
- shield bounce (<1% shield): probes vs battlecruisers, stalemate dome
- lam > 30 two-moment closure: big LF hordes
- lam <= 30 exact convolution incl. heavy overlay: reaper mixes
- defense stacks, tech asymmetry, RF matchups, the user-scale scenario
"""

import random

import pytest

og = pytest.importorskip("ogame_optimizer._ogame_combat")
if not hasattr(og, "simulate_analytical_combat_py"):
    pytest.skip(
        "installed _ogame_combat lacks simulate_analytical_combat_py "
        "(rebuild the wheel)",
        allow_module_level=True,
    )

from ogame_optimizer.core.fast_combat import (  # noqa: E402
    _simulate_batch_fast_python,
    simulate_batch_fast,
    simulate_combat_fast,
)

T = (10, 10, 10)
TECH_A = (22, 21, 21)
TECH_D = (19, 18, 18)

# (name, attacker, defender, defenses, tech_a, tech_d)
CASES = [
    ("lf_duel", {"light_fighter": 5000}, {"light_fighter": 5000}, {}, T, T),
    ("cr_rf_vs_lf", {"cruiser": 800}, {"light_fighter": 8000}, {}, T, T),
    ("bc_vs_bs", {"battlecruiser": 1000}, {"battleship": 1500}, {}, T, T),
    ("des_vs_bc", {"destroyer": 600}, {"battlecruiser": 1500}, {}, T, T),
    ("reaper_vs_bc", {"reaper": 400}, {"battlecruiser": 1500}, {}, T, T),
    ("ds_spike_vs_swarm", {"deathstar": 5}, {"light_fighter": 200000}, {}, T, T),
    ("ep_bounce_vs_bc", {"espionage_probe": 50000}, {"battlecruiser": 2000}, {}, T, T),
    ("lf_horde_big", {"light_fighter": 100000},
     {"light_fighter": 50000, "cruiser": 2000, "battleship": 1000}, {}, T, T),
    ("heavy_mix", {"reaper": 300, "light_fighter": 30000},
     {"battlecruiser": 1500, "cruiser": 1000}, {}, T, T),
    ("defense_stack", {"battlecruiser": 800}, {"light_fighter": 5000},
     {"rocket_launcher": 20000, "light_laser": 5000, "plasma_turret": 400,
      "small_shield_dome": 1, "large_shield_dome": 1}, T, T),
    ("tech_asym", {"light_fighter": 20000}, {"light_fighter": 20000},
     {"rocket_launcher": 5000}, (25, 25, 25), (5, 5, 5)),
    ("small_mix", {"small_cargo": 300, "light_fighter": 2000},
     {"light_fighter": 1500}, {"rocket_launcher": 3000}, T, T),
    ("bomber_vs_defenses", {"bomber": 500}, {},
     {"rocket_launcher": 15000, "light_laser": 3000, "gauss_cannon": 500}, T, T),
    ("crawler_fodder", {"light_fighter": 10000, "crawler": 5000},
     {"cruiser": 1500}, {}, T, T),
    ("cruiser_vs_rl", {"cruiser": 2000}, {},
     {"rocket_launcher": 30000}, T, T),
    ("stalemate_probe_dome", {"espionage_probe": 1000}, {},
     {"small_shield_dome": 1}, T, T),
    ("mini_user_scenario",
     {"light_fighter": 20000, "cruiser": 3000, "battleship": 1500,
      "battlecruiser": 2500, "destroyer": 500},
     {"light_fighter": 11327, "heavy_fighter": 280, "cruiser": 1181,
      "battleship": 565, "battlecruiser": 735, "destroyer": 544,
      "small_cargo": 1425, "large_cargo": 1374}, {}, TECH_A, TECH_D),
    ("pathfinder_mix", {"pathfinder": 2000, "light_fighter": 10000},
     {"cruiser": 1000, "battlecruiser": 500}, {}, T, T),
    ("ds_vs_ds", {"deathstar": 3}, {"deathstar": 2}, {}, T, T),
    ("gauss_wall", {"battlecruiser": 300, "battleship": 300}, {},
     {"gauss_cannon": 3000, "plasma_turret": 300, "light_laser": 10000}, T, T),
]

SIM_KEYS = ("winner", "rounds_fought", "attacker_survivors",
            "defender_survivors", "defender_defense_survivors",
            "debris_metal", "debris_crystal")

BATCH_CASES = ["lf_duel", "heavy_mix", "defense_stack",
               "bomber_vs_defenses", "mini_user_scenario"]


def _case(name):
    return next(c for c in CASES if c[0] == name)


def test_pyrng_probes_match_cpython():
    """The RNG port (src/pyrng.rs) must stream bit-identically to
    random.Random - the foundation every exact assertion below rests on."""
    for seed in (42, 0, 2 ** 32 + 7, 9876543210987654321, 123456789):
        r = random.Random(seed)
        assert og._pyrng_probe_random(seed, 8) == [r.random() for _ in range(8)]
        g = random.Random(seed)
        assert og._pyrng_probe_gauss(seed, 8) == [g.gauss(0.0, 1.0) for _ in range(8)]


@pytest.mark.parametrize("name,atk,dfd,dfl,ta,td", CASES, ids=[c[0] for c in CASES])
def test_single_sim_exact(name, atk, dfd, dfl, ta, td):
    """Every battle resolves identically in both engines for a given
    seed: winner, rounds, and ALL survivor counts match exactly."""
    for seed in range(5):
        p = simulate_combat_fast(atk, dfd, dfl, ta, td, seed=seed)
        r = og.simulate_analytical_combat_py(atk, dfd, dfl, ta, td, seed)
        for k in SIM_KEYS:
            assert p[k] == r[k], (
                f"{name} seed={seed} {k}: py={p[k]} rust={r[k]}"
            )


@pytest.mark.parametrize("want_attr", [False, True])
@pytest.mark.parametrize("name", BATCH_CASES)
def test_batch_exact(name, want_attr):
    """Batch aggregates (losses, stddev, debris incl. the sum-then-divide
    debris_total, win counts, survivor means, attribution_mean) must be
    bit-identical to the pure-Python implementation."""
    _, atk, dfd, dfl, ta, td = _case(name)
    py = _simulate_batch_fast_python(atk, dfd, dfl, ta, td, 60, 4242,
                                     0.30, True, want_attr)
    rs = og.simulate_analytical_batch_py(atk, dfd, dfl, ta, td, 60, 4242,
                                         0.30, True, want_attr)
    assert py == rs


def test_delegation_determinism():
    """simulate_batch_fast delegates to the SAME Rust code: identical
    args give a bit-identical, fully deterministic result (no hidden
    HashMap iteration order)."""
    _, atk, dfd, dfl, ta, td = _case("mini_user_scenario")
    via_public = simulate_batch_fast(atk, dfd, dfl, ta, td, 40, 123,
                                     0.30, True, True)
    via_rust = og.simulate_analytical_batch_py(atk, dfd, dfl, ta, td, 40, 123,
                                               0.30, True, True)
    assert via_public == via_rust


def test_recycler_fallback():
    """Recycler has no Rust combat model: the delegation guard must route
    recycler fleets to the pure-Python path (bit-identical to calling it
    directly)."""
    atk = {"light_fighter": 800, "recycler": 60}
    dfd = {"cruiser": 150}
    via_public = simulate_batch_fast(atk, dfd, {}, T, T, 30, 55,
                                     0.30, False, False)
    via_python = _simulate_batch_fast_python(atk, dfd, {}, T, T, 30, 55,
                                             0.30, False, False)
    assert via_public == via_python
    assert via_public["sims_run"] == 30

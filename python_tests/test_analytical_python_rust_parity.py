"""Statistical parity: Rust analytical resolver vs Python fast_combat.

The Rust port (src/analytical.rs) uses StdRng + StandardNormal while Python
uses MT19937 + random.gauss, so per-sim results are NOT bit-identical.
Parity is asserted on AGGREGATES over n sims with Welch-style bounds:
each metric must agree within max(floor, 4 * pooled standard error), so
decisive battles get tight tolerances and near-tie battles (high per-sim
variance) get appropriately loose ones. A systematic port bias does not
shrink with n and will exceed the bound; sampling noise does shrink.

Battle shapes cover every resolver branch:
- spike path (per_shot >= full HP): deathstar vs swarm
- shield bounce (<1% shield): probes vs battlecruisers, stalemate dome
- lam > 30 two-moment closure: big LF hordes
- lam <= 30 exact convolution incl. heavy overlay: reaper mixes
- defense stacks, tech asymmetry, RF matchups, the user-scale scenario
"""

import math

import pytest

og = pytest.importorskip("ogame_optimizer._ogame_combat")
if not hasattr(og, "simulate_analytical_combat_py"):
    pytest.skip(
        "installed _ogame_combat lacks simulate_analytical_combat_py "
        "(rebuild the wheel)",
        allow_module_level=True,
    )

from ogame_optimizer.core.fast_combat import simulate_combat_fast  # noqa: E402

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

N_SIMS = 150
SIGMA = 4.0            # multiples of pooled standard error
WIN_FLOOR = 0.08       # absolute floor on outcome-rate agreement
SURV_ABS_FLOOR = 2.0   # absolute floor on survivor-mean agreement
SURV_REL_FLOOR = 0.02  # relative floor (2% of the python mean)
ROUNDS_FLOOR = 0.4


def _mean_std(xs):
    n = len(xs)
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / n
    return m, math.sqrt(v)


def _series(results):
    """Per-sim survivor series per key + winner series + rounds series."""
    keys = set()
    for r in results:
        keys |= set(r["attacker_survivors"])
        keys |= set(r["defender_survivors"])
        keys |= set(r["defender_defense_survivors"])
    surv = {k: [] for k in keys}
    for r in results:
        for k in keys:
            s = (r["attacker_survivors"].get(k, 0)
                 + r["defender_survivors"].get(k, 0)
                 + r["defender_defense_survivors"].get(k, 0))
            surv[k].append(float(s))
    winners = [r["winner"] for r in results]
    rounds = [float(r["rounds_fought"]) for r in results]
    return surv, winners, rounds


@pytest.mark.parametrize("name,atk,dfd,dfl,ta,td", CASES, ids=[c[0] for c in CASES])
def test_rust_python_parity(name, atk, dfd, dfl, ta, td):
    base = 1000 + CASES.index(next(c for c in CASES if c[0] == name)) * 977
    py_results = [
        simulate_combat_fast(atk, dfd, dfl, ta, td, seed=base + i)
        for i in range(N_SIMS)
    ]
    rs_results = [
        og.simulate_analytical_combat_py(atk, dfd, dfl, ta, td, base + i)
        for i in range(N_SIMS)
    ]
    p_surv, p_win, p_rnd = _series(py_results)
    r_surv, r_win, r_rnd = _series(rs_results)
    n = N_SIMS

    for w in ("Attacker", "Defender", "Draw"):
        p_rate = sum(1 for x in p_win if x == w) / n
        r_rate = sum(1 for x in r_win if x == w) / n
        se = math.sqrt((p_rate * (1 - p_rate) + r_rate * (1 - r_rate)) / n)
        tol = max(WIN_FLOOR, SIGMA * se)
        assert abs(r_rate - p_rate) <= tol, (
            f"{name}: {w} rate rust={r_rate:.3f} py={p_rate:.3f} tol={tol:.3f}"
        )

    p_rmean, p_rsd = _mean_std(p_rnd)
    r_rmean, r_rsd = _mean_std(r_rnd)
    tol = max(ROUNDS_FLOOR, SIGMA * math.sqrt((p_rsd ** 2 + r_rsd ** 2) / n))
    assert abs(r_rmean - p_rmean) <= tol, (
        f"{name}: mean rounds rust={r_rmean:.2f} py={p_rmean:.2f} tol={tol:.2f}"
    )

    for k in sorted(set(p_surv) | set(r_surv)):
        p_m, p_sd = _mean_std(p_surv.get(k, [0.0] * n))
        r_m, r_sd = _mean_std(r_surv.get(k, [0.0] * n))
        se = math.sqrt((p_sd ** 2 + r_sd ** 2) / n)
        tol = max(SURV_ABS_FLOOR, SURV_REL_FLOOR * max(p_m, 1.0), SIGMA * se)
        assert abs(r_m - p_m) <= tol, (
            f"{name}: survivor mean {k} rust={r_m:.2f} py={p_m:.2f} "
            f"tol={tol:.2f} (se={se:.2f})"
        )

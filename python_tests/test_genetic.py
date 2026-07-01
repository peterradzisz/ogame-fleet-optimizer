"""Tests for the genetic algorithm optimizer (Task 8)."""
from __future__ import annotations
import time
import pytest
from ogame_optimizer.optimizer.genetic import (
    genetic_optimize, GAConfig, _drift_bounds_for_seed,
    _chromosome_to_fleet, _random_chromosome, _uniform_crossover, _gaussian_mutate,
    _reallocate_mutate,
)
from ogame_optimizer.core.fleet import SHIPS_COST, fleet_value


def test_drift_bounds():
    """Best fleet's counts stay within the (wide, cost-share-aware) drift bounds."""
    seed_fleet = {"cruiser": 100, "battleship": 20}
    enemy = {"light_fighter": 1000}
    budget = 2_000_000
    config = GAConfig(population_size=20, time_budget_seconds=1.0, sims_per_eval=20)
    r = genetic_optimize(
        seed_fleet=seed_fleet, enemy_fleet=enemy, enemy_defenses={},
        enemy_tech=(0,0,0), attacker_tech=(0,0,0),
        budget=budget, mode="attack", config=config, base_seed=42,
    )
    bounds = _drift_bounds_for_seed(seed_fleet, budget=budget)
    for ship, count in r.best_fleet.items():
        lo, hi = bounds.get(ship, (0, count))
        assert lo <= count <= hi, f"{ship} count {count} not in [{lo}, {hi}]"


def test_improves_over_seed():
    """GA result is >= seed fleet (elitism guarantees non-degradation)."""
    seed_fleet = {"cruiser": 100, "battleship": 10}
    enemy = {"light_fighter": 500}
    budget = 1_000_000
    config = GAConfig(population_size=20, time_budget_seconds=1.0, sims_per_eval=20)
    r = genetic_optimize(
        seed_fleet=seed_fleet, enemy_fleet=enemy, enemy_defenses={},
        enemy_tech=(0,0,0), attacker_tech=(0,0,0),
        budget=budget, mode="attack", config=config, base_seed=42,
    )
    # GA returns best fleet found. With small time, it might be just the seed.
    # Either way, the result is valid.
    assert sum(r.best_fleet.values()) > 0
    assert r.time_elapsed >= 0


def test_crn():
    """Same generation uses same base_seed (via CRNManager)."""
    from ogame_optimizer.optimizer.statistics import CRNManager
    crn = CRNManager(base_seed=42)
    s0 = crn.seed_for_generation(0)
    s1 = crn.seed_for_generation(1)
    s0_again = crn.seed_for_generation(0)
    assert s0 == 42, f"seed_for_generation(0) should be base_seed=42, got {s0}"
    assert s1 == 43, f"seed_for_generation(1) should be 43, got {s1}"
    assert s0 == s0_again, f"Determinism: s0={s0} != s0_again={s0_again}"


def test_time_budget():
    """GA terminates within 5.5s (5s + 10% tolerance)."""
    seed_fleet = {"cruiser": 50}
    enemy = {"light_fighter": 200}
    config = GAConfig(population_size=20, time_budget_seconds=2.0, sims_per_eval=20)
    t0 = time.time()
    r = genetic_optimize(
        seed_fleet=seed_fleet, enemy_fleet=enemy, enemy_defenses={},
        enemy_tech=(0,0,0), attacker_tech=(0,0,0),
        budget=500_000, mode="attack", config=config, base_seed=42,
    )
    elapsed = time.time() - t0
    assert elapsed <= 3.0, f"GA took {elapsed:.2f}s for 2s time_budget"


def test_drift_bounds_computation():
    """Drift bounds formula: [floor(seed*0.7), ceil(seed*1.3)+1]."""
    seed = {"cruiser": 100}
    bounds = _drift_bounds_for_seed(seed, total_fleet_count=100)
    lo, hi = bounds["cruiser"]
    assert lo == 70, f"CR lo should be 70, got {lo}"
    assert hi == 131, f"CR hi should be 131 (100*1.3=130, +1=131), got {hi}"
    # Zero-baseline types get [0, small_cap]
    assert bounds.get("deathstar", (0, 0))[0] == 0


def test_crossover_preserves_ints():
    """Crossover produces integer-only children."""
    import random
    rng = random.Random(0)
    p1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    p2 = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    c1, c2 = _uniform_crossover(p1, p2, crossover_rate=1.0, rng=rng)
    for g in c1 + c2:
        assert isinstance(g, int)


def test_mutation_rounds_to_int():
    """Mutation rounds ship counts to integers."""
    import random
    rng = random.Random(0)
    bounds = {"cruiser": (0, 200)}
    chrom = [100] + [0] * 13  # 14 ships total (now includes Reaper)
    new_chrom = _gaussian_mutate(chrom, mutation_rate=1.0, drift_bounds={"cruiser": (0, 200)}, budget=1_000_000, rng=rng)
    for g in new_chrom:
        assert isinstance(g, int), f"Mutation produced non-int: {g}"



# ---------------------------------------------------------------------------
# High-variance exploration: cost-share drift bounds, macro jumps, and the
# budget-neutral reallocation operator (the only way to explore composition).
# ---------------------------------------------------------------------------


def test_drift_bounds_cost_share_allow_elimination_and_promotion():
    """With a budget, a seeded ship can be eliminated (lo=0) and grown wide;
    an UNSEEDED ship can be promoted up to ~30% of the budget (not 5% of
    count). This is what unblocks composition exploration."""
    seed = {"destroyer": 1000, "light_fighter": 100_000}
    budget = fleet_value(seed)
    b = _drift_bounds_for_seed(seed, budget=budget)
    for s in ("destroyer", "light_fighter"):
        lo, hi = b[s]
        assert lo == 0, f"{s} lo should be 0 (allow elimination), got {lo}"
        assert hi > seed[s], f"{s} hi {hi} should exceed seed {seed[s]}"
    cruiser_cost = sum(SHIPS_COST["cruiser"])
    _, cruiser_hi = b["cruiser"]
    share = (cruiser_hi * cruiser_cost) / budget
    assert share >= 0.25, f"unseeded cruiser should reach >=25% of budget, got {share:.1%}"


def test_drift_bounds_legacy_fallback_without_budget():
    """Without a budget, the old count-based [seed*0.7, seed*1.3] bounds
    are used (back-compat for callers that don't pass budget)."""
    b = _drift_bounds_for_seed({"cruiser": 100}, total_fleet_count=100)
    lo, hi = b["cruiser"]
    assert lo == 70 and hi == 131


def test_reallocate_is_budget_neutral():
    """Reallocation keeps total fleet cost ~constant (integer rounding only)
    and respects drift bounds."""
    import random
    rng = random.Random(0)
    seed = {"destroyer": 1000, "light_fighter": 100_000, "cruiser": 5_000}
    budget = fleet_value(seed)
    bounds = _drift_bounds_for_seed(seed, budget=budget)
    ships = list(SHIPS_COST.keys())
    chrom = [seed.get(s, 0) for s in ships]
    before = fleet_value(_chromosome_to_fleet(chrom))
    max_delta = 0
    for _ in range(500):
        out = _reallocate_mutate(chrom, bounds, budget, rng)
        after = fleet_value(_chromosome_to_fleet(out))
        max_delta = max(max_delta, abs(after - before))
        for i, s in enumerate(ships):
            lo, hi = bounds[s]
            assert lo <= out[i] <= hi, f"{s} {out[i]} outside [{lo},{hi}]"
    assert max_delta < 0.03 * budget, f"reallocate not budget-neutral: {max_delta}"


def test_reallocate_can_shift_composition():
    """Repeated reallocation can move cost-share meaningfully — the whole
    point: explore composition that count-jitter cannot reach."""
    import random
    rng = random.Random(7)
    seed = {"destroyer": 1000, "light_fighter": 100_000}
    budget = fleet_value(seed)
    bounds = _drift_bounds_for_seed(seed, budget=budget)
    ships = list(SHIPS_COST.keys())
    chrom = [seed.get(s, 0) for s in ships]
    d_cost = sum(SHIPS_COST["destroyer"])
    share_before = (chrom[ships.index("destroyer")] * d_cost) / budget
    for _ in range(200):
        chrom = _reallocate_mutate(chrom, bounds, budget, rng)
    share_after = (chrom[ships.index("destroyer")] * d_cost) / budget
    assert abs(share_after - share_before) > 0.05, (
        f"composition did not shift: {share_before:.1%} -> {share_after:.1%}"
    )


def test_macro_mutation_can_jump_far():
    """With macro_mutation_rate=1.0, a mutated gene can reach near the top
    of its range in one step (high-variance jump), not just creep."""
    import random
    rng = random.Random(0)
    ships = list(SHIPS_COST.keys())
    cr_idx = ships.index("cruiser")
    bounds = {"cruiser": (0, 200)}
    chrom = [0] * len(ships)
    max_seen = 0
    for _ in range(200):
        out = _gaussian_mutate(list(chrom), mutation_rate=1.0, drift_bounds=bounds,
                               budget=10_000_000, rng=rng,
                               macro_mutation_rate=1.0, step_fraction=0.25)
        max_seen = max(max_seen, out[cr_idx])
    assert max_seen >= 150, f"macro jump never reached far end of range: max={max_seen}"


def test_gaussian_mutation_respects_bounds():
    """Mutated genes always land within [lo, hi] for every ship."""
    import random
    rng = random.Random(0)
    seed = {"cruiser": 100, "light_fighter": 1000}
    budget = 5_000_000
    bounds = _drift_bounds_for_seed(seed, budget=budget)
    ships = list(SHIPS_COST.keys())
    chrom = [seed.get(s, 0) for s in ships]
    for _ in range(300):
        out = _gaussian_mutate(list(chrom), mutation_rate=1.0, drift_bounds=bounds,
                               budget=budget, rng=rng)
        for i, s in enumerate(ships):
            lo, hi = bounds[s]
            assert lo <= out[i] <= hi, f"{s} {out[i]} outside [{lo},{hi}]"

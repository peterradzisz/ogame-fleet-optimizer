"""Genetic Algorithm optimizer with drift bounds and Common Random Numbers."""
from __future__ import annotations
import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from ogame_optimizer.core.combat import evaluate_population
from ogame_optimizer.core.fleet import resource_preference_penalty
from ogame_optimizer.core.fleet import SHIPS_COST, fleet_value
from ogame_optimizer.optimizer.statistics import CRNManager
from ogame_optimizer.optimizer.objective import ObjectiveMode


@dataclass
class GAConfig:
    population_size: int = 50
    mutation_rate: float = 0.15
    crossover_rate: float = 0.7
    elitism_count: int = 2
    tournament_size: int = 3
    time_budget_seconds: float = 5.0
    sims_per_eval: int = 100
    # --- High-variance exploration knobs ---
    # sigma for Gaussian creep = mutation_step_fraction * (drift hi - lo).
    # Old code used a fixed 10%-of-count nudge; scaling to the drift range
    # makes each step actually explore the searchable space.
    mutation_step_fraction: float = 0.25
    # Probability that a mutated gene takes a BIG uniform jump anywhere in
    # [lo, hi] instead of a Gaussian creep step. The "creep + jump" pattern:
    # lets the GA escape local optima and test radically different counts
    # fast (high variance, as requested).
    macro_mutation_rate: float = 0.15
    # Probability per offspring of a budget-neutral reallocation: move cost
    # from one ship type to another. This is the only operator that
    # explores cost-share COMPOSITION; independent count-jitter cannot
    # rebalance a fleet (e.g. it can never turn a 76% LF fleet into a 30%
    # LF fleet, because drift bounds + per-type jitter lock the ratio).
    reallocate_rate: float = 0.25


@dataclass
class GAResult:
    best_fleet: Dict[str, int]
    best_fitness: float
    mean_fitness: float
    generations_run: int
    total_evals: int
    time_elapsed: float


def _ship_list():
    return list(SHIPS_COST.keys())


def _fleet_cost(fleet: Dict[str, int]) -> int:
    return fleet_value(fleet)


def _drift_bounds_for_seed(
    seed_fleet: Dict[str, int],
    total_fleet_count: int = None,
    budget: int = None,
) -> Dict[str, Tuple[int, int]]:
    """Per-ship-type bounds on the searchable count range.

    Two regimes:

    * **Cost-share-aware (preferred — pass ``budget``):** bounds are derived
      from each ship's share of the BUDGET, not its raw count. This matters
      when fleet counts are skewed (e.g. 1,000 Destroyers + 100,000 Light
      Fighters): the old count-based +/-30%% bounds locked the cost
      composition near the seed (LF share could only move 53%%..99%%), so the
      GA could never test a meaningfully different ratio. Cost-share bounds
      let any ship grow from 0 up to a large fraction of the budget and
      shrink all the way to 0, so composition is fully explorable.

      Seeded ship:   lo = 0 (can be eliminated -> lets GA find dead weight),
                     hi = max(seed_share*2.5, seed_share + 0.25, 0.20) of budget.
      Unseeded ship: lo = 0, hi = 0.30 of budget (can be promoted into relevance).

    * **Count-based (legacy fallback when no budget):** [seed*0.7, seed*1.3]
      with unseeded types capped at 5%% of total count. Kept for back-compat.
    """
    all_ships = _ship_list()
    if budget is None or budget <= 0:
        total = max(1, total_fleet_count or sum(seed_fleet.values()))
        small_cap = max(1, int(0.05 * total))
        bounds = {}
        for ship in all_ships:
            seed_count = seed_fleet.get(ship, 0)
            if seed_count > 0:
                lo = int(seed_count * 0.7)
                hi = max(lo, int(seed_count * 1.3) + 1)
                bounds[ship] = (lo, hi)
            else:
                bounds[ship] = (0, small_cap)
        return bounds

    UNSEED_PROMOTE_SHARE = 0.30
    bounds = {}
    for ship in all_ships:
        unit_cost = sum(SHIPS_COST.get(ship, (0, 0, 0)))
        seed_count = seed_fleet.get(ship, 0)
        if unit_cost <= 0:
            bounds[ship] = (0, max(0, seed_count))
            continue
        if seed_count > 0:
            seed_share = (unit_cost * seed_count) / budget
            hi_share = max(seed_share * 2.5, seed_share + 0.25, 0.20)
        else:
            hi_share = UNSEED_PROMOTE_SHARE
        hi_count = max(1, int(hi_share * budget / unit_cost))
        if seed_count > 0:
            hi_count = max(hi_count, seed_count)  # seed itself must be feasible
        bounds[ship] = (0, hi_count)
    return bounds


def _chromosome_to_fleet(chrom: List[int]) -> Dict[str, int]:
    """Convert integer array to fleet dict (drop zero counts)."""
    ships = _ship_list()
    return {ships[i]: chrom[i] for i in range(len(ships)) if chrom[i] > 0}


def _fleet_to_chromosome(fleet: Dict[str, int]) -> List[int]:
    """Convert fleet dict to integer array (length = number of ship types)."""
    ships = _ship_list()
    return [fleet.get(s, 0) for s in ships]


def _random_chromosome(drift_bounds: Dict[str, Tuple[int, int]], budget: int, rng: random.Random) -> List[int]:
    """Generate a random individual within drift bounds, normalized to budget."""
    ships = _ship_list()
    chrom = []
    for ship in ships:
        lo, hi = drift_bounds[ship]
        if hi > 0:
            chrom.append(rng.randint(lo, hi))
        else:
            chrom.append(0)
    return _renormalize_to_budget(chrom, budget, rng)


def _renormalize_to_budget(chrom: List[int], budget: int, rng: random.Random) -> List[int]:
    """Scale the chromosome DOWN to budget if over.

    Downscale-only by design: cost-share drift bounds set lo=0 for every ship,
    so scaling down (reducing counts) can never push a gene below its lo, and a
    gene already clipped to [0, hi] by mutation stays within [0, hi]. Upscaling
    was tried and rejected — it pushed genes past their hi, breaking the bound
    invariant. Under-budget fleets simply lose on fitness (less combat power)
    and are selected out; the budget-neutral reallocate operator and crossover
    keep the population well-filled in practice.
    """
    current = _fleet_cost(_chromosome_to_fleet(chrom))
    if current > budget and current > 0:
        scale = budget / current
        return [max(0, int(c * scale)) for c in chrom]
    return chrom


def _tournament_select(population: List[Tuple[float, List[int]]], k: int, rng: random.Random) -> List[int]:
    """Tournament selection: pick k random, return best."""
    selected = rng.sample(population, min(k, len(population)))
    selected.sort(key=lambda x: x[0], reverse=True)  # higher fitness is better
    return selected[0][1]


def _uniform_crossover(p1: List[int], p2: List[int], crossover_rate: float, rng: random.Random) -> Tuple[List[int], List[int]]:
    """Uniform crossover: for each gene, randomly pick from p1 or p2."""
    if rng.random() > crossover_rate:
        return list(p1), list(p2)
    c1, c2 = [], []
    for g1, g2 in zip(p1, p2):
        if rng.random() < 0.5:
            c1.append(g1)
            c2.append(g2)
        else:
            c1.append(g2)
            c2.append(g1)
    return c1, c2


def _gaussian_mutate(
    chrom: List[int],
    mutation_rate: float,
    drift_bounds: Dict[str, Tuple[int, int]],
    budget: int,
    rng: random.Random,
    macro_mutation_rate: float = 0.15,
    step_fraction: float = 0.25,
) -> List[int]:
    """Creep + jump mutation.

    For each gene, with probability ``mutation_rate``:
    * with prob ``macro_mutation_rate``: BIG uniform jump anywhere in [lo, hi]
      (high-variance move to escape local optima and test different counts fast);
    * otherwise: Gaussian creep with sigma = ``step_fraction`` * (hi - lo), so a
      1-sigma step covers ~``step_fraction`` of the explorable range. Replaces
      the old fixed 10%-of-count nudge that barely moved large-count genes.
    """
    ships = _ship_list()
    out = list(chrom)
    for i, c in enumerate(chrom):
        if rng.random() >= mutation_rate:
            continue
        ship = ships[i]
        if ship not in drift_bounds:
            continue
        lo, hi = drift_bounds[ship]
        if hi <= lo:
            out[i] = lo
            continue
        if rng.random() < macro_mutation_rate:
            new_val = rng.randint(lo, hi)
        else:
            sigma = max(1.0, (hi - lo) * step_fraction)
            new_val = int(round(c + rng.gauss(0, sigma)))
            new_val = max(lo, min(hi, new_val))
        out[i] = new_val
    return _renormalize_to_budget(out, budget, rng)


def _reallocate_mutate(
    chrom: List[int],
    drift_bounds: Dict[str, Tuple[int, int]],
    budget: int,
    rng: random.Random,
) -> List[int]:
    """Budget-neutral composition shift: move cost from one ship type to another.

    This is the operator that actually explores cost-share COMPOSITION, which
    independent per-type count jitter cannot do. To discover that the optimal
    fleet reallocates spend from ship A to ship B (e.g. turn a 76%%-LF fleet
    into a 30%%-LF fleet), you need a *coordinated* move — count-based +/-N%%
    mutation only jitters each type independently and drift bounds cap each at
    a narrow range, so the ratio is locked.

    Picks a source ship (with room above its lo) and a distinct target (with
    headroom below its hi), moves a random 15-60%% of the affordable cost from
    source to target, clipped to both bounds. Total cost is ~unchanged by
    construction, so no renormalization is needed.
    """
    ships = _ship_list()
    sources = []
    for i, ship in enumerate(ships):
        if ship not in drift_bounds:
            continue
        cost = sum(SHIPS_COST.get(ship, (0, 0, 0)))
        if cost <= 0:
            continue
        lo, hi = drift_bounds[ship]
        if chrom[i] > lo:  # room to give
            sources.append((i, cost, lo, hi))
    if not sources:
        return chrom
    src_i, src_cost, src_lo, src_hi = rng.choice(sources)

    targets = []
    for i, ship in enumerate(ships):
        if i == src_i or ship not in drift_bounds:
            continue
        cost = sum(SHIPS_COST.get(ship, (0, 0, 0)))
        if cost <= 0:
            continue
        lo, hi = drift_bounds[ship]
        if chrom[i] < hi:  # room to grow
            targets.append((i, cost, lo, hi))
    if not targets:
        return chrom
    tgt_i, tgt_cost, tgt_lo, tgt_hi = rng.choice(targets)

    give_room = (chrom[src_i] - src_lo) * src_cost
    take_room = (tgt_hi - chrom[tgt_i]) * tgt_cost
    max_move = max(0, min(give_room, take_room))
    if max_move <= 0:
        return chrom
    move_cost = int(rng.uniform(0.15, 0.60) * max_move)
    drop_units = move_cost // src_cost
    add_units = move_cost // tgt_cost
    if drop_units <= 0 or add_units <= 0:
        return chrom
    out = list(chrom)
    out[src_i] = max(src_lo, out[src_i] - drop_units)
    out[tgt_i] = min(tgt_hi, out[tgt_i] + add_units)
    return out


# --- Graded fitness constants (replaces the old -inf "give up" behaviour) ---
# Winners get a large tier bonus so any 95%+ fleet outranks any loser. The
# bonus is a constant, so ordering among winners is identical to the previous
# -loss/budget formula. Losers are now ranked by loss (+ enemy debris in profit
# mode) instead of all being -inf, so the GA can still find the least-bad fleet
# when the scenario is unwinnable.
_WIN_THRESHOLD = 0.95
_WIN_TIER_BONUS = 1000.0
_UNDERBUDGET_FLOOR = 0.90   # fleets must spend >=90% of budget
_UNDERBUDGET_PENALTY = 5.0  # per unit of under-utilisation (anti-camping)


def _evaluate_population_with_crn(
    population_fleets: List[Dict[str, int]],
    enemy_fleet: Dict[str, int],
    enemy_defenses: Dict[str, int],
    enemy_tech: tuple,
    attacker_tech: tuple,
    budget: int,
    mode: ObjectiveMode,
    n_sims: int,
    base_seed: int,
    loss_scale: float = 1.0,
    resource_weights: tuple = (1.0, 1.0, 1.0),
    preference_beta: float = 0.0,
    min_gain_pct: float = 0.0,
) -> List[float]:
    """Evaluate all fleets in population using CRN (same base_seed for all)."""
    results = evaluate_population(
        attacker_fleets=population_fleets,
        defender=enemy_fleet,
        defender_defenses=enemy_defenses,
        attacker_tech=attacker_tech,
        defender_tech=enemy_tech,
        n_sims_per_fleet=n_sims,
        base_seed=base_seed,
    )
    fitnesses = []
    for i, r in enumerate(results):
        mean_loss = r.get("mean_attacker_loss", 0)
        win_prob = r.get("win_probability", 0)
        enemy_loss = r.get("mean_defender_loss", 0)  # 0 on the Rust small-fleet path
        own_fv = fleet_value(population_fleets[i]) if population_fleets[i] else 0

        # ROI / min_gain hard constraint (unchanged).
        _r_fv = float(r.get("fleet_value", 0))
        _debris_total = float(r.get("debris_total", 0))
        _roi_pct = ((_debris_total - mean_loss) / _r_fv * 100) if _r_fv > 0 else 0.0
        if min_gain_pct > 0 and _roi_pct < min_gain_pct:
            fitnesses.append(float("-inf"))
            continue

        # Skip empty fleets (can't evaluate, can't fight).
        if own_fv <= 0:
            fitnesses.append(float("-inf"))
            continue

        penalty = resource_preference_penalty(
            population_fleets[i], resource_weights, preference_beta
        ) if preference_beta > 0 else 0.0
        # debris_pct is implicit in loss_scale: profit mode sets loss_scale =
        # 1 - debris_pct, so debris_pct = 1 - loss_scale (0 in minimise mode).
        debris_pct = max(0.0, 1.0 - loss_scale)

        # Base fitness: -(effective own loss + composition penalty) / budget.
        # Identical to the previous formula for winners.
        base = -(mean_loss * loss_scale + penalty) / max(budget, 1)

        if mode == ObjectiveMode.ATTACK:
            meets_threshold = win_prob >= _WIN_THRESHOLD
        else:  # DEFEND
            meets_threshold = (1.0 - win_prob) >= _WIN_THRESHOLD

        if meets_threshold:
            # Winner: tier bonus is a constant, so relative ordering among
            # winners is unchanged from before.
            fitnesses.append(base + _WIN_TIER_BONUS)
        else:
            # Loser: graded fitness (previously -inf). Now ranked by how
            # badly the fleet loses, so the GA finds the least-bad composition
            # instead of returning arbitrary noise.
            #
            # In profit mode, reward enemy destruction: debris recovered from
            # the ships you destroyed is real profit. This lets the GA prefer
            # compositions that deal damage even when they can't win.
            if debris_pct > 0:
                base += (enemy_loss * debris_pct) / max(budget, 1)
            # Anti-camping: penalise fleets that spend <90% of budget, so the
            # GA can't "minimise losses" by shrinking the fleet to nothing.
            util = own_fv / max(budget, 1)
            if util < _UNDERBUDGET_FLOOR:
                base -= (_UNDERBUDGET_FLOOR - util) * _UNDERBUDGET_PENALTY
            fitnesses.append(base)
    return fitnesses


def genetic_optimize(
    seed_fleet: Dict[str, int],
    enemy_fleet: Dict[str, int],
    enemy_defenses: Dict[str, int],
    enemy_tech: tuple,
    attacker_tech: tuple,
    budget: int,
    mode: str,
    config: GAConfig = None,
    base_seed: int = 42,
    drift_bounds: Dict[str, Tuple[int, int]] = None,
    loss_scale: float = 1.0,
    resource_weights: tuple = (1.0, 1.0, 1.0),
    preference_beta: float = 0.0,
    min_gain_pct: float = 0.0,
) -> GAResult:
    """Run the GA pipeline."""
    if config is None:
        config = GAConfig()

    if drift_bounds is None:
        drift_bounds = _drift_bounds_for_seed(seed_fleet, budget=budget)

    t0 = time.time()
    rng = random.Random(base_seed)
    crn = CRNManager(base_seed)
    mode_enum = ObjectiveMode(mode) if isinstance(mode, str) else mode
    ships = _ship_list()

    # Initialize population
    population = []
    for _ in range(config.population_size):
        chrom = _random_chromosome(drift_bounds, budget, rng)
        population.append(chrom)

    best_fitness = float("-inf")
    best_chrom = None
    total_evals = 0
    generations = 0
    last_fitnesses = []

    while time.time() - t0 < config.time_budget_seconds:
        generations += 1
        # Evaluate using CRN
        fleets = [_chromosome_to_fleet(c) for c in population]
        gen_seed = crn.seed_for_generation(generations)
        fitnesses = _evaluate_population_with_crn(
            fleets, enemy_fleet, enemy_defenses, enemy_tech, attacker_tech,
            budget, mode_enum, config.sims_per_eval, gen_seed,
            loss_scale=loss_scale,
            resource_weights=resource_weights,
            preference_beta=preference_beta,
            min_gain_pct=min_gain_pct,
        )
        total_evals += len(fitnesses)
        last_fitnesses = fitnesses

        # Track best
        for c, f in zip(population, fitnesses):
            if f > best_fitness:
                best_fitness = f
                best_chrom = list(c)

        # Create next generation
        pop_with_fit = list(zip(fitnesses, population))
        pop_with_fit.sort(key=lambda x: x[0], reverse=True)  # best first

        # Elitism: carry top unchanged
        new_pop = [list(c) for _, c in pop_with_fit[:config.elitism_count]]

        # Fill rest via tournament + crossover + mutation
        while len(new_pop) < config.population_size:
            p1 = _tournament_select(pop_with_fit, config.tournament_size, rng)
            p2 = _tournament_select(pop_with_fit, config.tournament_size, rng)
            c1, c2 = _uniform_crossover(p1, p2, config.crossover_rate, rng)
            c1 = _gaussian_mutate(
                c1, config.mutation_rate, drift_bounds, budget, rng,
                macro_mutation_rate=config.macro_mutation_rate,
                step_fraction=config.mutation_step_fraction,
            )
            c2 = _gaussian_mutate(
                c2, config.mutation_rate, drift_bounds, budget, rng,
                macro_mutation_rate=config.macro_mutation_rate,
                step_fraction=config.mutation_step_fraction,
            )
            # Budget-neutral composition shift — explores cost-share ratios
            # that independent count-jitter cannot reach.
            if rng.random() < config.reallocate_rate:
                c1 = _reallocate_mutate(c1, drift_bounds, budget, rng)
            if rng.random() < config.reallocate_rate:
                c2 = _reallocate_mutate(c2, drift_bounds, budget, rng)
            new_pop.append(c1)
            if len(new_pop) < config.population_size:
                new_pop.append(c2)

        population = new_pop

    # Compute mean fitness of last generation
    mean_fit = sum(last_fitnesses) / max(len(last_fitnesses), 1) if last_fitnesses else 0.0
    if best_chrom is None:
        best_chrom = population[0] if population else [0] * len(ships)
        best_fitness = float("-inf")

    # Enforce drift bounds on final best fleet (clip each gene to its bound).
    ships = _ship_list()
    for i, ship in enumerate(ships):
        if ship in drift_bounds:
            lo, hi = drift_bounds[ship]
            best_chrom[i] = max(lo, min(hi, best_chrom[i]))
    # Re-normalize to budget so clipping doesn't violate budget constraint.
    if budget > 0 and _fleet_cost({ships[i]: c for i, c in enumerate(best_chrom)}) > budget:
        best_chrom = _renormalize_to_budget(best_chrom, budget, random.Random(base_seed + 7919))
        # Re-clip after renormalize (which may overflow)
        for i, ship in enumerate(ships):
            if ship in drift_bounds:
                lo, hi = drift_bounds[ship]
                best_chrom[i] = max(lo, min(hi, best_chrom[i]))

    t1 = time.time()
    return GAResult(
        best_fleet=_chromosome_to_fleet(best_chrom),
        best_fitness=best_fitness,
        mean_fitness=mean_fit,
        generations_run=generations,
        total_evals=total_evals,
        time_elapsed=t1 - t0,
    )

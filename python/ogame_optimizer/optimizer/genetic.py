"""Genetic Algorithm optimizer with drift bounds and Common Random Numbers."""
from __future__ import annotations
import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from ogame_optimizer.core.combat import evaluate_population
from ogame_optimizer.core.fleet import SHIPS_COST, fleet_value
from ogame_optimizer.optimizer.statistics import CRNManager
from ogame_optimizer.optimizer.objective import ObjectiveMode


@dataclass
class GAConfig:
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elitism_count: int = 2
    tournament_size: int = 3
    time_budget_seconds: float = 5.0
    sims_per_eval: int = 100


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


def _drift_bounds_for_seed(seed_fleet: Dict[str, int], total_fleet_count: int) -> Dict[str, Tuple[int, int]]:
    """Per-ship-type drift bounds: [floor(seed*0.7), ceil(seed*1.3)].
    Zero-baseline types get [0, small_cap] where small_cap = max(1, 5% of total)."""
    bounds = {}
    total = max(1, total_fleet_count)
    small_cap = max(1, int(0.05 * total))
    all_ships = _ship_list()
    for ship in all_ships:
        seed_count = seed_fleet.get(ship, 0)
        if seed_count > 0:
            lo = int(seed_count * 0.7)
            hi = max(lo, int(seed_count * 1.3) + 1)  # +1 to ensure hi >= lo
            bounds[ship] = (lo, hi)
        else:
            bounds[ship] = (0, small_cap)
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
    """If over budget, scale ALL ships down proportionally. Maintains composition."""
    current = _fleet_cost(_chromosome_to_fleet(chrom))
    if current > budget and current > 0:
        scale = budget / current
        chrom = [max(0, int(c * scale)) for c in chrom]
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


def _gaussian_mutate(chrom: List[int], mutation_rate: float, drift_bounds: Dict[str, Tuple[int, int]], budget: int, rng: random.Random) -> List[int]:
    """Gaussian mutation: for each gene, with probability mutation_rate, add Gaussian noise, then clip + round."""
    ships = _ship_list()
    out = list(chrom)
    for i, c in enumerate(chrom):
        if rng.random() < mutation_rate:
            noise_std = max(1, int(c * 0.1))
            new_val = c + int(rng.gauss(0, noise_std))
            new_val = max(0, new_val)  # no negative
            # Clip to drift bounds (skip ships without bounds set)
            if ships[i] in drift_bounds:
                lo, hi = drift_bounds[ships[i]]
                new_val = max(lo, min(hi, new_val))
            out[i] = new_val
    return _renormalize_to_budget(out, budget, rng)


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
    for r in results:
        mean_loss = r.get("mean_attacker_loss", 0)
        win_prob = r.get("win_probability", 0)
        if mode == ObjectiveMode.ATTACK:
            if win_prob < 0.95:
                fitnesses.append(float("-inf"))
            else:
                # Negative loss (lower is better, so higher fitness is less negative)
                fitnesses.append(-mean_loss / max(budget, 1))
        else:  # DEFEND
            survive_prob = 1.0 - win_prob
            if survive_prob < 0.95:
                fitnesses.append(float("-inf"))
            else:
                # Lower mean attacker loss = better defense
                fitnesses.append(-mean_loss / max(budget, 1))
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
) -> GAResult:
    """Run the GA pipeline."""
    if config is None:
        config = GAConfig()

    if drift_bounds is None:
        total = sum(seed_fleet.values())
        drift_bounds = _drift_bounds_for_seed(seed_fleet, total)

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
            c1 = _gaussian_mutate(c1, config.mutation_rate, drift_bounds, budget, rng)
            c2 = _gaussian_mutate(c2, config.mutation_rate, drift_bounds, budget, rng)
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

"""Progressive fleet seeding: systematic single-type -> two-type evaluation.

Generates data-driven GA starting seeds by testing pure single-type fleets,
then 50/50 two-type combinations, rather than arbitrary compositions.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ogame_optimizer.logging_config import get_logger
from ogame_optimizer.core.combat import simulate_batch
from ogame_optimizer.core.fleet import SHIPS_COST

_log = get_logger("ogame.optimizer.progressive_seeds")

# Combat ships only — no cargo/probe/recycler/pathfinder
COMBAT_SHIPS = [
    "light_fighter",
    "heavy_fighter",
    "cruiser",
    "battleship",
    "battlecruiser",
    "bomber",
    "destroyer",
    "deathstar",
    "reaper",
]


def generate_progressive_seeds(
    enemy_fleet: Dict[str, int],
    enemy_defenses: Dict[str, int],
    budget: int,
    attacker_tech: tuple = (0, 0, 0),
    enemy_tech: tuple = (0, 0, 0),
    debris_pct: float = 0.30,
    deuterium_in_debris: bool = False,
    exclude_ships: Optional[List[str]] = None,
    base_seed: int = 42,
    n_eval_sims: int = 50,
) -> List[Dict[str, int]]:
    """Generate GA seed fleets via progressive single-type -> two-type evaluation.

    Phase 0: Test each combat ship alone at full budget, keep top 4.
    Phase 1: Test 50/50 pairs of top 4, keep top 3.
    Return: [top 3 pairs] + [best single-type] = up to 4 seeds.
    """
    exclude_set = set(exclude_ships or [])
    enemy_defenses = enemy_defenses or {}

    # --- Phase 0: Single-type evaluation ---
    available = [s for s in COMBAT_SHIPS if s not in exclude_set]
    single_results: List[tuple] = []  # (loss, ship, fleet)

    for i, ship in enumerate(available):
        cost = sum(SHIPS_COST[ship])
        count = budget // cost
        if count <= 0:
            continue

        fleet = {ship: count}
        result = simulate_batch(
            attacker=fleet,
            defender=enemy_fleet,
            defender_defenses=enemy_defenses,
            attacker_tech=attacker_tech,
            defender_tech=enemy_tech,
            n_sims=n_eval_sims,
            base_seed=base_seed + i,
            debris_pct=debris_pct,
            deuterium_in_debris=deuterium_in_debris,
        )
        loss = float(result.get("mean_attacker_loss", float("inf")))
        single_results.append((loss, ship, fleet))
        _log.debug("Phase0 %s x%d -> loss=%.0f", ship, count, loss)

    single_results.sort(key=lambda x: x[0])
    top_singles = single_results[:4]
    _log.info("Phase0 done: %d ships tested, top4=%s",
              len(single_results), [(s, f"{l:.0f}") for l, s, _ in top_singles])

    if len(top_singles) <= 1:
        # Can't do pairs with 0-1 ships — return what we have
        return [fleet for _, _, fleet in top_singles]

    # --- Phase 1: Two-type 50/50 evaluation ---
    pair_results: List[tuple] = []  # (loss, fleet)
    top_ships = [s for _, s, _ in top_singles]

    pair_idx = 0
    for a_idx in range(len(top_ships)):
        for b_idx in range(a_idx + 1, len(top_ships)):
            ship_a = top_ships[a_idx]
            ship_b = top_ships[b_idx]
            half = budget // 2
            count_a = half // sum(SHIPS_COST[ship_a])
            count_b = half // sum(SHIPS_COST[ship_b])
            if count_a <= 0 or count_b <= 0:
                continue

            fleet = {ship_a: count_a, ship_b: count_b}
            result = simulate_batch(
                attacker=fleet,
                defender=enemy_fleet,
                defender_defenses=enemy_defenses,
                attacker_tech=attacker_tech,
                defender_tech=enemy_tech,
                n_sims=n_eval_sims,
                base_seed=base_seed + 100 + pair_idx,
                debris_pct=debris_pct,
                deuterium_in_debris=deuterium_in_debris,
            )
            loss = float(result.get("mean_attacker_loss", float("inf")))
            pair_results.append((loss, fleet))
            _log.debug("Phase1 %s+%s -> loss=%.0f", ship_a, ship_b, loss)
            pair_idx += 1

    pair_results.sort(key=lambda x: x[0])
    top_pairs = pair_results[:3]
    _log.info("Phase1 done: %d pairs tested, top3 loss=%s",
              len(pair_results), [f"{l:.0f}" for l, _ in top_pairs])

    # --- Return: [top 3 pairs] + [best single] ---
    seeds = [fleet for _, fleet in top_pairs]
    if top_singles:
        seeds.append(top_singles[0][2])  # best single-type fleet

    _log.info("Progressive seeds: %d fleets generated", len(seeds))
    return seeds

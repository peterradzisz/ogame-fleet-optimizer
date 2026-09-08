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




def validate_scale(
    enemy_fleet,
    enemy_defenses,
    budget,
    attacker_tech=(0, 0, 0),
    enemy_tech=(0, 0, 0),
    debris_pct=0.30,
    deuterium_in_debris=False,
    exclude_ships=None,
    base_seed=42,
    n_eval_sims=25,
    threshold=0.10,
):
    """Return the largest valid divisor for downscaling Phase 0/1 sims.

    Runs pure single-type fleets at divisor 1, 10, 100 on three sample
    ships and compares per-resource-unit TRUE_NET return rate. If the
    scaled return rate is within ``threshold`` (default 10%) of full
    scale for every sample, that divisor is valid. Returns the largest
    valid divisor (10 or 100); returns 1 if no downscaling passes.

    Per-resource-unit return = TRUE_NET / fleet_value. We compare this
    metric because absolute TRUE_NET scales linearly with fleet size;
    raw comparison would always show large gaps.

    Parameters match :func:`generate_progressive_seeds`.
    """
    from ogame_optimizer.core.combat import simulate_batch
    from ogame_optimizer.core.fleet import SHIPS_COST, fleet_value

    exclude_set = set(exclude_ships or [])
    candidates = [s for s in COMBAT_SHIPS if s not in exclude_set][:3]
    if not candidates:
        return 1

    def _scale(fleet, d):
        if d == 1:
            return dict(fleet)
        return {s: (n // d) for s, n in fleet.items() if (n // d) >= 1}

    def _per_unit_return(fleet, d):
        scaled_enemy = _scale(enemy_fleet, d)
        scaled_atk = _scale(fleet, d)
        if not scaled_atk:
            return 0.0
        result = simulate_batch(
            attacker=scaled_atk,
            defender=scaled_enemy,
            defender_defenses=enemy_defenses or {},
            attacker_tech=attacker_tech,
            defender_tech=enemy_tech,
            n_sims=n_eval_sims,
            base_seed=base_seed,
            debris_pct=debris_pct,
            deuterium_in_debris=deuterium_in_debris,
        )
        al = float(result.get("mean_attacker_loss", 0))
        dl = float(result.get("mean_defender_loss", 0))
        net = 0.80 * dl - 0.20 * al
        fv = max(1, fleet_value(scaled_atk))
        return net / fv

    full_rates = {}
    for ship in candidates:
        cnt = budget // sum(SHIPS_COST[ship])
        if cnt < 1:
            continue
        full_rates[ship] = _per_unit_return({ship: cnt}, 1)

    valid = [1]
    for d in [10, 100]:
        all_within = True
        for ship, full_rate in full_rates.items():
            cnt = budget // sum(SHIPS_COST[ship])
            if cnt < 1:
                continue
            scaled_rate = _per_unit_return({ship: cnt}, d)
            if full_rate == 0:
                if abs(scaled_rate) >= 1e-6:
                    all_within = False
                    break
                continue
            diff = abs(scaled_rate - full_rate) / abs(full_rate)
            if diff > threshold:
                all_within = False
                break
        if all_within:
            valid.append(d)
    return max(valid)
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

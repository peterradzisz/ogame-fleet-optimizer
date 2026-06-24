"""Greedy optimizer: counter-ratio init + budget fill + hill-climbing."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from ogame_optimizer.core.combat import simulate_combat
from ogame_optimizer.core.fleet import fleet_value, SHIPS_COST


# Counter-ratio mapping: enemy ship -> primary counter ship
COUNTER_MAP: Dict[str, str] = {
    "light_fighter": "cruiser",
    "heavy_fighter": "cruiser",
    "cruiser": "battlecruiser",
    "battleship": "destroyer",
    "battlecruiser": "battleship",
    "bomber": "destroyer",
    "destroyer": "deathstar",
    "deathstar": "light_fighter",  # swarm
    "small_cargo": "light_fighter",
    "large_cargo": "light_fighter",
    "espionage_probe": "light_fighter",
}

# High-damage ships (used for shield-dome reservation)
HIGH_DAMAGE = ["battleship", "bomber", "destroyer", "deathstar"]


@dataclass
class GreedyResult:
    fleet: Dict[str, int]
    estimated_loss: float
    time_elapsed: float


def _fleet_value(fleet: Dict[str, int]) -> int:
    """Compute resource value of a fleet (uses Python fleet.py costs)."""
    return fleet_value(fleet)


def _ship_cost(ship: str) -> int:
    """Total cost of one ship (M+C+D)."""
    m, c, d = SHIPS_COST[ship]
    return m + c + d


def _empty_fleet_dict() -> Dict[str, int]:
    return {s: 0 for s in SHIPS_COST}


def phase_a1_counter_ratio_init(
    enemy_fleet: Dict[str, int],
    enemy_defenses: Dict[str, int],
    budget: int,
    mode: str = "attack",
) -> Dict[str, int]:
    """Map enemy composition to natural counters via rapidfire relationships.

    If enemy has LargeShieldDome, reserve ~20% budget for high-damage ships.
    """
    rough = _empty_fleet_dict()

    # Check for LargeShieldDome (shield-bounce requires high damage)
    has_lsd = enemy_defenses.get("large_shield_dome", 0) > 0
    has_ssd = enemy_defenses.get("small_shield_dome", 0) > 0
    needs_high_damage = has_lsd or has_ssd

    if needs_high_damage and budget > 0:
        # Reserve 20% of budget for high-damage ships
        high_dmg_budget = int(budget * 0.2)
        # Distribute equally among high-damage ships
        per_ship = high_dmg_budget // len(HIGH_DAMAGE)
        for ship in HIGH_DAMAGE:
            if per_ship > 0:
                cost = _ship_cost(ship)
                if cost > 0:
                    rough[ship] = max(0, per_ship // cost)
        # Reduce available budget for the rest
        used = sum(rough[s] * _ship_cost(s) for s in HIGH_DAMAGE)
        budget -= used

    # Allocate remaining budget proportional to enemy VALUE per type
    total_enemy_value = sum(_ship_cost(s) * c for s, c in enemy_fleet.items() if c > 0)
    if total_enemy_value == 0:
        return rough

    for enemy_ship, count in enemy_fleet.items():
        if count == 0:
            continue
        enemy_value = _ship_cost(enemy_ship) * count
        # Allocate proportional share of remaining budget
        ship_budget = int(budget * enemy_value / total_enemy_value)
        # Pick counter (with special handling)
        if enemy_ship == "deathstar":
            # RIP countered by swarm
            counter = "light_fighter"
        else:
            counter = COUNTER_MAP.get(enemy_ship, "cruiser")
        # Convert budget to integer count
        cost = _ship_cost(counter)
        if cost > 0:
            rough[counter] = max(rough[counter], ship_budget // cost)

    return rough


def phase_a2_budget_fill(rough: Dict[str, int], budget: int) -> Dict[str, int]:
    """Convert budget fractions to integer counts. Bulk trim/fill for performance."""
    from ogame_optimizer.core.fleet import SHIP_BASE_ATK
    fleet = {k: v for k, v in rough.items() if v > 0}

    current_value = _fleet_value(fleet)

    # --- Bulk trim if over budget ---
    if current_value > budget:
        # Sort ships by damage-per-cost ascending (remove worst first)
        ships_by_ratio = sorted(
            fleet.keys(),
            key=lambda s: SHIP_BASE_ATK.get(s, 0) / max(_ship_cost(s), 1)
        )
        for ship in ships_by_ratio:
            if current_value <= budget:
                break
            cost = _ship_cost(ship)
            if cost == 0:
                continue
            overflow = current_value - budget
            remove_count = min(fleet[ship], overflow // cost + 1)
            fleet[ship] -= remove_count
            current_value -= remove_count * cost
            if fleet[ship] <= 0:
                del fleet[ship]

    # --- Bulk top up if under budget ---
    if current_value < budget:
        # Find best damage-per-cost ship
        best_ship = None
        best_ratio = -1
        for ship in SHIPS_COST:
            cost = _ship_cost(ship)
            damage = SHIP_BASE_ATK.get(ship, 0)
            ratio = damage / max(cost, 1)
            if ratio > best_ratio and cost > 0:
                best_ratio = ratio
                best_ship = ship
        if best_ship and _ship_cost(best_ship) > 0:
            remaining = budget - current_value
            add_count = remaining // _ship_cost(best_ship)
            if add_count > 0:
                fleet[best_ship] = fleet.get(best_ship, 0) + add_count

    return fleet


def phase_a3_local_search(
    fleet: Dict[str, int],
    enemy_fleet: Dict[str, int],
    enemy_defenses: Dict[str, int],
    enemy_tech: tuple,
    attacker_tech: tuple,
    seed: int,
    time_budget_s: float = 1.0,
    budget: int = 0,
) -> Dict[str, int]:
    """Hill climbing: try swap/augment/trim moves, keep improvements."""
    start = time.time()
    best = dict(fleet)
    best_loss = _evaluate_single(best, enemy_fleet, enemy_defenses, enemy_tech, attacker_tech, seed)

    top_k = 5
    no_improvement_count = 0
    max_no_improvement = 50  # stop after 50 iterations without improvement

    while time.time() - start < time_budget_s and no_improvement_count < max_no_improvement:
        candidates = []

        # Try augmenting each ship type (only if within budget)
        for ship in SHIPS_COST:
            if budget > 0:
                new_val = _fleet_value(best) + _ship_cost(ship)
                if new_val > budget:
                    continue
            c = best.get(ship, 0) + 1
            new = dict(best)
            new[ship] = c
            candidates.append((ship, "augment", c, new))

        # Try trimming each ship type (if count > 0)
        for ship, count in list(best.items()):
            if count > 0:
                new = dict(best)
                new[ship] = count - 1
                if new[ship] == 0:
                    del new[ship]
                candidates.append((ship, "trim", count - 1, new))

        # Try swaps: +1 ship A, -1 ship B (if B has > 0)
        for ship_a in SHIPS_COST:
            cost_a = _ship_cost(ship_a)
            if budget > 0 and _fleet_value(best) + cost_a > budget + _ship_cost(next(iter(best), ship_a)):
                # Only allow swaps that don't increase value beyond budget
                pass
            if _fleet_value({**best, ship_a: best.get(ship_a, 0) + 1}) > _fleet_value(best) + cost_a:
                continue
            for ship_b in list(best.keys()):
                if best.get(ship_b, 0) > 0 and ship_a != ship_b:
                    new = dict(best)
                    new[ship_a] = new.get(ship_a, 0) + 1
                    new[ship_b] = new[ship_b] - 1
                    if new[ship_b] == 0:
                        del new[ship_b]
                    candidates.append((ship_a, f"swap_{ship_b}", 1, new))

        # Filter candidates by budget
        if budget > 0:
            candidates = [(s, op, c, f) for s, op, c, f in candidates if _fleet_value(f) <= budget]

        # Evaluate candidates (with time guard)
        scored = []
        deadline_breached = False
        for ship, op, c, new_fleet in candidates:
            if time.time() - start > time_budget_s:
                deadline_breached = True
                break
            loss = _evaluate_single(new_fleet, enemy_fleet, enemy_defenses, enemy_tech, attacker_tech, seed)
            scored.append((loss, ship, op, new_fleet))
        scored.sort(key=lambda x: x[0])

        # Apply best if it improves
        if scored and scored[0][0] < best_loss:
            best = scored[0][3]
            best_loss = scored[0][0]
            no_improvement_count = 0
        else:
            no_improvement_count += 1

    return best


def _evaluate_single(
    fleet: Dict[str, int],
    enemy_fleet: Dict[str, int],
    enemy_defenses: Dict[str, int],
    enemy_tech: tuple,
    attacker_tech: tuple,
    seed: int,
) -> float:
    """Evaluate a single fleet via one combat simulation. Returns mean attacker loss."""
    if _fleet_value(fleet) == 0:
        return float("inf")
    r = simulate_combat(
        attacker=fleet,
        defender=enemy_fleet,
        defender_defenses=enemy_defenses,
        attacker_tech=attacker_tech,
        defender_tech=enemy_tech,
        seed=seed,
    )
    # Compute attacker loss as initial value - survivor value
    initial_value = _fleet_value(fleet)
    survivor_value = _fleet_value(r.get("attacker_survivors", {}))
    return float(initial_value - survivor_value)


def greedy_optimize(
    enemy_fleet: Dict[str, int],
    enemy_defenses: Dict[str, int],
    enemy_tech: tuple = (0, 0, 0),
    attacker_tech: tuple = (0, 0, 0),
    budget: int = 100_000,
    mode: str = "attack",
    seed: int = 42,
    time_budget_s: float = 1.0,
) -> GreedyResult:
    """Run the full 3-phase greedy pipeline."""
    t0 = time.time()

    # Phase A1: counter-ratio init
    rough = phase_a1_counter_ratio_init(enemy_fleet, enemy_defenses, budget, mode)

    # Phase A2: budget fill
    fleet = phase_a2_budget_fill(rough, budget)

    # Phase A3: local search
    fleet = phase_a3_local_search(fleet, enemy_fleet, enemy_defenses, enemy_tech, attacker_tech, seed, time_budget_s, budget=budget)

    # Final evaluation
    final_loss = _evaluate_single(fleet, enemy_fleet, enemy_defenses, enemy_tech, attacker_tech, seed)
    t1 = time.time()

    # Clean up: remove zero-count ships
    fleet = {s: c for s, c in fleet.items() if c > 0}

    return GreedyResult(
        fleet=fleet,
        estimated_loss=final_loss,
        time_elapsed=t1 - t0,
    )

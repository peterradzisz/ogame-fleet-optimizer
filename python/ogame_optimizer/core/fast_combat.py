"""Fast analytical combat resolver for large fleets.

Instead of simulating each individual shot O(units), computes expected
damage per ship-type-pair O(types^2) with Gaussian noise for variance.
Mathematically equivalent for large fleets (law of large numbers).

Used automatically by combat.py when total fleet size exceeds FAST_THRESHOLD.
"""
from __future__ import annotations

import math
import random
from typing import Dict, Optional, Tuple

# Ship stats (must match src/ships.rs)
SHIP_STATS: Dict[str, dict] = {
    "light_fighter":    {"atk": 50,     "shield": 10,    "hull": 400},
    "heavy_fighter":    {"atk": 150,    "shield": 25,    "hull": 1000},
    "cruiser":          {"atk": 400,    "shield": 50,    "hull": 2700},
    "battleship":       {"atk": 1000,   "shield": 200,   "hull": 6000},
    "battlecruiser":    {"atk": 700,    "shield": 400,   "hull": 7000},
    "bomber":           {"atk": 1000,   "shield": 500,   "hull": 7500},
    "destroyer":        {"atk": 2000,   "shield": 500,   "hull": 11000},
    "deathstar":        {"atk": 200000, "shield": 50000, "hull": 900000},
    "small_cargo":      {"atk": 5,      "shield": 10,    "hull": 400},
    "large_cargo":      {"atk": 5,      "shield": 25,    "hull": 1200},
    "espionage_probe":  {"atk": 1,      "shield": 0,     "hull": 100},
    "pathfinder":       {"atk": 200,    "shield": 100,   "hull": 2300},   # structure 23,000/10 (FIXED per Fandom)
    "recycler":         {"atk": 1,      "shield": 10,    "hull": 1600},   # structure 16,000/10
    "reaper":           {"atk": 2800,   "shield": 700,   "hull": 14000},  # structure 140,000/10 (FIXED per Fandom; user-verified W14/A16/S13 scaling)
    "solar_satellite":  {"atk": 1,      "shield": 1,     "hull": 200},    # structure 2,000/10
    "crawler":          {"atk": 1,      "shield": 1,     "hull": 400},    # structure 4,000/10
}

DEFENSE_STATS: Dict[str, dict] = {
    "rocket_launcher":   {"atk": 80,   "shield": 20,    "hull": 200},  # structure 2,000/10
    "light_laser":       {"atk": 100,  "shield": 25,    "hull": 200},  # structure 2,000/10
    "heavy_laser":       {"atk": 250,  "shield": 100,   "hull": 800},  # structure 8,000/10
    "gauss_cannon":      {"atk": 1100, "shield": 200,   "hull": 3500},  # structure 35,000/10
    "ion_cannon":        {"atk": 150,  "shield": 500,   "hull": 800},
    "plasma_turret":     {"atk": 3000, "shield": 300,   "hull": 10000},
    "small_shield_dome": {"atk": 0,    "shield": 2000,  "hull": 2000},
    "large_shield_dome": {"atk": 0,    "shield": 10000, "hull": 10000},  # structure 100,000/10
}

# Rapidfire table: (shooter, target) -> rf_value
# Each entry means shooter has rapidfire N against target.
# Expected shots multiplier when ALL targets are that type = N+1.
# Damage is distributed proportionally across target types.
# Rapidfire table: (shooter, target) -> rf_value
# Verified against OGame Fandom wiki for modern OGame (post-v0.84).
# Key fixes from prior version:
#   - Reaper: was anti-fighter (LF=3, HF=2), actually anti-capital (BS=7, Bo=4, De=3)
#   - Deathstar vs Battlecruiser: was 250, actually 15 per Fandom
#   - Pathfinder was missing entirely from shooter side
#   - Solar Satellite and Crawler were not modeled (needed for accurate RF chains)
RAPIDFIRE: Dict[tuple, int] = {
    # Light Fighter: vs EP=5, SS=5, Crawler=5
    ("light_fighter", "espionage_probe"): 5,
    ("light_fighter", "solar_satellite"): 5,
    ("light_fighter", "crawler"): 5,
    # Heavy Fighter: vs SC=3, EP=5, SS=5, Crawler=5
    ("heavy_fighter", "small_cargo"): 3,
    ("heavy_fighter", "espionage_probe"): 5,
    ("heavy_fighter", "solar_satellite"): 5,
    ("heavy_fighter", "crawler"): 5,
    # Cruiser: vs LF=6, EP=5, SS=5, Crawler=5, RL=10
    ("cruiser", "light_fighter"): 6,
    ("cruiser", "espionage_probe"): 5,
    ("cruiser", "solar_satellite"): 5,
    ("cruiser", "crawler"): 5,
    ("cruiser", "rocket_launcher"): 10,
    # Battleship: vs EP=5, SS=5, Crawler=5, PF=5
    ("battleship", "espionage_probe"): 5,
    ("battleship", "solar_satellite"): 5,
    ("battleship", "crawler"): 5,
    ("battleship", "pathfinder"): 5,
    # Battlecruiser: vs EP=5, SS=5, Crawler=5, SC=3, LC=3, HF=4, CR=4, BS=7
    ("battlecruiser", "espionage_probe"): 5,
    ("battlecruiser", "solar_satellite"): 5,
    ("battlecruiser", "crawler"): 5,
    ("battlecruiser", "small_cargo"): 3,
    ("battlecruiser", "large_cargo"): 3,
    ("battlecruiser", "heavy_fighter"): 4,
    ("battlecruiser", "cruiser"): 4,
    ("battlecruiser", "battleship"): 7,
    # Bomber: vs EP=5, SS=5, Crawler=5, defenses (RL=20, LL=20, HL=10, IC=10, GC=5, PT=5)
    ("bomber", "espionage_probe"): 5,
    ("bomber", "solar_satellite"): 5,
    ("bomber", "crawler"): 5,
    ("bomber", "rocket_launcher"): 20,
    ("bomber", "light_laser"): 20,
    ("bomber", "heavy_laser"): 10,
    ("bomber", "ion_cannon"): 10,
    ("bomber", "gauss_cannon"): 5,
    ("bomber", "plasma_turret"): 5,
    # Destroyer: vs EP=5, SS=5, Crawler=5, BC=2, LL=10
    ("destroyer", "espionage_probe"): 5,
    ("destroyer", "solar_satellite"): 5,
    ("destroyer", "crawler"): 5,
    ("destroyer", "battlecruiser"): 2,
    ("destroyer", "light_laser"): 10,
    # Reaper (post-v0.84, modern OGame) - FIXED: was anti-fighter, actually anti-capital
    ("reaper", "espionage_probe"): 5,
    ("reaper", "solar_satellite"): 5,
    ("reaper", "crawler"): 5,
    ("reaper", "battleship"): 7,
    ("reaper", "bomber"): 4,
    ("reaper", "destroyer"): 3,
    # Pathfinder (post-v0.84, modern OGame) - NEW
    ("pathfinder", "espionage_probe"): 5,
    ("pathfinder", "solar_satellite"): 5,
    ("pathfinder", "crawler"): 5,
    ("pathfinder", "light_fighter"): 3,
    ("pathfinder", "heavy_fighter"): 2,
    ("pathfinder", "cruiser"): 3,
    # Deathstar: vs EP=1250, SS=1250, Crawler=1250, LF=200, HF=100,
    # CR=33, BS=30, BC=15 (FIXED: was 250), Bo=25, De=5, SC=250, LC=250,
    # Pathfinder=30, Reaper=30, defenses as listed
    ("deathstar", "espionage_probe"): 1250,
    ("deathstar", "solar_satellite"): 1250,
    ("deathstar", "crawler"): 1250,
    ("deathstar", "light_fighter"): 200,
    ("deathstar", "heavy_fighter"): 100,
    ("deathstar", "cruiser"): 33,
    ("deathstar", "battleship"): 30,
    ("deathstar", "battlecruiser"): 15,  # FIXED: was 250
    ("deathstar", "pathfinder"): 30,
    ("deathstar", "reaper"): 30,
    ("deathstar", "bomber"): 25,
    ("deathstar", "destroyer"): 5,
    ("deathstar", "small_cargo"): 250,
    ("deathstar", "large_cargo"): 250,
    ("deathstar", "rocket_launcher"): 200,
    ("deathstar", "light_laser"): 200,
    ("deathstar", "heavy_laser"): 100,
    ("deathstar", "ion_cannon"): 100,
    ("deathstar", "gauss_cannon"): 50,
}




# Ship costs: (metal, crystal, deuterium) for debris calculation
SHIP_COSTS_MCD = {
    "light_fighter": (3000, 1000, 0),
    "heavy_fighter": (6000, 4000, 0),
    "cruiser": (20000, 7000, 2000),
    "battleship": (45000, 15000, 0),
    "battlecruiser": (30000, 40000, 15000),
    "bomber": (50000, 25000, 15000),
    "destroyer": (60000, 50000, 15000),
    "deathstar": (5000000, 4000000, 1000000),
    "small_cargo": (2000, 2000, 0),
    "large_cargo": (6000, 6000, 0),
    "espionage_probe": (0, 1000, 0),
    "pathfinder": (8000, 15000, 8000),  # FIXED per Fandom
    "recycler": (10000, 6000, 2000),
    "reaper": (85000, 55000, 20000),  # FIXED per Fandom: added 20K deuterium
    "solar_satellite": (0, 2000, 500),
    "crawler": (2000, 2000, 1000),
}

DEFENSE_COSTS_MCD = {
    "rocket_launcher": (2000, 0, 0),
    "light_laser": (1500, 500, 0),
    "heavy_laser": (6000, 2000, 0),
    "gauss_cannon": (20000, 15000, 2000),
    "ion_cannon": (5000, 3000, 0),
    "plasma_turret": (50000, 50000, 30000),
    "small_shield_dome": (10000, 10000, 0),
    "large_shield_dome": (50000, 50000, 0),
}

# Default debris percentages (standard OGame = 30%)
DEFAULT_DEBRIS_PCT = 0.30



def calculate_debris(
    attacker_initial: dict,
    attacker_survivors: dict,
    defender_initial: dict,
    defender_survivors: dict,
    defender_def_initial: dict = None,
    defender_def_survivors: dict = None,
    debris_pct: float = DEFAULT_DEBRIS_PCT,
    deuterium_in_debris: bool = False,
) -> dict:
    """Calculate debris field from destroyed ships and defenses."""
    def _lost_mcd(initial, survivors, cost_table):
        mcd_lost = [0, 0, 0]
        for ship, init_count in initial.items():
            if init_count <= 0 or ship not in cost_table:
                continue
            surv_count = survivors.get(ship, 0)
            destroyed = max(0, init_count - surv_count)
            if destroyed > 0:
                costs = cost_table[ship]
                mcd_lost[0] += costs[0] * destroyed
                mcd_lost[1] += costs[1] * destroyed
                mcd_lost[2] += costs[2] * destroyed
        return mcd_lost

    atk_lost = _lost_mcd(attacker_initial, attacker_survivors, SHIP_COSTS_MCD)
    def_lost = _lost_mcd(defender_initial, defender_survivors, SHIP_COSTS_MCD)
    if defender_def_initial:
        def_lost_def = _lost_mcd(defender_def_initial, defender_def_survivors or {}, DEFENSE_COSTS_MCD)
        def_lost = [def_lost[i] + def_lost_def[i] for i in range(3)]

    total_lost = [atk_lost[i] + def_lost[i] for i in range(3)]

    debris_metal = int(total_lost[0] * debris_pct)
    debris_crystal = int(total_lost[1] * debris_pct)
    debris_deuterium = int(total_lost[2] * debris_pct) if deuterium_in_debris else 0

    return {
        "debris_metal": debris_metal,
        "debris_crystal": debris_crystal,
        "debris_deuterium": debris_deuterium,
        "debris_total": debris_metal + debris_crystal + debris_deuterium,
    }

FAST_THRESHOLD = 500  # above this many total units, use analytical resolver


def _total_units(*fleets: Dict[str, int]) -> int:
    return sum(sum(v for v in f.values() if isinstance(v, int) and v > 0) for f in fleets)


def _make_side(fleet: Dict[str, int], defenses: Dict[str, int], tech: Tuple[int, int, int]):
    """Build combat state dict from fleet + defenses."""
    atk_mult = 1 + tech[0] * 0.1
    shield_mult = 1 + tech[1] * 0.1
    hull_mult = 1 + tech[2] * 0.1

    side = {}
    for k, v in fleet.items():
        if v > 0 and k in SHIP_STATS:
            s = SHIP_STATS[k]
            side[k] = {
                "count": v,
                "shields": s["shield"] * v * shield_mult,
                "hull": s["hull"] * v * hull_mult,  # FIXED: hull stat is already armor (structure/10)
                "base_shield": s["shield"] * shield_mult,
                "unit_hull": s["hull"] * hull_mult,  # FIXED: hull stat is already armor (structure/10)
                "atk": s["atk"] * atk_mult,
                "hits": 0.0,
            }
    for k, v in defenses.items():
        if v > 0 and k in DEFENSE_STATS:
            s = DEFENSE_STATS[k]
            side[k] = {
                "count": v,
                "shields": s["shield"] * v * shield_mult,
                "hull": s["hull"] * v * hull_mult,  # FIXED: hull stat is already armor (structure/10)
                "base_shield": s["shield"] * shield_mult,
                "unit_hull": s["hull"] * hull_mult,  # FIXED: hull stat is already armor (structure/10)
                "atk": s["atk"] * atk_mult,
                "hits": 0.0,
            }
    return side


def _fire(attacker_side: dict, defender_side: dict, rng: random.Random):
    """Grouped fire with per-shot overkill handling.

    All attacker types' fire against each defender type is resolved in one
    pass (eliminating sequential fire-ordering bias), but damage is split
    into two regimes so high-damage single-target weapons behave correctly:

    * SPIKE - shots that deal >= a defender unit's full HP each kill exactly
      one unit (overkill discarded). Previously the pooled-damage model
      recycled this overkill, making Deathstars etc. absurdly strong vs
      swarms.
    * CHIP  - sub-lethal shots pool into the defender stack's shield then
      hull (law of large numbers).

    Rapidfire model: if ship A has RF=N against type B present at fraction
    f, A's continuation probability is f*(N-1)/N, so its expected shot
    multiplier is 1 / (1 - sum_f (N-1)/N) = N for a pure target. This
    matches the Rust core and ogamespec.
    """
    total_def_count = sum(u["count"] for u in defender_side.values())
    if total_def_count == 0:
        return

    # Pre-compute fractions (target distribution proportions)
    fractions = {}
    for k, d in defender_side.items():
        fractions[k] = d["count"] / total_def_count if d["count"] > 0 else 0.0

    # Global rapidfire model (correct OGame RF mechanics).
    #
    # In OGame, each ship fires at a RANDOM target. If the target is an RF
    # type, there is (RF-1)/RF probability to fire again at another RANDOM
    # target. The chain continues until a non-RF target is hit or the
    # continuation roll fails.
    #
    # The expected total shots per ship is:
    #   shot_mult = 1 / (1 - sum(frac_i * (RF_i - 1) / RF_i))
    # where the sum is over ALL RF target types present in the defender fleet.
    # This single multiplier applies to ALL defender types proportionally.
    #
    # Verified via Monte Carlo simulation (100K ships, < 1% error).
    # The previous per-pair model gave each pair its own full RF multiplier,
    # overestimating total damage by ~1.4x and distorting the damage
    # distribution (too much damage to RF targets, too little to non-RF).
    atk_info = {}  # atk_info[atk_type][def_type] = (per_shot_dmg, effective_shots)
    for k_atk, atk in attacker_side.items():
        if atk["count"] == 0 or atk["atk"] <= 0:
            continue
        atk_info[k_atk] = {}
        per_shot = atk["atk"]
        # Compute global shot multiplier from ALL RF targets in defender fleet
        cont_prob = 0.0
        for k_def, frac in fractions.items():
            rf = RAPIDFIRE.get((k_atk, k_def), 0)
            if rf > 1:
                cont_prob += frac * (rf - 1) / rf
        shot_multiplier = 1.0 / (1.0 - cont_prob) if cont_prob < 0.95 else 20.0
        for k_def, frac in fractions.items():
            if frac <= 0:
                continue
            effective_shots = atk["count"] * shot_multiplier * frac
            atk_info[k_atk][k_def] = (per_shot, effective_shots)

    # For each defender type: resolve incoming fire with per-shot overkill
    # handling.
    #
    # Two regimes are modelled per (attacker_type, defender_type) pair:
    #
    # * SPIKE - a single shot deals >= one defender unit's full effective HP
    #   (shield + hull). In real OGame each such shot kills exactly ONE unit
    #   and the excess damage is wasted. The previous implementation pooled
    #   ALL damage and divided by per-unit HP, which recycled that excess and
    #   made high-damage weapons (notably the 200k-attack Deathstar) vastly
    #   too lethal vs swarms of cheap ships (e.g. 223 RIPs "wiping" 123k BCs).
    #
    # * CHIP - a single shot is sub-lethal. Damage accumulates in a pool
    #   (shield first, then hull); this is the law-of-large-numbers regime
    #   where the analytical approximation is valid.
    #
    # Spike kills are resolved first (each consumes one whole unit, bypassing
    # the shared shield pool), then chip damage wears down the survivors.
    for k_def, d in defender_side.items():
        if d["count"] == 0 or fractions.get(k_def, 0.0) == 0.0:
            continue
        frac = fractions[k_def]
        count = d["count"]
        unit_shield = d["base_shield"]   # full per-unit shield (regen'd each round)
        unit_hull = d["unit_hull"]       # full per-unit hull
        # unit_eff_hp uses full hull for spike/chip classification.
        # Hull damage accumulation is handled separately via hull_pool below.
        unit_eff_hp = unit_shield + unit_hull

        spike_kills = 0.0   # shots that each obliterate one whole unit
        chip_dmg = 0.0      # accumulated sub-lethal damage
        chip_shots = 0.0    # total chip shots (for per-hit damage estimate)

        for k_atk, pair_info in atk_info.items():
            if k_def not in pair_info:
                continue
            per_shot, effective_shots = pair_info[k_def]
            # OGame shield bounce: a shot below 1% of the unit's max shield
            # is completely absorbed and wasted.
            if per_shot < unit_shield * 0.01:
                continue
            if effective_shots < 0.5:
                continue
            if per_shot >= unit_eff_hp:
                # SPIKE: one shot kills one unit; overkill is discarded.
                spike_kills += effective_shots
            else:
                # CHIP: pools into shield/hull of the surviving stack.
                chip_dmg += per_shot * effective_shots
                chip_shots += effective_shots

        # Resolve spike kills. Each consumes one whole unit (its shield AND
        # its hull), so this damage never enters the shared shield pool.
        kills = min(spike_kills, count)
        survivors = count - kills
        if survivors <= 0:
            d["count"] = 0
            d["hull"] = 0
            d["shields"] = 0
            d["hits"] = 0.0
            continue

        # Apply chip damage to the survivors' pooled shield, then hull.
        if chip_dmg > 0:
            sigma = math.sqrt(max(chip_dmg * (1.0 - frac), 1.0))
            chip_dmg = max(0.0, chip_dmg + rng.gauss(0.0, sigma))
            shield_pool = unit_shield * survivors
            absorbed = min(chip_dmg, shield_pool)
            hull_dmg = chip_dmg - absorbed
            # Hull hits = how many shots got through shields to damage hull
            if chip_shots > 0 and hull_dmg > 0:
                avg_dmg_per_shot = chip_dmg / chip_shots
                hull_hits = hull_dmg / avg_dmg_per_shot if avg_dmg_per_shot > 0 else 0.0
            else:
                hull_hits = 0.0
        else:
            hull_dmg = 0.0
            hull_hits = 0.0

        # Start from accumulated hull (d["hull"]), adjusted for spike kills
        # (spike kills remove whole units, reducing hull proportionally).
        hull_pool = (d["hull"] / count) * survivors - hull_dmg
        if hull_pool <= 0:
            d["count"] = 0
            d["hull"] = 0
            d["shields"] = 0
            d["hits"] = 0.0
            continue

        max_hull = unit_hull * survivors
        hull_ratio = min(1.0, hull_pool / max_hull) if max_hull > 0 else 0.0

        # === Full Poisson damage distribution model ===
        # In OGame, each shot targets a RANDOM individual ship. Damage follows
        # a Poisson distribution: P(k hits) = e^-lam * lam^k / k!
        # Ships with few hits survive at high hull; ships with many hits die.
        # The pooled model averages this, treating ALL ships as having lam hits.
        # We instead compute survival by summing over each possible hit count.
        #
        # Key parameters:
        #   lam = accumulated hull-penetrating hits per ship
        #   dmg_per_hit = average hull damage per hit = (1-hull_ratio)/lam
        #   For k hits: remaining = 1 - k * dmg_per_hit
        #     remaining >= 0.7 -> survive (1.0)
        #     0 < remaining < 0.7 -> survive with explosion probability
        #     remaining <= 0 -> destroyed

        prev_hits = d.get("hits", 0.0) * (survivors / count) if count > 0 else 0.0
        accumulated_shots = prev_hits + hull_hits
        lam = accumulated_shots / survivors if survivors > 0 else 0.0

        if lam < 0.01 or hull_ratio >= 1.0:
            # No meaningful damage - all survive
            survival_frac = 1.0
            hull_factor = hull_ratio
        else:
            # Average hull damage per hit (as fraction of unit_hull)
            dmg_per_hit = (1.0 - hull_ratio) / lam if lam > 0 else 0.0
            # Cap iterations at point where remaining hull goes negative
            max_k = min(int(1.0 / dmg_per_hit) + 2, 60) if dmg_per_hit > 0 else 1

            survival_frac = 0.0
            hull_factor = 0.0
            # Poisson P(k) computed iteratively
            p_k = math.exp(-min(lam, 20.0))  # P(0)
            for k in range(max_k + 1):
                remaining = 1.0 - k * dmg_per_hit
                if remaining >= 0.7:
                    survive = 1.0
                    surv_hull = remaining
                elif remaining > 0:
                    p_explode = ((0.7 - remaining) / 0.7) * (1.0 - remaining)
                    survive = remaining * (1.0 - p_explode)
                    surv_hull = remaining
                else:
                    survive = 0.0
                    surv_hull = 0.0
                survival_frac += p_k * survive
                hull_factor += p_k * survive * surv_hull
                # Next Poisson term: P(k+1) = P(k) * lam / (k+1)
                p_k *= lam / (k + 1)

            # Normalize hull_factor to average hull of SURVIVORS
            if survival_frac > 0:
                hull_factor = hull_factor / survival_frac
            else:
                hull_factor = 0.0

        survival_frac = min(1.0, survival_frac)
        new_count = int(survivors * survival_frac)
        d["count"] = new_count
        # Hull accumulation
        if new_count > 0:
            d["hull"] = unit_hull * new_count * hull_factor
        else:
            d["hull"] = 0
        d["shields"] = unit_shield * new_count
        # Update accumulated hits (survivorship bias: survivors had fewer hits)
        if new_count > 0 and survivors > 0:
            d["hits"] = accumulated_shots * (new_count / survivors)
        else:
            d["hits"] = 0.0


def _regen_shields(side: dict):
    """Regenerate shields to full (OGame rule: shields regen each round)."""
    for u in side.values():
        u["shields"] = u["base_shield"] * u["count"]


def simulate_combat_fast(
    attacker: Dict[str, int],
    defender: Dict[str, int],
    defender_defenses: Optional[Dict[str, int]] = None,
    attacker_tech: Tuple[int, int, int] = (0, 0, 0),
    defender_tech: Tuple[int, int, int] = (0, 0, 0),
    seed: int = 42,
) -> dict:
    """Analytical combat simulation. Same return format as Rust simulate_combat."""
    defender_defenses = defender_defenses or {}
    rng = random.Random(seed)

    atk_side = _make_side(attacker, {}, attacker_tech)
    def_side = _make_side(defender, defender_defenses, defender_tech)

    rounds_fought = 0
    stalemate = False
    for rnd in range(6):
        rounds_fought = rnd + 1
        if not any(u["count"] > 0 for u in atk_side.values()):
            break
        if not any(u["count"] > 0 for u in def_side.values()):
            break

        # Snapshot counts for draw detection
        atk_before = sum(u["count"] for u in atk_side.values())
        def_before = sum(u["count"] for u in def_side.values())

        # SIMULTANEOUS fire: both sides use start-of-round counts
        # Snapshot defender state before attacker fires
        def_start = {k: dict(u) for k, u in def_side.items()}
        # Attacker fires → damages defender
        _fire(atk_side, def_side, rng)
        # Save defender's post-attack damaged state
        def_damaged = {k: dict(u) for k, u in def_side.items()}
        # Restore defender's start-of-round counts for their counter-fire
        for k, u in def_side.items():
            u["count"] = def_start.get(k, {}).get("count", 0)
            u["shields"] = def_start.get(k, {}).get("shields", 0)
            u["hull"] = def_start.get(k, {}).get("hull", 0)
            u["hits"] = def_start.get(k, {}).get("hits", 0.0)
        # Defender fires at full strength → damages attacker
        _fire(def_side, atk_side, rng)
        # Restore defender to the damaged state (from attacker's fire)
        for k, u in def_side.items():
            u["count"] = def_damaged.get(k, {}).get("count", 0)
            u["shields"] = def_damaged.get(k, {}).get("shields", 0)
            u["hull"] = def_damaged.get(k, {}).get("hull", 0)
            u["hits"] = def_damaged.get(k, {}).get("hits", 0.0)

        # Check for stalemate (no damage either side)
        atk_after = sum(u["count"] for u in atk_side.values())
        def_after = sum(u["count"] for u in def_side.values())
        if atk_after == atk_before and def_after == def_before:
            stalemate = True  # neither side can damage the other, but keep fighting all 6 rounds

        _regen_shields(atk_side)
        _regen_shields(def_side)

    atk_surv = {k: u["count"] for k, u in atk_side.items() if u["count"] > 0}
    def_ship_surv = {k: u["count"] for k, u in def_side.items() if u["count"] > 0 and k in SHIP_STATS}
    def_def_surv = {k: u["count"] for k, u in def_side.items() if u["count"] > 0 and k in DEFENSE_STATS}

    atk_total = sum(atk_surv.values())
    def_total = sum(def_ship_surv.values()) + sum(def_def_surv.values())

    if atk_total > 0 and def_total == 0:
        winner = "Attacker"
    elif def_total > 0 and atk_total == 0:
        winner = "Defender"
    elif atk_total == 0 and def_total == 0:
        winner = "Draw"
    elif stalemate:
        winner = "Draw"
    else:
        # OGame rule: if both sides survive 6 rounds, it is a DRAW regardless
        # of fleet sizes (matches the Rust per-unit core). The previous
        # 'more ships wins' fallthrough made huge fodder fleets 'beat'
        # unbeatable enemies just by having more survivors (e.g. 2M probes
        # 'winning' vs 50k BCs despite bouncing off their shields).
        winner = "Draw"

    return {
        "winner": winner,
        "rounds_fought": rounds_fought,
        "attacker_survivors": atk_surv,
        "defender_survivors": def_ship_surv,
        "defender_defense_survivors": def_def_surv,
        "debris_metal": 0,  # Updated below with actual values
        "debris_crystal": 0,
    }


def simulate_batch_fast(
    attacker: Dict[str, int],
    defender: Dict[str, int],
    defender_defenses: Optional[Dict[str, int]] = None,
    attacker_tech: Tuple[int, int, int] = (0, 0, 0),
    defender_tech: Tuple[int, int, int] = (0, 0, 0),
    n_sims: int = 100,
    base_seed: int = 42,
    debris_pct: float = DEFAULT_DEBRIS_PCT,
    deuterium_in_debris: bool = False,
) -> dict:
    """Run N analytical sims and return aggregate stats (same format as Rust batch)."""
    from ogame_optimizer.core.fleet import SHIPS_COST

    atk_value = sum(sum(SHIPS_COST.get(k, (0, 0, 0))) * v for k, v in attacker.items())
    def_value = sum(sum(SHIPS_COST.get(k, (0, 0, 0))) * v for k, v in defender.items())
    losses = []
    def_losses = []
    debris_metal_sum = debris_crystal_sum = debris_deut_sum = 0
    wins = losses_count = draws = 0
    # Per-type MEAN survivors (fractional) — exposed in result so callers
    # can derive survival_pct per ship type. Previously missing from the
    # fast path, which made the "Surviving (after 6 rounds)" column show
    # 0 for every ship on large fleets that use the fast resolver.
    from collections import defaultdict
    atk_surv_sum: dict = defaultdict(float)
    def_surv_sum: dict = defaultdict(float)
    def_def_surv_sum: dict = defaultdict(float)

    for i in range(n_sims):
        r = simulate_combat_fast(
            attacker, defender, defender_defenses,
            attacker_tech, defender_tech,
            seed=base_seed + i,
        )
        surv_value = sum(
            sum(SHIPS_COST.get(k, (0, 0, 0))) * v
            for k, v in r["attacker_survivors"].items()
        )
        loss = atk_value - surv_value
        losses.append(loss)

        def_surv_value = sum(
            sum(SHIPS_COST.get(k, (0, 0, 0))) * v
            for k, v in r.get("defender_survivors", {}).items()
        )
        def_losses.append(def_value - def_surv_value)

        # Compute debris per-sim for accurate averaging
        db = calculate_debris(
            attacker, r.get("attacker_survivors", {}),
            defender, r.get("defender_survivors", {}),
            defender_defenses, r.get("defender_defense_survivors", {}),
            debris_pct, deuterium_in_debris,
        )
        debris_metal_sum += db["debris_metal"]
        debris_crystal_sum += db["debris_crystal"]
        debris_deut_sum += db["debris_deuterium"]

        # Accumulate per-type survivors for averaging
        for _s, _n in (r.get("attacker_survivors") or {}).items():
            atk_surv_sum[_s] += _n
        for _s, _n in (r.get("defender_survivors") or {}).items():
            def_surv_sum[_s] += _n
        for _s, _n in (r.get("defender_defense_survivors") or {}).items():
            def_def_surv_sum[_s] += _n

        if r["winner"] == "Attacker":
            wins += 1
        elif r["winner"] == "Defender":
            losses_count += 1
        else:
            draws += 1

    mean_loss = sum(losses) / n_sims if n_sims > 0 else 0
    variance = sum((l - mean_loss) ** 2 for l in losses) / n_sims if n_sims > 0 else 0
    stddev = math.sqrt(variance)
    mean_def_loss = sum(def_losses) / n_sims if n_sims > 0 else 0

    return {
        "mean_attacker_loss": mean_loss,
        "stddev_attacker_loss": stddev,
        "mean_defender_loss": mean_def_loss,
        "win_probability": wins / n_sims if n_sims > 0 else 0,
        "wins": wins,
        "losses": losses_count,
        "draws": draws,
        "sims_run": n_sims,
        "seed_used": base_seed,
        "debris_metal": int(debris_metal_sum / n_sims) if n_sims > 0 else 0,
        "debris_crystal": int(debris_crystal_sum / n_sims) if n_sims > 0 else 0,
        "debris_deuterium": int(debris_deut_sum / n_sims) if n_sims > 0 else 0,
        "debris_total": int((debris_metal_sum + debris_crystal_sum + debris_deut_sum) / n_sims) if n_sims > 0 else 0,
        "attacker_survivors_mean": {s: n / n_sims for s, n in atk_surv_sum.items()},
        "defender_survivors_mean": {s: n / n_sims for s, n in def_surv_sum.items()},
        "defender_defense_survivors_mean": {s: n / n_sims for s, n in def_def_surv_sum.items()},
    }


def evaluate_population_fast(
    attacker_fleets: list,
    defender: Dict[str, int],
    defender_defenses: Optional[Dict[str, int]] = None,
    attacker_tech: Tuple[int, int, int] = (0, 0, 0),
    defender_tech: Tuple[int, int, int] = (0, 0, 0),
    n_sims_per_fleet: int = 100,
    base_seed: int = 42,
) -> list:
    """Evaluate multiple attacker fleets vs same defender (for GA)."""
    results = []
    for idx, fleet in enumerate(attacker_fleets):
        r = simulate_batch_fast(
            fleet, defender, defender_defenses,
            attacker_tech, defender_tech,
            n_sims_per_fleet,
            base_seed + idx * 7919,
        )
        results.append({
            "mean_attacker_loss": r["mean_attacker_loss"],
            "mean_defender_loss": r["mean_defender_loss"],
            "stddev_attacker_loss": r["stddev_attacker_loss"],
            "win_probability": r["win_probability"],
            "sims_run": n_sims_per_fleet,
        })
    return results


def should_use_fast(attacker: Dict[str, int], defender: Dict[str, int], defenses: Optional[Dict[str, int]] = None) -> bool:
    """Check if total fleet size is large enough to warrant analytical resolver."""
    total = _total_units(attacker, defender, defenses or {})
    return total > FAST_THRESHOLD


__all__ = [
    "simulate_combat_fast", "simulate_batch_fast", "evaluate_population_fast",
    "should_use_fast", "FAST_THRESHOLD",
]

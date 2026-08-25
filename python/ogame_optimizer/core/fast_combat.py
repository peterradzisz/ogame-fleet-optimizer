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



def _poisson_ge(lam: float, m: int) -> float:
    """P(k >= m) for k ~ Poisson(lam), integer m >= 0.

    Exact for integer m via the finite series
        P(k >= m) = 1 - e^-lam * sum_{j=0}^{m-1} lam^j / j!
    Pure math module - no new dependencies. lam is clamped at 500 where
    e^-lam underflows (the tail is 1.0 to double precision there).
    """
    if m <= 0:
        return 1.0
    if lam <= 0.0:
        return 0.0
    if lam > 500.0:
        return 1.0
    term = math.exp(-lam)
    cdf = term
    for j in range(1, m):
        term *= lam / j
        cdf += term
    return max(0.0, 1.0 - min(1.0, cdf))


SUBSTEPS = 1  # single-step per round: sub-stepped mean-field shield depletion
# destroys the per-unit multi-hit tail (calibrated + verified vs Rust)

# --- Compound-Poisson damage-tail closure dials ---------------------------
# Calibrated against the official ogame.org big-battle report
# (python_tests/test_official_battles.py): CR 81.4 / BS 94.9 / BC 96.4 /
# PF 79.9 percent attacker keep. HEAVY_SHOT_TAU decides which chip streams
# are convolved as explicit "heavy" lineages in the exact (lam <= 30) path;
# OVERLAY_BETA damps their cross-round stratification; HAZARD_MID positions
# the lam>30 explosion hazard inside the round's damage interval; V_CAP caps
# the carried damage-fraction variance (a [0,1] quantity can never exceed
# 0.25).
HEAVY_SHOT_TAU = 0.25
OVERLAY_BETA = 1.0
HAZARD_MID = 0.5
V_CAP = 0.25

_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


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
    # Sub-stepped fire resolution (Rust sequential-fire parity). Real
    # OGame fire is sequential: defenders dying mid-round concentrate the
    # fixed start-of-round shot volume onto the remaining units, and RF
    # chains decay as RF targets die out. Resolving each round in SUBSTEPS
    # lockstep chunks - re-aiming per chunk against CURRENT counts - 
    # captures this feedback. A flat single-step spread made cheap
    # shield-less fodder (espionage probes) look far tankier than the
    # Rust per-unit core says it is (EP fodder "helped" analytically
    # while Rust showed +15% losses from it).
    shooters = {k: u for k, u in attacker_side.items()
                if u["count"] > 0 and u["atk"] > 0}
    if not shooters:
        return
    noise_sigma = 0.15 / math.sqrt(SUBSTEPS)
    for _step in range(SUBSTEPS):
        total_def_count = sum(u["count"] for u in defender_side.values())
        if total_def_count <= 0:
            return
        fractions = {
            k: (u["count"] / total_def_count if u["count"] > 0 else 0.0)
            for k, u in defender_side.items()
        }
        # Global RF multiplier from the CURRENT target distribution.
        sub_shots = {}
        for k_atk, atk in shooters.items():
            cont_prob = 0.0
            for k_def, frac in fractions.items():
                rf = RAPIDFIRE.get((k_atk, k_def), 0)
                if rf > 1 and frac > 0:
                    cont_prob += frac * (rf - 1) / rf
            mult = 1.0 / (1.0 - cont_prob) if cont_prob < 0.95 else 20.0
            sub_shots[k_atk] = atk["count"] * mult / SUBSTEPS

        for k_def, d in defender_side.items():
            if d["count"] <= 0 or fractions.get(k_def, 0.0) <= 0.0:
                continue
            frac = fractions[k_def]
            count = d["count"]
            unit_shield = d["base_shield"]
            unit_hull = d["unit_hull"]
            unit_eff_hp = unit_shield + unit_hull
            c_rem = d.get("shield_rem", unit_shield)

            spike_kills = 0.0
            chip_streams = []
            for k_atk, atk in shooters.items():
                per_shot = atk["atk"]
                aimed = sub_shots[k_atk] * frac
                # OGame shield bounce: a shot below 1% of the unit's max
                # shield is completely absorbed and wasted.
                if per_shot < unit_shield * 0.01:
                    continue
                if aimed < 0.5:
                    continue
                if per_shot >= unit_eff_hp:
                    spike_kills += aimed
                else:
                    chip_streams.append((per_shot, aimed))

            kills = min(spike_kills, count)
            survivors = count - kills
            if survivors <= 0:
                d["count"] = 0
                d["hull"] = 0
                d["shields"] = 0
                d["hits"] = 0.0
                d["shield_rem"] = 0.0
                d["dmg_bins"] = []
                continue

            # === Per-unit Poisson path model (spec: src/combat.rs apply_damage) ===
            # Aggregate chip streams into one Poisson stream: lam shots of mean
            # size s_eff per survivor. Within the round each unit's shield
            # (fully regenerated) absorbs the first ceil(C/s_eff) shots' worth;
            # the spill (k*s - C)+ damages the hull on top of the unit's prior
            # damage. A unit dies when total damage reaches 100%. Every shot
            # arriving while the unit is below 70% hull rolls an explosion
            # check with chance equal to the CURRENT damage fraction (Rust
            # apply_damage semantics). Prior damage is carried as a small
            # HISTOGRAM (damage fraction -> unit weight) so cross-round
            # compounding variance is preserved: units hit hard in early rounds
            # are the ones that cross the 70% explosion threshold later - a
            # survivor-mean field would compress exactly this tail. Counts stay
            # FLOAT through the rounds; the old per-round int() truncation was
            # the source of the scale-flat "exactly 800k" Reaper loss.
            if survivors <= 0.5:
                d["count"] = 0
                d["hull"] = 0
                d["shields"] = 0
                d["hits"] = 0.0
                d["shield_rem"] = 0.0
                continue
            lam_tot = 0.0
            w_dmg = 0.0
            for per_shot, aimed in chip_streams:
                lam_tot += aimed
                w_dmg += aimed * per_shot
            if lam_tot <= 0.0:
                d["count"] = survivors
                d["shields"] = unit_shield * survivors
                continue
            s_eff = w_dmg / lam_tot
            lam = min(lam_tot / survivors, 500.0)
            # Small cross-sim variance (CRN-friendly): perturb the shot rate.
            perturb = max(0.25, 1.0 + rng.gauss(0.0, noise_sigma))
            lam *= perturb
            H = unit_hull
            if H <= 0:
                continue
            # Compound-stream split for the exact (lam <= 30) path: chip shots
            # big enough to reach the explosion zone on their own
            # (per_shot >= HEAVY_SHOT_TAU * H) are convolved as an explicit
            # heavy lineage instead of being smeared into the damage-weighted
            # mean shot s_eff. Averaging e.g. a 0.7*H reaper shot with
            # probe/cargo chips understates the per-unit spill tail by orders
            # of magnitude (P(k >= 6 | lam 0.7) ~ 1e-5 vs the true ~1%
            # single-hit mass) - the official big battle shows exactly this
            # signature (CR/PF bleed to mixed heavy fire while BS/BC stay
            # safe). Single-enemy-type battles (every duel parity fixture)
            # produce one chip stream: lam_h_tot = 0 and the exact path below
            # stays bit-identical to the previous single-stream resolver.
            lam_h_tot = 0.0
            w_h = 0.0
            for per_shot, aimed in chip_streams:
                if per_shot >= HEAVY_SHOT_TAU * H:
                    lam_h_tot += aimed
                    w_h += aimed * per_shot
            j_strip = (
                max(1, int(math.ceil(c_rem / s_eff - 1e-12)))
                if c_rem > 0 else 0
            )

            bins = d.get("dmg_bins")
            if not bins:
                bins = [(0.0, 1.0)]

            if lam > 30.0:
                # === 2-moment compound-Poisson tail closure ==================
                # Replaces the old modal-lineage mean collapse (which destroyed
                # all cross-sectional variance). With S = (k*s - C)+ for
                # k ~ Poisson(lam) and m = ceil(C/s), the Poisson identities
                #   E[k 1{k>=m}]   = lam * P(k >= m-1)
                #   E[k^2 1{k>=m}] = lam^2*P(k >= m-2) + lam*P(k >= m-1)
                # give the closed-form spill moments
                #   E[S]  = s*lam*P(k>=m-1) - C*P(k>=m)
                #   E[S2] = s^2*(lam^2*P(k>=m-2) + lam*P(k>=m-1))
                #           - 2*s*C*lam*P(k>=m-1) + C^2*P(k>=m)
                # Track per-type damage-fraction mean x_bar and variance v;
                # hull deaths = Normal upper tail P(x >= 1); explosion hazard
                # = exp(-n_s * x_eff) with n_s the expected spilling-shot
                # count and x_eff the mid-interval damage fraction (Rust
                # apply_damage semantics: each shot landing while hull < 70%
                # rolls explode with chance = the then-current damage
                # fraction, so no hazard below x = 0.3).
                wsum = max(sum(w for _, w in bins), 1e-9)
                x_bar = min(0.99, sum(x * w for x, w in bins) / wsum)
                v_bar = sum(w * (x - x_bar) ** 2 for x, w in bins) / wsum
                m = j_strip
                Pm = _poisson_ge(lam, m)
                Pm1 = _poisson_ge(lam, m - 1)
                Pm2 = _poisson_ge(lam, m - 2)
                ES = s_eff * lam * Pm1 - c_rem * Pm
                ES2 = (
                    s_eff * s_eff * (lam * lam * Pm2 + lam * Pm1)
                    - 2.0 * s_eff * c_rem * lam * Pm1
                    + c_rem * c_rem * Pm
                )
                var_S = max(0.0, ES2 - ES * ES)
                x_new = x_bar + ES / H
                v_new = min(V_CAP, v_bar + var_S / (H * H))
                x_post = x_new
                v_post = v_new
                if v_new <= 1e-12:
                    surv_tail = 0.0 if x_new >= 1.0 else 1.0
                else:
                    sig = math.sqrt(v_new)
                    a = (1.0 - x_new) / sig
                    if a > 8.3:
                        # upper tail below double resolution: everyone survives
                        # (0.5*erfc underflows to 0.0 beyond a ~ 27, which the
                        # zero-out below would misread as annihilation)
                        surv_tail = 1.0
                    elif a < -8.3:
                        surv_tail = 0.0
                    else:
                        surv_tail = max(0.0, min(1.0, 0.5 * math.erfc(a / _SQRT2)))
                    if 1e-9 < surv_tail < 1.0 - 1e-9:
                        # survivors' conditional moments: the upper-truncated
                        # Normal at x = 1 (exact first/second moments)
                        phi_a = math.exp(-0.5 * a * a) / _SQRT2PI
                        hh = phi_a / (1.0 - surv_tail)
                        x_post = x_new - sig * hh
                        v_post = v_new * max(0.0, 1.0 - a * hh - hh * hh)
                n_s = lam * Pm1 - (m - 1 if m >= 1 else 0) * Pm
                x_eff = x_bar + HAZARD_MID * (min(x_new, 1.0) - x_bar)
                if n_s > 0.0 and x_eff > 0.3:
                    haz = math.exp(-n_s * x_eff)
                else:
                    haz = 1.0
                surv_round = surv_tail * haz
                if surv_round <= 1e-9:
                    d["count"] = 0
                    d["hull"] = 0
                    d["shields"] = 0
                    d["hits"] = 0.0
                    d["shield_rem"] = 0.0
                    d["dmg_bins"] = []
                    continue
                new_count = survivors * surv_round
                # Explosion thinning is selective on high-x units; approximate
                # the variance reduction by survival squared (pragmatic,
                # documented approximation of removing the top of the spread).
                x_post = min(max(x_post, 0.0), 0.99)
                v_post = min(V_CAP, v_post * haz * haz)
                sig2 = math.sqrt(v_post)
                if sig2 > 1e-4 and 0.0 < x_post - sig2 and x_post + sig2 < 0.99:
                    # carry the state as a 2-point histogram so a later exact
                    # round (or this branch again) inherits both the mean and
                    # the spread of the survivors
                    d["dmg_bins"] = [(x_post - sig2, 0.5), (x_post + sig2, 0.5)]
                else:
                    d["dmg_bins"] = [(x_post, 1.0)]
                d["count"] = new_count
                d["dmg_frac"] = x_post
                d["hull"] = H * new_count * max(0.0, 1.0 - x_post)
                d["shields"] = unit_shield * new_count
                d["hits"] = lam * new_count
                d["shield_rem"] = max(0.0, c_rem - s_eff * lam)
                continue

            support = int(lam + 8.0 * math.sqrt(lam)) + 2
            k_cap = min(support, 2000)

            # Normalise the carried histogram: the bin weights describe the
            # damage DISTRIBUTION of the current survivors. Prior survival is
            # already baked into d["count"]; re-multiplying stale weights (<1)
            # into the count every round produced a phantom ~0.45%/round loss
            # even with no incoming fire (observed vs the Rust core).
            wsum = sum(w for _, w in bins)
            if wsum <= 0:
                bins = [(0.0, 1.0)]
                wsum = 1.0

            new_pairs = []
            if lam_h_tot > 0.0:
                # Joint compound convolution: j heavy shots (Poisson lam_h of
                # size s_h, strip the shield first, each rolls the explosion
                # check at the damage fraction it leaves behind) x k bulk
                # shots (Poisson lam_b of size s_b, single-stream semantics
                # below). Surviving lineages are carried as separate histogram
                # bins ("partial tail de-collapse") so later rounds keep
                # hazarding exactly the ships that took the heavy hits.
                lam_h = (lam_h_tot / survivors) * perturb
                s_h = w_h / lam_h_tot
                lam_b = max(0.0, lam - lam_h)
                s_b = (w_dmg - w_h) / (lam_tot - lam_h_tot) if lam_b > 0.0 else 0.0
                p_exp_b = math.exp(-lam_b)
                j_max = min(int(lam_h + 5.0 * math.sqrt(lam_h)) + 2, 64)
                for x_prev, w_b in bins:
                    if w_b <= 0.0 or x_prev >= 1.0:
                        continue
                    w_b = w_b / wsum
                    p_j = math.exp(-lam_h)
                    for j in range(j_max + 1):
                        if p_j < 1e-14 and j > lam_h:
                            break
                        c_j = max(0.0, c_rem - j * s_h)
                        x_h = x_prev + max(0.0, j * s_h - c_rem) / H
                        path_h = 1.0
                        for i in range(1, j + 1):
                            x_i = x_prev + max(0.0, i * s_h - c_rem) / H
                            if x_i > 0.3:
                                path_h *= max(0.0, 1.0 - min(x_i, 0.99))
                            if path_h <= 0.0:
                                break
                        if x_h < 1.0 and path_h > 0.0:
                            if lam_b > 0.0:
                                k_kill = int(((1.0 - x_h) * H + c_j) / s_b) + 2
                                k_max = min(k_cap, max(k_kill, 1))
                                j_strip_b = (
                                    max(1, int(math.ceil(c_j / s_b - 1e-12)))
                                    if c_j > 0 else 0
                                )
                                p_k = p_exp_b
                                path = path_h
                                for k in range(k_max + 1):
                                    x_k = x_h + max(0.0, k * s_b - c_j) / H
                                    if x_k >= 1.0:
                                        break
                                    if k >= j_strip_b and x_k > 0.3:
                                        path *= max(0.0, 1.0 - min(x_k, 0.99))
                                    if path <= 0.0:
                                        break
                                    new_pairs.append((x_k, w_b * p_j * p_k * path))
                                    p_k *= lam_b / (k + 1)
                            else:
                                new_pairs.append((x_h, w_b * p_j * path_h))
                        p_j *= lam_h / (j + 1)
            else:
                p_exp = math.exp(-lam)
                for x_prev, w_b in bins:
                    if w_b <= 0.0 or x_prev >= 1.0:
                        continue
                    w_b = w_b / wsum
                    if s_eff > 0:
                        k_kill = int(((1.0 - x_prev) * H + c_rem) / s_eff) + 2
                    else:
                        k_kill = k_cap + 1
                    k_max = min(k_cap, max(k_kill, 1))
                    p_k = p_exp
                    path = 1.0
                    surv_b = 0.0
                    dmg_b = 0.0
                    for k in range(k_max + 1):
                        x_k = x_prev + max(0.0, k * s_eff - c_rem) / H
                        if x_k >= 1.0:
                            break
                        if k >= j_strip and x_k > 0.3:
                            path *= max(0.0, 1.0 - min(x_k, 0.99))
                        if path <= 0.0:
                            break
                        surv_b += p_k * path
                        dmg_b += p_k * path * x_k
                        p_k *= lam / (k + 1)
                    if surv_b > 1e-12:
                        # dmg_b == 0.0 with surv_b > 0: every shot within the
                        # k-support was fully shield-absorbed (x_k = 0 for all
                        # covered k, e.g. one probe shot of 1 vs a 10-shield
                        # LF stack). That lineage survives UNDAMAGED and must
                        # still be appended - dropping it left new_pairs empty
                        # and the fallback below annihilated the whole stack
                        # ("1 probe kills 500 LFs" phantom Attacker wins).
                        x_pair = min(dmg_b / surv_b, 0.99) if dmg_b > 0.0 else 0.0
                        new_pairs.append((x_pair, w_b * surv_b))

            if not new_pairs:
                d["count"] = 0
                d["hull"] = 0
                d["shields"] = 0
                d["hits"] = 0.0
                d["shield_rem"] = 0.0
                d["dmg_bins"] = []
                continue

            # Damp the cross-round damage stratification. The exact R-fold
            # convolution (undamped per-k lineages) overstates persistent
            # stratification vs the Rust per-unit core, while collapsing each
            # round to the survivor mean discards it entirely; calibrated on
            # the canonical pure duels (reaper/destroyer vs BC at 300/600/
            # 1000/3000), the Rust results sit at the geometric mean of the
            # two, i.e. deviations from the round mean damped by ~50-80%.
            # The heavy-overlay path carries its lineages (nearly) undamped:
            # damping there would re-create the mean collapse the overlay
            # exists to fix (OVERLAY_BETA dial).
            BETA_SPREAD = 0.8 if s_eff <= unit_shield * 1.0 else 0.5
            beta = OVERLAY_BETA if lam_h_tot > 0.0 else BETA_SPREAD
            if new_pairs:
                w_tot = sum(w for _, w in new_pairs)
                x_bar = sum(x * w for x, w in new_pairs) / max(w_tot, 1e-12)
                new_pairs = [
                    (x_bar + beta * (x - x_bar), w) for x, w in new_pairs
                ]

            # Rebin into fixed 5%-wide damage buckets to bound growth. Each
            # bucket stores the WEIGHTED MEAN x of the pairs assigned to it
            # (not the bucket centre): this conserves the distribution mean
            # without injecting artificial spread, which would inflate the
            # explosion-zone weight round over round.
            bucket = {}
            for x, w in new_pairs:
                key = min(int(x / 0.05), 19)
                sx, sw = bucket.get(key, (0.0, 0.0))
                bucket[key] = (sx + x * w, sw + w)
            d["dmg_bins"] = sorted(
                (sx / sw, sw) for _, (sx, sw) in bucket.items() if sw > 0
            )

            surv_total = sum(w for _, w in d["dmg_bins"])
            new_count = survivors * min(1.0, surv_total)
            x_mean = sum(x * w for x, w in d["dmg_bins"]) / max(surv_total, 1e-9)
            d["count"] = new_count
            d["dmg_frac"] = x_mean
            d["hull"] = H * new_count * max(0.0, 1.0 - x_mean)
            d["shields"] = unit_shield * new_count
            d["hits"] = lam * new_count
            d["shield_rem"] = max(0.0, c_rem - s_eff * lam)



def _regen_shields(side: dict):
    """Regenerate shields to full (OGame rule: shields regen each round)."""
    for u in side.values():
        u["shields"] = u["base_shield"] * u["count"]
        u["shield_rem"] = u["base_shield"]


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

        # ATTACKER-FIRST volleys (official OGame semantics, verified against
        # ogame.org combat simulator reports - see src/combat.rs). The
        # attacker's full volley resolves first; only the defender types
        # that SURVIVE it return fire. Sequential _fire calls implement
        # this directly: the second call reads the defender's post-volley
        # counts/damage. (The old snapshot/restore dance made both sides
        # fire at full start-of-round strength, overstating defenders.)
        _fire(atk_side, def_side, rng)
        _fire(def_side, atk_side, rng)

        # Check for stalemate (no damage either side)
        atk_after = sum(u["count"] for u in atk_side.values())
        def_after = sum(u["count"] for u in def_side.values())
        if atk_after == atk_before and def_after == def_before:
            stalemate = True  # neither side can damage the other, but keep fighting all 6 rounds

        _regen_shields(atk_side)
        _regen_shields(def_side)

    # Public schema: integer survivor counts. Counts stay FLOAT through the
    # rounds (expectation model); round stochastically exactly once here
    # (floor(x + U) is unbiased across sims). A hard floor quantises any
    # fractional death to one FULL unit in EVERY sim - under attacker-first
    # semantics, duel-scale attacker losses are fractions of a unit, so the
    # floor bias dominated the error (+1567% measured on destroyer@300).
    def _sround(x: float) -> int:
        return int(x + rng.random())

    atk_surv = {k: _sround(u["count"]) for k, u in atk_side.items() if u["count"] > 0.5}
    def_ship_surv = {k: _sround(u["count"]) for k, u in def_side.items() if u["count"] > 0.5 and k in SHIP_STATS}
    def_def_surv = {k: _sround(u["count"]) for k, u in def_side.items() if u["count"] > 0.5 and k in DEFENSE_STATS}

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

"""Optimizer orchestration: greedy -> GA -> final validation.

Logs every phase boundary so failures are easy to trace.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ogame_optimizer.logging_config import get_logger
from ogame_optimizer.core.combat import simulate_batch
from ogame_optimizer.core.fleet import compute_budget, SHIPS_COST, weighted_fleet_value, resource_preference_penalty, fleet_value
from ogame_optimizer.core.fast_combat import RAPIDFIRE, SHIP_STATS, DEFENSE_STATS
from ogame_optimizer.optimizer.greedy import greedy_optimize
from ogame_optimizer.optimizer.genetic import genetic_optimize, _drift_bounds_for_seed


_log = get_logger("ogame.optimizer.orchestration")


def _merge_fleet(base: Dict[str, int], additions: Dict[str, int]) -> Dict[str, int]:
    """Merge a fixed base fleet with GA-produced additions (additive counts).

    Only positive counts are carried. Used when base_fleet mode is active so
    that every combat validation sees the full fleet the player would field.
    """
    if not base:
        return dict(additions)
    merged = {k: v for k, v in base.items() if v > 0}
    for k, v in additions.items():
        if v > 0:
            merged[k] = merged.get(k, 0) + v
    return merged


def _resource_split(fleet: Dict[str, int]) -> tuple[int, int, int]:
    """Split a fleet's total cost into (metal, crystal, deuterium).

    Uses ``SHIPS_COST``; ship keys missing from the table contribute 0
    (defensive - optimizer fleets only contain known keys). By construction
    the three components sum to ``fleet_value(fleet)``.
    """
    metal = crystal = deuterium = 0
    for ship, count in fleet.items():
        cost = SHIPS_COST.get(ship)
        if cost and count > 0:
            metal += cost[0] * count
            crystal += cost[1] * count
            deuterium += cost[2] * count
    return metal, crystal, deuterium


def _budget_shares(fleet: Dict[str, int]) -> Dict[str, float]:
    """Cost-share vector of a fleet: share(t) = cost(t) / fleet total cost.

    share(t) = (sum SHIPS_COST[t] * count) / total cost of the fleet.
    Types with zero count (or unknown cost) are omitted; shares sum to
    1.0 for any non-empty fleet of known ships. Returns {} for a
    zero-cost / empty fleet (callers treat that as "no composition").
    """
    total = 0.0
    costs: Dict[str, float] = {}
    for ship, count in fleet.items():
        cost = SHIPS_COST.get(ship)
        if cost and count and count > 0:
            c = float(sum(cost)) * count
            costs[ship] = costs.get(ship, 0.0) + c
            total += c
    if total <= 0:
        return {}
    return {s: c / total for s, c in costs.items()}


def _share_distance(fleet_a: Dict[str, int], fleet_b: Dict[str, int]) -> float:
    """Total-variation distance between two fleets' cost-share vectors.

    ``0.5 * sum(|shareA(t) - shareB(t)|)`` over the UNION of ship types
    (a type missing from one fleet contributes share 0 on that side).
    Range [0, 1]: 0 = identical cost composition, 1 = fully disjoint
    ship types. Used as the diversity gate metric for fleet alternatives.
    """
    shares_a = _budget_shares(fleet_a)
    shares_b = _budget_shares(fleet_b)
    return 0.5 * sum(
        abs(shares_a.get(t, 0.0) - shares_b.get(t, 0.0))
        for t in set(shares_a) | set(shares_b)
    )


def _compute_kill_estimates(
    recommended_fleet: Dict[str, int],
    enemy_fleet: Dict[str, int],
    attribution_mean: Dict[str, Dict[str, float]],
    enemy_defenses: Dict[str, int],
    enemy_tech: tuple,
    attacker_tech: tuple,
    defender_fleet_analysis: Dict[str, Dict[str, float]],
) -> Dict[str, Dict]:
    """Estimate per-attacker-ship kills, cost-per-kill and damage share.

    Primary source: the engine's per-pair attribution matrix
    (``attribution_mean[atk][def]`` = mean damage attributed by the
    analytical engine). When the engine did not produce one (Rust path /
    small fleets), fall back to a static damage-potential heuristic:

        potential = count_atk * atk(tech-adjusted) * rf_mult
        rf_mult   = RAPIDFIRE[(atk, def)] for RF pairs, else 1 (engine
                    convention: an RF pair fires ~rf shots per volley vs a
                    pure target; a no-RF pair still fires once - a raw x0
                    for non-RF pairs would zero out most columns)
        bounce    = shot < 1% of the defender's tech-adjusted shield -> 0
                    (OGame shield-bounce rule, mirrors fast_combat._fire)

    Kills are attributed column-normalized: each defender FLEET type's
    ``destroyed_count`` is split across attacker types proportionally to
    their potential against that type. Zero-division guards: a defender
    type whose column total is 0 contributes no kills to any attacker; an
    attacker with ``kills_est == 0`` gets ``cost_per_kill = None``. All
    rows are estimates either way (the UI labels both sources identically).
    """
    atk_types = [s for s, c in recommended_fleet.items() if c > 0]
    if not atk_types:
        return {}

    def_fleet_types = [d for d, c in enemy_fleet.items() if c > 0]
    def_all = def_fleet_types + [d for d, c in (enemy_defenses or {}).items() if c > 0]
    if not def_all:
        return {s: {"kills_est": 0.0, "cost_per_kill": None, "damage_share": 0.0}
                for s in atk_types}

    # --- Damage potential per (atk, def) pair --------------------------------
    potential: Dict[str, Dict[str, float]] = {s: {d: 0.0 for d in def_all} for s in atk_types}
    if attribution_mean:
        for s in atk_types:
            row = attribution_mean.get(s) or {}
            for d in def_all:
                try:
                    potential[s][d] = float(row.get(d, 0.0) or 0.0)
                except (TypeError, ValueError):
                    potential[s][d] = 0.0
    else:
        # Heuristic fallback (no engine attribution matrix available).
        atk_mult = 1.0 + (attacker_tech[0] if attacker_tech else 0) * 0.1
        shield_mult = 1.0 + (enemy_tech[1] if len(enemy_tech) > 1 else 0) * 0.1
        for s in atk_types:
            stats = SHIP_STATS.get(s)
            if not stats:
                continue
            count_s = recommended_fleet[s]
            per_shot = stats["atk"] * atk_mult
            for d in def_all:
                d_stats = SHIP_STATS.get(d) or DEFENSE_STATS.get(d)
                if not d_stats:
                    continue
                if per_shot < d_stats["shield"] * shield_mult * 0.01:
                    continue  # shield bounce: shot fully absorbed
                rf = RAPIDFIRE.get((s, d), 0)
                mult = rf if rf > 1 else 1
                potential[s][d] = count_s * per_shot * mult

    # Column totals (per defender type, across attacker types) and row
    # damage totals (per attacker, across ALL def types incl. defenses).
    col_total = {d: sum(potential[s][d] for s in atk_types) for d in def_all}
    row_total = {s: sum(potential[s].values()) for s in atk_types}
    grand_total = sum(row_total.values())

    result: Dict[str, Dict] = {}
    for s in atk_types:
        kills_est = 0.0
        for d in def_fleet_types:
            entry = defender_fleet_analysis.get(d) or {}
            try:
                destroyed = int(entry.get("destroyed_count", 0))
            except (TypeError, ValueError):
                destroyed = 0
            if destroyed <= 0:
                continue
            col = col_total.get(d, 0.0)
            if col <= 0.0:
                continue  # bounced / no-potential column: contributes 0
            kills_est += (potential[s][d] / col) * destroyed
        ship_total_cost = sum(SHIPS_COST.get(s, (0, 0, 0))) * recommended_fleet[s]
        result[s] = {
            "kills_est": float(kills_est),
            "cost_per_kill": (ship_total_cost / kills_est) if kills_est > 0 else None,
            "damage_share": (row_total[s] / grand_total) if grand_total > 0 else 0.0,
        }
    return result


@dataclass
class AlternativeResult:
    """A quality-gated alternative fleet composition ("Option B/C").

    Produced by the alternatives engine (``_generate_alternatives``) AFTER
    the primary result is frozen. Every statistic comes from an
    independent final-validation batch of THIS fleet (same techs / economy
    / debris settings as the primary, at ``max(150, final_sims // 2)``
    sims). ``kill_estimates`` reuses the PRIMARY run's defender analysis:
    destroyed counts are primary-run artifacts, so they are an estimate
    here (documented; the attribution matrix itself IS the alternative's
    own). ``difference_vs_primary`` is the cost-share distance
    (``_share_distance``) to the primary fleet - >= the 0.10 diversity
    gate whenever this alternative was accepted through it.
    """

    label: str
    fleet: Dict[str, int]
    win_probability: float
    expected_loss_mean: float
    expected_loss_stddev: float
    confidence_interval_95: tuple
    fleet_cost_metal: int
    fleet_cost_crystal: int
    fleet_cost_deuterium: int
    kill_estimates: Dict[str, Dict]
    difference_vs_primary: float


@dataclass
class OptimizationResult:
    recommended_fleet: Dict[str, int]
    expected_loss_mean: float
    expected_loss_stddev: float
    win_probability: float
    confidence_interval_95: List[float]
    sims_run_final: int
    greedy_baseline_loss: float
    ga_improvement_pct: float
    time_elapsed_greedy: float
    time_elapsed_ga: float
    total_time: float
    seed_used: int
    mode: str = "attack"
    fleet_value: int = 0
    fleet_lost_pct: float = 0.0
    ships_lost_count: int = 0
    ships_initial_count: int = 0
    debris_metal: int = 0
    debris_crystal: int = 0
    debris_deuterium: int = 0
    debris_total: int = 0
    net_profit: int = 0
    net_profit_pct: float = 0.0
    recyclers_needed: int = 0
    recycler_capacity: int = 20000
    recyclers_cost_metal: int = 0
    recyclers_cost_crystal: int = 0
    recyclers_cost_deuterium: int = 0
    recyclers_cost_total: int = 0
    fleet_analysis: Dict[str, Dict[str, float]] = field(default_factory=dict)
    defender_fleet_analysis: Dict[str, Dict[str, float]] = field(default_factory=dict)
    defender_defense_analysis: Dict[str, Dict[str, float]] = field(default_factory=dict)
    raw_loss_mean: float = 0.0
    # min_gain constraint result
    min_gain_required: float = 0.0
    min_gain_met: bool = True
    actual_roi_pct: float = 0.0
    win_threshold_met: bool = True
    resource_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    preference_beta: float = 0.05
    fleet_weighted_value: float = 0.0
    resource_preference_penalty: float = 0.0
    resource_preference_match_score: float = 1.0
    # base_fleet mode: the player's existing fleet (locked, always fielded)
    base_fleet: Dict[str, int] = field(default_factory=dict)
    base_fleet_cost: int = 0
    base_fleet_count: int = 0
    recommended_additions: Dict[str, int] = field(default_factory=dict)
    # Costs & kills transparency: per-resource cost splits of the recommended
    # fleet (merged, = fleet_value split by M/C/D) and of the additions line,
    # plus per-attacker-ship kill attribution estimates.
    fleet_cost_metal: int = 0
    fleet_cost_crystal: int = 0
    fleet_cost_deuterium: int = 0
    additions_cost_metal: int = 0
    additions_cost_crystal: int = 0
    additions_cost_deuterium: int = 0
    kill_estimates: Dict[str, Dict] = field(default_factory=dict)
    # Fleet alternatives ("Option B/C"): genuinely-different compositions,
    # each independently validated. Populated only on the main result site
    # (early exits / skips keep it empty by design).
    alternatives: List[AlternativeResult] = field(default_factory=list)


def _validate_inputs(
    enemy_fleet: Dict[str, int],
    enemy_defenses: Dict[str, int],
    budget: int,
) -> None:
    if not enemy_fleet and not enemy_defenses:
        _log.error("Validation failed: no enemy (empty fleet + defenses)")
        raise ValueError("No enemy to fight: enemy_fleet and enemy_defenses both empty")
    if budget <= 0:
        _log.error("Validation failed: budget=%d (must be positive)", budget)
        raise ValueError("Budget must be positive")
    cheapest = min(sum(SHIPS_COST[k]) for k in SHIPS_COST)
    if budget < cheapest:
        _log.error("Validation failed: budget=%d < cheapest ship cost=%d", budget, cheapest)
        raise ValueError(
            f"Budget insufficient for any fleet (need at least {cheapest} for one ship, got {budget})"
        )
    _log.debug("Inputs OK: enemy_fleet=%s defenses=%s budget=%d", enemy_fleet, enemy_defenses, budget)


def _sensitivity_analysis(
    fleet: Dict[str, int],
    enemy_fleet: Dict[str, int],
    enemy_defenses: Dict[str, int],
    attacker_tech: tuple,
    enemy_tech: tuple,
    base_loss: float,
    debris_pct: float,
    deuterium_in_debris: bool,
    base_seed: int = 42,
    n_sims: int = 200,
    skip_ships: Optional[set] = None,
    base_fleet: Optional[Dict[str, int]] = None,
    loss_scale: float = 1.0,
    resource_weights: tuple = (1.0, 1.0, 1.0),
    preference_beta: float = 0.0,
    max_total_seconds: Optional[float] = None,
) -> Dict[str, Dict]:
    """For each ship in fleet, measure impact of removing it (redistribute to best remaining).

    Returns {ship: {"impact_pct": float, "tag": str, "redistributed_to": str, "loss_breakdown": dict}}.

    When base_fleet is provided: only analyzes ships with additions (count >
    base count). When removing such a ship, only the additions are removed
    (base is preserved). skip_ships (legacy) excludes ship types entirely.

    Tags:
    - critical: removing increases losses >20%
    - important: 5-20% increase
    - negligible: -5% to +5% change
    - dead_weight: non-fodder ship where fleet improves without it (<-5%)
    - fodder: cheap screening ship with negative impact - serves as cannon fodder, not truly dead weight
    """
    _skip = skip_ships or set()
    # In base_fleet mode, only analyze ships with additions (count > base count)
    present_ships = []
    for s, cnt in fleet.items():
        if cnt <= 0 or s in _skip:
            continue
        if base_fleet and cnt <= base_fleet.get(s, 0):
            continue
        present_ships.append(s)
    if len(present_ships) <= 1:
        return {}

    # Ships that serve as cannon fodder - negative impact is expected, not a flaw
    FODDER_SHIPS = {"light_fighter", "heavy_fighter", "small_cargo", "large_cargo", "espionage_probe"}

    # Ship value contributions for redistribution target selection
    ship_values = {s: sum(SHIPS_COST.get(s, (0, 0, 0))) * fleet[s] for s in present_ships}

    # Get baseline per-type losses via single detailed simulation
    # Convert base_loss (raw) to effective loss for consistent comparison
    _base_pen = resource_preference_penalty(fleet, resource_weights, preference_beta) if preference_beta > 0 else 0.0
    _base_effective_loss = base_loss * loss_scale + _base_pen
    base_per_type = {}
    try:
        from ogame_optimizer.core.fast_combat import simulate_combat_fast
        base_detail = simulate_combat_fast(
            fleet, enemy_fleet, enemy_defenses, attacker_tech, enemy_tech,
            seed=base_seed + 60000,
        )
        base_survivors = base_detail.get("attacker_survivors", {})
        base_per_type = {s: fleet.get(s, 0) - base_survivors.get(s, 0) for s in fleet}
    except Exception:
        pass

    # Adaptive n_sims: at scale (1.2M-unit enemies) one sim costs ~18ms, so a
    # flat 200 sims x 13 types = ~47s. Probe the per-sim cost once and shrink
    # n_sims so the whole per-type loop fits inside max_total_seconds.
    if max_total_seconds is not None:
        try:
            _probe_t0 = time.time()
            simulate_batch(
                attacker=fleet,
                defender=enemy_fleet,
                defender_defenses=enemy_defenses,
                attacker_tech=attacker_tech,
                defender_tech=enemy_tech,
                n_sims=3,
                base_seed=base_seed + 77000,
                debris_pct=debris_pct,
                deuterium_in_debris=deuterium_in_debris,
            )
            _per_sim = (time.time() - _probe_t0) / 3.0
            if _per_sim > 1e-6:
                _n_types = len(present_ships)
                _n_eff = max(30, min(n_sims, int(max_total_seconds / (_n_types * _per_sim))))
                _log.info(
                    "Sensitivity: per_sim=%.1fms -> n_sims %d -> %d for %d types (cap %.1fs)",
                    _per_sim * 1000.0, n_sims, _n_eff, _n_types, max_total_seconds,
                )
                n_sims = _n_eff
        except Exception:
            pass  # probe failed -> keep n_sims as-is

    analysis = {}
    for idx, ship in enumerate(present_ships):
        # Redistribute target = highest value remaining ship
        remaining = {s: v for s, v in ship_values.items() if s != ship}
        if not remaining:
            continue
        target = max(remaining, key=remaining.get)

        # Build variant: remove this ship. In base_fleet mode, keep base count.
        if base_fleet and ship in base_fleet:
            base_count = base_fleet[ship]
            variant = dict(fleet)
            variant[ship] = base_count
            freed_budget = sum(SHIPS_COST.get(ship, (0, 0, 0))) * (fleet[ship] - base_count)
        else:
            variant = {k: v for k, v in fleet.items() if k != ship}
            freed_budget = sum(SHIPS_COST.get(ship, (0, 0, 0))) * fleet[ship]
        target_cost = sum(SHIPS_COST.get(target, (0, 0, 0)))
        extra_count = 0
        if target_cost > 0:
            extra_count = freed_budget // target_cost
            variant[target] = variant.get(target, 0) + extra_count

        result = simulate_batch(
            attacker=variant,
            defender=enemy_fleet,
            defender_defenses=enemy_defenses,
            attacker_tech=attacker_tech,
            defender_tech=enemy_tech,
            n_sims=n_sims,
            base_seed=base_seed + 50000 + idx,
            debris_pct=debris_pct,
            deuterium_in_debris=deuterium_in_debris,
        )
        raw_loss = float(result.get("mean_attacker_loss", base_loss))
        # Apply same effective-loss formula as the GA so tags are consistent
        _pen = resource_preference_penalty(variant, resource_weights, preference_beta) if preference_beta > 0 else 0.0
        variant_loss = raw_loss * loss_scale + _pen
        if base_loss > 0:
            impact_pct = ((variant_loss - base_loss) / base_loss) * 100
        elif variant_loss > 0:
            impact_pct = 999.0
        else:
            impact_pct = 0.0

        # Compute per-type loss breakdown
        loss_breakdown = {}
        if base_per_type:
            try:
                variant_detail = simulate_combat_fast(
                    variant, enemy_fleet, enemy_defenses, attacker_tech, enemy_tech,
                    seed=base_seed + 60000 + idx,
                )
                variant_survivors = variant_detail.get("attacker_survivors", {})
                variant_per_type = {s: variant.get(s, 0) - variant_survivors.get(s, 0) for s in variant}
                all_types = set(list(base_per_type.keys()) + list(variant_per_type.keys()))
                for s in all_types:
                    delta = variant_per_type.get(s, 0) - base_per_type.get(s, 0)
                    if abs(delta) >= 1:
                        loss_breakdown[s] = int(delta)
            except Exception:
                pass

        # Tagging: fodder ships get "fodder" tag, not "dead_weight"
        if impact_pct > 20:
            tag = "critical"
        elif impact_pct > 5:
            tag = "important"
        elif impact_pct < -5:
            if ship in FODDER_SHIPS:
                tag = "fodder"
            else:
                tag = "dead_weight"
        else:
            tag = "negligible"

        analysis[ship] = {
            "impact_pct": round(impact_pct, 1),
            "tag": tag,
            "redistributed_to": target,
            "extra_count": extra_count,
            "loss_breakdown": loss_breakdown,
        }
        _log.info("  Sensitivity %s: %+.1f%% -> %s (redist->%s +%d)",
                  ship, impact_pct, tag, target, extra_count)

    return analysis

def _rust_verify(
    fleet: Dict[str, int],
    enemy_fleet: Dict[str, int],
    enemy_defenses: Dict[str, int],
    attacker_tech: tuple,
    enemy_tech: tuple,
    analytical_loss: float,
    n_sims: int = 3,
    base_seed: int = 42,
    max_ships: int = 1000,
) -> None:
    """Run real per-unit Rust simulation and log comparison vs analytical.

    Only runs if total fleet size is small enough for per-unit sim to be fast.
    Logs the comparison but does NOT override analytical results (logging-only for now).
    """
    total_ships = sum(fleet.values()) + sum(enemy_fleet.values()) + sum(enemy_defenses.values())
    if total_ships > max_ships:
        _log.info("  Rust verify: SKIPPED (fleet %d ships > limit %d)", total_ships, max_ships)
        return

    try:
        from ogame_optimizer.core.combat import (
            _normalize_ship_keys, _strip_unknown_for_rust,
            _normalize_defense_keys, _to_tech_tuple,
        )
        from ogame_optimizer import _ogame_combat

        result = _ogame_combat.simulate_batch_py(
            _normalize_ship_keys(_strip_unknown_for_rust(fleet)),
            _normalize_ship_keys(_strip_unknown_for_rust(enemy_fleet)),
            _normalize_defense_keys(enemy_defenses),
            _to_tech_tuple(attacker_tech),
            _to_tech_tuple(enemy_tech),
            n_sims,
            base_seed,
        )
        rust_loss = float(result.get("mean_attacker_loss", 0))
        rust_win = float(result.get("win_probability", 0))
        rust_stddev = float(result.get("stddev_attacker_loss", 0))

        _log.info("  Rust verify (%d sims, %d ships): loss=%.0f win=%.0f%% stddev=%.0f",
                  n_sims, total_ships, rust_loss, rust_win * 100, rust_stddev)
        _log.info("  Analytical:      loss=%.0f", analytical_loss)

        if analytical_loss > 0:
            diff_pct = abs(rust_loss - analytical_loss) / analytical_loss * 100
            if diff_pct > 50:
                _log.warning("  *** DISCREPANCY: Rust vs analytical differ by %.0f%%! ***", diff_pct)
            else:
                _log.info("  Agreement: within %.0f%%", diff_pct)
        elif rust_loss > 0:
            _log.warning("  *** DISCREPANCY: Analytical says 0 loss but Rust says %.0f! ***", rust_loss)
        else:
            _log.info("  Agreement: both predict ~0 loss")
    except Exception as e:
        _log.warning("  Rust verify: FAILED - %s", str(e)[:100])



def _prune_dead_weight(
    fleet: Dict[str, int],
    sensitivity: Dict[str, Dict],
    threshold: float = -3.0,
) -> tuple:
    """Build a pruned fleet with dead-weight ships removed and budget redistributed.

    Uses the sensitivity analysis to identify ships whose removal IMPROVES the
    fleet (impact < threshold). Redistributes their freed budget to the
    highest-impact positive ship. This is the "drop the negative %, add to the
    positive best" move the GA can't reliably make in a single pass because its
    reallocate operator works incrementally (15-60% per step) rather than
    wholesale-eliminating a ship type.

    Returns (pruned_fleet, pruned_ship_names). Returns (None, []) if nothing
    to prune or the prune would leave the fleet empty.
    """
    dead_weight = sorted(
        [(s, info.get("impact_pct", 0))
         for s, info in sensitivity.items()
         if isinstance(info, dict) and info.get("impact_pct", 0) < threshold],
        key=lambda x: x[1],  # worst (most negative) first
    )
    if not dead_weight:
        return None, []

    pruned_names = {s for s, _ in dead_weight}
    pruned = {s: c for s, c in fleet.items() if s not in pruned_names and c > 0}
    if not pruned:
        # Every present type reads dead-weight: single-removal artifacts on
        # a bad base fleet (removing ANY one type and redistributing looks
        # better, so the greedy prune would gut the fleet). Fall back to the
        # dominant type - a pure fleet of the main type beats the poisoned
        # mix (verified: cruiser+battlecruiser+stray at 40% win becomes
        # pure cruiser at 100% win).
        if not fleet:
            return None, []
        dominant = max(fleet, key=lambda s: fleet[s])
        if fleet[dominant] <= 0 or len([c for c in fleet.values() if c > 0]) < 2:
            return None, []
        pruned_names = {s for s, c in fleet.items() if c > 0 and s != dominant}
        pruned = {dominant: fleet[dominant]}

    # Redistribute freed budget to the highest-impact POSITIVE ship present.
    positive = [(s, sensitivity[s].get("impact_pct", 0))
                for s in pruned
                if s in sensitivity and isinstance(sensitivity[s], dict)
                and sensitivity[s].get("impact_pct", 0) > 0]
    if positive:
        target = max(positive, key=lambda x: x[1])[0]
    else:
        target = max(pruned, key=lambda s: sum(SHIPS_COST.get(s, (0, 0, 0))) * pruned[s])

    freed = sum(sum(SHIPS_COST.get(s, (0, 0, 0))) * fleet.get(s, 0) for s in pruned_names)
    target_cost = sum(SHIPS_COST.get(target, (0, 0, 0)))
    if target_cost > 0 and freed > 0:
        pruned[target] = pruned.get(target, 0) + freed // target_cost

    return pruned, list(pruned_names)


# --- Fleet alternatives ("Option B/C") ----------------------------------
#
# Diversity gate: a candidate must differ from the primary (and from every
# already-accepted alternative) by a cost-share total-variation distance of
# at least this much. 0.10 = e.g. 10% of the budget moved between types.
_MIN_ALT_SHARE_DISTANCE = 0.10
# Quality gate: an alternative's effective loss may exceed the primary's by
# at most this factor (trading a little optimality for a different comp).
_MAX_ALT_LOSS_FACTOR = 1.10
# Max number of accepted alternatives ("Option B", "Option C").
_MAX_ALTERNATIVES = 2
# Forced-pass GA budget: clamp(ga_time_budget * 0.6, 0.5, 4.0) per pass.
_FORCED_PASS_BUDGET_MIN = 0.5
_FORCED_PASS_BUDGET_MAX = 4.0


def _generate_alternatives(
    primary_fleet: Dict[str, int],
    primary_loss: float,
    primary_win_met: bool,
    defender_fleet_analysis: Dict[str, Dict[str, float]],
    enemy_fleet: Dict[str, int],
    enemy_defenses: Dict[str, int],
    enemy_tech: tuple,
    attacker_tech: tuple,
    mode: str,
    base_fleet: Optional[Dict[str, int]],
    ga_budget: int,
    exclude_ships,
    seed_candidates: List[tuple],
    ga_time_budget: float,
    final_sims: int,
    debris_pct: float,
    deuterium_in_debris: bool,
    loss_scale: float,
    resource_weights: tuple,
    preference_beta: float,
    min_gain_pct: float,
    base_seed: int,
) -> List[AlternativeResult]:
    """Build up to 2 genuinely-different, quality-gated fleet alternatives.

    Called AFTER the primary's final validation + sensitivity so every
    primary number it compares against is frozen. Pipeline per candidate:

    1. Diversity gate (cheap, composition-only - no sims): cost-share
       distance vs primary AND vs every accepted alternative must be
       >= ``_MIN_ALT_SHARE_DISTANCE``.
    2. Independent validation: same ``simulate_batch`` machinery as the
       primary final validation, with ``max(150, final_sims // 2)`` sims
       and identical techs / economy / debris settings (distinct,
       deterministic base_seed offsets).
    3. Quality gate: ``alt_loss <= _MAX_ALT_LOSS_FACTOR * primary_loss``
       AND the alternative's win-threshold status equals the primary's
       (same 0.95 rule for the mode - an unwinnable-vs-winnable swap is
       never offered as an "alternative").

    Candidate sources (in order, stop at 2 accepted):
    * Harvest: the top 4 non-primary per-seed bests from Phase B's
      multi-start loop (threaded out via ``seed_candidates`` as
      ``(name, additions, merged_fleet, validated_loss)`` tuples), ranked
      by their Phase-B validated loss.
    * Forced pass B: a short GA run whose exclude set adds the primary's
      dominant-by-cost ship type (unioned with user excludes).
    * Forced pass C (only if an Option B was accepted): excludes the
      dominant types of BOTH the primary and Option B.

    Forced passes reuse the Phase B shape (progressive seed generation
      with the exclude set -> per-seed explore -> refine of the best) at
      ``clamp(ga_time_budget * 0.6, 0.5, 4.0)`` seconds, with deterministic
      base_seed offsets (+1001 / +1002) so runs are reproducible.

    ``kill_estimates`` of an accepted alternative reuse the PRIMARY's
    ``defender_fleet_analysis`` (destroyed counts are primary-run
    artifacts; the alternative's own attribution matrix is used for the
    damage split, so the heuristic fallback applies on the Rust path /
    small fleets exactly as for the primary).

    Skip conditions (return empty, logged): no additions budget (0.0x
    simulation), primary win-threshold not met (unwinnable -> noise),
    empty primary fleet, or base_fleet mode where the primary ended up
    with no additions.
    """
    primary_fleet = {s: c for s, c in primary_fleet.items() if c > 0}

    # --- Skip conditions (cheap, logged, before any work) -----------------
    if ga_budget <= 0:
        _log.info("--- Alternatives: SKIPPED (no additions budget: 0.0x simulation) ---")
        return []
    if not primary_win_met:
        _log.info("--- Alternatives: SKIPPED (primary win threshold not met - "
                  "unwinnable scenario, alternatives would be noise) ---")
        return []
    if not primary_fleet:
        _log.info("--- Alternatives: SKIPPED (empty primary fleet) ---")
        return []
    if base_fleet:
        _add = {s: primary_fleet.get(s, 0) - base_fleet.get(s, 0) for s in primary_fleet}
        if not any(c > 0 for c in _add.values()):
            _log.info("--- Alternatives: SKIPPED (additions mode with no additions) ---")
            return []

    from ogame_optimizer.optimizer.genetic import GAConfig
    from ogame_optimizer.optimizer.progressive_seeds import generate_progressive_seeds

    accepted: List[AlternativeResult] = []
    # Fleet list for the pairwise diversity gate: primary first, then each
    # accepted alternative's fleet (index i -> label accepted[i-1]).
    gate_fleets: List[Dict[str, int]] = [primary_fleet]

    def _fleet_key(f: Dict[str, int]) -> tuple:
        return tuple(sorted((s, c) for s, c in f.items() if c > 0))

    def _enforce_budget(fleet: Dict[str, int]) -> Dict[str, int]:
        """Mirror the primary's post-GA budget enforcement: proportional
        scale-down of an over-budget fleet (additions only in base_fleet
        mode - the base is a sunk cost). Usually a no-op; defensive."""
        if base_fleet:
            adds = {s: max(0, fleet.get(s, 0) - base_fleet.get(s, 0)) for s in fleet}
            cost = fleet_value(adds)
            if cost > ga_budget and cost > 0:
                scale = ga_budget / cost
                adds = {k: int(v * scale) for k, v in adds.items() if int(v * scale) > 0}
                return _merge_fleet(base_fleet, adds)
            return fleet
        cost = fleet_value(fleet)
        if cost > ga_budget and cost > 0:
            scale = ga_budget / cost
            return {k: int(v * scale) for k, v in fleet.items() if int(v * scale) > 0}
        return fleet

    def _dominant_type(fleet: Dict[str, int]) -> Optional[str]:
        """Ship type holding the largest share of the fleet's total cost."""
        best_ship, best_cost = None, -1.0
        for ship, count in fleet.items():
            if count <= 0:
                continue
            cost = float(sum(SHIPS_COST.get(ship, (0, 0, 0)))) * count
            if cost > best_cost:
                best_ship, best_cost = ship, cost
        return best_ship

    def _evaluate_candidate(name: str, cand_fleet: Dict[str, int], eval_idx: int) -> Optional[AlternativeResult]:
        """Diversity gate -> validate -> quality gate for one candidate."""
        cand = {s: c for s, c in cand_fleet.items() if c > 0}
        if not cand:
            _log.info("--- Alternatives: candidate=%s rejected (empty fleet) ---", name)
            return None
        dist = _share_distance(cand, primary_fleet)
        if dist < _MIN_ALT_SHARE_DISTANCE:
            _log.info("--- Alternatives: candidate=%s dist=%.2f rejected(diversity<%.2f vs primary) ---",
                      name, dist, _MIN_ALT_SHARE_DISTANCE)
            return None
        for i, other in enumerate(gate_fleets[1:], start=1):
            d_other = _share_distance(cand, other)
            if d_other < _MIN_ALT_SHARE_DISTANCE:
                _log.info("--- Alternatives: candidate=%s dist=%.2f rejected(diversity<%.2f vs %s) ---",
                          name, dist, _MIN_ALT_SHARE_DISTANCE, accepted[i - 1].label)
                return None

        # Diversity OK -> the expensive part: independent final validation
        # (same machinery/settings as the primary, half the sims, floor 150).
        n_alt_sims = max(150, final_sims // 2)
        alt_final = simulate_batch(
            attacker=cand,
            defender=enemy_fleet,
            defender_defenses=enemy_defenses,
            attacker_tech=attacker_tech,
            defender_tech=enemy_tech,
            n_sims=n_alt_sims,
            base_seed=base_seed + 8888 + 17 * eval_idx,
            debris_pct=debris_pct,
            deuterium_in_debris=deuterium_in_debris,
        )
        alt_raw_loss = float(alt_final.get("mean_attacker_loss", 0))
        alt_penalty = resource_preference_penalty(cand, resource_weights, preference_beta)
        alt_loss = alt_raw_loss * loss_scale + alt_penalty
        alt_stddev = float(alt_final.get("stddev_attacker_loss", 0))
        alt_wp = float(alt_final.get("win_probability", 0.0))
        alt_win_met = (alt_wp >= 0.95) if mode == "attack" else ((1.0 - alt_wp) >= 0.95)

        if alt_loss > _MAX_ALT_LOSS_FACTOR * primary_loss:
            _log.info("--- Alternatives: candidate=%s dist=%.2f rejected(loss=%.0f > %.2fx%.0f) ---",
                      name, dist, alt_loss, _MAX_ALT_LOSS_FACTOR, primary_loss)
            return None
        if alt_win_met != primary_win_met:
            _log.info("--- Alternatives: candidate=%s dist=%.2f rejected(win-threshold status "
                      "differs from primary: %s vs %s) ---", name, dist, alt_win_met, primary_win_met)
            return None

        label = "Option B" if not accepted else "Option C"
        stderr = alt_stddev / max(1, n_alt_sims ** 0.5)
        cost_m, cost_c, cost_d = _resource_split(cand)
        alt_kills = _compute_kill_estimates(
            recommended_fleet=cand,
            enemy_fleet=enemy_fleet,
            attribution_mean=alt_final.get("attribution_mean", {}) or {},
            enemy_defenses=enemy_defenses,
            enemy_tech=enemy_tech,
            attacker_tech=attacker_tech,
            defender_fleet_analysis=defender_fleet_analysis,
        )
        result = AlternativeResult(
            label=label,
            fleet=cand,
            win_probability=alt_wp,
            expected_loss_mean=alt_loss,
            expected_loss_stddev=alt_stddev,
            confidence_interval_95=(alt_loss - 1.96 * stderr, alt_loss + 1.96 * stderr),
            fleet_cost_metal=cost_m,
            fleet_cost_crystal=cost_c,
            fleet_cost_deuterium=cost_d,
            kill_estimates=alt_kills,
            difference_vs_primary=dist,
        )
        _log.info("--- Alternatives: candidate=%s dist=%.2f accepted(%s: loss=%.0f win=%.1f%%) ---",
                  name, dist, label, alt_loss, alt_wp * 100)
        return result

    def _forced_pass(pass_name: str, forced_excludes: set, pass_seed: int) -> Optional[Dict[str, int]]:
        """Short GA run with extra ship-type excludes (Phase B shape).

        Seeds from its own progressive-seed generation (exclude set
        applied), quick-explores each seed, then refines the best with the
        remaining budget - mirroring Phase B1/B2. Returns the best MERGED
        fleet (base + additions in base_fleet mode) or None.
        """
        all_excludes = set(exclude_ships or []) | set(forced_excludes)
        pass_budget = max(_FORCED_PASS_BUDGET_MIN,
                          min(_FORCED_PASS_BUDGET_MAX, ga_time_budget * 0.6))
        _log.info("--- Alternatives: forced pass %s (budget=%.1fs excludes=%s) ---",
                  pass_name, pass_budget, sorted(all_excludes))
        try:
            prog = generate_progressive_seeds(
                enemy_fleet=enemy_fleet,
                enemy_defenses=enemy_defenses,
                budget=ga_budget,
                attacker_tech=attacker_tech,
                enemy_tech=enemy_tech,
                debris_pct=debris_pct,
                deuterium_in_debris=deuterium_in_debris,
                exclude_ships=sorted(all_excludes),
                base_seed=pass_seed,
            )
        except Exception as seed_exc:
            _log.warning("  Forced pass %s: seed generation failed: %s", pass_name, seed_exc)
            return None
        prog = [{s: c for s, c in f.items() if s not in all_excludes and c > 0} for f in prog]
        prog = [f for f in prog if f and sum(f.values()) > 0]
        if not prog:
            _log.info("  Forced pass %s: no viable seeds (excludes leave nothing)", pass_name)
            return None

        def _validate(fleet: Dict[str, int], n_sims: int) -> float:
            batch = simulate_batch(
                attacker=fleet,
                defender=enemy_fleet,
                defender_defenses=enemy_defenses,
                attacker_tech=attacker_tech,
                defender_tech=enemy_tech,
                n_sims=n_sims,
                base_seed=base_seed + 7777,
                debris_pct=debris_pct,
                deuterium_in_debris=deuterium_in_debris,
            )
            pen = resource_preference_penalty(fleet, resource_weights, preference_beta)
            return float(batch.get("mean_attacker_loss", float("inf"))) * loss_scale + pen

        def _ga(seed_fleet: Dict[str, int], cfg: GAConfig, ga_seed: int):
            drift = _drift_bounds_for_seed(seed_fleet, budget=ga_budget)
            for s in all_excludes:
                drift[s] = (0, 0)
            return genetic_optimize(
                seed_fleet=seed_fleet,
                enemy_fleet=enemy_fleet,
                enemy_defenses=enemy_defenses,
                enemy_tech=enemy_tech,
                attacker_tech=attacker_tech,
                budget=ga_budget,
                mode=mode,
                config=cfg,
                base_seed=ga_seed,
                drift_bounds=drift,
                loss_scale=loss_scale,
                resource_weights=resource_weights,
                preference_beta=preference_beta,
                min_gain_pct=min_gain_pct,
                base_fleet=base_fleet,
            )

        # Explore phase (mirrors Phase B1: min(15% of budget, 2s) per seed).
        explore_time = min(pass_budget * 0.15, 2.0)
        best_fleet: Optional[Dict[str, int]] = None
        best_loss = float("inf")
        for i, seed_f in enumerate(prog):
            if explore_time >= 0.5:
                ga_out = _ga(seed_f, GAConfig(
                    time_budget_seconds=explore_time, sims_per_eval=20, population_size=20,
                    mutation_rate=0.30, mutation_step_fraction=0.30, macro_mutation_rate=0.20,
                    reallocate_rate=0.30,
                ), pass_seed + 101 + i)
                additions = ga_out.best_fleet
            else:
                additions = seed_f
            merged = _merge_fleet(base_fleet, additions) if base_fleet else dict(additions)
            loss = _validate(merged, 100)
            if loss < best_loss:
                best_fleet, best_loss = merged, loss
        if best_fleet is None:
            return None

        # Refine phase (mirrors Phase B2: refine + polish on the best).
        best_additions = {s: max(0, best_fleet.get(s, 0) - (base_fleet or {}).get(s, 0))
                          for s in best_fleet}
        best_additions = {s: c for s, c in best_additions.items() if c > 0}
        refine_time = pass_budget - explore_time * len(prog)
        refine_rounds = [
            ("refine", 0.50, 50, 0.15, 0.18, 0.12, 0.20),
            ("polish", 0.50, 100, 0.08, 0.10, 0.05, 0.10),
        ]
        for r_idx, (rname, t_frac, sims_eval, mut_rate, step_frac, macro_rate, realloc_rate) in enumerate(refine_rounds):
            t_alloc = refine_time * t_frac
            if t_alloc < 0.5 or not best_additions:
                continue
            ga_out = _ga(best_additions, GAConfig(
                time_budget_seconds=t_alloc, sims_per_eval=sims_eval, population_size=30,
                mutation_rate=mut_rate, mutation_step_fraction=step_frac,
                macro_mutation_rate=macro_rate, reallocate_rate=realloc_rate,
            ), pass_seed + 201 + r_idx)
            merged = _merge_fleet(base_fleet, ga_out.best_fleet) if base_fleet else dict(ga_out.best_fleet)
            if not any(c > 0 for c in merged.values()):
                continue
            loss = _validate(merged, 200)
            if loss < best_loss:
                best_fleet, best_loss = merged, loss
                best_additions = {s: max(0, merged.get(s, 0) - (base_fleet or {}).get(s, 0))
                                  for s in merged}
                best_additions = {s: c for s, c in best_additions.items() if c > 0}
        return best_fleet

    _log.info("--- Alternatives: generating (primary loss=%.0f, up to %d) ---",
              primary_loss, _MAX_ALTERNATIVES)

    # --- Source 1: harvested Phase-B per-seed bests (already computed) ----
    seen_keys = {_fleet_key(primary_fleet)}
    harvested = []
    for name, additions, merged, loss in seed_candidates:
        fleet = _enforce_budget({s: c for s, c in (merged or additions or {}).items() if c > 0})
        if not fleet:
            continue
        if loss == float("inf") or loss != loss:  # inf / nan guard
            continue
        key = _fleet_key(fleet)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        harvested.append((name, fleet, loss))
    harvested.sort(key=lambda item: item[2])
    harvested = harvested[:4]
    if harvested:
        _log.info("  Harvested %d Phase-B candidates (best loss=%.0f)",
                  len(harvested), harvested[0][2])

    eval_idx = 0
    for name, fleet, _loss in harvested:
        if len(accepted) >= _MAX_ALTERNATIVES:
            break
        result = _evaluate_candidate("harvest:" + name, fleet, eval_idx)
        eval_idx += 1
        if result is not None:
            accepted.append(result)
            gate_fleets.append(result.fleet)

    # --- Source 2: forced pass B (exclude primary's dominant type) --------
    dominant_primary = _dominant_type(primary_fleet)
    if len(accepted) < _MAX_ALTERNATIVES and dominant_primary:
        cand = _forced_pass("B", {dominant_primary}, base_seed + 1001)
        if cand:
            cand = _enforce_budget(cand)
            result = _evaluate_candidate("forced:B", cand, eval_idx)
            eval_idx += 1
            if result is not None:
                accepted.append(result)
                gate_fleets.append(result.fleet)
    elif not dominant_primary:
        _log.info("--- Alternatives: forced pass B skipped (no dominant type) ---")

    # --- Source 3: forced pass C (exclude primary + Option B dominants) ---
    if len(accepted) < _MAX_ALTERNATIVES and accepted and dominant_primary:
        dominant_b = _dominant_type(accepted[0].fleet)
        excludes_c = {dominant_primary} | ({dominant_b} if dominant_b else set())
        cand = _forced_pass("C", excludes_c, base_seed + 1002)
        if cand:
            cand = _enforce_budget(cand)
            result = _evaluate_candidate("forced:C", cand, eval_idx)
            eval_idx += 1
            if result is not None:
                accepted.append(result)
                gate_fleets.append(result.fleet)

    _log.info("--- Alternatives: done (%d accepted: %s) ---",
              len(accepted), [a.label for a in accepted] or "none")
    return accepted


def optimize(
    enemy_fleet: Dict[str, int],
    enemy_defenses: Optional[Dict[str, int]] = None,
    enemy_tech: tuple = (0, 0, 0),
    attacker_tech: tuple = (0, 0, 0),
    budget_multiplier: float = 1.0,
    mode: str = "attack",
    base_seed: int = 42,
    ga_time_budget: float = 5.0,
    final_sims: int = 1000,
    exclude_ships=None,
    seed_fleet=None,
    debris_pct: float = 0.30,
    deuterium_in_debris: bool = False,
    optimization_target: str = "maximize_profit",
    # Hard constraint: minimum return after recycling as a percentage of
    # fleet value. 0 = disabled (default). When > 0, the optimizer only
    # accepts fleets whose actual ROI >= this threshold.
    min_gain_pct: float = 0.0,
    hyperspace_tech: int = 11,
    resource_weights: tuple[float, float, float] = (2.0, 1.0, 1.0),
    preference_beta: float = 0.05,
    collector_class: bool = False,
    base_fleet: Optional[Dict[str, int]] = None,
    # Fleet alternatives ("Option B/C"): generate up to 2 additional
    # genuinely-different, quality-gated compositions after the primary
    # result is frozen (adds ~2x (0.5-4s GA + validation) wall time).
    include_alternatives: bool = True,
) -> OptimizationResult:
    enemy_defenses = enemy_defenses or {}
    t0 = time.time()
    _log.info("=== OPTIMIZE START mode=%s multiplier=%s seed=%d ===", mode, budget_multiplier, base_seed)
    _log.info("Enemy fleet: %s", enemy_fleet)
    _log.info("Enemy defenses: %s", enemy_defenses)
    _log.info("Techs: attacker=%s defender=%s", attacker_tech, enemy_tech)

    budget = compute_budget(enemy_fleet, enemy_defenses, budget_multiplier)
    _log.info("Budget computed: %d (multiplier=%s)", budget, budget_multiplier)
    # In profit mode, effective loss = raw_loss * (1 - debris_pct)
    # because debris_pct of your losses are recyclable
    _loss_scale = (1.0 - debris_pct) if optimization_target == "maximize_profit" else 1.0
    _log.info("Target: %s (loss_scale=%.2f, debris_pct=%.0f%%, min_gain_pct=%.1f%%)",
              optimization_target, _loss_scale, debris_pct * 100, min_gain_pct)
    _log.info("Resource weights: M=%.2f C=%.2f D=%.2f (composition penalty, beta=%.2f)",
              resource_weights[0], resource_weights[1], resource_weights[2], preference_beta)
    if budget_multiplier == 0 and base_fleet:
        _log.info('0.0x SIMULATION: multiplier=0 with base_fleet - pure evaluation mode')
    elif budget_multiplier == 0 and not base_fleet:
        raise ValueError(
            "0.0X multiplier requires a base fleet. Switch to the "
            "Start from My Fleet tab and enter your ships first."
        )
    else:
        _validate_inputs(enemy_fleet, enemy_defenses, budget)

    # --- base_fleet mode: the player has an existing fleet (locked, always
    # fielded) and wants to know what to BUILD on top of it. The GA
    # optimises additions only; the base is merged into every combat
    # evaluation. The budget_multiplier directly controls the ADDITIONS
    # budget (= enemy_value * multiplier), independent of base cost.
    # So 1.0x = "build up to enemy_value of new ships", 2.0x = "build up
    # to 2x enemy_value", 0.0x = "don't add anything, just evaluate base".
    base_cost = 0
    base_count = 0
    if base_fleet:
        from ogame_optimizer.core.fleet import fleet_value as _fv_base
        base_cost = _fv_base(base_fleet)
        base_count = sum(base_fleet.values())
        _log.info("Base fleet: cost=%d count=%d ships", base_cost, base_count)
    _ga_budget = budget  # multiplier * enemy_value, independent of base
    if base_fleet:
        _log.info("Additions budget: %d (multiplier*enemy_value); base: %d ships for %d res",
                  _ga_budget, base_count, base_cost)

    # --- Early-exit: if base fleet already wins, don't add anything ---
    # Adding ships to a fleet that already wins only adds more losses to report
    # without improving the win. The user sees contradictory recommendations
    # (additions proposed but tagged dead_weight).
    if base_fleet and _ga_budget > 0:
        _base_check = simulate_batch(
            attacker=dict(base_fleet), defender=enemy_fleet,
            defender_defenses=enemy_defenses, attacker_tech=attacker_tech,
            defender_tech=enemy_tech, n_sims=200, base_seed=base_seed + 7777,
            debris_pct=debris_pct, deuterium_in_debris=deuterium_in_debris,
        )
        _base_wp = float(_base_check.get("win_probability", 0))
        _already_wins = (_base_wp >= 0.95) if mode == "attack" else ((1.0 - _base_wp) >= 0.95)
        if _already_wins:
            _log.info("Base fleet already wins (%.1f%%) - skipping optimization, returning base fleet as-is", _base_wp * 100)
            # Build a minimal result with just the base fleet
            t_done = time.time()
            _base_fv = fleet_value(base_fleet)
            _base_cost_split = _resource_split(base_fleet)
            _base_raw_loss = float(_base_check.get("mean_attacker_loss", 0))
            _base_stddev = float(_base_check.get("stddev_attacker_loss", 0))
            _base_pen = resource_preference_penalty(base_fleet, resource_weights, preference_beta)
            _base_eff_loss = _base_raw_loss * _loss_scale + _base_pen
            _stderr = _base_stddev / max(1, final_sims ** 0.5)
            _debris_total = int(_base_check.get("debris_total", 0))
            _net_profit = _debris_total - _base_raw_loss

            # Compute recycler requirements (matches main path formula)
            _recycler_cap = int(20000 * (1 + hyperspace_tech * 0.05) * (1.25 if collector_class else 1.0))
            _recyclers_needed = (_debris_total + _recycler_cap - 1) // _recycler_cap if _recycler_cap > 0 else 0
            _rec_cost_tuple = (10000, 6000, 2000)  # Metal, Crystal, Deuterium per recycler
            _recyclers_cost_metal = _recyclers_needed * _rec_cost_tuple[0]
            _recyclers_cost_crystal = _recyclers_needed * _rec_cost_tuple[1]
            _recyclers_cost_deuterium = _recyclers_needed * _rec_cost_tuple[2]
            _recyclers_cost_total = _recyclers_cost_metal + _recyclers_cost_crystal + _recyclers_cost_deuterium

            # Compute sensitivity analysis for the base fleet.
            # Skip base ships from impact tags - they are locked and can't be
            # removed, so showing 'dead_weight' for them is misleading.
            # Only ships with additions (count > base count) would be analyzed.
            _base_sens = _sensitivity_analysis(
                fleet=dict(base_fleet), enemy_fleet=enemy_fleet,
                enemy_defenses=enemy_defenses, attacker_tech=attacker_tech,
                enemy_tech=enemy_tech, base_loss=_base_raw_loss,
                debris_pct=debris_pct, deuterium_in_debris=deuterium_in_debris,
                base_seed=base_seed, n_sims=200,
                loss_scale=_loss_scale, resource_weights=resource_weights,
                preference_beta=preference_beta,
                skip_ships=set(base_fleet.keys()),
            )

            # Compute defender analysis (per-ship survival)
            _def_analysis = {}
            try:
                _atk_surv_mean = _base_check.get("attacker_survivors_mean", {}) or {}
                _def_surv_mean = _base_check.get("defender_survivors_mean", {}) or {}
                _base_ships_lost = 0
                for _ship, _count in base_fleet.items():
                    if _count > 0:
                        _surv = float(_atk_surv_mean.get(_ship, 0))
                        _surv_pct = round((_surv / _count) * 100, 1)
                        if _ship not in _base_sens:
                            _base_sens[_ship] = {}
                        _base_sens[_ship]["survival_pct"] = _surv_pct
                        _base_sens[_ship]["surviving_count"] = round(_surv)
                        _base_ships_lost += max(0, _count - round(_surv))
                for _ship, _count in enemy_fleet.items():
                    if _count > 0:
                        _surv = float(_def_surv_mean.get(_ship, 0))
                        _surv_pct = round((_surv / _count) * 100, 1)
                        _def_analysis[_ship] = {
                            "count": _count,
                            "surviving_count": round(_surv),
                            "destroyed_count": int(_count - round(_surv)),
                            "survival_pct": _surv_pct,
                        }
            except Exception:
                pass

            return OptimizationResult(
                recommended_fleet=dict(base_fleet),
                fleet_value=_base_fv,
                fleet_lost_pct=(_base_raw_loss / _base_fv * 100) if _base_fv > 0 else 0,
                ships_lost_count=int(_base_check.get("ships_lost", 0)),
                ships_initial_count=base_count,
                debris_metal=int(_base_check.get("debris_metal", 0)),
                debris_crystal=int(_base_check.get("debris_crystal", 0)),
                debris_deuterium=int(_base_check.get("debris_deuterium", 0)),
                debris_total=_debris_total,
                net_profit=_net_profit,
                net_profit_pct=(_net_profit / _base_fv * 100) if _base_fv > 0 else 0,
                recyclers_needed=_recyclers_needed,
                recycler_capacity=_recycler_cap,
                recyclers_cost_metal=_recyclers_cost_metal,
                recyclers_cost_crystal=_recyclers_cost_crystal,
                recyclers_cost_deuterium=_recyclers_cost_deuterium,
                recyclers_cost_total=_recyclers_cost_total,
                raw_loss_mean=_base_raw_loss,
                win_threshold_met=True,
                resource_weights=tuple(resource_weights),
                preference_beta=preference_beta,
                fleet_weighted_value=weighted_fleet_value(base_fleet, resource_weights),
                resource_preference_penalty=_base_pen,
                resource_preference_match_score=1.0,
                expected_loss_mean=_base_eff_loss,
                expected_loss_stddev=_base_stddev,
                win_probability=_base_wp,
                confidence_interval_95=[_base_eff_loss - 1.96 * _stderr, _base_eff_loss + 1.96 * _stderr],
                sims_run_final=final_sims,
                greedy_baseline_loss=0.0,
                ga_improvement_pct=0.0,
                time_elapsed_greedy=0.0,
                time_elapsed_ga=0.0,
                total_time=t_done - t0,
                seed_used=base_seed,
                mode=mode,
                fleet_analysis=_base_sens,
                defender_fleet_analysis=_def_analysis,
                defender_defense_analysis={
                    _d: {
                        'count': _c,
                        'surviving_count': round(float(_base_check.get('defender_defense_survivors_mean', {}).get(_d, 0))),
                        'destroyed_count': int(_c - round(float(_base_check.get('defender_defense_survivors_mean', {}).get(_d, 0)))),
                        'survival_pct': round(float(_base_check.get('defender_defense_survivors_mean', {}).get(_d, 0)) / _c * 100, 1),
                    }
                    for _d, _c in enemy_defenses.items() if _c > 0
                },
                base_fleet=dict(base_fleet),
                base_fleet_cost=base_cost,
                base_fleet_count=base_count,
                recommended_additions={},
                fleet_cost_metal=_base_cost_split[0],
                fleet_cost_crystal=_base_cost_split[1],
                fleet_cost_deuterium=_base_cost_split[2],
                additions_cost_metal=0,
                additions_cost_crystal=0,
                additions_cost_deuterium=0,
                kill_estimates={},
            )

    # --- Refine-seed normalisation (before Phase A adopts it) ---
    # 1. base_fleet mode: the client sends the previously recommended
    #    MERGED fleet (base + additions). Strip base counts, else the
    #    base is merged on top again downstream and counted twice in
    #    every combat evaluation.
    # 2. Budget feasibility: an over-budget seed would be validated and
    #    set global_best_loss to a loss that no budget-constrained GA
    #    candidate can beat (unbeatable benchmark). Scale it down at
    #    adoption time (same int() truncation style as the post-GA
    #    scale-down). In base_fleet mode only the ADDITIONS are scaled
    #    against the additions budget; the base is a sunk cost.
    _refine_seed = None
    if seed_fleet and not (base_fleet and _ga_budget == 0):
        _refine_seed = {k: v for k, v in seed_fleet.items() if v > 0}
        if base_fleet:
            _refine_seed = {k: v - base_fleet.get(k, 0) for k, v in _refine_seed.items()}
            _refine_seed = {k: v for k, v in _refine_seed.items() if v > 0}
        _seed_cost = fleet_value(_refine_seed) if _refine_seed else 0
        if _seed_cost > budget and _seed_cost > 0:
            _log.warning("refine seed over budget, scaled down (cost=%d > budget=%d)",
                         _seed_cost, budget)
            _seed_scale = budget / _seed_cost
            _refine_seed = {k: int(v * _seed_scale) for k, v in _refine_seed.items()
                            if int(v * _seed_scale) > 0}
        if not _refine_seed:
            _log.warning("refine seed empty after normalisation - falling back to greedy")
            _refine_seed = None

    # Phase A: greedy (or use provided seed_fleet for refinement)
    if base_fleet and _ga_budget == 0:
        _log.info('--- Phase A: SKIPPED (0.0x simulation, no additions budget) ---')
        from ogame_optimizer.optimizer.greedy import GreedyResult
        greedy_result = GreedyResult(fleet={}, estimated_loss=0.0, time_elapsed=0.0)
    elif _refine_seed:
        _log.info("--- Phase A: Using provided seed_fleet (refinement) ---")
        from ogame_optimizer.optimizer.greedy import GreedyResult
        greedy_result = GreedyResult(fleet=dict(_refine_seed), estimated_loss=0.0, time_elapsed=0.0)
        # Evaluate it to get baseline loss (merged with base for combat)
        from ogame_optimizer.optimizer.greedy import _evaluate_single
        _seed_eval_fleet = _merge_fleet(base_fleet, _refine_seed) if base_fleet else _refine_seed
        greedy_result.estimated_loss = _evaluate_single(
            _seed_eval_fleet, enemy_fleet, enemy_defenses, enemy_tech, attacker_tech, base_seed
        )
    else:
        _log.info("--- Phase A: Greedy (budget=1.0s) ---")
        greedy_result = greedy_optimize(
            enemy_fleet=enemy_fleet,
            enemy_defenses=enemy_defenses,
            enemy_tech=enemy_tech,
            attacker_tech=attacker_tech,
            budget=_ga_budget,
            mode=mode,
            seed=base_seed,
            time_budget_s=1.0,
        )
    t1 = time.time()
    _log.info("Phase A done in %.2fs: seed_fleet=%s loss=%.0f",
              t1 - t0, greedy_result.fleet, greedy_result.estimated_loss)

    # Phase B: Multi-START + multi-round GA
    # Try different fleet compositions (greedy LF-heavy, all-cruiser, all-BC, balanced)
    # Then refine the best one with increasing fidelity
    _log.info("--- Phase B: Multi-start GA (budget=%.1fs) ---", ga_time_budget)

    # Zero out excluded ships from greedy seed
    if exclude_ships:
        _log.info("Excluding ships from optimization: %s", exclude_ships)
        for s in exclude_ships:
            if s in greedy_result.fleet:
                greedy_result.fleet[s] = 0
        greedy_result.fleet = {k: v for k, v in greedy_result.fleet.items() if v > 0}

    from ogame_optimizer.core.fleet import SHIPS_COST, fleet_value as _fv2
    from ogame_optimizer.optimizer.genetic import GAConfig, _drift_bounds_for_seed
    from ogame_optimizer.optimizer.progressive_seeds import generate_progressive_seeds

    # Generate data-driven starting compositions via progressive seeding
    # Phase 0: pure single-type fleets (all-BC, all-Destroyer, etc.)
    # Phase 1: 50/50 two-type combos of the best singles
    _log.info("--- Progressive seeding (budget=%d) ---", budget)
    progressive = generate_progressive_seeds(
        enemy_fleet=enemy_fleet,
        enemy_defenses=enemy_defenses,
        budget=_ga_budget,
        attacker_tech=attacker_tech,
        enemy_tech=enemy_tech,
        debris_pct=debris_pct,
        deuterium_in_debris=deuterium_in_debris,
        exclude_ships=exclude_ships,
        base_seed=base_seed,
    )

    seeds = {}
    # Always keep greedy as a seed (different algorithm - counter-ratio based)
    if greedy_result.fleet:
        seeds["greedy"] = greedy_result.fleet
    # Add progressive seeds (named by their composition)
    for i, prog_fleet in enumerate(progressive):
        name = "prog_" + "_".join(sorted(prog_fleet.keys())[:2])
        seeds[name] = prog_fleet
    # Remove any seed that's empty
    seeds = {name: s for name, s in seeds.items() if s}

    _log.info("Multi-start seeds: %s", {k: f"{sum(v.values())} ships" for k, v in seeds.items()})

    # Track global best (merged = base + additions for combat; additions
    # tracked separately so GA seeds / drift bounds operate on additions only).
    global_best_additions = dict(greedy_result.fleet)
    global_best_fleet = _merge_fleet(base_fleet, greedy_result.fleet) if base_fleet else dict(greedy_result.fleet)
    if base_fleet:
        _log.info('DIAG: post-greedy global_best_fleet=%s (cost=%d)', {k:v for k,v in global_best_fleet.items() if v}, fleet_value(global_best_fleet) if global_best_fleet else 0)
    # Validate greedy baseline with proper simulation count (not single-sim artifact)
    greedy_validation = simulate_batch(
        attacker=global_best_fleet,
        defender=enemy_fleet,
        defender_defenses=enemy_defenses,
        attacker_tech=attacker_tech,
        defender_tech=enemy_tech,
        n_sims=200,
        base_seed=base_seed + 7777,
    )
    # Apply resource-preference penalty (composition objective). Score the
    # fleet combat actually sees (base + additions merge), matching how GA
    # candidates are penalised on their merged fleets.
    _greedy_penalty = resource_preference_penalty(global_best_fleet, resource_weights, preference_beta)
    global_best_loss = float(greedy_validation.get("mean_attacker_loss", greedy_result.estimated_loss)) * _loss_scale + _greedy_penalty
    _log.info("Baseline (greedy, validated 200 sims): eff_loss=%.0f (raw=%.0f + penalty=%.0f) win=%.0f%%",
              global_best_loss,
              float(greedy_validation.get("mean_attacker_loss", greedy_result.estimated_loss)) * _loss_scale,
              _greedy_penalty,
              float(greedy_validation.get("win_probability", 0)) * 100)

    # Phase B1: Quick exploration from each seed (parallel strategies)
    explore_time = min(ga_time_budget * 0.15, 2.0)  # 15% of budget per seed
    # Alternatives harvest: every per-seed best computed below is retained
    # (name, additions, merged fleet, validated loss) - zero cost, purely
    # additive bookkeeping consumed by _generate_alternatives later.
    _alt_seed_candidates: List[tuple] = []
    for seed_name, seed_fleet in seeds.items():
        if explore_time < 0.5:
            continue
        seed_drift = _drift_bounds_for_seed(seed_fleet, budget=_ga_budget)
        if exclude_ships:
            for s in exclude_ships:
                seed_drift[s] = (0, 0)

        _log.info("  Start '%s': %.1fs explore", seed_name, explore_time)
        ga_round = genetic_optimize(
            seed_fleet=seed_fleet,
            enemy_fleet=enemy_fleet,
            enemy_defenses=enemy_defenses,
            enemy_tech=enemy_tech,
            attacker_tech=attacker_tech,
            budget=_ga_budget,
            mode=mode,
            config=GAConfig(
                time_budget_seconds=explore_time, sims_per_eval=20, population_size=20,
                mutation_rate=0.30, mutation_step_fraction=0.30, macro_mutation_rate=0.20,
                reallocate_rate=0.30,
            ),
            base_seed=base_seed + abs(hash(seed_name)) % 9999,
            drift_bounds=seed_drift,
            loss_scale=_loss_scale,
            resource_weights=resource_weights,
            preference_beta=preference_beta,
            min_gain_pct=min_gain_pct,
            base_fleet=base_fleet,
        )

        # Quick validate (merge base for combat)
        _ga_merged = _merge_fleet(base_fleet, ga_round.best_fleet) if base_fleet else ga_round.best_fleet
        validation = simulate_batch(
            attacker=_ga_merged,
            defender=enemy_fleet,
            defender_defenses=enemy_defenses,
            attacker_tech=attacker_tech,
            defender_tech=enemy_tech,
            n_sims=100,
            base_seed=base_seed + 7777,
        )
        _round_penalty = resource_preference_penalty(_ga_merged, resource_weights, preference_beta)
        validated_loss = float(validation.get("mean_attacker_loss", float("inf"))) * _loss_scale + _round_penalty
        _alt_seed_candidates.append((seed_name, dict(ga_round.best_fleet), dict(_ga_merged), validated_loss))

        if validated_loss < global_best_loss:
            global_best_additions = dict(ga_round.best_fleet)
            global_best_fleet = dict(_ga_merged)
            global_best_loss = validated_loss
            _log.info("  Start '%s': IMPROVED to %.0f (raw=%.0f + penalty=%.0f)", seed_name, global_best_loss, float(validation.get("mean_attacker_loss", 0)) * _loss_scale, _round_penalty)
        else:
            _log.info("  Start '%s': %.0f (no improvement)", seed_name, validated_loss)

    # Phase B2: Refine the best seed with increasing fidelity
    best_drift = _drift_bounds_for_seed(global_best_additions, budget=_ga_budget)
    if exclude_ships:
        for s in exclude_ships:
            best_drift[s] = (0, 0)

    refine_time = ga_time_budget - explore_time * len(seeds)
    # Annealed high-variance schedule: refine still explores composition
    # (moderate steps + reallocation), polish converges with small steps.
    # (name, time_frac, sims/eval, mut_rate, step_frac, macro_rate, realloc_rate)
    rounds = [
        ("refine", 0.50, 50,  0.15, 0.18, 0.12, 0.20),
        ("polish", 0.50, 100, 0.08, 0.10, 0.05, 0.10),
    ]
    for rname, t_frac, sims_eval, mut_rate, step_frac, macro_rate, realloc_rate in rounds:
        t_alloc = refine_time * t_frac
        if t_alloc < 0.5:
            continue
        _log.info("  Round '%s': %.1fs, %d sims/eval, mut=%.2f step=%.2f macro=%.2f realloc=%.2f",
                  rname, t_alloc, sims_eval, mut_rate, step_frac, macro_rate, realloc_rate)
        ga_round = genetic_optimize(
            seed_fleet=global_best_additions,
            enemy_fleet=enemy_fleet,
            enemy_defenses=enemy_defenses,
            enemy_tech=enemy_tech,
            attacker_tech=attacker_tech,
            budget=_ga_budget,
            mode=mode,
            config=GAConfig(
                time_budget_seconds=t_alloc, sims_per_eval=sims_eval, population_size=30,
                mutation_rate=mut_rate, mutation_step_fraction=step_frac,
                macro_mutation_rate=macro_rate, reallocate_rate=realloc_rate,
            ),
            base_seed=base_seed + abs(hash(rname)) % 9999,
            drift_bounds=best_drift,
            loss_scale=_loss_scale,
            resource_weights=resource_weights,
            preference_beta=preference_beta,
            min_gain_pct=min_gain_pct,
            base_fleet=base_fleet,
        )

        # Validate merged fleet (base + additions)
        _ga_merged = _merge_fleet(base_fleet, ga_round.best_fleet) if base_fleet else ga_round.best_fleet
        validation = simulate_batch(
            attacker=_ga_merged,
            defender=enemy_fleet,
            defender_defenses=enemy_defenses,
            attacker_tech=attacker_tech,
            defender_tech=enemy_tech,
            n_sims=200,
            base_seed=base_seed + 7777,
            debris_pct=debris_pct,
            deuterium_in_debris=deuterium_in_debris,
        )
        _round_penalty = resource_preference_penalty(_ga_merged, resource_weights, preference_beta)
        validated_loss = float(validation.get("mean_attacker_loss", float("inf"))) * _loss_scale + _round_penalty

        if validated_loss < global_best_loss:
            global_best_additions = dict(ga_round.best_fleet)
            global_best_fleet = dict(_ga_merged)
            global_best_loss = validated_loss
            _log.info("  Round '%s': IMPROVED to %.0f (raw=%.0f + penalty=%.0f)", rname, global_best_loss, float(validation.get("mean_attacker_loss", 0)) * _loss_scale, _round_penalty)
        else:
            _log.info("  Round '%s': no improvement", rname)

    class _Compat:
        pass
    ga_result = _Compat()
    ga_result.best_fleet = global_best_fleet
    ga_result.best_fitness = -global_best_loss
    if base_fleet:
        _log.info('DIAG: pre-final ga_result.best_fleet=%s', {k: v for k, v in ga_result.best_fleet.items() if v})

    t2 = time.time()
    _log.info("Phase B done in %.2fs: best_loss=%.0f", t2 - t1, global_best_loss)

    # Strict budget enforcement: proportional scale if over.
    # In base_fleet mode, ONLY the additions are scaled - the base fleet is a
    # sunk cost (player already built it) and must be preserved as-is.
    from ogame_optimizer.core.fleet import fleet_value as _fv
    if base_fleet:
        _additions = {s: max(0, ga_result.best_fleet.get(s, 0) - base_fleet.get(s, 0))
                      for s in ga_result.best_fleet}
        _add_cost = _fv(_additions)
        if _add_cost > _ga_budget and _add_cost > 0:
            _log.warning("Additions over budget: %d > %d, scaling down (base preserved)", _add_cost, _ga_budget)
            _scale = _ga_budget / _add_cost
            _additions = {k: max(0, int(v * _scale)) for k, v in _additions.items() if int(v * _scale) > 0}
            ga_result.best_fleet = _merge_fleet(base_fleet, _additions)
            _log.info("Additions scaled to: %d (base preserved at %d)", _fv(_additions), base_cost)
        else:
            _log.info("Additions within budget: %d <= %d (base preserved)", _add_cost, _ga_budget)
    else:
        _fleet_val = _fv(ga_result.best_fleet)
        if _fleet_val > budget and _fleet_val > 0:
            _log.warning("Fleet over budget: %d > %d, scaling down", _fleet_val, budget)
            scale = budget / _fleet_val
            _fleet = {k: max(0, int(v * scale)) for k, v in ga_result.best_fleet.items() if int(v * scale) > 0}
            ga_result.best_fleet = _fleet
            _log.info("Budget scaled to: %d", _fv(ga_result.best_fleet))

    # Phase C: Prune dead-weight ships and refine from the cleaned fleet.
    # The GA explores incrementally and can't reliably wholesale-eliminate a
    # ship type in 5s. The sensitivity analysis identifies exactly which ships
    # hurt (negative impact); this phase acts on that finding by dropping them,
    # redistributing their budget to the best positive-impact ship, and running
    # a short GA refinement from the pruned starting point.
    _log.info("--- Phase C: Prune & refine ---")
    _prune_t0 = time.time()
    # >= 2 types: even a two-type fleet benefits when one type is dead
    # weight (e.g. cruiser+battlecruiser vs fodder: pruning the BC
    # redistributed budget turns a 40%-win fleet into a 100%-win one).
    if ga_time_budget >= 1.5 and len([c for c in ga_result.best_fleet.values() if c > 0]) >= 2:
        _prune_base = simulate_batch(
            attacker=ga_result.best_fleet, defender=enemy_fleet,
            defender_defenses=enemy_defenses, attacker_tech=attacker_tech,
            defender_tech=enemy_tech, n_sims=200, base_seed=base_seed + 7777,
            debris_pct=debris_pct, deuterium_in_debris=deuterium_in_debris,
        )
        _prune_base_loss = float(_prune_base.get("mean_attacker_loss", 0))
        _prune_sens = _sensitivity_analysis(
            fleet=ga_result.best_fleet, enemy_fleet=enemy_fleet,
            enemy_defenses=enemy_defenses, attacker_tech=attacker_tech,
            enemy_tech=enemy_tech, base_loss=_prune_base_loss,
            debris_pct=debris_pct, deuterium_in_debris=deuterium_in_debris,
            base_seed=base_seed, n_sims=200,
            base_fleet=base_fleet if base_fleet else None,
            loss_scale=_loss_scale, resource_weights=resource_weights,
            preference_beta=preference_beta,
            max_total_seconds=20.0,
        )
        _pruned, _pruned_names = _prune_dead_weight(ga_result.best_fleet, _prune_sens)
        if _pruned and sum(_pruned.values()) > 0:
            _log.info("  Pruning %d dead-weight ships: %s", len(_pruned_names), _pruned_names)
            # Budget-enforce the pruned fleet.
            # In base_fleet mode, ONLY additions are scaled (base is sunk cost).
            if base_fleet:
                _pruned_add = {s: max(0, _pruned.get(s, 0) - base_fleet.get(s, 0))
                               for s in _pruned}
                _pa_cost = _fv(_pruned_add)
                if _pa_cost > _ga_budget and _pa_cost > 0:
                    _log.warning("  Pruned additions over budget: %d > %d, scaling", _pa_cost, _ga_budget)
                    _scale = _ga_budget / _pa_cost
                    _pruned_add = {k: max(0, int(v * _scale)) for k, v in _pruned_add.items() if int(v * _scale) > 0}
                    _pruned = _merge_fleet(base_fleet, _pruned_add)
            else:
                _pfv = _fv(_pruned)
                if _pfv > budget and _pfv > 0:
                    _scale = budget / _pfv
                    _pruned = {k: max(0, int(v * _scale)) for k, v in _pruned.items() if int(v * _scale) > 0}
            # Validate the pruned fleet DIRECTLY (no GA refinement). The
            # sensitivity analysis already identified the right move (remove
            # ship X, redistribute to the best positive-impact ship); applying
            # it and validating is more reliable than running a short GA that
            # explores AWAY from the good pruned starting point.
            _prune_val = simulate_batch(
                attacker=_pruned, defender=enemy_fleet,
                defender_defenses=enemy_defenses, attacker_tech=attacker_tech,
                defender_tech=enemy_tech, n_sims=200, base_seed=base_seed + 7777,
                debris_pct=debris_pct, deuterium_in_debris=deuterium_in_debris,
            )
            _prune_pen = resource_preference_penalty(_pruned, resource_weights, preference_beta)
            _prune_eff_loss = float(_prune_val.get("mean_attacker_loss", float("inf"))) * _loss_scale + _prune_pen
            if _prune_eff_loss < global_best_loss:
                global_best_fleet = dict(_pruned)
                global_best_loss = _prune_eff_loss
                ga_result.best_fleet = global_best_fleet
                _log.info("  Prune: IMPROVED to %.0f (was %.0f)",
                          global_best_loss, _prune_base_loss * _loss_scale + _prune_pen)
            else:
                _log.info("  Prune: no improvement (kept original)")
        else:
            _log.info("  No dead-weight to prune")
    else:
        _log.info("  Skipped (fleet has <=2 ship types)")
    _log.info("Phase C done in %.2fs", time.time() - _prune_t0)

    # Final validation
    _log.info("--- Final validation (%d sims) ---", final_sims)
    _log.info("DIAG: final validation fleet=%s", {k: v for k, v in ga_result.best_fleet.items() if v})
    final = simulate_batch(
        attacker=ga_result.best_fleet,
        defender=enemy_fleet,
        defender_defenses=enemy_defenses,
        attacker_tech=attacker_tech,
        defender_tech=enemy_tech,
        n_sims=final_sims,
        base_seed=base_seed + 9999,
        debris_pct=debris_pct,
        deuterium_in_debris=deuterium_in_debris,
    )
    t3 = time.time()
    _log.info("Validation done in %.2fs: mean_loss=%.0f stddev=%.0f win_prob=%.3f",
              t3 - t2,
              float(final.get("mean_attacker_loss", 0)),
              float(final.get("stddev_attacker_loss", 0)),
              float(final.get("win_probability", 0)))

    # === min_gain constraint: verify the recommended fleet meets the threshold ===
    # ROI = (debris_total - loss) / fleet_value
    # We compute it from the final validation result and compare against the
    # user-specified min_gain_pct. Log a WARNING if not met (optimizer failed).
    _final_fleet_value_check = _fv(ga_result.best_fleet)
    _final_debris = int(final.get("debris_total", 0))
    _final_loss = float(final.get("mean_attacker_loss", 0))
    _final_roi_pct = ((_final_debris - _final_loss) / _final_fleet_value_check * 100) if _final_fleet_value_check > 0 else 0.0
    _min_gain_met = (_final_roi_pct >= min_gain_pct) if min_gain_pct > 0 else True
    if min_gain_pct > 0:
        if _min_gain_met:
            _log.info("min_gain constraint: MET (ROI=%.1f%% >= required=%.1f%%)",
                      _final_roi_pct, min_gain_pct)
        else:
            _log.warning("min_gain constraint: NOT MET (ROI=%.1f%% < required=%.1f%%). "
                         "No fleet within budget could satisfy both win-rate and ROI constraints.",
                         _final_roi_pct, min_gain_pct)


    mean_loss_raw = float(final.get("mean_attacker_loss", 0))
    # Apply resource-preference tiebreaker to displayed loss so it matches what
    # the GA actually optimised. The raw mean_loss (without *_loss_scale) is
    # preserved as raw_loss_mean for transparency.
    _best_penalty = resource_preference_penalty(ga_result.best_fleet, resource_weights, preference_beta)
    mean_loss = mean_loss_raw * _loss_scale + _best_penalty
    stddev_loss = float(final.get("stddev_attacker_loss", 0))
    win_prob = float(final.get("win_probability", 0.0))

    # Win-threshold check: warn if the scenario is unwinnable at this budget.
    if mode == "attack":
        _win_met = win_prob >= 0.95
    else:
        _win_met = (1.0 - win_prob) >= 0.95
    if not _win_met:
        _log.warning("WIN THRESHOLD NOT MET (win_prob=%.1f%%, need >=95%% for mode=%s). "
                     "Scenario appears unwinnable at budget_multiplier=%s - the GA optimised "
                     "for the least-bad outcome (min loss / max enemy debris). "
                     "Increase budget_multiplier for a winning fleet.",
                     win_prob * 100, mode, budget_multiplier)
    stderr = stddev_loss / max(1, final_sims ** 0.5)
    ci = [mean_loss - 1.96 * stderr, mean_loss + 1.96 * stderr]

    improvement_pct = 0.0
    if greedy_result.estimated_loss > 0:
        improvement_pct = (
            (greedy_result.estimated_loss * _loss_scale - mean_loss) / max(greedy_result.estimated_loss * _loss_scale, 1) * 100.0
        )

    _log.info("=== OPTIMIZE DONE total=%.2fs greedy=%.2fs ga=%.2fs win_prob=%.3f improvement=%.1f%% ===",
              t3 - t0, t1 - t0, t2 - t1, win_prob, improvement_pct)

    from ogame_optimizer.core.fleet import fleet_value as _fv3
    _final_fv = _fv3(ga_result.best_fleet)
    _final_wv = weighted_fleet_value(ga_result.best_fleet, resource_weights)
    # Compute match score (0..1) for the response: 1.0 = perfect preference match
    _wM, _wC, _wD = resource_weights
    _max_w = max(_wM, _wC, _wD)
    if _max_w > 1.0 and _final_fv > 0:
        _weighted = weighted_fleet_value(ga_result.best_fleet, resource_weights)
        _ratio = _weighted / _final_fv
        _match_score = max(0.0, min(1.0, (_ratio - 1.0) / (_max_w - 1.0)))
    else:
        _match_score = 1.0
    _lost_pct = (mean_loss_raw / _final_fv * 100) if _final_fv > 0 else 0

    recycler_cap = int(20000 * (1 + hyperspace_tech * 0.05) * (1.25 if collector_class else 1.0))
    debris_total_val = int(final.get("debris_total", 0))
    recyclers = (debris_total_val + recycler_cap - 1) // recycler_cap if recycler_cap > 0 else 0
    # Recycler build cost (M/C/D) for the needed count
    _rec_cost = SHIPS_COST.get("recycler", (10000, 6000, 2000))
    recyclers_cost_metal = recyclers * _rec_cost[0]
    recyclers_cost_crystal = recyclers * _rec_cost[1]
    recyclers_cost_deuterium = recyclers * _rec_cost[2]
    recyclers_cost_total = recyclers_cost_metal + recyclers_cost_crystal + recyclers_cost_deuterium

    # Sensitivity analysis: which ships are critical vs dead weight?
    _log.info("--- Sensitivity analysis ---")
    fleet_analysis = _sensitivity_analysis(
        fleet=ga_result.best_fleet,
        enemy_fleet=enemy_fleet,
        enemy_defenses=enemy_defenses,
        attacker_tech=attacker_tech,
        enemy_tech=enemy_tech,
        base_loss=mean_loss_raw,
        debris_pct=debris_pct,
        deuterium_in_debris=deuterium_in_debris,
        base_seed=base_seed,
        base_fleet=base_fleet if base_fleet else None,
        loss_scale=_loss_scale, resource_weights=resource_weights,
        preference_beta=preference_beta,
        max_total_seconds=20.0,
    )

    # Compute per-ship survival rates (for shield marker in UI) using the
    # batch's averaged per-type survivors. The single-seed fast-combat call
    # was unstable (one sim != 500-sim mean), and we now have averaged
    # survivors in the final batch result.
    defender_fleet_analysis: Dict[str, Dict[str, float]] = {}
    try:
        _atk_surv_mean = final.get("attacker_survivors_mean", {}) or {}
        _def_surv_mean = final.get("defender_survivors_mean", {}) or {}
        _ships_lost = 0
        _ships_init = 0
        for _ship, _count in ga_result.best_fleet.items():
            if _count > 0:
                _surv = float(_atk_surv_mean.get(_ship, 0))
                _surv_pct = round((_surv / _count) * 100, 1)
                if _ship not in fleet_analysis:
                    fleet_analysis[_ship] = {}
                fleet_analysis[_ship]["survival_pct"] = _surv_pct
                fleet_analysis[_ship]["surviving_count"] = round(_surv)
                _ships_init += _count
                _ships_lost += max(0, _count - round(_surv))
        # Build defender analysis: per-ship surviving count and survival pct
        for _ship, _count in enemy_fleet.items():
            if _count > 0:
                _surv = float(_def_surv_mean.get(_ship, 0))
                _surv_pct = round((_surv / _count) * 100, 1)
                defender_fleet_analysis[_ship] = {
                    "count": _count,
                    "surviving_count": round(_surv),
                    "destroyed_count": int(_count - round(_surv)),
                    "survival_pct": _surv_pct,
                }
    except Exception:
        pass

    # Build defender defense analysis (per-defense survival)
    defender_defense_analysis: Dict[str, Dict[str, float]] = {}
    try:
        _def_def_surv_mean = final.get("defender_defense_survivors_mean", {}) or {}
        for _def, _count in enemy_defenses.items():
            if _count > 0:
                _surv = float(_def_def_surv_mean.get(_def, 0))
                _surv_pct = round((_surv / _count) * 100, 1)
                defender_defense_analysis[_def] = {
                    "count": _count,
                    "surviving_count": round(_surv),
                    "destroyed_count": int(_count - round(_surv)),
                    "survival_pct": _surv_pct,
                }
    except Exception:
        pass

    # Costs & kills transparency: cost splits + kill attribution estimates.
    # attribution_mean comes from the analytical engine (absent on the Rust
    # path / small fleets - the heuristic fallback inside covers those).
    _fleet_cost_split = _resource_split(ga_result.best_fleet)
    _recommended_additions = (
        {s: max(0, ga_result.best_fleet.get(s, 0) - base_fleet.get(s, 0))
         for s in ga_result.best_fleet
         if ga_result.best_fleet.get(s, 0) > base_fleet.get(s, 0)}
    ) if base_fleet else dict(ga_result.best_fleet)
    _additions_cost_split = _resource_split(_recommended_additions)
    _kill_estimates = _compute_kill_estimates(
        recommended_fleet=ga_result.best_fleet,
        enemy_fleet=enemy_fleet,
        attribution_mean=final.get("attribution_mean", {}) or {},
        enemy_defenses=enemy_defenses,
        enemy_tech=enemy_tech,
        attacker_tech=attacker_tech,
        defender_fleet_analysis=defender_fleet_analysis,
    )

    # --- Fleet alternatives: up to 2 genuinely-different compositions ------
    # Runs AFTER final validation + sensitivity so every primary number the
    # gates compare against is frozen. Any failure here must NEVER kill the
    # optimization - log and continue with an empty alternatives list.
    alternatives: List[AlternativeResult] = []
    if include_alternatives:
        try:
            alternatives = _generate_alternatives(
                primary_fleet=ga_result.best_fleet,
                primary_loss=mean_loss,
                primary_win_met=_win_met,
                defender_fleet_analysis=defender_fleet_analysis,
                enemy_fleet=enemy_fleet,
                enemy_defenses=enemy_defenses,
                enemy_tech=enemy_tech,
                attacker_tech=attacker_tech,
                mode=mode,
                base_fleet=base_fleet,
                ga_budget=_ga_budget,
                exclude_ships=exclude_ships,
                seed_candidates=_alt_seed_candidates,
                ga_time_budget=ga_time_budget,
                final_sims=final_sims,
                debris_pct=debris_pct,
                deuterium_in_debris=deuterium_in_debris,
                loss_scale=_loss_scale,
                resource_weights=resource_weights,
                preference_beta=preference_beta,
                min_gain_pct=min_gain_pct,
                base_seed=base_seed,
            )
        except Exception as alt_exc:  # noqa: BLE001 - alternatives are best-effort
            _log.error("Alternatives generation failed (continuing without): %s",
                       alt_exc, exc_info=True)
            alternatives = []

    return OptimizationResult(
        recommended_fleet=ga_result.best_fleet,
        fleet_value=_final_fv,
        fleet_lost_pct=_lost_pct,
        ships_lost_count=_ships_lost,
        ships_initial_count=_ships_init,
        debris_metal=int(final.get("debris_metal", 0)),
        net_profit=int(final.get("debris_total", 0)) - int(mean_loss_raw),
        net_profit_pct=((final.get("debris_total", 0) - mean_loss_raw) / _final_fv * 100) if _final_fv > 0 else 0,
        recyclers_needed=recyclers,
        recycler_capacity=recycler_cap,
        recyclers_cost_metal=recyclers_cost_metal,
        recyclers_cost_crystal=recyclers_cost_crystal,
        recyclers_cost_deuterium=recyclers_cost_deuterium,
        recyclers_cost_total=recyclers_cost_total,
        debris_crystal=int(final.get("debris_crystal", 0)),
        debris_deuterium=int(final.get("debris_deuterium", 0)),
        debris_total=int(final.get("debris_total", 0)),
        raw_loss_mean=mean_loss_raw,
        min_gain_required=min_gain_pct,
        min_gain_met=_min_gain_met,
        actual_roi_pct=_final_roi_pct,
        win_threshold_met=_win_met,
        resource_weights=tuple(resource_weights),
        preference_beta=preference_beta,
        fleet_weighted_value=_final_wv,
        resource_preference_penalty=_best_penalty,
        resource_preference_match_score=_match_score,
        expected_loss_mean=mean_loss,
        expected_loss_stddev=stddev_loss,
        win_probability=win_prob,
        confidence_interval_95=ci,
        sims_run_final=final_sims,
        greedy_baseline_loss=float(greedy_result.estimated_loss),
        ga_improvement_pct=improvement_pct,
        time_elapsed_greedy=t1 - t0,
        time_elapsed_ga=t2 - t1,
        total_time=t3 - t0,
        seed_used=base_seed,
        mode=mode,
        fleet_analysis=fleet_analysis,
        defender_fleet_analysis=defender_fleet_analysis,
        defender_defense_analysis=defender_defense_analysis,
        base_fleet=base_fleet if base_fleet else {},
        base_fleet_cost=base_cost,
        base_fleet_count=base_count,
        recommended_additions=_recommended_additions,
        fleet_cost_metal=_fleet_cost_split[0],
        fleet_cost_crystal=_fleet_cost_split[1],
        fleet_cost_deuterium=_fleet_cost_split[2],
        additions_cost_metal=_additions_cost_split[0],
        additions_cost_crystal=_additions_cost_split[1],
        additions_cost_deuterium=_additions_cost_split[2],
        kill_estimates=_kill_estimates,
        alternatives=alternatives,
    )


__all__ = ["optimize", "OptimizationResult", "AlternativeResult"]

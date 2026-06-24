"""Python wrapper for the Rust combat core (PyO3 bridge).

Exposes:
- simulate_combat: single battle
- simulate_batch: N battles, aggregate stats
- evaluate_population: GA batch API (multiple attackers vs same defender)

The Rust extension (``ogame_optimizer._ogame_combat``) is built via
maturin from ``src/lib.rs``.  Its public pyfunctions are suffixed with
``_py``; this module re-exports them under the canonical Task 6 names
and accepts the project's public Python convention of **snake_case**
keys (``light_fighter``, ``rocket_launcher``) -- translating to
**PascalCase** at the boundary in BOTH directions.  PascalCase is also
accepted unchanged for callers that prefer zero translation overhead.
"""
from __future__ import annotations

from typing import Any, Optional

try:
    from ogame_optimizer import _ogame_combat as _rust
except ImportError:
    import ogame_combat as _rust  # type: ignore


# ---------------------------------------------------------------------------
# snake_case <-> PascalCase translation tables.
#
# The PascalCase strings MUST match ``ShipType::as_str()`` and
# ``DefenseType::as_str()`` in ``src/ships.rs`` exactly.
# ---------------------------------------------------------------------------

_SHIP_SNAKE_TO_PASCAL: dict[str, str] = {
    "small_cargo":     "SmallCargo",
    "large_cargo":     "LargeCargo",
    "light_fighter":   "LightFighter",
    "heavy_fighter":   "HeavyFighter",
    "cruiser":         "Cruiser",
    "battleship":      "Battleship",
    "battlecruiser":   "Battlecruiser",
    "bomber":          "Bomber",
    "destroyer":       "Destroyer",
    "deathstar":       "Deathstar",
    "espionage_probe": "EspionageProbe",
    "pathfinder": "Pathfinder",
    "recycler": "Recycler",
    "reaper": "Reaper",
}

_SHIP_PASCAL_TO_SNAKE: dict[str, str] = {v: k for k, v in _SHIP_SNAKE_TO_PASCAL.items()}

_DEFENSE_SNAKE_TO_PASCAL: dict[str, str] = {
    "rocket_launcher":   "RocketLauncher",
    "light_laser":       "LightLaser",
    "heavy_laser":       "HeavyLaser",
    "gauss_cannon":      "GaussCannon",
    "ion_cannon":        "IonCannon",
    "plasma_turret":     "PlasmaTurret",
    "small_shield_dome": "SmallShieldDome",
    "large_shield_dome": "LargeShieldDome",
}

_DEFENSE_PASCAL_TO_SNAKE: dict[str, str] = {v: k for k, v in _DEFENSE_SNAKE_TO_PASCAL.items()}


def _normalize_ship_keys(fleet: Optional[dict]) -> dict:
    """Translate snake_case ship keys to PascalCase; pass PascalCase through."""
    out: dict[str, int] = {}
    for k, v in (fleet or {}).items():
        if k in _SHIP_SNAKE_TO_PASCAL:
            out[_SHIP_SNAKE_TO_PASCAL[k]] = int(v)
        else:
            out[k] = int(v)
    return out


def _normalize_defense_keys(defenses: Optional[dict]) -> dict:
    """Translate snake_case defense keys to PascalCase; pass PascalCase through."""
    out: dict[str, int] = {}
    for k, v in (defenses or {}).items():
        if k in _DEFENSE_SNAKE_TO_PASCAL:
            out[_DEFENSE_SNAKE_TO_PASCAL[k]] = int(v)
        else:
            out[k] = int(v)
    return out


def _translate_ship_keys_back(fleet: Optional[dict]) -> dict:
    """Translate PascalCase ship keys back to snake_case; pass snake_case through."""
    out: dict[str, int] = {}
    for k, v in (fleet or {}).items():
        out[_SHIP_PASCAL_TO_SNAKE.get(k, k)] = int(v)
    return out


def _translate_defense_keys_back(defenses: Optional[dict]) -> dict:
    """Translate PascalCase defense keys back to snake_case; pass snake_case through."""
    out: dict[str, int] = {}
    for k, v in (defenses or {}).items():
        out[_DEFENSE_PASCAL_TO_SNAKE.get(k, k)] = int(v)
    return out


def _translate_result(result: Optional[dict]) -> dict:
    """Wrap a Rust combat result so all fleet/defense dicts use snake_case keys."""
    if not result:
        return result or {}
    if "attacker_survivors" in result:
        result["attacker_survivors"] = _translate_ship_keys_back(result["attacker_survivors"])
    if "defender_survivors" in result:
        result["defender_survivors"] = _translate_ship_keys_back(result["defender_survivors"])
    if "defender_defense_survivors" in result:
        result["defender_defense_survivors"] = _translate_defense_keys_back(result["defender_defense_survivors"])
    return result


from ogame_optimizer.core.fast_combat import (
    simulate_combat_fast, simulate_batch_fast, evaluate_population_fast,
    should_use_fast,
)


_RUST_UNKNOWN_SHIPS = {"pathfinder", "recycler", "Pathfinder", "Recycler"}


def _strip_unknown_for_rust(fleet):
    """Remove only pathfinder/recycler (known to Python but not Rust core).
    Truly unknown ships pass through so Rust raises a clear ValueError."""
    return {k: v for k, v in (fleet or {}).items() if k not in _RUST_UNKNOWN_SHIPS}


def _to_tech_tuple(tech) -> tuple:
    return tuple(tech)


def simulate_combat(
    attacker: dict,
    defender: dict,
    defender_defenses: Optional[dict] = None,
    attacker_tech=(0, 0, 0),
    defender_tech=(0, 0, 0),
    seed: int = 42,
) -> dict:
    """Simulate a single OGame combat.

    Args:
        attacker: ``{ship_name: count}`` e.g. ``{"light_fighter": 100}`` or
            ``{"LightFighter": 100}`` (both accepted).
        defender: ``{ship_name: count}``
        defender_defenses: ``{defense_name: count}`` e.g. ``{"rocket_launcher": 50}``
        attacker_tech: ``(weapon, shield, armor)`` levels
        defender_tech: ``(weapon, shield, armor)`` levels
        seed: RNG seed (deterministic)

    Returns:
        Dict with ``winner`` (``"Attacker" | "Defender" | "Draw"``),
        ``rounds_fought``, ``attacker_survivors``, ``defender_survivors``,
        ``defender_defense_survivors``, ``debris_metal``, ``debris_crystal``.
        All nested fleet/defense dicts use snake_case keys.
    """
    # Fast path for large fleets
    if should_use_fast(attacker, defender, defender_defenses):
        return simulate_combat_fast(attacker, defender, defender_defenses, attacker_tech, defender_tech, int(seed))
    return _translate_result(_rust.simulate_combat_py(
        _normalize_ship_keys(_strip_unknown_for_rust(attacker)),
        _normalize_ship_keys(_strip_unknown_for_rust(defender)),
        _normalize_defense_keys(defender_defenses),
        _to_tech_tuple(attacker_tech),
        _to_tech_tuple(defender_tech),
        int(seed),
    ))


def simulate_batch(
    attacker: dict,
    defender: dict,
    defender_defenses: Optional[dict] = None,
    attacker_tech=(0, 0, 0),
    defender_tech=(0, 0, 0),
    n_sims: int = 100,
    base_seed: int = 42,
    debris_pct: float = 0.30,
    deuterium_in_debris: bool = False,
) -> dict:
    """Run N simulations and return aggregate statistics.

    The GIL is released for the entire simulation loop; this is a single
    Python->Rust call regardless of ``n_sims``.

    Returns dict with ``mean_attacker_loss``, ``stddev_attacker_loss``,
    ``mean_defender_loss``, ``win_probability``, ``wins``, ``losses``,
    ``draws``, ``sims_run``, ``seed_used``.
    """
    # Fast path for large fleets
    if should_use_fast(attacker, defender, defender_defenses):
        return simulate_batch_fast(attacker, defender, defender_defenses, attacker_tech, defender_tech, int(n_sims), int(base_seed), debris_pct=debris_pct, deuterium_in_debris=deuterium_in_debris)
    result = _translate_result(_rust.simulate_batch_py(
        _normalize_ship_keys(_strip_unknown_for_rust(attacker)),
        _normalize_ship_keys(_strip_unknown_for_rust(defender)),
        _normalize_defense_keys(defender_defenses),
        _to_tech_tuple(attacker_tech),
        _to_tech_tuple(defender_tech),
        int(n_sims),
        int(base_seed),
    ))

    # Compute averaged debris and per-type mean survivors across K detail sims.
    # Previous behaviour called a SINGLE sim for per-type survivors, then
    # computed debris from those survivors — which made `debris_total`
    # decoupled from the batch's averaged `mean_attacker_loss` /
    # `win_probability`. When that lone sim randomly produced a different
    # outcome (e.g. attacker won in the debris sim, lost in 499/500 batch
    # sims), the displayed debris was nonsense. See issue: "Net Profit
    # positive when Win Probability = 0%". Fast path is already correct
    # (simulate_batch_fast averages per sim), only the Rust path was wrong.
    from collections import defaultdict
    try:
        from ogame_optimizer.core.fast_combat import calculate_debris
        _stripped_a = _strip_unknown_for_rust(attacker)
        _stripped_d = _strip_unknown_for_rust(defender)
        # Cap K for performance: enough to be stable, cheap enough not to
        # dominate the cost of the call (the batch itself does the heavy
        # lifting in Rust). 50 is the same cap the fast path uses
        # implicitly by virtue of running its own loop.
        K = min(int(n_sims), 50)
        if K < 1:
            K = 1
        _dm_sum = _dc_sum = _dd_sum = 0
        _atk_surv: dict[str, float] = defaultdict(float)
        _def_surv: dict[str, float] = defaultdict(float)
        for k in range(K):
            detail = _translate_result(_rust.simulate_combat_py(
                _normalize_ship_keys(_strip_unknown_for_rust(attacker)),
                _normalize_ship_keys(_strip_unknown_for_rust(defender)),
                _normalize_defense_keys(defender_defenses),
                _to_tech_tuple(attacker_tech),
                _to_tech_tuple(defender_tech),
                int(base_seed) + k,  # distinct seed per sim
            ))
            _debris = calculate_debris(
                _stripped_a, detail.get("attacker_survivors", {}),
                _stripped_d, detail.get("defender_survivors", {}),
                defender_defenses, detail.get("defender_defense_survivors", {}),
                debris_pct, deuterium_in_debris,
            )
            _dm_sum += _debris["debris_metal"]
            _dc_sum += _debris["debris_crystal"]
            _dd_sum += _debris["debris_deuterium"]
            for _s, _n in (detail.get("attacker_survivors") or {}).items():
                _atk_surv[_s] += _n
            for _s, _n in (detail.get("defender_survivors") or {}).items():
                _def_surv[_s] += _n
        result["debris_metal"] = int(_dm_sum / K)
        result["debris_crystal"] = int(_dc_sum / K)
        result["debris_deuterium"] = int(_dd_sum / K)
        result["debris_total"] = int((_dm_sum + _dc_sum + _dd_sum) / K)
        # Per-type MEAN survivors (fractional) — callers can convert to
        # integer counts or survival percentages as needed.
        result["attacker_survivors_mean"] = {s: n / K for s, n in _atk_surv.items()}
        result["defender_survivors_mean"] = {s: n / K for s, n in _def_surv.items()}
    except Exception:
        result["debris_metal"] = 0
        result["debris_crystal"] = 0
        result["debris_deuterium"] = 0
        result["debris_total"] = 0
        result["attacker_survivors_mean"] = {}
        result["defender_survivors_mean"] = {}

    return result


def evaluate_population(
    attacker_fleets: list,
    defender: dict,
    defender_defenses: Optional[dict] = None,
    attacker_tech=(0, 0, 0),
    defender_tech=(0, 0, 0),
    n_sims_per_fleet: int = 100,
    base_seed: int = 42,
) -> list:
    """Evaluate a population of attacker fleets vs the same defender.

    Each attacker fleet is simulated ``n_sims_per_fleet`` times in a
    single Rust call (no per-fleet round-trip).  Returns a list of result
    dicts (one per attacker fleet), each with ``mean_attacker_loss``,
    ``stddev_attacker_loss``, ``win_probability``, ``sims_run``.
    """
    # Fast path for large fleets
    if any(should_use_fast(f, defender, defender_defenses) for f in attacker_fleets):
        return evaluate_population_fast(attacker_fleets, defender, defender_defenses, attacker_tech, defender_tech, int(n_sims_per_fleet), int(base_seed))
    raw_results = _rust.evaluate_population_py(
        [_normalize_ship_keys(_strip_unknown_for_rust(f)) for f in attacker_fleets],
        _normalize_ship_keys(defender),
        _normalize_defense_keys(defender_defenses),
        _to_tech_tuple(attacker_tech),
        _to_tech_tuple(defender_tech),
        int(n_sims_per_fleet),
        int(base_seed),
    )
    return [_translate_result(r) for r in raw_results]


__all__ = ["simulate_combat", "simulate_batch", "evaluate_population"]

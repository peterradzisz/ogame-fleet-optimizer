"""OGame fleet, defense, and budget primitives.

This module is the Python-side authoritative source for the optimizer's
budget math.  The Rust combat simulator (``src/ships.rs``) carries its own
copy of the unit costs for parity checks; if the two drift apart, the
``test_ships_cost_matches_rust_source_of_truth`` test in
``python_tests/test_fleet.py`` will fail immediately.

Keys
----
Public ship and defense keys are lowercase snake_case (``"light_fighter"``,
``"rocket_launcher"``, ``"deathstar"``).  The Rust core uses PascalCase
identifiers (``"LightFighter"``); conversion happens at the PyO3 bridge
(Task 6) — never silently inside this module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional


# ---------------------------------------------------------------------------
# Cost tables — verified against src/ships.rs.
# ---------------------------------------------------------------------------

# Each value is ``(metal, crystal, deuterium)`` raw resources per unit, with
# no build multiplier applied.  Order matches ``ShipType::ALL`` in ships.rs.
SHIPS_COST: Dict[str, tuple] = {
    "small_cargo":      (2_000,        2_000,      0),
    "large_cargo":      (6_000,        6_000,      0),
    "light_fighter":    (3_000,        1_000,      0),
    "heavy_fighter":    (6_000,        4_000,      0),
    "cruiser":          (20_000,       7_000,      2_000),
    "battleship":       (45_000,       15_000,     0),
    "battlecruiser":    (30_000,       40_000,     15_000),
    "bomber":           (50_000,       25_000,     15_000),
    "destroyer":        (60_000,       50_000,     15_000),
    "deathstar":        (5_000_000,    4_000_000,  1_000_000),
    "pathfinder": (8000, 15000, 8000),  # FIXED per Fandom,
    "recycler": (10000, 6000, 2000),
    "espionage_probe":  (0,            1_000,      0),
    "reaper":           (85_000,        55_000,     20_000),  # FIXED per Fandom
}

# Order matches ``DefenseType::ALL`` in ships.rs.
DEFENSES_COST: Dict[str, tuple] = {
    "rocket_launcher":     (2_000,    0,          0),
    "light_laser":         (1_500,    500,        0),
    "heavy_laser":         (6_000,    2_000,      0),
    "gauss_cannon":        (20_000,   15_000,     2_000),
    "ion_cannon":          (5_000,    3_000,      0),
    "plasma_turret":       (50_000,   50_000,     30_000),
    "small_shield_dome":   (10_000,   10_000,     0),
    "large_shield_dome":   (50_000,   50_000,     0),
}

# Tolerance for "is this multiplier on the 0.5 grid?" — accommodates IEEE 754
# rounding noise from values like ``0.1 + 0.1 + 0.1 + 0.1 + 0.1``.
_GRID_TOLERANCE: float = 1e-9


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit_cost(unit_key: str, table: Mapping[str, tuple]) -> int:
    """Return the per-unit M+C+D total for ``unit_key`` from ``table``."""
    try:
        metal, crystal, deuterium = table[unit_key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown unit key {unit_key!r}; expected one of {sorted(table)!r}"
        ) from exc
    return metal + crystal + deuterium


def _validate_counts(
    counts: Mapping[str, int],
    table: Mapping[str, tuple],
    label: str,
) -> None:
    """Raise ``ValueError`` on unknown keys or non-int / negative counts."""
    for key, count in counts.items():
        if key not in table:
            raise ValueError(
                f"Unknown {label} key {key!r}; expected one of {sorted(table)!r}"
            )
        # ``bool`` is a subclass of ``int`` in Python; reject it explicitly so
        # ``Fleet(ships={'light_fighter': True})`` is a clear error rather
        # than a silently-misleading ``1``.
        if isinstance(count, bool) or not isinstance(count, int):
            raise ValueError(
                f"{label} count for {key!r} must be an int, got {type(count).__name__}"
            )
        if count < 0:
            raise ValueError(
                f"{label} count for {key!r} must be non-negative, got {count}"
            )


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class Fleet:
    """A combat fleet composition: ``Dict[ship_key, count]``.

    ``ship_key`` must be one of the keys in :data:`SHIPS_COST`.  Counts are
    non-negative Python ``int`` (no floats, no bools).
    """

    ships: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_counts(self.ships, SHIPS_COST, label="ship")

    def total_cost(self) -> int:
        """Return raw M+C+D total cost of this fleet."""
        return sum(
            count * _unit_cost(key, SHIPS_COST)
            for key, count in self.ships.items()
        )


# ---------------------------------------------------------------------------
# Free-function API (used by optimizer heuristics that work on plain dicts)
# ---------------------------------------------------------------------------


def fleet_value(
    fleet: Mapping[str, int],
    defenses: Optional[Mapping[str, int]] = None,
) -> int:
    """Return raw M+C+D value of ``fleet`` plus ``defenses`` (both optional).

    Used to compute "what does this enemy cost to replace?" for the budget
    ceiling.  ``defenses`` is ``None`` when the player is scouting a fleet
    with no stationary defenses (common in ACS / expedition scenarios).
    """
    _validate_counts(fleet, SHIPS_COST, label="ship")
    total = sum(
        count * _unit_cost(key, SHIPS_COST) for key, count in fleet.items()
    )
    if defenses is not None:
        _validate_counts(defenses, DEFENSES_COST, label="defense")
        total += sum(
            count * _unit_cost(key, DEFENSES_COST)
            for key, count in defenses.items()
        )
    return total


def weighted_fleet_value(
    fleet: Mapping[str, int],
    weights: tuple[float, float, float],
    defenses: Optional[Mapping[str, int]] = None,
) -> float:
    """Return resource-weighted value of ``fleet`` (+ optional ``defenses``).

    Formula: ``sum(count * (w_M * M_cost + w_C * C_cost + w_D * D_cost))``.

    Unlike :func:`fleet_value`, which sums metal + crystal + deuterium as
    interchangeable units, this honours a per-resource preference vector
    so callers can say "I value metal 2x because I'm overflowing on it".
    Default weights ``(1, 1, 1)`` reproduce the raw value exactly.

    Used by the optimizer's soft resource-preference tiebreaker
    (see :func:`resource_preference_multiplier`). Does NOT affect the
    raw budget ceiling — all recommended fleets remain buildable.
    """
    wM, wC, wD = (float(weights[0]), float(weights[1]), float(weights[2]))
    if wM < 0 or wC < 0 or wD < 0:
        raise ValueError(
            f"weights must be non-negative, got {weights!r}"
        )
    _validate_counts(fleet, SHIPS_COST, label="ship")
    total = 0.0
    for key, count in fleet.items():
        m, c, d = SHIPS_COST[key]
        total += count * (wM * m + wC * c + wD * d)
    if defenses is not None:
        _validate_counts(defenses, DEFENSES_COST, label="defense")
        for key, count in defenses.items():
            m, c, d = DEFENSES_COST[key]
            total += count * (wM * m + wC * c + wD * d)
    return total


def resource_preference_penalty(
    fleet: Mapping[str, int],
    weights: tuple[float, float, float],
    beta: float = 0.05,
) -> float:
    """Return an additive resource penalty in raw resource units.

    The penalty is *added* to the fleet's effective loss so the GA
    actually prefers compositions that match the user's resource bias,
    instead of merely tiebreaking between equally-good fleets (which was
    the limitation of the old multiplicative approach — it could not
    budge a near-zero-loss solution like "24 deathstars").

    Behaviour:
      - ``beta == 0``        ⇒ penalty = 0 (current behaviour, no bias).
      - ``beta == 0.05``     ⇒ up to 5 % of fleet cost penalty for a
                                fleet whose resource composition is the
                                *opposite* of the user's preference.
      - ``beta == 0.20``     ⇒ up to 20 % of cost (strong bias — the
                                optimizer will trade significant combat
                                effectiveness for resource composition).
      - Uniform weights (e.g. ``(1, 1, 1)``) ⇒ penalty = 0 always.

    ``preference_match_score`` is a normalised 0-1 measure of how well
    the fleet's resource mix matches the weights. 0 = no match (e.g.
    pure crystal under 2:1:1), 1 = perfect match (pure metal under
    2:1:1). The penalty is ``beta * fleet_cost * (1 - score)``.

    With 2:1:1 weights and ``beta=0.15``:
      - Pure LF (75 % metal) → score 0.75 → penalty ~3.75 % of cost
      - Pure Cruiser (69 %) → score 0.69 → penalty ~4.7 % of cost
      - Pure Destroyer (48 %)→ score 0.48 → penalty ~7.8 % of cost
      - Pure Probe (0 %)    → score 0.00 → penalty 15.0 % of cost
    """
    if beta <= 0:
        return 0.0

    wM, wC, wD = (float(weights[0]), float(weights[1]), float(weights[2]))
    max_w = max(wM, wC, wD)
    # If the highest weight is 1 (or all weights equal / sub-1), the user
    # has no expressed preference and the penalty is zero by design.
    if max_w <= 1.0:
        return 0.0

    raw = fleet_value(fleet)
    if raw <= 0:
        return 0.0

    weighted = weighted_fleet_value(fleet, weights)
    # ratio ranges from 1.0 (fleet has none of the max-weighted resource)
    # to max_w (fleet is 100 % the max-weighted resource).
    ratio = weighted / raw
    score = (ratio - 1.0) / (max_w - 1.0)
    score = max(0.0, min(1.0, score))

    return beta * raw * (1.0 - score)


def validate_multiplier(multiplier: float) -> None:
    """Validate that ``multiplier`` is positive and on the 0.1-step grid.

    Allowed values: ``0.1, 0.2, ..., 1.0, 1.5, 2.0, ...``.  Off-grid values
    like ``0.37, 0.55, 1.25`` are rejected because they would produce a
    non-integer-currency budget (e.g. 300,000.00000000003 metal).

    Float-precision tolerance accommodates arithmetic like
    ``0.1 + 0.1 + 0.1 + 0.1 + 0.1`` that the user might intend as 0.5.
    """
    if not isinstance(multiplier, (int, float)) or isinstance(multiplier, bool):
        raise ValueError(
            f"multiplier must be a number, got {type(multiplier).__name__}"
        )
    if multiplier < 0:
        raise ValueError(
            f"multiplier must be non-negative, got {multiplier}"
        )
    tenth = float(multiplier) * 10.0
    if not math.isclose(tenth, round(tenth), abs_tol=_GRID_TOLERANCE):
        raise ValueError(
            f"multiplier {multiplier} is not on the 0.1-step grid "
            f"(allowed: 0.1, 0.2, ..., 1.0, 1.5, ...)"
        )


def compute_budget(
    enemy_fleet: Mapping[str, int],
    enemy_defenses: Optional[Mapping[str, int]] = None,
    multiplier: float = 1.0,
) -> int:
    """Return the budget ceiling for the counter-fleet.

    Mathematically: ``fleet_value(enemy_fleet, enemy_defenses) * multiplier``.

    The result is always an ``int`` (truncated if the float arithmetic
    produces fractional dust).  Validation of ``multiplier`` is delegated to
    :func:`validate_multiplier` so the failure mode is identical regardless
    of whether the caller goes through this entry point or pre-validates.
    """
    validate_multiplier(multiplier)
    raw = fleet_value(enemy_fleet, enemy_defenses)
    return int(raw * multiplier)


def validate_fleet_in_budget(
    fleet: Mapping[str, int],
    budget: int,
) -> bool:
    """Return ``True`` iff ``fleet``'s total cost is ``<= budget``."""
    _validate_counts(fleet, SHIPS_COST, label="ship")
    if not isinstance(budget, int) or isinstance(budget, bool):
        raise ValueError(
            f"budget must be an int, got {type(budget).__name__}"
        )
    if budget < 0:
        raise ValueError(f"budget must be non-negative, got {budget}")
    return sum(
        count * _unit_cost(key, SHIPS_COST) for key, count in fleet.items()
    ) <= budget


__all__ = [
    "DEFENSES_COST",
    "SHIPS_COST",
    "SHIP_BASE_ATK",
    "SHIP_DRIVE_DATA",
    "DEFAULT_DRIVE_TECHS",
    "effective_ship_speed",
    "derive_penalty_factors",
    "Fleet",
    "compute_budget",
    "fleet_value",
    "weighted_fleet_value",
    "resource_preference_penalty",
    "fleet_penalty_multiplier",
    "validate_fleet_in_budget",
    "validate_multiplier",
]
# Base attack values for each ship (from src/ships.rs base_attack)
SHIP_BASE_ATK: Dict[str, int] = {
    "small_cargo": 5, "large_cargo": 5,
    "light_fighter": 50, "heavy_fighter": 150,
    "cruiser": 400, "battleship": 1000, "battlecruiser": 700,
    "bomber": 1000, "destroyer": 2000, "deathstar": 200000,
    "espionage_probe": 0,  # Probe can't attack
    "pathfinder": 200,     # FIXED per Fandom (was 300, also had duplicate-key bug)
    "reaper": 2800,        # FIXED per Fandom (was missing — anti-capital ship)
    "recycler": 1,
    "solar_satellite": 1,  # Can't meaningfully attack
    "crawler": 1,          # Can't meaningfully attack
}



# ---------------------------------------------------------------------------
# Fuel / speed penalty: drive-tech-aware (rework 2026-09-09)
# ---------------------------------------------------------------------------
# Effective speed = base speed x drive multiplier (Combustion +10%/lvl,
# Impulse +20%/lvl, Hyperspace +30%/lvl). Three hulls switch drives at
# thresholds: Small Cargo (Impulse 5), Bomber (Hyperspace 8), Recycler
# (Impulse 17, then Hyperspace 15). Large Cargo never switches.
#
# Sources: OGame wiki "Base Speed" / "Ships" pages, retrieved 2026-09-09.
# The wiki's "speed at minimum research" values are locked as drift
# anchors in python_tests/test_fuel_speed_penalty.py:
#   HF@ID2=14,000  PF@HD2=19,200  SC@ID5=20,000  Recycler@ID17=17,600
#   Recycler@HD15=33,000  Bomber@HD8=17,000  Deathstar@HD7=310
#
# Penalty factors are DERIVED from the attacker's drive techs instead of
# the old static table, fixing two of its errors: Battleship was listed
# "very slow" although it flies at exactly Battlecruiser speed (same
# 10,000 base, same Hyperspace drive - only its fuel is 2x), and Light
# Fighter was treated as BC-fast although at typical techs it is 25-40%
# slower (Combustion +10%/lvl cannot keep pace with Hyperspace +30%/lvl;
# only very high Combustion closes the gap).

DRIVE_MULT_PER_LEVEL = {"combustion": 0.10, "impulse": 0.20, "hyperspace": 0.30}

# Used when a request carries no drive techs (API back-compat).
DEFAULT_DRIVE_TECHS: Dict[str, int] = {"combustion": 16, "impulse": 14, "hyperspace": 12}

# ship -> {"drive", "base" speed, "fuel" (deuterium), "switches"} where
# switches are (drive, min_level, new_base, new_fuel) applied in order.
SHIP_DRIVE_DATA: Dict[str, Dict] = {
    "light_fighter":   {"drive": "combustion", "base": 12_500, "fuel": 20, "switches": []},
    "heavy_fighter":   {"drive": "impulse",    "base": 10_000, "fuel": 75, "switches": []},
    "cruiser":         {"drive": "impulse",    "base": 15_000, "fuel": 300, "switches": []},
    "battleship":      {"drive": "hyperspace", "base": 10_000, "fuel": 500, "switches": []},
    "battlecruiser":   {"drive": "hyperspace", "base": 10_000, "fuel": 250, "switches": []},
    "bomber":          {"drive": "impulse",    "base": 4_000, "fuel": 700,
                        "switches": [("hyperspace", 8, 5_000, 700)]},
    "destroyer":       {"drive": "hyperspace", "base": 5_000, "fuel": 1000, "switches": []},
    "deathstar":       {"drive": "hyperspace", "base": 100, "fuel": 1, "switches": []},
    "reaper":          {"drive": "hyperspace", "base": 7_000, "fuel": 900, "switches": []},
    "pathfinder":      {"drive": "hyperspace", "base": 12_000, "fuel": 300, "switches": []},
    "small_cargo":     {"drive": "combustion", "base": 5_000, "fuel": 10,
                        "switches": [("impulse", 5, 10_000, 20)]},
    "large_cargo":     {"drive": "combustion", "base": 7_500, "fuel": 50, "switches": []},
    "recycler":        {"drive": "combustion", "base": 2_000, "fuel": 300,
                        "switches": [("impulse", 17, 4_000, 300),
                                     ("hyperspace", 15, 6_000, 300)]},
    "espionage_probe": {"drive": "combustion", "base": 100_000_000, "fuel": 1, "switches": []},
}

# Calibration: factor = 1 + ALPHA*log2(v_bc/v)+ + BETA*log2(fuel/fuel_bc)+
# clamped to [1.0, _PENALTY_CAP]. At DEFAULT_DRIVE_TECHS this yields
# roughly: deathstar 1.10, recycler 1.05, destroyer 1.04, bomber 1.03,
# reaper 1.03, large_cargo 1.02, battleship 1.01, light_fighter 1.008,
# heavy_fighter 1.004, cruiser/pathfinder 1.003, probe/BC 1.00.
_PENALTY_ALPHA = 0.015  # per log2(x) slower than the BC reference
_PENALTY_BETA = 0.012   # per log2(x) thirstier than the BC reference
_PENALTY_CAP = 1.15


def effective_ship_speed(drive_techs: Optional[Dict[str, int]] = None) -> Dict[str, float]:
    """Effective speed per ship at the given drive techs (switches applied).

    Unflyable units (solar satellite, crawler) are not in SHIP_DRIVE_DATA
    and therefore not in the result.
    """
    dt = drive_techs or DEFAULT_DRIVE_TECHS
    out: Dict[str, float] = {}
    for ship, prof in SHIP_DRIVE_DATA.items():
        drive, base = prof["drive"], prof["base"]
        for sw_drive, min_lvl, new_base, _fuel in prof["switches"]:
            if dt.get(sw_drive, 0) >= min_lvl:
                drive, base = sw_drive, new_base
        out[ship] = float(base) * (1.0 + DRIVE_MULT_PER_LEVEL[drive] * dt.get(drive, 0))
    return out


def derive_penalty_factors(drive_techs: Optional[Dict[str, int]] = None) -> Dict[str, float]:
    """Per-ship fuel/speed penalty factors at the given drive techs."""
    speeds = effective_ship_speed(drive_techs)
    v_ref = speeds["battlecruiser"]
    f_ref = SHIP_DRIVE_DATA["battlecruiser"]["fuel"]
    factors: Dict[str, float] = {}
    for ship, v in speeds.items():
        s_term = math.log2(v_ref / v) if v < v_ref else 0.0
        fuel = SHIP_DRIVE_DATA[ship]["fuel"]
        f_term = math.log2(fuel / f_ref) if fuel > f_ref else 0.0
        factors[ship] = min(_PENALTY_CAP, 1.0 + _PENALTY_ALPHA * s_term + _PENALTY_BETA * f_term)
    return factors


def fleet_penalty_multiplier(
    fleet,
    pct: float = 0.0,
    drive_techs: Optional[Dict[str, int]] = None,
) -> float:
    """Return weighted-average penalty factor for fleet, scaled by pct.

    Computes sum(count * penalty[ship]) / sum(count) where penalty[ship]
    is linearly interpolated between 1.0 (no effect, pct=0) and
    derive_penalty_factors(drive_techs)[ship] (max effect, pct=10). pct is clamped to
    [0, 10]; values <= 0 return 1.0 immediately (no effect).

    Count-weighted (not cost-weighted): a fleet of 1 BC + 1 Deathstar
    yields (1.00 + 1.10) / 2 = 1.05 regardless of resource spend. This is
    intentional: the user is penalising "presence" of slow/deut-expensive
    ships because even one Deathstar slows the entire fleet travel time.

    Parameters
    ----------
    fleet : Mapping[str, int]
        Attacker composition. Empty/zero-count ships ignored.
    pct : float
        User-facing percentage 0-10. 0 disables. 5 applies half the table.
        10 applies the table in full.
    drive_techs : Optional[Dict[str, int]]
        Attacker drive levels {"combustion", "impulse", "hyperspace"}.
        None -> DEFAULT_DRIVE_TECHS.

    Returns
    -------
    float
        Multiplier >= 1.0. Multiply mean_attacker_loss by this to bias
        toward cheap-and-fast ships at the cost of slow/deut-expensive ones.
    """
    if pct <= 0:
        return 1.0
    if pct > 10.0:
        pct = 10.0
    factors = derive_penalty_factors(drive_techs)
    total = 0
    weighted = 0.0
    for ship, count in fleet.items():
        if count <= 0:
            continue
        base = factors.get(ship, 1.0)
        factor = 1.0 + (base - 1.0) * (pct / 10.0)
        weighted += factor * count
        total += count
    if total == 0:
        return 1.0
    return weighted / total

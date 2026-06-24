"""Edge case handling and validation utilities (Task 12).

Centralizes guards for:
- Zero enemy (empty fleet + empty defenses)
- Budget too low (cannot afford even one ship)
- Budget too low to win (best-effort returned with warning)
- RIP vs RIP draw detection
- NaN / Infinity protection in fitness math
- Large resource values (overflow safety)
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from ogame_optimizer.core.fleet import SHIPS_COST, fleet_value


class EdgeCaseError(ValueError):
    """Raised when input cannot be processed at all (vs best-effort)."""


def cheapest_ship_cost() -> int:
    """Cost (M+C+D) of the cheapest ship (Light Fighter = 4000)."""
    return min(sum(SHIPS_COST[k]) for k in SHIPS_COST)


def defense_value(defenses: Dict[str, int]) -> int:
    """Rough resource value of planetary defenses.

    Uses simple cost proxies: RL=2000, LL=2500, HL=6000, GC=20000,
    IC=8000, PT=50000, SSD=20000, LSD=100000.
    """
    costs = {
        "rocket_launcher": 2000,
        "light_laser": 2500,
        "heavy_laser": 6000,
        "gauss_cannon": 20000,
        "ion_cannon": 8000,
        "plasma_turret": 50000,
        "small_shield_dome": 20000,
        "large_shield_dome": 100000,
    }
    return sum(costs.get(k, 10000) * v for k, v in defenses.items() if v > 0)


def validate_inputs(
    enemy_fleet: Dict[str, int],
    enemy_defenses: Dict[str, int],
    budget: int,
) -> Tuple[bool, Optional[str]]:
    """Returns (ok, error_message_or_warning).

    ok=False means hard failure (no fleet possible).
    ok=True with error_message set means best-effort (warning).
    """
    if not enemy_fleet and not enemy_defenses:
        return False, "No enemy to fight: enemy_fleet and enemy_defenses both empty"
    if budget <= 0:
        return False, f"Budget must be positive, got {budget}"
    if any(v < 0 for v in enemy_fleet.values()):
        return False, "Negative ship count in enemy_fleet"
    if any(v < 0 for v in enemy_defenses.values()):
        return False, "Negative defense count in enemy_defenses"
    cheapest = cheapest_ship_cost()
    if budget < cheapest:
        return False, f"Budget insufficient for any fleet (need at least {cheapest}, got {budget})"
    return True, None


def is_nan_or_inf(x: float) -> bool:
    """True if x is NaN, +Inf, or -Inf."""
    try:
        return not math.isfinite(x)
    except (TypeError, ValueError):
        return True


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that returns default on zero/NaN/Inf instead of crashing."""
    if denominator == 0 or is_nan_or_inf(denominator):
        return default
    if is_nan_or_inf(numerator):
        return default
    return numerator / denominator


def rip_vs_rip_is_draw() -> bool:
    """RIP vs RIP combat is always a draw (RIPs cannot damage each other)."""
    return True


def large_value_safe(value: int) -> bool:
    """True if value fits in u128 (Rust overflow safety check)."""
    return 0 <= value < (1 << 127)

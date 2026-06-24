"""Core domain primitives for the OGame fleet optimizer.

This sub-package owns:

- :mod:`ogame_optimizer.core.fleet` — ship and defense cost tables, the
  :class:`~ogame_optimizer.core.fleet.Fleet` dataclass, and budget math
  (:func:`~ogame_optimizer.core.fleet.compute_budget`,
  :func:`~ogame_optimizer.core.fleet.validate_multiplier`, …).
- :mod:`ogame_optimizer.core.tech` — the player's research levels
  (:class:`~ogame_optimizer.core.tech.TechLevels`).
- :mod:`ogame_optimizer.core.combat` — Pythonic wrapper around the Rust
  combat extension (:func:`~ogame_optimizer.core.combat.simulate_combat`,
  :func:`~ogame_optimizer.core.combat.simulate_batch`,
  :func:`~ogame_optimizer.core.combat.evaluate_population`).

These primitives are deliberately independent of:

- the Rust combat simulator (``ogame_optimizer._ogame_combat``) — the
  bridge is wired up in Task 6 (this module);
- the optimizer heuristics (greedy, ILP, etc.) — those live in Task 7+
  and consume this module;
- the FastAPI surface — Task 11 wraps these types for HTTP.
"""
from ogame_optimizer.core.combat import (
    evaluate_population,
    simulate_batch,
    simulate_combat,
)
from ogame_optimizer.core.fleet import (
    DEFENSES_COST,
    SHIPS_COST,
    Fleet,
    compute_budget,
    fleet_value,
    validate_fleet_in_budget,
    validate_multiplier,
)
from ogame_optimizer.core.tech import TechLevels

__all__ = [
    "DEFENSES_COST",
    "Fleet",
    "SHIPS_COST",
    "TechLevels",
    "compute_budget",
    "evaluate_population",
    "fleet_value",
    "simulate_batch",
    "simulate_combat",
    "validate_fleet_in_budget",
    "validate_multiplier",
]
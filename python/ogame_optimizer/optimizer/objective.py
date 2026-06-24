"""Dual-mode objective: evaluate a fleet and return a :class:`VarianceReport`.

This is the thin glue layer between the Rust combat core
(:func:`ogame_optimizer.core.combat.simulate_batch`) and the GA / greedy /
API layers. It owns two things:

* :class:`ObjectiveMode` — the ``attack`` / ``defend`` switch the whole
  optimizer is parameterised by.
* :func:`evaluate_fleet` — runs ``n_sims`` combats for one fleet and returns
  a :class:`~ogame_optimizer.optimizer.statistics.VarianceReport` carrying
  mean / stddev / percentiles / CI for that fleet's per-sim attacker losses.

It deliberately does **not** compute the fitness scalar here — that lives in
:func:`ogame_optimizer.optimizer.statistics.compute_fitness`, which is where
the hard constraint (win/survive ≥ 0.95 or ``-inf``) is enforced. Keeping
"describe the fleet" (this module) separate from "score the fleet"
(``compute_fitness``) means the same report can feed a GA, a greedy sweep, or
a human-readable API without re-running simulations.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, Tuple

from ogame_optimizer.core.combat import simulate_batch

from .statistics import VarianceReport

__all__ = ["ObjectiveMode", "evaluate_fleet"]


class ObjectiveMode(str, Enum):
    """What side of the battle we are optimising for.

    ``str`` mixin so ``ObjectiveMode.ATTACK == "attack"`` and the value
    serialises cleanly to JSON / CLI args.
    """

    ATTACK = "attack"
    DEFEND = "defend"


def evaluate_fleet(
    fleet: Mapping[str, int],
    enemy: Mapping[str, int],
    enemy_defenses: Mapping[str, int] | None,
    enemy_tech: Tuple[int, int, int],
    own_tech: Tuple[int, int, int],
    budget: int,
    mode: ObjectiveMode,
    n_sims: int,
    seed: int,
) -> VarianceReport:
    """Evaluate a fleet via :func:`simulate_batch` and return a report.

    Runs ``n_sims`` combats of ``fleet`` attacking ``enemy`` (plus
    ``enemy_defenses``) and aggregates the per-sim **attacker** loss into a
    :class:`VarianceReport`.

    Why attacker loss regardless of mode
    ------------------------------------
    The combat engine always reports losses from the attacker's perspective.
    In **defend** mode the caller is the defender, so the "attacker" in the
    simulation is the *enemy*; the caller's own losses are then the
    ``mean_defender_loss`` field of the batch result, which
    :func:`~ogame_optimizer.optimizer.statistics.compute_fitness` reads
    directly. This function therefore just hands back the full batch-derived
    report; mode only changes how ``compute_fitness`` interprets it.

    Parameters
    ----------
    fleet
        Attacker fleet (snake_case or PascalCase ship names; the combat core
        normalises both).
    enemy
        Defender fleet.
    enemy_defenses
        Defender planetary defenses, or ``None`` / ``{}`` for none.
    enemy_tech
        Defender ``(weapon, shield, armor)`` levels.
    own_tech
        Attacker ``(weapon, shield, armor)`` levels.
    budget
        Resource budget. **Not** used for the simulation, only stashed on the
        report so downstream ``compute_fitness`` has it without the caller
        having to thread it through again. Must be positive.
    mode
        :class:`ObjectiveMode.ATTACK` or :class:`ObjectiveMode.DEFEND`.
        Stored on the returned report for traceability.
    n_sims
        Number of combats to simulate. Must be ≥ 1.
    seed
        Base seed for the combat RNG. For GA fairness, pass a
        :class:`~ogame_optimizer.optimizer.statistics.CRNManager`-derived
        per-generation seed so all individuals in a generation share it.

    Returns
    -------
    VarianceReport
        Aggregate attacker-loss statistics over ``n_sims`` combats, with
        ``budget`` and ``mode`` attached as metadata
        (``report.meta["budget"]``, ``report.meta["mode"]``).

    Raises
    ------
    ValueError
        If ``n_sims < 1`` or ``budget <= 0``.
    """
    if n_sims < 1:
        raise ValueError(f"n_sims must be >= 1, got {n_sims}")
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")
    if enemy_defenses is None:
        enemy_defenses = {}

    batch: dict[str, Any] = simulate_batch(
        attacker=fleet,
        defender=enemy,
        defender_defenses=enemy_defenses,
        attacker_tech=own_tech,
        defender_tech=enemy_tech,
        n_sims=n_sims,
        base_seed=seed,
    )

    report = VarianceReport.from_batch(batch)
    # Attach the full batch + mode/budget as metadata so downstream
    # compute_fitness has everything it needs without a second call.
    report.meta = {
        "batch": batch,
        "mode": str(mode.value),
        "budget": int(budget),
        "seed": int(seed),
    }
    return report

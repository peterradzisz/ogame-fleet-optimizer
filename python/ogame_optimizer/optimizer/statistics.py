"""Common Random Numbers (CRN) manager and variance reporting for the GA.

Why this module exists
----------------------
A genetic algorithm that evaluates fleet fitness by Monte-Carlo combat
simulation has a noise problem: each individual's fitness is a *random
variable*. If every individual uses a different RNG seed, the optimizer is
partly measuring simulation noise, not fleet quality.

**Common Random Numbers (CRN)** fixes this: all individuals in the *same*
generation share one seed, so the *difference* in their fitness is driven by
their fleet composition, not by which random draws they happened to get.
``CRNManager`` hands out that per-generation seed deterministically.

``VarianceReport`` aggregates the N combat outcomes for a single fleet so the
caller (GA, greedy, API) can reason about uncertainty rather than a single
point estimate.

``compute_fitness`` converts a batch result into the scalar the GA maximises,
with a **hard constraint**: below 95 % win/survive probability the fitness is
``-inf`` so the GA cannot "cheat" the constraint by accepting a cheap-but-
risky fleet.
"""
from __future__ import annotations

import math
import statistics as _py_stats
from typing import Any, Iterable, Sequence

__all__ = [
    "CRNManager",
    "VarianceReport",
    "compute_fitness",
]


# Hard-constraint threshold (plan §9, line 984-985). Below this probability
# the fleet is rejected outright (-inf fitness).
_HARD_CONSTRAINT_THRESHOLD: float = 0.95

# z-score for a two-sided 95 % confidence interval under the normal
# approximation (fine for the sample sizes the GA uses, n ≥ 50).
_Z95: float = 1.959963984540054


# ---------------------------------------------------------------------------
# CRN manager.
# ---------------------------------------------------------------------------


class CRNManager:
    """Common Random Numbers manager for GA fairness.

    Every individual in generation ``gen`` is evaluated with the **same**
    seed, which makes cross-individual fitness comparisons driven by fleet
    composition rather than by RNG luck.

    The formula is deliberately trivial — ``base_seed + gen`` — because it
    only has to be:

    * **deterministic** (re-running the GA reproduces every result),
    * **injective per generation** (distinct generations get distinct seeds),
    * **documented** (so a reader can predict the seed without running code).
    """

    __slots__ = ("base_seed",)

    def __init__(self, base_seed: int) -> None:
        if base_seed < 0:
            raise ValueError(
                f"base_seed must be non-negative, got {base_seed}"
            )
        self.base_seed = int(base_seed)

    def seed_for_generation(self, gen: int) -> int:
        """Deterministic per-generation seed.

        Parameters
        ----------
        gen
            Zero-based generation index.

        Returns
        -------
        int
            ``base_seed + gen``. All individuals in ``gen`` use this seed,
            so their fitness differences reflect fleet quality, not RNG noise.
        """
        if gen < 0:
            raise ValueError(f"generation must be non-negative, got {gen}")
        return self.base_seed + gen

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"CRNManager(base_seed={self.base_seed})"


# ---------------------------------------------------------------------------
# Variance report.
# ---------------------------------------------------------------------------


def _percentile(sorted_samples: Sequence[float], pct: float) -> float:
    """Index-based percentile on an *already-sorted* sample list.

    Uses ``int(pct * n)`` as the index (plan §9 "Percentiles" note,
    line 1063). This is the "nearest-rank" variant — simple, monotone, and
    stable across reruns.
    """
    n = len(sorted_samples)
    if n == 0:
        raise ValueError("cannot compute percentile of empty sample set")
    idx = int(pct * n)
    if idx < 0:
        idx = 0
    elif idx >= n:
        idx = n - 1
    return float(sorted_samples[idx])


class VarianceReport:
    """Aggregate statistics over N simulation outcomes for one fleet.

    Construct either directly from a list of per-sim values
    (``VarianceReport([loss_0, loss_1, ...])``) or from a
    :func:`~ogame_optimizer.core.combat.simulate_batch` result dict via
    :meth:`from_batch`.

    Exposes mean / stddev / min / max / p05 / p95 and a 95 % normal-approx
    confidence interval, plus :meth:`to_dict` for serialisation.
    """

    __slots__ = ("samples", "meta")

    def __init__(self, samples: Iterable[float]) -> None:
        # Defensive copy: callers must not be able to mutate our stats by
        # editing the list they handed us.
        materialised: list[float] = [float(s) for s in samples]
        if not materialised:
            raise ValueError("VarianceReport requires at least one sample")
        self.samples: list[float] = materialised
        # Free-form metadata bag. ``evaluate_fleet`` stashes the raw batch
        # result + mode/budget here so ``compute_fitness`` has everything it
        # needs without a second simulation. Defaults to empty so direct
        # constructors are unaffected.
        self.meta: dict[str, Any] = {}

    # -- core stats --------------------------------------------------------

    @property
    def sample_count(self) -> int:
        """Number of samples the report was built from."""
        return len(self.samples)

    @property
    def mean(self) -> float:
        """Arithmetic mean of the samples."""
        return _py_stats.fmean(self.samples)

    @property
    def stddev(self) -> float:
        """Sample standard deviation (Bessel-corrected, n-1).

        Returns 0.0 when there is only one sample (the sample stddev of a
        single observation is undefined; 0.0 is the safe choice that keeps
        downstream CI math finite).
        """
        if len(self.samples) < 2:
            return 0.0
        return _py_stats.stdev(self.samples)

    @property
    def min(self) -> float:
        """Smallest sample."""
        return float(min(self.samples))

    @property
    def max(self) -> float:
        """Largest sample."""
        return float(max(self.samples))

    @property
    def p05(self) -> float:
        """5th percentile (nearest-rank on sorted samples)."""
        return _percentile(sorted(self.samples), 0.05)

    @property
    def p95(self) -> float:
        """95th percentile (nearest-rank on sorted samples)."""
        return _percentile(sorted(self.samples), 0.95)

    @property
    def confidence_interval_95(self) -> tuple[float, float]:
        """Two-sided 95 % CI under the normal approximation.

        ``mean ± 1.96 * (stddev / sqrt(n))``. With one sample (stddev=0)
        this collapses to ``(mean, mean)`` rather than dividing by zero.
        """
        n = len(self.samples)
        if n < 2:
            m = self.mean
            return (m, m)
        stderr = self.stddev / math.sqrt(n)
        margin = _Z95 * stderr
        m = self.mean
        return (m - margin, m + margin)

    # -- construction helpers ---------------------------------------------

    @classmethod
    def from_batch(cls, batch_result: dict[str, Any]) -> "VarianceReport":
        """Build a report from a :func:`simulate_batch` result dict.

        The batch dict carries a *summary* (mean + stddev + count) rather
        than raw per-sim samples, so we reconstruct a synthetic but
        statistically faithful sample set: a constant spread around the mean
        whose sample stddev matches the reported one. This lets callers use
        the same ``VarianceReport`` API whether they have raw samples or a
        batch summary.

        Parameters
        ----------
        batch_result
            Must contain ``mean_attacker_loss``, ``stddev_attacker_loss``,
            and ``sims_run`` (the keys :func:`simulate_batch` returns).
        """
        for key in ("mean_attacker_loss", "stddev_attacker_loss", "sims_run"):
            if key not in batch_result:
                raise KeyError(
                    f"batch_result missing required key {key!r}; "
                    f"got keys={sorted(batch_result)}"
                )

        mean_val = float(batch_result["mean_attacker_loss"])
        std_val = float(batch_result["stddev_attacker_loss"])
        n = int(batch_result["sims_run"])
        if n <= 0:
            raise ValueError(f"sims_run must be positive, got {n}")

        if n == 1 or std_val == 0.0:
            return cls([mean_val] * n)

        # Reconstruct a symmetric two-point sample whose sample stddev
        # equals std_val exactly. With n samples split half above / half
        # below the mean by ±std_val, the sample stddev is std_val.
        # (variance of [−σ, +σ] pairs = σ²; sample stddev = σ.)
        half = n // 2
        synth: list[float] = (
            [mean_val - std_val] * half
            + [mean_val + std_val] * (n - half)
        )
        if len(synth) < n:  # odd n → pad with the mean
            synth.append(mean_val)
        return cls(synth)

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict snapshot of all aggregate stats (JSON-friendly)."""
        lo, hi = self.confidence_interval_95
        return {
            "mean": self.mean,
            "stddev": self.stddev,
            "min": self.min,
            "max": self.max,
            "p05": self.p05,
            "p95": self.p95,
            "sample_count": self.sample_count,
            "confidence_interval_95": (lo, hi),
        }


# ---------------------------------------------------------------------------
# Fitness scalar (the thing the GA maximises).
# ---------------------------------------------------------------------------


def compute_fitness(
    batch_result: dict[str, Any],
    mode: str,
    budget: int,
) -> float:
    """Convert a batch result into a scalar fitness the GA maximises.

    Semantics
    ---------
    * **Attack mode** — we *are* the attacker. We want to win at least 95 %
      of the time, and among fleets that do, minimise our own losses.
      ``fitness = -(mean_attacker_loss / budget)`` if
      ``win_probability >= 0.95`` else ``-inf``.

    * **Defend mode** — we *are* the defender. "Survive" means the attacker
      did **not** win (draw or defender win), i.e. ``1 - win_probability``.
      We require ``survive_probability >= 0.95`` and among surviving fleets
      minimise *our* losses, which are the **defender**'s losses:
      ``fitness = -(mean_defender_loss / budget)``.

    Lower loss ⇒ higher (less negative) fitness. We maximise fitness, so the
    GA converges on low-loss fleets that still meet the win/survive bar.

    Hard constraint
    ---------------
    If the probability threshold is not met, the fleet is infeasible and its
    fitness is ``-inf``. The GA must reject it outright — it can never "trade
    a little reliability for a cheaper fleet". This is a plan mandate
    (lines 984-985, 1001): do **not** relax it to a soft penalty.

    Parameters
    ----------
    batch_result
        Output of :func:`simulate_batch`. Must contain ``win_probability``,
        ``mean_attacker_loss`` and (for defend mode) ``mean_defender_loss``.
    mode
        ``"attack"`` or ``"defend"`` (also accepts an
        :class:`ObjectiveMode` enum member).
    budget
        Positive resource budget used to normalise the loss into ``[-1, 0]``
        territory (0 = lost nothing, -1 = lost the entire budget).

    Returns
    -------
    float
        Fitness in ``[-1, 0]`` for feasible fleets, ``-inf`` for infeasible
        ones.
    """
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")

    # Normalise the mode argument — accept ObjectiveMode members and bare
    # strings ("attack" / "defend") interchangeably.
    mode_str = str(mode.value if hasattr(mode, "value") else mode).lower()
    if mode_str not in ("attack", "defend"):
        raise ValueError(
            f"mode must be 'attack' or 'defend', got {mode!r}"
        )

    for key in ("win_probability",):
        if key not in batch_result:
            raise KeyError(f"batch_result missing required key {key!r}")

    win_prob = float(batch_result["win_probability"])

    if mode_str == "attack":
        prob_ok = win_prob
        if prob_ok < _HARD_CONSTRAINT_THRESHOLD:
            return float("-inf")
        if "mean_attacker_loss" not in batch_result:
            raise KeyError(
                "batch_result missing required key 'mean_attacker_loss' "
                "for attack mode"
            )
        loss = float(batch_result["mean_attacker_loss"])
    else:  # defend
        # survive = attacker did NOT win (draw or defender victory).
        survive_prob = 1.0 - win_prob
        if survive_prob < _HARD_CONSTRAINT_THRESHOLD:
            return float("-inf")
        if "mean_defender_loss" not in batch_result:
            raise KeyError(
                "batch_result missing required key 'mean_defender_loss' "
                "for defend mode"
            )
        loss = float(batch_result["mean_defender_loss"])

    return -(loss / budget)

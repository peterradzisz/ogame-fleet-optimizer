"""Pydantic schemas for the FastAPI surface (Task 11)."""
from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class ShipCounts(BaseModel):
    """Dict of {ship_type: count} with validation."""
    ships: Dict[str, int] = Field(default_factory=dict)

    @field_validator("ships")
    @classmethod
    def validate_ships(cls, v: Dict[str, int]) -> Dict[str, int]:
        valid = {"small_cargo", "large_cargo", "light_fighter", "heavy_fighter", "cruiser", "battleship", "battlecruiser", "bomber", "destroyer", "deathstar", "pathfinder", "reaper", "recycler", "espionage_probe", "solar_satellite", "crawler"}
        for k, val in v.items():
            if k not in valid:
                raise ValueError(f"Unknown ship type: {k}")
            if val < 0:
                raise ValueError(f"Ship count must be >= 0, got {val} for {k}")
        return v


class DefenseCounts(BaseModel):
    defenses: Dict[str, int] = Field(default_factory=dict)

    @field_validator("defenses")
    @classmethod
    def validate_defenses(cls, v: Dict[str, int]) -> Dict[str, int]:
        valid = {"rocket_launcher", "light_laser", "heavy_laser", "gauss_cannon", "ion_cannon", "plasma_turret", "small_shield_dome", "large_shield_dome"}
        for k, val in v.items():
            if k not in valid:
                raise ValueError(f"Unknown defense type: {k}")
            if val < 0:
                raise ValueError(f"Defense count must be >= 0, got {val} for {k}")
        return v


class TechLevelsSchema(BaseModel):
    weapon: int = Field(default=0, ge=0)
    shield: int = Field(default=0, ge=0)
    armor: int = Field(default=0, ge=0)


class OptimizeRequest(BaseModel):
    enemy_fleet: ShipCounts
    enemy_defenses: DefenseCounts = Field(default_factory=DefenseCounts)
    attacker_tech: TechLevelsSchema = Field(default_factory=TechLevelsSchema)
    defender_tech: TechLevelsSchema = Field(default_factory=TechLevelsSchema)
    budget_multiplier: float = 1.0
    mode: str = "attack"
    seed: Optional[int] = 42
    ga_time_budget: float = 5.0
    final_sims: int = 1000
    exclude_ships: Optional[List[str]] = None
    seed_fleet: Optional[Dict[str, int]] = None
    debris_pct: float = 0.30
    deuterium_in_debris: bool = False
    optimization_target: str = "maximize_profit"
    # Minimum gain after recycling, as a percentage (0-100).
    # 0 = no constraint (default, backward compatible).
    # > 0 = hard constraint: the optimizer rejects any fleet whose
    #   (debris_total - loss) / fleet_value is below this percentage.
    # The constraint is applied AS A HARD GATE in the GA fitness function
    # (fleets below threshold get fitness = -inf). Combined with the
    # existing 95% win-probability gate, this guarantees the recommended
    # fleet meets both survival and ROI thresholds (when possible).
    # Common values: 0 (disabled), 10, 20, 30, 40, 50, ... 100.
    min_gain_pct: float = 0.0
    # Resource preference multipliers (M, C, D). Default 2:1:1 biases toward
    # metal-heavy fleets when combat performance is similar. 1:1:1 disables.
    # Step 0.1; values must be non-negative.
    resource_weights: Optional[List[float]] = None
    # Strength of the resource-preference penalty (0.0-1.0). Default 0.05.
    # 0 = no preference applied; 0.10 = mild; 0.20+ = strong bias.
    # Penalty = beta * fleet_cost * (1 - preference_match_score), so a fleet
    # that exactly matches the weights pays 0 penalty.
    preference_beta: Optional[float] = None
    hyperspace_tech: int = 11
    collector_class: bool = False

    @field_validator("budget_multiplier")
    @classmethod
    def validate_multiplier(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("budget_multiplier must be positive")
        if abs(v / 0.1 - round(v / 0.1)) > 1e-6:
            raise ValueError(f"budget_multiplier must be a 0.1-step value (0.1, 0.2, ..., 1.0, 1.5, ...), got {v}")
        return v

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        if v not in ("attack", "defend"):
            raise ValueError(f"mode must be attack or defend, got {v}")
        return v

    @field_validator("min_gain_pct")
    @classmethod
    def validate_min_gain(cls, v: float) -> float:
        if v < 0 or v > 100:
            raise ValueError(f"min_gain_pct must be in [0, 100], got {v}")
        return v


class OptimizeResponse(BaseModel):
    recommended_fleet: Dict[str, int]
    expected_loss_mean: float
    expected_loss_stddev: float
    win_probability: float
    confidence_interval_95: List[float]
    sims_run_final: int
    greedy_baseline_loss: float
    ga_improvement_pct: float
    time_elapsed_total: float
    seed_used: int
    mode: str
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
    fleet_analysis: Dict[str, Dict] = Field(default_factory=dict)
    defender_fleet_analysis: Dict[str, Dict] = Field(default_factory=dict)
    resource_weights: List[float] = Field(default_factory=lambda: [2.0, 1.0, 1.0])
    preference_beta: float = 0.05
    fleet_weighted_value: float = 0.0
    resource_preference_penalty: float = 0.0
    resource_preference_match_score: float = 1.0
    raw_loss_mean: float = 0.0
    # min_gain constraint result
    min_gain_required: float = 0.0     # echo back the constraint
    min_gain_met: bool = True            # True if recommended fleet meets it
    # Actual ROI of the recommended fleet = (debris - loss) / fleet_value * 100
    # (already in net_profit_pct, but provided under a more descriptive name)
    actual_roi_pct: float = 0.0


class CombatRequest(BaseModel):
    attacker: ShipCounts
    defender: ShipCounts
    defender_defenses: DefenseCounts = Field(default_factory=DefenseCounts)
    attacker_tech: TechLevelsSchema = Field(default_factory=TechLevelsSchema)
    defender_tech: TechLevelsSchema = Field(default_factory=TechLevelsSchema)
    n_sims: int = Field(default=100, ge=1, le=10000)
    seed: Optional[int] = 42


class CombatResponse(BaseModel):
    mean_attacker_loss: float
    stddev_attacker_loss: float
    mean_defender_loss: float
    win_probability: float
    wins: int
    losses: int
    draws: int
    sims_run: int
    seed_used: int

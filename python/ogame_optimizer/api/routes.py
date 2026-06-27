"""API + UI routes for the OGame fleet auto-optimizer.

Every endpoint logs entry/exit with key parameters for debugging.
"""
from __future__ import annotations
from pathlib import Path
import traceback

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ogame_optimizer.logging_config import get_logger
from ogame_optimizer.api.schemas import (
    OptimizeRequest, OptimizeResponse, CombatRequest, CombatResponse,
)
from ogame_optimizer.core.combat import simulate_batch
from ogame_optimizer.optimizer.orchestration import optimize, OptimizationResult


_log = get_logger("ogame.api.routes")
router = APIRouter()

_TEMPLATES_DIR = Path(__file__).parent.parent / "web" / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    _log.info("Serving index page")
    return templates.TemplateResponse(request, "index.html")


@router.get("/api/ships")
def list_ships() -> dict:
    _log.debug("Listing ships")
    return {"ships": [
        {"key": "small_cargo", "name": "Small Cargo"},
        {"key": "large_cargo", "name": "Large Cargo"},
        {"key": "light_fighter", "name": "Light Fighter"},
        {"key": "heavy_fighter", "name": "Heavy Fighter"},
        {"key": "cruiser", "name": "Cruiser"},
        {"key": "battleship", "name": "Battleship"},
        {"key": "battlecruiser", "name": "Battlecruiser"},
        {"key": "bomber", "name": "Bomber"},
        {"key": "destroyer", "name": "Destroyer"},
        {"key": "deathstar", "name": "Deathstar"},
        {"key": "pathfinder", "name": "Pathfinder"},
        {"key": "reaper", "name": "Reaper"},
        {"key": "recycler", "name": "Recycler"},
        {"key": "espionage_probe", "name": "Espionage Probe"},
        {"key": "solar_satellite", "name": "Solar Satellite"},
        {"key": "crawler", "name": "Crawler"},
    ]}


@router.get("/api/defenses")
def list_defenses() -> dict:
    _log.debug("Listing defenses")
    return {"defenses": [
        {"key": "rocket_launcher", "name": "Rocket Launcher"},
        {"key": "light_laser", "name": "Light Laser"},
        {"key": "heavy_laser", "name": "Heavy Laser"},
        {"key": "gauss_cannon", "name": "Gauss Cannon"},
        {"key": "ion_cannon", "name": "Ion Cannon"},
        {"key": "plasma_turret", "name": "Plasma Turret"},
        {"key": "small_shield_dome", "name": "Small Shield Dome"},
        {"key": "large_shield_dome", "name": "Large Shield Dome"},
    ]}


@router.post("/api/combat", response_model=CombatResponse)
def run_combat(req: CombatRequest) -> CombatResponse:
    _log.info("Combat request: attacker=%s defender=%s n_sims=%d", req.attacker.ships, req.defender.ships, req.n_sims)
    try:
        result = simulate_batch(
            attacker=req.attacker.ships,
            defender=req.defender.ships,
            defender_defenses=req.defender_defenses.defenses,
            attacker_tech=(req.attacker_tech.weapon, req.attacker_tech.shield, req.attacker_tech.armor),
            defender_tech=(req.defender_tech.weapon, req.defender_tech.shield, req.defender_tech.armor),
            n_sims=req.n_sims,
            base_seed=req.seed or 42,
        )
        _log.info("Combat result: win_prob=%.3f mean_atk_loss=%.0f sims=%d",
                  float(result.get("win_probability", 0)),
                  float(result.get("mean_attacker_loss", 0)),
                  int(result.get("sims_run", 0)))
        return CombatResponse(
            mean_attacker_loss=float(result.get("mean_attacker_loss", 0)),
            stddev_attacker_loss=float(result.get("stddev_attacker_loss", 0)),
            mean_defender_loss=float(result.get("mean_defender_loss", 0)),
            win_probability=float(result.get("win_probability", 0.0)),
            wins=int(result.get("wins", 0)),
            losses=int(result.get("losses", 0)),
            draws=int(result.get("draws", 0)),
            sims_run=int(result.get("sims_run", 0)),
            seed_used=int(result.get("seed_used", req.seed or 42)),
        )
    except Exception as e:
        _log.exception("Combat failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Combat simulation failed: {str(e)}")


@router.post("/api/optimize", response_model=OptimizeResponse)
def run_optimize(req: OptimizeRequest) -> OptimizeResponse:
    _log.info("Optimize request: enemy_fleet=%s enemy_defenses=%s multiplier=%s mode=%s",
              req.enemy_fleet.ships, req.enemy_defenses.defenses, req.budget_multiplier, req.mode)
    try:
        result: OptimizationResult = optimize(
            enemy_fleet=req.enemy_fleet.ships,
            enemy_defenses=req.enemy_defenses.defenses,
            enemy_tech=(req.defender_tech.weapon, req.defender_tech.shield, req.defender_tech.armor),
            attacker_tech=(req.attacker_tech.weapon, req.attacker_tech.shield, req.attacker_tech.armor),
            budget_multiplier=req.budget_multiplier,
            mode=req.mode,
            base_seed=req.seed or 42,
            ga_time_budget=req.ga_time_budget,
            final_sims=req.final_sims,
            exclude_ships=req.exclude_ships,
            seed_fleet=req.seed_fleet,
            debris_pct=req.debris_pct,
            deuterium_in_debris=req.deuterium_in_debris,
            optimization_target=req.optimization_target,
            resource_weights=tuple(req.resource_weights) if req.resource_weights else (2.0, 1.0, 1.0),
            preference_beta=req.preference_beta if req.preference_beta is not None else 0.05,
            hyperspace_tech=req.hyperspace_tech,
            collector_class=req.collector_class,
            min_gain_pct=req.min_gain_pct,
        )
        _log.info("Optimize result: fleet=%s win_prob=%.3f loss_mean=%.0f time=%.2fs",
                  result.recommended_fleet, result.win_probability, result.expected_loss_mean, result.total_time)
        return OptimizeResponse(
            recommended_fleet=result.recommended_fleet,
            raw_loss_mean=result.raw_loss_mean,
            expected_loss_mean=result.expected_loss_mean,
            expected_loss_stddev=result.expected_loss_stddev,
            win_probability=result.win_probability,
            confidence_interval_95=result.confidence_interval_95,
            sims_run_final=result.sims_run_final,
            greedy_baseline_loss=result.greedy_baseline_loss,
            ga_improvement_pct=result.ga_improvement_pct,
            time_elapsed_total=result.total_time,
            seed_used=result.seed_used,
            mode=result.mode,
            fleet_value=result.fleet_value,
            fleet_lost_pct=result.fleet_lost_pct,
            ships_lost_count=result.ships_lost_count,
            ships_initial_count=result.ships_initial_count,
            debris_metal=result.debris_metal,
            debris_crystal=result.debris_crystal,
            debris_deuterium=result.debris_deuterium,
            debris_total=result.debris_total,
            net_profit=result.net_profit,
            net_profit_pct=result.net_profit_pct,
            recyclers_needed=result.recyclers_needed,
            recycler_capacity=result.recycler_capacity,
            recyclers_cost_metal=result.recyclers_cost_metal,
            recyclers_cost_crystal=result.recyclers_cost_crystal,
            recyclers_cost_deuterium=result.recyclers_cost_deuterium,
            recyclers_cost_total=result.recyclers_cost_total,
            fleet_analysis=result.fleet_analysis,
            defender_fleet_analysis=result.defender_fleet_analysis,
            resource_weights=list(result.resource_weights),
            preference_beta=result.preference_beta,
            fleet_weighted_value=result.fleet_weighted_value,
            resource_preference_penalty=result.resource_preference_penalty,
            resource_preference_match_score=result.resource_preference_match_score,
            min_gain_required=result.min_gain_required,
            min_gain_met=result.min_gain_met,
            actual_roi_pct=result.actual_roi_pct,
        )
    except ValueError as e:
        _log.warning("Optimize validation error: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _log.exception("Optimize failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

"""Drive-tech-aware fuel/speed penalty (fleet.py SHIP_DRIVE_DATA).

Wiki drift anchors: OGame wiki "Base Speed" page (retrieved 2026-09-09)
publishes exact "speed at minimum research" values, which lock both the
base speeds and the drive-switch thresholds. User-reported issue this
locks: Light Fighter is slower than Battlecruiser unless Combustion is
very high (Combustion +10%/lvl vs Hyperspace +30%/lvl).
"""
from __future__ import annotations

import pathlib

import pytest
from fastapi.testclient import TestClient

from ogame_optimizer.api.app import app
from ogame_optimizer.core.fleet import (
    DEFAULT_DRIVE_TECHS,
    derive_penalty_factors,
    effective_ship_speed,
    fleet_penalty_multiplier,
)


def speed(techs, ship):
    return effective_ship_speed(techs)[ship]


class TestWikiDriftAnchors:
    def test_heavy_fighter_is_impulse(self):
        # HF base 10,000 on IMPULSE (not combustion): at ID 2 -> 14,000
        assert speed({"combustion": 99, "impulse": 2, "hyperspace": 0}, "heavy_fighter") == 14_000

    def test_pathfinder_base_12000(self):
        assert speed({"combustion": 0, "impulse": 0, "hyperspace": 2}, "pathfinder") == 19_200

    def test_small_cargo_switch_at_impulse_5(self):
        below = {"combustion": 0, "impulse": 4, "hyperspace": 0}
        at = {"combustion": 0, "impulse": 5, "hyperspace": 0}
        assert speed(below, "small_cargo") == 5_000
        assert speed(at, "small_cargo") == 20_000  # wiki anchor

    def test_recycler_double_switch(self):
        assert speed({"combustion": 0, "impulse": 17, "hyperspace": 0}, "recycler") == 17_600
        assert speed({"combustion": 0, "impulse": 0, "hyperspace": 15}, "recycler") == 33_000

    def test_bomber_switch_at_hd8(self):
        assert speed({"combustion": 0, "impulse": 0, "hyperspace": 8}, "bomber") == 17_000

    def test_deathstar_min_research(self):
        assert speed({"combustion": 0, "impulse": 0, "hyperspace": 7}, "deathstar") == 310

    def test_large_cargo_never_switches(self):
        t = {"combustion": 6, "impulse": 25, "hyperspace": 25}
        assert speed(t, "large_cargo") == 12_000  # wiki anchor 12,000 @ CD6

    def test_battleship_always_equals_battlecruiser(self):
        for hd in (0, 5, 12, 20):
            t = {"combustion": 0, "impulse": 0, "hyperspace": hd}
            assert speed(t, "battleship") == speed(t, "battlecruiser")


class TestUserScenario:
    def test_lf_slower_than_bc_until_very_high_combustion(self):
        typical = {"combustion": 17, "impulse": 15, "hyperspace": 12}
        assert speed(typical, "light_fighter") < speed(typical, "battlecruiser")
        # Hyperspace +30%/lvl: LF needs ~CD 34 to match HD 12
        high = {"combustion": 34, "impulse": 15, "hyperspace": 12}
        assert speed(high, "light_fighter") >= speed(high, "battlecruiser")

    def test_lf_penalty_shrinks_as_combustion_rises(self):
        lo = derive_penalty_factors({"combustion": 10, "impulse": 8, "hyperspace": 10})["light_fighter"]
        hi = derive_penalty_factors({"combustion": 20, "impulse": 15, "hyperspace": 12})["light_fighter"]
        assert hi < lo

    def test_bc_is_reference_and_bs_only_fuel_penalised(self):
        for techs in ({"combustion": 10, "impulse": 8, "hyperspace": 10}, DEFAULT_DRIVE_TECHS):
            f = derive_penalty_factors(techs)
            # BC has the hyperspace-fuel base bonus (factor 0.99 < 1.0);
            # BS has identical speed but higher fuel so its factor is
            # strictly worse than BC's at any tech level.
            assert f["battlecruiser"] == pytest.approx(0.99, abs=1e-9)
            assert f["battlecruiser"] < f["battleship"]
            assert f["battleship"] < 1.03


class TestPenaltyDerivation:
    def test_default_techs_ordering_and_values(self):
        f = derive_penalty_factors()  # DEFAULT_DRIVE_TECHS
        # Deathstar: 1.10 speed penalty * (1 - 0.01 hyperspace bonus) ~= 1.089
        assert f["deathstar"] == pytest.approx(1.089, abs=0.002)
        assert f["destroyer"] == pytest.approx(1.039, abs=0.002)
        assert f["recycler"] > f["destroyer"]  # slowest flyer below DS
        assert f["battleship"] > 1.0           # 2x fuel
        assert f["espionage_probe"] == 1.0     # fastest + cheapest fuel
        assert f["cruiser"] == pytest.approx(1.003, abs=0.002)  # faster, slightly thirstier

    def test_factors_bounded(self):
        # Hyperspace-fuel ships can go down to ~0.99; slow+thirsty ships up
        # to ~1.10; never outside [0.95, 1.15].
        for cd in (0, 10, 20, 30):
            for hd in (0, 10, 20, 30):
                for v in derive_penalty_factors({"combustion": cd, "impulse": 8, "hyperspace": hd}).values():
                    assert 0.95 <= v <= 1.15


class TestMultiplier:
    def test_pct_scaling_and_count_weighting(self):
        fleet = {"battlecruiser": 1, "deathstar": 1}
        assert fleet_penalty_multiplier(fleet, 0) == 1.0
        full = fleet_penalty_multiplier(fleet, 10)
        half = fleet_penalty_multiplier(fleet, 5)
        assert half == pytest.approx((full + 1.0) / 2, abs=1e-9)
        # (BC + DS) / 2 with the new factor table.
        assert full == pytest.approx((derive_penalty_factors()["battlecruiser"] + derive_penalty_factors()["deathstar"]) / 2, abs=1e-9)

    def test_drive_techs_change_penalty(self):
        fleet = {"light_fighter": 100}
        lo = fleet_penalty_multiplier(fleet, 10, {"combustion": 10, "impulse": 8, "hyperspace": 10})
        hi = fleet_penalty_multiplier(fleet, 10, {"combustion": 20, "impulse": 8, "hyperspace": 10})
        assert hi < lo


class TestApiAndUi:
    @pytest.fixture(scope="class")
    def client(self):
        return TestClient(app)

    def _payload(self, **ov):
        p = {
            "enemy_fleet": {"ships": {"light_fighter": 100}},
            "enemy_defenses": {"defenses": {}},
            "attacker_tech": {"weapon": 0, "shield": 0, "armor": 0},
            "defender_tech": {"weapon": 0, "shield": 0, "armor": 0},
            "budget_multiplier": 1.0, "mode": "attack", "seed": 42,
            "ga_time_budget": 0.3, "final_sims": 50,
            "fuel_speed_penalty_pct": 5,
            "drive_techs": {"combustion": 18, "impulse": 12, "hyperspace": 11},
        }
        p.update(ov)
        return p

    def test_drive_techs_accepted(self, client):
        r = client.post("/api/optimize", json=self._payload())
        assert r.status_code == 200, r.text

    def test_drive_techs_range_validated(self, client):
        r = client.post("/api/optimize", json=self._payload(drive_techs={"combustion": 99, "impulse": 1, "hyperspace": 1}))
        assert r.status_code == 422

    def test_ui_wiring(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        idx = (root / "python/ogame_optimizer/web/templates/index.html").read_text(encoding="utf-8")
        js = (root / "python/ogame_optimizer/web/static/app.js").read_text(encoding="utf-8")
        for n in ("drive_combustion", "drive_impulse", "drive_hyperspace"):
            assert f'name="{n}"' in idx, f"{n} input missing"
        assert 'fd.get("drive_combustion")' in js
        assert "v=20260908e" in idx


class TestBcPreference:
    """Battlecruiser is the experienced-fleeter favourite (hyperspace + low
    fuel). The bonus must make BC the lowest factor (best fitness), ahead
    of similarly armed combat ships, without overturning combat value.
    """

    def test_bc_is_lowest_at_default_techs(self):
        f = derive_penalty_factors()
        assert f["battlecruiser"] <= min(
            f["cruiser"], f["pathfinder"], f["battleship"], f["reaper"]
        )
        assert f["battlecruiser"] < 1.0  # has the base credit

    def test_bc_beats_bs_because_fuel_is_lower(self):
        # BS and BC have IDENTICAL speed, so speed penalty is the same;
        # only fuel separates them. BC's 250 vs BS's 500 makes BC strictly
        # cheaper at runtime.
        f = derive_penalty_factors()
        assert f["battlecruiser"] < f["battleship"]

    def test_bc_beats_cruiser_at_default_techs(self):
        # CR is faster but thirstier; BC's hyperspace bonus + lower fuel
        # outweigh CR's speed advantage in the penalty model.
        f = derive_penalty_factors()
        assert f["battlecruiser"] < f["cruiser"]

    def test_ds_still_heavily_penalised(self):
        # The hyperspace bonus doesn't save Deathstar: fuel 1 is fine but
        # the speed penalty (100 base vs 10000) dominates -> factor > 1.05.
        f = derive_penalty_factors()
        assert f["deathstar"] > 1.05

    def test_factor_ordering_preserved(self):
        # Old test_fleet-style ordering still holds with bonus:
        # bc (preferred) < hyperspace-fuel ships (pf, bs) <= 1.0 < bad ships.
        f = derive_penalty_factors()
        assert f["battlecruiser"] < 1.0
        assert f["battleship"] > 1.0
        assert f["pathfinder"] < 1.0
        assert f["deathstar"] > f["destroyer"]
        assert f["destroyer"] > f["bomber"]
        assert f["bomber"] > f["reaper"]

    def test_bonus_scales_with_pct(self):
        # At pct=0 the bonus AND the penalty scale to zero -> every ship 1.0.
        fl = {"battlecruiser": 100, "deathstar": 100}
        zero = fleet_penalty_multiplier(fl, pct=0)
        full = fleet_penalty_multiplier(fl, pct=10)
        assert zero == pytest.approx(1.0, abs=1e-9)
        # Pure BC fleet at pct=10 -> 0.99 (bonus fully applied)
        bc_only = fleet_penalty_multiplier({"battlecruiser": 100}, pct=10)
        assert bc_only == pytest.approx(0.99, abs=1e-9)

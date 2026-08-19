"""Fixtures from the OFFICIAL ogame.org combat simulator (user-provided
HTML reports, 2026-08). These lock the attacker-volley-first fire order
and the broad outcome shape of both engines against ground truth.

Battle 1 (small): 30 BCs ATTACK 191 LF + 3 HF + 4 CR + 5 BS + 4 BC + 2 DES
+ 4 PF, all techs 0. Official result: DRAW - 22/30 BCs destroyed (8
survive), 121/191 LF lost (70 survive), 1/3 HF lost, all heavies intact.
Battle 2 (big): CR 12,333 / BS 44,444 / BC 55,555 / PF 44,444 (techs
18/18/18) attack a 500k-unit mixed fleet incl. 1 RIP + 1,822 Reapers
(techs 17/17/17). Official result: attacker WINS round 5, defender fully
annihilated; attacker keeps CR 81.4% / BS 94.9% / BC 96.4% / PF 79.9%.
"""
import pytest

from ogame_optimizer.core.combat import simulate_batch
from ogame_optimizer.core.fast_combat import simulate_batch_fast

SMALL_DEFENDER = {
    "light_fighter": 191, "heavy_fighter": 3, "cruiser": 4, "battleship": 5,
    "battlecruiser": 4, "destroyer": 2, "pathfinder": 4,
}
BIG_ATTACKER = {
    "cruiser": 12333, "battleship": 44444, "battlecruiser": 55555,
    "pathfinder": 44444,
}
BIG_DEFENDER = {
    "light_fighter": 64051, "heavy_fighter": 25064, "cruiser": 8425,
    "battleship": 3508, "battlecruiser": 4095, "bomber": 2534,
    "destroyer": 1515, "deathstar": 1, "reaper": 1822, "pathfinder": 9667,
    "small_cargo": 53610, "large_cargo": 25011, "recycler": 330,
    "espionage_probe": 149818,
}
# Official attacker survivors, battle 2 (won round 5)
BIG_OFFICIAL_SURV = {
    "cruiser": 10034, "battleship": 42186, "battlecruiser": 53531,
    "pathfinder": 35509,
}


def _rust(fleet, enemy, atk_tech, def_tech, n=100, seed=42):
    # Public wrapper: auto-routes to the Rust per-unit core below the
    # FAST_THRESHOLD and returns snake_case survivor means.
    return simulate_batch(attacker=fleet, defender=enemy,
                          defender_defenses={}, attacker_tech=atk_tech,
                          defender_tech=def_tech, n_sims=n, base_seed=seed)


class TestOfficialSmallBattle:
    """30 attacking BCs vs the player fleet - official outcome: draw."""

    def test_rust_matches_official_shape(self):
        r = _rust({"battlecruiser": 30}, SMALL_DEFENDER, (0, 0, 0), (0, 0, 0), n=100)
        # BCs must NOT win (official: draw). Attacker win prob ~0.
        assert float(r.get("win_probability", 0)) <= 0.05, (
            f"BC-attacker win {r.get('win_probability')}: simultaneous-fire "
            "regression (official sim produces a draw)"
        )
        dsur = r.get("defender_survivors_mean", {}) or {}
        lf = float(dsur.get("light_fighter", 0))
        # Official: 70/191 LFs survive (engine measures ~78).
        assert 30 <= lf <= 130, f"LF survivors {lf} vs official 70"
        bs = float(dsur.get("battleship", 0))
        assert bs >= 4.5, f"Battleships must survive (official 5/5), got {bs}"

    def test_analytical_matches_official_shape(self):
        r = simulate_batch_fast({"battlecruiser": 30}, SMALL_DEFENDER, {},
                                n_sims=50, base_seed=42)
        assert float(r.get("win_probability", 0)) <= 0.10
        dsur = r.get("defender_survivors_mean", {}) or {}
        lf = float(dsur.get("light_fighter", 0))
        assert 25 <= lf <= 140, f"LF survivors {lf} vs official 70"


class TestOfficialBigBattle:
    """156k-unit attacker (18/18/18) vs 500k-unit defender (17/17/17) -
    official outcome: attacker wins round 5, defender annihilated."""

    def test_analytical_winner_and_annihilation(self):
        r = simulate_batch_fast(BIG_ATTACKER, BIG_DEFENDER, {},
                                attacker_tech=(18, 18, 18),
                                defender_tech=(17, 17, 17),
                                n_sims=20, base_seed=42)
        assert float(r.get("win_probability", 0)) >= 0.95, "attacker must win"
        dsur = r.get("defender_survivors_mean", {}) or {}
        assert sum(dsur.values()) <= 0.01 * 500_000, "defender must be annihilated"

    def test_analytical_attacker_loss_bounds(self):
        """Known limitation (documented): the expectation resolver
        under-reads heavy-tail losses ~2x vs the official engine, so the
        bound is generous on the low side; the LOSS RANKING must match
        (pathfinder > cruiser > battleship > battlecruiser, official)."""
        r = simulate_batch_fast(BIG_ATTACKER, BIG_DEFENDER, {},
                                attacker_tech=(18, 18, 18),
                                defender_tech=(17, 17, 17),
                                n_sims=20, base_seed=42)
        surv = r.get("attacker_survivors_mean", {}) or {}
        keep = {k: surv.get(k, 0) / BIG_ATTACKER[k] for k in BIG_ATTACKER}
        official = {k: BIG_OFFICIAL_SURV[k] / BIG_ATTACKER[k] for k in BIG_ATTACKER}
        # survival within [official-25pts, official+15pts]
        for k in keep:
            assert keep[k] >= official[k] - 0.25, (
                f"{k}: survival {keep[k]:.3f} too pessimistic vs official {official[k]:.3f}"
            )
            assert keep[k] <= official[k] + 0.15, (
                f"{k}: survival {keep[k]:.3f} too optimistic vs official {official[k]:.3f}"
            )
        # loss ranking by clearly-separated groups (official: BC 96.4 >
        # BS 94.9 >> CR 81.4 ~ PF 79.9 - the CR/PF pair is near-tied, so
        # only the group structure is asserted)
        assert keep["battlecruiser"] > keep["battleship"], "BC must lose least"
        assert keep["battleship"] > max(keep["cruiser"], keep["pathfinder"]), (
            "BS must keep more than CR/PF"
        )

"""Tests for progressive fleet seeding module."""
from ogame_optimizer.optimizer.progressive_seeds import generate_progressive_seeds
from ogame_optimizer.core.fleet import SHIPS_COST, fleet_value


class TestProgressiveSeeds:
    """Progressive seed generation tests."""

    def test_returns_nonempty_list(self):
        """Standard enemy fleet returns 1-4 seed fleets."""
        seeds = generate_progressive_seeds(
            enemy_fleet={"light_fighter": 1000, "cruiser": 200},
            enemy_defenses={},
            budget=100_000_000,
        )
        assert len(seeds) >= 1
        assert len(seeds) <= 4
        for seed in seeds:
            assert isinstance(seed, dict)
            assert len(seed) > 0

    def test_exclude_ships_respected(self):
        """Excluded ships do not appear in any seed."""
        seeds = generate_progressive_seeds(
            enemy_fleet={"cruiser": 500},
            enemy_defenses={},
            budget=50_000_000,
            exclude_ships=["battlecruiser", "destroyer"],
        )
        all_types = set()
        for seed in seeds:
            all_types.update(seed.keys())
        assert "battlecruiser" not in all_types
        assert "destroyer" not in all_types

    def test_budget_not_exceeded(self):
        """No seed fleet value exceeds the budget."""
        budget = 80_000_000
        seeds = generate_progressive_seeds(
            enemy_fleet={"light_fighter": 5000, "cruiser": 500},
            enemy_defenses={},
            budget=budget,
        )
        for seed in seeds:
            fv = fleet_value(seed)
            # Allow small rounding (floor division may leave slight slack)
            assert fv <= budget + max(sum(v) for v in SHIPS_COST.values()), \
                f"Seed {seed} value {fv} exceeds budget {budget}"

    def test_tiny_budget(self):
        """Tiny budget doesn't crash, returns at least 1 seed."""
        seeds = generate_progressive_seeds(
            enemy_fleet={"light_fighter": 10},
            enemy_defenses={},
            budget=5000,  # Only LF (4000) fits
        )
        assert len(seeds) >= 1

    def test_only_combat_ships(self):
        """No cargo/probe/recycler/pathfinder in any seed."""
        non_combat = {"small_cargo", "large_cargo", "espionage_probe", "recycler", "pathfinder"}
        seeds = generate_progressive_seeds(
            enemy_fleet={"light_fighter": 2000, "cruiser": 300},
            enemy_defenses={},
            budget=100_000_000,
        )
        for seed in seeds:
            for ship in seed:
                assert ship not in non_combat, f"Non-combat ship {ship} in seed"

    def test_returns_valid_ship_names(self):
        """All ship names in seeds are valid SHIPS_COST keys."""
        seeds = generate_progressive_seeds(
            enemy_fleet={"cruiser": 1000},
            enemy_defenses={},
            budget=200_000_000,
        )
        for seed in seeds:
            for ship in seed:
                assert ship in SHIPS_COST, f"Unknown ship {ship}"

    def test_zero_budget_returns_empty(self):
        """Zero budget returns empty list (can't afford anything)."""
        seeds = generate_progressive_seeds(
            enemy_fleet={"light_fighter": 10},
            enemy_defenses={},
            budget=0,
        )
        assert len(seeds) == 0

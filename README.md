# OGame Fleet Auto-Optimizer

A local web tool that recommends the optimal counter-fleet for any OGame battle. Paste the enemy fleet/defense report, set your budget and server settings, and get a battle-tested fleet composition with loss estimates, debris calculations, and recycler planning.

## Features

### Fleet Optimization
- **Dual-mode**: Optimize your **attack fleet** or your **defending forces**
- **Progressive seeding**: Systematically tests pure single-type fleets, then 50/50 two-type combos, then launches GA from the best compositions
- **Multi-round GA**: Explore (10 sims/eval, 20% mutation) then Refine (50 sims, 10%) then Polish (100 sims, 5%), never regresses
- **Budget enforcement**: Proportional scaling maintains composition ratio within budget
- **Ship exclusion**: Checkbox any ship type to exclude from recommendations
- **Optimization targets**: "Minimize Loss" or "Maximize Profit" (effective loss = raw x (1 - debris%))

### Combat Model (Full OGame Accuracy)
- **13 ship types**: LF, HF, Cruiser, Battleship, Battlecruiser, Bomber, Destroyer, Deathstar, Small/Large Cargo, Espionage Probe + Pathfinder + Recycler
- **8 defense types**: Rocket Launcher, Light/Heavy Laser, Gauss, Ion, Plasma Turret, Small/Large Shield Dome
- **6 combat rounds** with simultaneous resolution (both sides fire at full start-of-round strength)
- **Rapidfire** chains (BCs get 3-7x extra shots vs fodder)
- **Shield bounce** < 1% rule (LF cannot damage Large Shield Dome)
- **Armor thresholds** at 30% / 70%
- **Draws** supported
- **Analytical fast-path**: For large fleets (>500 ships), uses expected-damage-with-Gaussian-noise O(types^2) instead of per-unit Monte Carlo -- 500x faster

### Results and Analysis
- **Impact-based coloring**: Each recommended ship is tagged by combat sensitivity:
  - GREEN **Critical**: Removing increases losses by >20% -- definitely build
  - **Important** (subtle green): 5-20% impact
  - **Dead weight** (red): Removing and redistributing budget actually *reduces* losses -- skip these
- **Debris calculation**: Configurable debris % (default 80%), includes Metal + Crystal + Deuterium
- **Net profit**: debris_total - loss with +/- percentage
- **Recycler planning**: Based on Hyperspace Tech (+5%/level) and Collector class (+25%)
- **Win probability** with 95% confidence interval

### Web UI
- **Paste parser**: Paste OGame reports directly (handles comma-separated numbers)
- **Server settings**: Debris %, Deuterium in debris, Hyperspace Tech, Collector class
- **Dark theme** with metric card layout
- **Refine button**: Re-optimize from previous result with different budget
- **File logging**: Rotating log (5MB x 5) with phase-by-phase traces

## Prerequisites

- **Rust** (stable, 1.70+) -- https://rustup.rs
- **Python** 3.10+
- **maturin** -- pip install maturin

## Quick Start

```bash
cd ogame-fleet-optimizer

python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
pip install maturin
maturin develop --release

uvicorn ogame_optimizer.api.app:app --host 127.0.0.1 --port 8000 --reload
```

Open http://127.0.0.1:8000 in your browser.

**Windows shortcut**: Run `run.bat` (starts server with auto-reload). Run `test.bat` for tests.

## Usage

### Web UI

1. **Paste enemy fleet/defenses** or enter counts manually
2. **Set tech levels** for both sides (Weapon / Shield / Armor, 0-25+)
3. **Pick budget multiplier**: 0.5x to 5.0x
4. **Configure server settings**: Debris %, Deuterium in debris, Hyperspace Tech, Collector class
5. **Choose optimization target**: Minimize Loss or Maximize Profit
6. **Optionally exclude ships** you don't want
7. Click **Optimize** (takes 2-10s)

### Reading the Results

| Metric | What it means |
|--------|--------------|
| **Recommended Fleet** | Ship types and counts to build and send |
| **Impact %** | How much each ship helps (green) or hurts (red) |
| **Win Probability** | Green >=95%, yellow 80-95%, red <80% |
| **Expected Loss** | Mean +/- stddev across 200-1000 simulations |
| **Debris Total** | Metal + Crystal + Deuterium at your debris % |
| **Net Profit** | debris - loss (positive = profitable raid) |
| **Recyclers Needed** | Based on Hyperspace Tech and Collector class |

### API

**POST /api/optimize** -- Full optimization:

```json
{
  "enemy_fleet": {"ships": {"light_fighter": 5772, "cruiser": 1709}},
  "enemy_defenses": {"defenses": {"rocket_launcher": 100}},
  "attacker_tech": {"weapon": 15, "shield": 12, "armor": 14},
  "defender_tech": {"weapon": 10, "shield": 10, "armor": 10},
  "budget_multiplier": 1.0,
  "mode": "attack",
  "debris_pct": 0.8,
  "deuterium_in_debris": true,
  "optimization_target": "maximize_profit",
  "hyperspace_tech": 12,
  "collector_class": true,
  "final_sims": 1000
}
```

Response includes `recommended_fleet`, `fleet_analysis` (per-ship impact tags), `expected_loss_mean`, `win_probability`, `debris_total`, `net_profit`, `recyclers_needed`.

**POST /api/combat** -- Run N simulations of a specific battle.

**GET /api/ships**, **GET /api/defenses** -- Full metadata.

## How It Works

### Optimization Pipeline

1. **Budget**: Enemy fleet + defense value x multiplier
2. **Greedy** (~1s): Counter-ratio init via rapidfire + local search
3. **Progressive + GA** (~5s): Single-type seeds -> two-type combos -> multi-round GA (explore/refine/polish)
4. **Sensitivity Analysis** (~1s): Tag each ship critical/important/negligible/dead_weight
5. **Final validation** (200-1000 sims): Tight confidence interval

### Combat Resolution

- **Small fleets** (<=500 units): Rust Monte Carlo via PyO3
- **Large fleets** (>500 units): Python analytical resolver -- O(types^2), 500x faster

### Tech Stack

| Layer | Technology |
|-------|-----------|
| Combat core | Rust (PyO3, rand, rayon) |
| Optimization | Python (greedy + GA) |
| API | FastAPI + Pydantic |
| UI | Vanilla JS + Jinja2 |
| Bridge | maturin / PyO3 |

## Testing

```bash
.venv/Scripts/activate
pytest python_tests/ -v    # 142 Python tests
cargo test --release       # 52 Rust tests
```

## Project Layout

```
ogame-fleet-optimizer/
  Cargo.toml                   # Rust deps
  src/                         # Rust combat core
    lib.rs                     # PyO3 module
    combat.rs                  # simulate_combat, simulate_batch
    ships.rs                   # 13 ships + 8 defenses + stats
    rapidfire.rs               # Rapidfire table
  python/ogame_optimizer/
    core/                      # combat.py, fast_combat.py, fleet.py, tech.py
    optimizer/                 # greedy.py, genetic.py, progressive_seeds.py,
                                orchestration.py, statistics.py, objective.py
    api/                       # app.py, routes.py, schemas.py
    web/                       # templates/, static/
    logging_config.py
  python_tests/                # 142 pytest tests
  run.bat / test.bat           # Windows launchers
```

## Limitations

- No lifeform techs
- No officers / admirals / commanders
- No fuel cost / flight time
- No round-by-round visualization
- No OGame API integration (paste only)
- Resource model: raw Metal + Crystal + Deuterium (no weighting)

## License

MIT

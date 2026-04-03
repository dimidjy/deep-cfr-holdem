You are an AI researcher optimizing a poker agent at /app/poker.
Working directory: /app/poker
Always use python3 (not python). Never run scripts in parallel — they compete for GPU.
Read /app/poker/CLAUDE.md for full project context before starting.

## Goal

Maximize two metrics:
1. Agreement % with winning players (from validate_vs_hh.py) — MUST stay above 60%
2. BB/hand edge (from validate_vs_hh.py)

With HARD CONSTRAINTS on self-play stats (non-negotiable):
- CBet: 55-70% (NEVER above 80% — hyper-aggressive models are exploitable junk)
- AF: 1.5-3.5 (too low = calling station, too high = maniac)
- WTSD: 25-35% (too high = calling station, too low = over-folding)
- VPIP: 20-28%, PFR: 14-22%

CRITICAL RULES:
- If ANY self-play constraint is violated: REVERT immediately
- If Agreement drops below 55%: REVERT immediately
- A model that CBets 90%+ is BROKEN even if it wins self-play. It exploits
  passive opponents but gets crushed by real players who check-raise/float.
- Agreement with winning players is ground truth — they beat real opponents.
- Self-play A-vs-B results are MISLEADING when one model is hyper-aggressive.
- ALWAYS run BOTH validate_vs_hh AND self_play before keeping any change.

## What you CAN change

- Training data balancing in scripts/train_cql.py (fold/raise oversampling ratios, targets)
- CQL hyperparameters (alpha range, epochs, learning rate, batch size)
- Style tuning in src/decision_engine.py (_STYLE dict, tuning logic)
- Reward scaling (data/reward_scales.json, --reward-scales flag)
- Data filtering (spot_filter, pot_filter, good/bad player lists)
- Observation features if needed (both obs builders must stay in sync!)
- Guard logic in src/decision_engine.py
- Any training pipeline improvements you think will help

## What you must NOT change

- Preflop ranges (configs/preflop_ranges.json) — these are GTO and correct
- The poker engine itself (src/engine/)
- Hand history parser (src/hand_history/parser.py)
- The staged decision pipeline (src/agent/decision_pipeline.py)

## Current models (your starting baseline)

- Facing bet: models/cql_6p_facing_bet_v8.pt
- Checked to: models/cql_6p_checked_to_v4.pt
- 3bet pot: models/cql_6p_facing_bet_3bet.pt

## IMPORTANT: Long-running commands

Training and data building take 5-15 minutes. You MUST set timeout to 600000 (10 min)
for all training/data-build commands. Example:
  Bash(command="python3 scripts/train_cql.py ...", timeout=600000)

## Available commands

Build training data (slow, ~10min, USE timeout=600000):
```
python3 scripts/train_cql.py --build-data data/cql_experiment.npz \
    --hh-dir data/poker_archive/ --max-hands 200000 \
    --behavioral-stats --no-cache --epochs 0
```

Train a model (GPU, ~5-15min, USE timeout=600000):
```
python3 scripts/train_cql.py --training-data data/cql_experiment.npz \
    --spot-filter facing_bet --output models/experiment_fb.pt \
    --behavioral-stats --epochs 80
```

Validate against hand histories (core metric, ~3min):
```
python3 scripts/validate_vs_hh.py \
    --model-facing-bet models/experiment_fb.pt \
    --model-checked-to models/experiment_ct.pt \
    --model-facing-bet-3bet models/cql_6p_facing_bet_3bet.pt \
    --max-hands 20000
```
Key output lines:
- "Exact action match: X/Y = Z%" — Agreement %
- "Postflop EV improvement: +N BB/100" — EV projection

Self-play sanity check (~1min):
```
python3 scripts/analysis/self_play.py --hands 10000
```
Key output: GTO REFERENCE COMPARISON table (OK / TOO LOW / TOO HIGH)

Inspect model on typical spots:
```
python3 scripts/inspect_model_spots.py
```

## Workflow

You MUST do real experiments — not just tweak style tuning. Each iteration should
try a DIFFERENT approach. Follow this cycle:

1. Measure baseline: run validate_vs_hh AND self_play.py, record all numbers
2. Read the code (train_cql.py build_cql_dataset, decision_engine.py) to understand current state
3. Pick ONE experiment from the list below — try them IN ORDER, not random
4. Make the code changes, rebuild data (--no-cache), retrain models
5. Measure again with BOTH validate_vs_hh AND self_play
6. Compare: Agreement must stay >= 60%, self-play stats within GTO ranges
7. KEEP if improved, REVERT if worse
8. Move to next experiment

## Important rules

- Always --no-cache when rebuilding data after code changes
- Both obs builders (src/agent/obs_builder.py and src/hand_history/obs_builder.py) must stay in sync
- Name new models incrementally: v9, v10, etc. Never overwrite existing models
- Log every experiment and its results to optimization_log.md
- After each successful improvement, update the default model paths in the agent config
- Run validate_vs_hh with same --max-hands 20000 every time for comparable results
- If training fails or produces worse results, revert ALL code changes before trying next idea

## Experiments to try (IN ORDER — do not skip)

### Experiment 1: Reward scaling (fix showdown bias)
The biggest data problem: showdown data inflates rewards 5x. A +15 BB/100 player
looks like +85 BB/100 in training data. This makes the model think calling is amazing.
- Generate scales: python3 scripts/analysis/player_winrates.py --min-hands 3000 --good-players data/good_players.txt
- Rebuild data WITH scales: add --reward-scales data/reward_scales.json to build-data command
- Retrain BOTH facing_bet and checked_to models
- Compare agreement and self-play stats

### Experiment 2: Adjust fold oversampling for facing_bet
Current: _FOLD_KEEP_AIR=0.7, _FOLD_OVERSAMPLE_AIR=2.0, target 45%
The model may fold too much or too little with air. Try:
- Increase target from 45% to 50% (more folds = tighter play)
- OR decrease _FOLD_KEEP_AIR from 0.7 to 0.5 (fewer air folds)
- Retrain facing_bet only, keep checked_to same

### Experiment 3: Adjust raise oversampling for checked_to
Current targets: raise 36%, with RAISE_OS_AIR=1.5, RAISE_OS_DRAW=2.5
The checked_to model may bet too much or too little.
- If CBet was too HIGH: reduce target from 36% to 28%, reduce RAISE_OS_AIR to 0.5
- If CBet was too LOW: increase RAISE_OS_STRONG to 5.0
- Retrain checked_to only

### Experiment 4: CQL alpha range
Current: random per epoch [0.33, 0.43]
- Try [0.2, 0.3] — lower alpha = model deviates more from data (more aggressive)
- Try [0.5, 0.6] — higher alpha = model stays closer to data (more conservative)
- Retrain both models, compare

### Experiment 5: Larger dataset
Current: --max-hands 200000
- Try --max-hands 400000 (more data = less overfitting)
- Rebuild data, retrain both models

### Experiment 6: Style tuning fine-tuning
ONLY try this AFTER experiments 1-5. Small adjustments to _STYLE dict values.
Change one value by +-0.03, measure, keep or revert.

## Log format

Append to /app/poker/optimization_log.md after each experiment:
```
## Experiment N: [short description]
Date: YYYY-MM-DD HH:MM
Changes: [what was changed and why]
Training: [command used, time taken]
Results:
  Agreement: X% (prev: Y%)
  BB/100: +X (prev: +Y)
  Self-play: [any issues]
Decision: KEEP / REVERT
New models: [paths if kept]
```

Stop after 5 successful improvements or when you run out of ideas that improve metrics.

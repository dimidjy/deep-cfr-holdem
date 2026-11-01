# Architecture

## Data flow

```
             ┌────────────── training ──────────────┐
 engine ───▶ Deep CFR traversals ───▶ advantage / strategy nets ───▶ checkpoint
 (src/engine)   (scripts/train_gpu.py, src/training)                     │
                                                                         ▼
 hand histories ─▶ parser ─▶ obs builder ─▶ supervised / CQL / IQL ─▶ checkpoint
 (src/hand_history)                          (scripts/train_*.py)        │
                                                                         ▼
                       DecisionEngine + staged decision pipeline  ◀──────┘
                       (src/decision_engine.py, src/agent/decision_pipeline.py)
                                   │
         ┌──────────────┬──────────┴─────────┬───────────────────┐
     self-play      tournament         HH validation        spot evaluator / web UI
```

## Observation schema

`src/obs_schema.py` is the single source of truth for block widths, the
ordered feature names and the derived `IDX_*` constants. Three builders must
produce identical vectors for the same game state:

| Builder | Used by |
|---|---|
| `src/engine/fast_enhanced_engine.py` | Deep CFR training |
| `src/hand_history/obs_builder.py` | offline datasets from hand histories |
| `src/agent/obs_builder.py` | runtime agent (spot evaluator, web UI) |

The 6-max layout: position (8) + pot odds (3) + hand strength (5) + stack (2)
+ equity block (9) + street (4) + game state (6) + betting history (5) +
behavioral stats (8) + betting line (3) = **53 dims**. The equity block holds
hand strength, made-hand category, flush/straight participation, top pair,
board pairing/flush/straight texture and rank within category.

`tests/test_obs_parity.py` checks layout width across the three builders,
the index→feature-name mapping, multiway equity agreement between engine
and agent, and made-hand label parity.

## Action space

`0 Fold, 1 Call, 2 Raise ½ pot, 3 Raise pot, 4 All-in, 5 Raise min (2BB),
7 Raise ⅓ pot`. The networks output the first four (fold/call/½/pot). All-in
enters only through data-driven promotion, and min-raise only through
preflop ranges. Strategy arrays always have `NUM_ACTIONS = 6` slots.

## Decision pipeline

`build_pipeline()` in `src/agent/decision_pipeline.py` declares the order in
one list. Each stage is a small class with `applies()` and `run()` over a
`DecisionState` bundle:

1. `ModelStrategyStage`: preflop lookup / ranges, or model inference
2. `ArchetypeSkewStage`: optional opponent-style skew (evaluation opponents)
3. `BluffSuppressionStage`, `EVSearchStage`: optional guards (off by default)
4. `AllInPromoteStage`, `AllInCallFilterStage`: stack-commitment guards
5. `SampleActionStage`: argmax or capped sampling
6. `BlockBetPromoStage`
7. `PostflopSizingStage`: data-driven bet size and all-in promotion
8. `RaiseThirdSizingStage`

The pipeline never touches a `GameState`, so self-play fills the same bundle
from the simulator and runs exactly the code the spot evaluator runs.

## Lookups

- **Preflop** (`src/preflop_lookup.py`): action distributions per spot,
  keyed by `(hole class, position, action history, stack bucket,
  num_to_act)` with an L0→L2 back-off hierarchy. Hand histories only show
  hole cards at showdown, so fold frequencies are recovered with a
  *play-rate gate* that uses AA's per-combo frequency as the deal-rate
  reference. Static ranges (`src/preflop_ranges.py`) are the fallback.
- **Postflop sizing** (`src/postflop_sizing.py`): the model picks the action
  and the lookup only picks the size. It never changes fold/call into a
  raise. Keyed by `(street, hand bucket, draw, board texture, IP/OOP, SPR,
  action context)`.
- Spot-key construction lives in one module per lookup
  (`*_spot_key.py`), shared by the build scripts and the runtime.

## Evaluation

`scripts/tournament.py` runs three phases over a directory of checkpoints,
parallelized with `ProcessPoolExecutor` and resumable (skip-if-exists):

1. spot validation against annotated spots
2. hand-history validation (agreement %, EV vs. showdown outcomes)
3. round-robin head-to-head self-play (50k hands per pair)

It writes `summary.{json,csv,md}` with the H2H matrix.
`scripts/benchmark_baselines.py` plays a checkpoint against scripted
opponents (random, always-call, always-raise) and archetype-skewed copies of
itself. It reports BB/100 with a 95% confidence interval computed from
per-hand results, with one sample per hand summed over the model's three seats.
`scripts/make_figures.py` renders the README figures from `docs/results/`.

`scripts/analysis/self_play.py` reports VPIP / PFR / 3-bet / CBet / AF / WTSD
and, with `--range-analysis`, 13×13 range grids per position and history.

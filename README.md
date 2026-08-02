# deep-cfr-holdem

Research code for training and evaluating agents for **6-max No-Limit Texas
Hold'em**, an imperfect-information, multi-agent game. It covers the whole
loop: a fast game engine, Deep CFR self-play training on GPU, offline RL from
hand-history data, a staged decision pipeline, and an evaluation harness
(self-play statistics, head-to-head tournaments, GTO solver checks).

Built March–August 2026 as a personal research project, about 57k lines of
Python.

## Highlights

- **Deep CFR on GPU.** External-sampling MCCFR traversals run in
  multiprocess workers. The networks are residual advantage and strategy
  nets, the buffers are reservoir-sampled, and training resumes from
  checkpoints. Reward shaping (bluff/value/draw-aware bonuses) is all
  config-driven. See `scripts/train_gpu.py` and `src/training/`.
- **Self-play against a K-best opponent pool.** Burst-loop training keeps a
  diverse pool of past checkpoints across the aggression axis, so the trainee
  does not overfit to a single opponent style (`scripts/train_burst_loop.py`).
- **Offline RL and imitation.** Supervised policy, CQL and IQL trainers learn
  from parsed hand histories, with cohort filtering, counterfactual folds
  and reward weighting (`scripts/train_supervised.py`, `train_cql.py`,
  `train_iql.py`). Imitation on annotated spots can be mixed into CFR
  training.
- **Fast engine.** A NumPy engine with 53-dim observations: position, pot
  odds, hand strength, equity, board texture, betting line and behavioral
  stats. A neural equity model replaces Monte Carlo equity in the hot path
  (`src/engine/`, `scripts/train_equity_model.py`).
- **One observation schema.** Three code paths build observations: the
  training engine, the hand-history generator and the runtime agent.
  `src/obs_schema.py` owns the layout and `tests/test_obs_parity.py`
  enforces that all three agree. See *Lessons learned* for why.
- **Staged decision pipeline.** Model strategy, then conditional overrides,
  then data-driven sizing, as an ordered list of small stages
  (`src/agent/decision_pipeline.py`). Self-play and spot evaluation run the
  same pipeline.
- **Evaluation harness.** Self-play GTO-band statistics (VPIP, PFR, CBet, AF,
  WTSD), a parallel round-robin tournament with resumable jobs, validation
  against hand histories and the public Pluribus logs, and a TexasSolver
  bridge for GTO spot checks (`scripts/tournament.py`,
  `scripts/analysis/self_play.py`, `src/gto/`).
- **Tooling.** A FastAPI web UI to launch jobs, stream metrics and explore
  model decisions spot by spot, plus a parametric spot library and
  LLM-assisted spot labeling.

## Repository layout

```
src/
  engine/          fast game engine + enhanced observation features
  training/        Deep CFR (networks, buffers, batched traversal)
  agent/           game-state model, runtime observation builder, decision pipeline
  models/          unified adapters for Deep CFR / CQL / supervised checkpoints
  hand_history/    hand-history parser and observation builder
  gto/             TexasSolver bridge (spot inputs, range builder, extraction)
  search/          depth-limited and Monte Carlo rollout search
  decision_engine.py, preflop_*.py, postflop_*.py   lookups and sizing
  obs_schema.py    single source of truth for the observation layout
scripts/           training, data building, evaluation and analysis CLIs
tests/             355 tests (engine, CFR utils, lookups, obs parity, GTO)
webui/             job runner and spot explorer
configs/           training configs and preflop range tables
```

Details are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and
[docs/TRAINING.md](docs/TRAINING.md).

## Quick start

```bash
pip install -r requirements.txt
make test                      # 355 tests, ~20 s on CPU

make train                     # Deep CFR on GPU (configs/training_config.json)
make selfplay MODEL=models/best/checkpoint_iter_4000.pt HANDS=10000
make tournament                # round-robin over models/best
make webui                     # http://127.0.0.1:8000
```

Checkpoints, hand-history archives and the SQLite lookup databases are not
checked in (see `.gitignore`). The scripts that build them are in `scripts/`
(`build_training_data.py`, `build_preflop_lookup.py`,
`build_postflop_lookup.py`, `train_equity_model.py`).

## Results

Tournament of eight late-stage checkpoints from one training arc. Each
model played every other model head-to-head for 50k hands, plus 100k hands
of hand-history validation (`tournament/tornament_v34/summary.md`):

| Model | Spot % | H2H edge (BB/100) | W-T-L | CBet | AF | WTSD |
|---|---:|---:|:---:|---:|---:|---:|
| iter_4000 | 89.9 | **+4.45** | 8-5-1 | 61.5 | 2.26 | 32.0 |
| iter_4000 (2) | 88.0 | +3.91 | 9-4-1 | 62.8 | 2.31 | 32.4 |
| iter_4200 | 90.8 | +3.64 | 7-5-2 | 68.4 | 2.47 | 31.2 |
| iter_3900 | 85.3 | −1.00 | 3-6-5 | 58.9 | 2.79 | 30.2 |
| iter_4100 | 90.4 | −3.31 | 1-6-7 | 62.7 | 2.34 | 32.1 |

The selected checkpoint (`iter_4000`) lands inside all four postflop target
bands used for model selection (CBet 55–70%, AF 2.0–3.5, WTSD 25–32%).

## Lessons learned

- **Train/serve skew is the expensive bug.** Equity at `obs[18]` was
  computed heads-up in training but multiway at runtime, so a 6-way top pair
  read 0.70 in one and 0.19 in the other. The fix was structural: one
  schema module, and a parity test that compares all three builders on
  the same states.
- **Head-to-head wins can mislead.** Late checkpoints drift toward
  over-aggression (CBet 62% → 79% between iterations 3400 and 4200). They
  beat passive siblings in self-play but are exploitable. Model selection
  therefore requires self-play statistics inside target bands *before*
  looking at head-to-head edge.
- **Shaping needs controlled A/B tests.** Several reward shapers looked good
  in isolation and were reverted after tournament comparison. Keeping them
  as config flags (default off) made that cheap.

## Scope

This repository is a game-AI and reinforcement-learning research project.
It contains no code for connecting to, reading from or acting on any online
poker platform.

## License

MIT

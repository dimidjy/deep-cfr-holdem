"""Web runner for the poker project's scripts.

Start it with:

    python3 -m webui.server                 # http://0.0.0.0:8000
    python3 -m webui.server --port 9000

There is NO authentication. Bind it to localhost and reach it over an SSH
tunnel (`ssh -L 8000:localhost:8000 user@server`), or put it behind a
reverse proxy with auth. It can launch training jobs and delete nothing,
but anything that can start a job can also burn your GPU for days.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from webui.jobs import MANAGER, REPO_ROOT

STATIC = Path(__file__).resolve().parent / 'static'
CONFIG_PATH = REPO_ROOT / 'configs' / 'training_config.json'

app = FastAPI(title='Poker Runner')


# ─────────────────────────────────────────────────────────────────────────────
# Defaults for the forms — read from the real config so the UI never drifts
# from what the scripts would do on their own.
# ─────────────────────────────────────────────────────────────────────────────
def _load_config() -> Dict[str, Any]:
    try:
        return json.loads(CONFIG_PATH.read_text())
    except Exception:
        return {}


def _list_checkpoints() -> List[str]:
    out = []
    for d in ('models/best', 'models', 'models/v3'):
        p = REPO_ROOT / d
        if not p.is_dir():
            continue
        for f in sorted(p.glob('*.pt')):
            rel = str(f.relative_to(REPO_ROOT))
            if rel not in out:
                out.append(rel)
    return out


@app.get('/api/defaults')
def defaults() -> Dict[str, Any]:
    cfg = _load_config()
    tr = cfg.get('training', {})
    opp = cfg.get('opponent', {})
    return {
        'config_path': str(CONFIG_PATH.relative_to(REPO_ROOT)),
        'training': {
            'num_iterations': tr.get('num_iterations', 7000),
            'traversals_per_iteration': tr.get('traversals_per_iteration', 5000),
            'batch_size': tr.get('batch_size', 65536),
            'train_passes': tr.get('train_passes', 7),
            'learning_rate': tr.get('learning_rate', 0.0003),
            'buffer_size': tr.get('buffer_size', 10000000),
            'checkpoint_interval': tr.get('checkpoint_interval', 100),
            'lr_scheduler_t_max': tr.get('lr_scheduler_t_max', 5000),
            'imitation_weight': tr.get('imitation_weight', 0.0),
            'imitation_passes': tr.get('imitation_passes', 1),
            'imitation_batch_size': tr.get('imitation_batch_size', 256),
            'random_preflop_folds': tr.get('random_preflop_folds', '0:100'),
        },
        'opponent': {
            'deck_reshuffle': opp.get('deck_reshuffle', True),
            'bluff_penalty': opp.get('bluff_penalty', 0.0),
            'call_bonus': opp.get('call_bonus', 0.0),
            'value_bonus': opp.get('value_bonus', 0.0),
            'fold_penalty': opp.get('fold_penalty', 0.0),
            'draw_call_boost': opp.get('draw_call_boost', 0.0),
            'draw_bet_boost': opp.get('draw_bet_boost', 0.0),
            'barrel_fold_bonus': opp.get('barrel_fold_bonus', 0.0),
            'mid_pair_fold_bonus': opp.get('mid_pair_fold_bonus', 0.0),
            'allin_monster_rate': opp.get('allin_monster_rate', 0.0),
            'allin_monster_threshold': opp.get('allin_monster_threshold', 0.5),
            'station_rate': opp.get('station_rate', 0.0),
            'station_call_boost': opp.get('station_call_boost', 1.75),
            'bluff_draw_exempt': opp.get('bluff_draw_exempt', True),
        },
        'checkpoints': _list_checkpoints(),
        'equity_model': 'models/equity_model_v3.pt'
        if (REPO_ROOT / 'models/equity_model_v3.pt').exists() else '',
        'cpu_count': __import__('os').cpu_count(),
        # Surfaced in the UI as warnings — they are real project pitfalls.
        'notes': [
            'Only one job runs at a time — scripts compete for GPU/CPU.',
            'Pass an equity model: without it obs[18] is heads-up equity even '
            'in multiway pots, which does not match the runtime observation builder.',
            'Stop training if self-play CBet drifts above 70% — that is the '
            'known over-aggression failure mode.',
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Request models
# ─────────────────────────────────────────────────────────────────────────────
class TrainReq(BaseModel):
    # config-level (written into a per-run config file)
    num_iterations: int = 7000
    traversals_per_iteration: int = 5000
    batch_size: int = 65536
    train_passes: int = 7
    learning_rate: float = 0.0003
    buffer_size: int = 10000000
    checkpoint_interval: int = 100
    lr_scheduler_t_max: int = 5000
    imitation_weight: float = 0.0
    imitation_passes: int = 1
    imitation_batch_size: int = 256
    random_preflop_folds: str = '0:100'
    # opponent / shapers (config-level too)
    deck_reshuffle: bool = True
    bluff_penalty: float = 0.0
    call_bonus: float = 0.0
    value_bonus: float = 0.0
    fold_penalty: float = 0.0
    draw_call_boost: float = 0.0
    draw_bet_boost: float = 0.0
    barrel_fold_bonus: float = 0.0
    mid_pair_fold_bonus: float = 0.0
    allin_monster_rate: float = 0.0
    allin_monster_threshold: float = 0.5
    station_rate: float = 0.0
    station_call_boost: float = 1.75
    bluff_draw_exempt: bool = True
    # CLI-level
    workers: int = 0                     # 0 = script default
    device: str = ''                     # '' = auto
    models_dir: str = ''
    resume: str = ''
    equity_model: str = ''
    kbest_seed: List[str] = []
    kbest_update_interval: int = 0
    buffer_save_interval: int = 0
    preflop_ranges: bool = True
    label: str = ''


class ValidateReq(BaseModel):
    kind: str                            # self_play | validate_hh | spots | spot_logs
    model: str
    hands: int = 10000
    max_hands: int = 50000
    range_analysis: bool = False
    label: str = ''


def _fmt(v: Any) -> str:
    return str(v)


@app.post('/api/run/train')
def run_train(req: TrainReq) -> Dict[str, Any]:
    if MANAGER.is_running():
        raise HTTPException(409, 'A job is already running.')

    run_dir = MANAGER.new_run_dir('train')

    # Build a per-run config so the run is reproducible and auditable.
    cfg = _load_config()
    cfg.setdefault('training', {}).update({
        'num_iterations': req.num_iterations,
        'traversals_per_iteration': req.traversals_per_iteration,
        'batch_size': req.batch_size,
        'train_passes': req.train_passes,
        'learning_rate': req.learning_rate,
        'buffer_size': req.buffer_size,
        'checkpoint_interval': req.checkpoint_interval,
        'lr_scheduler_t_max': req.lr_scheduler_t_max,
        'imitation_weight': req.imitation_weight,
        'imitation_passes': req.imitation_passes,
        'imitation_batch_size': req.imitation_batch_size,
        'random_preflop_folds': req.random_preflop_folds,
    })
    cfg.setdefault('opponent', {}).update({
        'deck_reshuffle': req.deck_reshuffle,
        'bluff_penalty': req.bluff_penalty,
        'call_bonus': req.call_bonus,
        'value_bonus': req.value_bonus,
        'fold_penalty': req.fold_penalty,
        'draw_call_boost': req.draw_call_boost,
        'draw_bet_boost': req.draw_bet_boost,
        'barrel_fold_bonus': req.barrel_fold_bonus,
        'mid_pair_fold_bonus': req.mid_pair_fold_bonus,
        'allin_monster_rate': req.allin_monster_rate,
        'allin_monster_threshold': req.allin_monster_threshold,
        'station_rate': req.station_rate,
        'station_call_boost': req.station_call_boost,
        'bluff_draw_exempt': req.bluff_draw_exempt,
    })
    if req.models_dir:
        cfg.setdefault('paths', {})['models'] = req.models_dir
    run_cfg = run_dir / 'config.json'
    run_cfg.write_text(json.dumps(cfg, indent=2))

    cmd = ['python3', 'scripts/train_gpu.py', '--config', str(run_cfg)]
    if req.workers:
        cmd += ['--workers', str(req.workers)]
    if req.device:
        cmd += ['--device', req.device]
    if req.models_dir:
        cmd += ['--models-dir', req.models_dir]
    if req.resume:
        cmd += ['--resume', req.resume]
    if req.equity_model:
        cmd += ['--equity-model', req.equity_model]
    if req.kbest_seed:
        cmd += ['--kbest-seed'] + list(req.kbest_seed)
    if req.kbest_update_interval:
        cmd += ['--kbest-update-interval', str(req.kbest_update_interval)]
    if req.buffer_save_interval:
        cmd += ['--buffer-save-interval', str(req.buffer_save_interval)]
    if not req.preflop_ranges:
        cmd += ['--no-preflop-ranges']

    label = req.label or f'train {req.num_iterations} iters'
    job = MANAGER.start('train', label, cmd, req.model_dump(), run_dir)
    return {'ok': True, 'job': job.public()}


@app.post('/api/run/validate')
def run_validate(req: ValidateReq) -> Dict[str, Any]:
    if MANAGER.is_running():
        raise HTTPException(409, 'A job is already running.')
    if not req.model and req.kind != 'spot_logs':
        raise HTTPException(400, 'A model checkpoint is required.')

    run_dir = MANAGER.new_run_dir(req.kind)

    if req.kind == 'self_play':
        cmd = ['python3', 'scripts/analysis/self_play.py',
               '--model', req.model, '--hands', str(req.hands)]
        if req.range_analysis:
            cmd.append('--range-analysis')
    elif req.kind == 'validate_hh':
        # --betting-line is REQUIRED for 53-dim models; without it the script
        # dies with a size mismatch. Always pass it.
        cmd = ['python3', 'scripts/validate_vs_hh.py',
               '--model', req.model, '--max-hands', str(req.max_hands),
               '--behavioral-stats', '--betting-line']
    elif req.kind == 'spots':
        cmd = ['python3', 'scripts/inspect_model_spots.py', '--model', req.model]
    elif req.kind == 'spot_logs':
        cmd = ['python3', 'scripts/validate_spot_logs.py']
        if req.model:
            cmd += ['--model', req.model]
    else:
        raise HTTPException(400, f'Unknown validation kind: {req.kind}')

    label = req.label or f'{req.kind}: {Path(req.model).name if req.model else "-"}'
    job = MANAGER.start(req.kind, label, cmd, req.model_dump(), run_dir)
    return {'ok': True, 'job': job.public()}


# ─────────────────────────────────────────────────────────────────────────────
# Spot inspector — how does the model play each kind of spot?
# ─────────────────────────────────────────────────────────────────────────────
_EVALUATORS: Dict[str, Any] = {}


def _get_evaluator(model_path: str):
    """One evaluator per model — constructing it loads the checkpoint."""
    key = model_path or '__default__'
    if key not in _EVALUATORS:
        from src.spot_evaluator import SpotEvaluator
        _EVALUATORS[key] = SpotEvaluator.for_model(model_path)
    return _EVALUATORS[key]


class SpotsReq(BaseModel):
    model: str = ''
    categories: List[str] = []
    streets: List[str] = ['flop']
    positions: List[str] = ['IP', 'OOP']
    player_counts: List[int] = [6]
    facings: List[str] = []
    pot_bb: float = 7.0
    stack_bb: float = 100.0


@app.get('/api/spots/catalog')
def spots_catalog() -> Dict[str, Any]:
    from src.spot_library import catalog
    return catalog()


@app.post('/api/spots/evaluate')
def spots_evaluate(req: SpotsReq) -> Dict[str, Any]:
    """Run the model over the requested slice. Synchronous on purpose: this is
    inference over a few hundred spots, not a job — it must not take the
    single-job slot away from a training run."""
    from src.spot_library import generate
    from src.spot_evaluator import summarise

    spots = generate(
        categories=req.categories or None,
        streets=req.streets or None,
        positions=req.positions or None,
        player_counts=req.player_counts or None,
        facings=req.facings or None,
        pot_bb=req.pot_bb, stack_bb=req.stack_bb)
    if len(spots) > 600:
        raise HTTPException(
            400, f'{len(spots)} spots requested; narrow the filters to <=600.')

    try:
        ev = _get_evaluator(req.model)
    except Exception as exc:
        raise HTTPException(500, f'could not load model: {exc}')

    results = [ev.evaluate(s) for s in spots]
    return {
        'results': [r.to_dict() for r in results],
        'stats': summarise(results),
        'count': len(results),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Spot library — a curated, labelled regression suite
# ─────────────────────────────────────────────────────────────────────────────
class SpotRow(BaseModel):
    label: str = ''
    category: str = 'custom'
    street: str = 'flop'
    position: str = 'IP'
    num_players: int = 2
    facing: str = 'checked_to'
    hole: List[str] = []
    board: List[str] = []
    pot_bb: float = 7.0
    call_bb: float = 0.0
    stack_bb: float = 100.0
    right_action: Optional[int] = None
    notes: str = ''
    tags: str = ''


class SeedReq(BaseModel):
    categories: List[str] = []
    streets: List[str] = ['flop']
    positions: List[str] = ['IP', 'OOP']
    player_counts: List[int] = [6]
    facings: List[str] = []
    pot_bb: float = 7.0
    stack_bb: float = 100.0


class LibValidateReq(BaseModel):
    model: str = ''
    category: str = ''


@app.get('/api/library/spots')
def library_list(category: str = '', labelled_only: bool = False) -> Dict[str, Any]:
    from src.spot_store import get_store
    st = get_store(str(REPO_ROOT / 'data' / 'spot_library.sqlite'))
    return {'spots': st.list(category=category, labelled_only=labelled_only),
            'categories': st.categories(), 'stats': st.stats()}


@app.post('/api/library/spots')
def library_create(row: SpotRow) -> Dict[str, Any]:
    from src.spot_store import get_store
    st = get_store(str(REPO_ROOT / 'data' / 'spot_library.sqlite'))
    return {'spot': st.create(row.model_dump())}


@app.put('/api/library/spots/{spot_id}')
def library_update(spot_id: int, row: SpotRow) -> Dict[str, Any]:
    from src.spot_store import get_store
    st = get_store(str(REPO_ROOT / 'data' / 'spot_library.sqlite'))
    updated = st.update(spot_id, row.model_dump())
    if updated is None:
        raise HTTPException(404, 'spot not found')
    return {'spot': updated}


@app.delete('/api/library/spots/{spot_id}')
def library_delete(spot_id: int) -> Dict[str, Any]:
    from src.spot_store import get_store
    st = get_store(str(REPO_ROOT / 'data' / 'spot_library.sqlite'))
    if not st.delete(spot_id):
        raise HTTPException(404, 'spot not found')
    return {'ok': True}


@app.post('/api/library/seed')
def library_seed(req: SeedReq) -> Dict[str, Any]:
    from src.spot_store import get_store
    st = get_store(str(REPO_ROOT / 'data' / 'spot_library.sqlite'))
    added = st.seed_from_generator(
        categories=req.categories or None, streets=req.streets or None,
        positions=req.positions or None, player_counts=req.player_counts or None,
        facings=req.facings or None, pot_bb=req.pot_bb, stack_bb=req.stack_bb)
    return {'added': added, 'stats': st.stats()}


class LabelFromModelReq(BaseModel):
    model: str = ''
    category: str = ''
    overwrite: bool = False


@app.post('/api/library/label_from_model')
def library_label_from_model(req: LabelFromModelReq) -> Dict[str, Any]:
    """Seed right_action from what the model currently does.

    Uses the argmax of the strategy, not the sampled action: the pipeline
    samples from a mix, so labelling from a single draw would record a
    minority action as "correct" and be unreproducible.

    These are starting points to edit, not ground truth — a suite labelled
    this way scores 100% against the model that produced it.
    """
    from src.spot_store import get_store, row_to_spot
    st = get_store(str(REPO_ROOT / 'data' / 'spot_library.sqlite'))
    rows = st.list(category=req.category)
    ev = _get_evaluator(req.model)

    labelled = skipped = 0
    for row in rows:
        if row.get('right_action') is not None and not req.overwrite:
            skipped += 1
            continue
        res = ev.evaluate(row_to_spot(row))
        strat = res.pipeline if any(res.pipeline) else res.raw
        if not any(strat):
            skipped += 1
            continue
        action = int(max(range(len(strat)), key=lambda i: strat[i]))
        st.update(row['id'], {'right_action': action,
                              'tags': (row.get('tags') or '') + ' model-labelled'})
        labelled += 1
    return {'labelled': labelled, 'skipped': skipped, 'stats': st.stats()}


@app.post('/api/library/validate')
def library_validate(req: LibValidateReq) -> Dict[str, Any]:
    """Score the model against the labelled spots. Only labelled rows count —
    an unlabelled spot has no right answer to compare with."""
    from src.spot_store import get_store, row_to_spot
    st = get_store(str(REPO_ROOT / 'data' / 'spot_library.sqlite'))
    rows = st.list(category=req.category, labelled_only=True)
    if not rows:
        return {'results': [], 'summary': {'n': 0, 'correct': 0, 'wrong': 0},
                'by_category': []}

    ev = _get_evaluator(req.model)
    results, by_cat = [], {}
    correct = 0
    for row in rows:
        res = ev.evaluate(row_to_spot(row))
        want = row['right_action']
        got = res.pipeline_action
        # Any raise counts as matching any raise: the label records intent
        # (be aggressive here), not the exact sizing.
        ok = (got == want) or (want in (2, 3, 4) and got in (2, 3, 4))
        correct += ok
        slot = by_cat.setdefault(row['category'], {'category': row['category'],
                                                   'n': 0, 'correct': 0})
        slot['n'] += 1
        slot['correct'] += ok
        d = res.to_dict()
        d['spot']['db_id'] = row['id']
        d['spot']['label'] = row['label']
        d['right_action'] = want
        d['ok'] = bool(ok)
        results.append(d)

    for slot in by_cat.values():
        slot['pct'] = round(100 * slot['correct'] / max(slot['n'], 1), 1)
    return {
        'results': results,
        'summary': {'n': len(results), 'correct': correct,
                    'wrong': len(results) - correct,
                    'pct': round(100 * correct / max(len(results), 1), 1)},
        'by_category': sorted(by_cat.values(), key=lambda s: -s['n']),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Vector explorer — edit an observation feature by feature
# ─────────────────────────────────────────────────────────────────────────────
class VectorReq(BaseModel):
    model: str = ''
    obs: List[float]
    legal: List[int] = [0, 1, 2, 3, 4]


@app.get('/api/vector/schema')
def vector_schema() -> Dict[str, Any]:
    from src import obs_schema as sch
    names = sch.feature_names()
    key = {sch.IDX_EQUITY, sch.IDX_MADE_HAND, sch.IDX_POT_ODDS,
           sch.IDX_FLUSH_DRAW, sch.IDX_STRAIGHT_DRAW, sch.IDX_TOP_PAIR,
           sch.IDX_BOARD_PAIR, sch.IDX_STACK_TO_POT}
    return {
        'features': [{'idx': i, 'name': n, 'key': i in key}
                     for i, n in enumerate(names)],
        'dim': sch.STANDARD_OBS_DIM,
        'made_hand_cutoffs': [
            {'value': v, 'label': l} for v, l in sch.MADE_HAND_TIERS],
    }


@app.get('/api/vector/samples')
def vector_samples(limit: int = 60) -> Dict[str, Any]:
    """Real observation vectors from logged spots, as starting points."""
    import json as _json
    from src import obs_schema as sch
    out = []
    debug_dir = REPO_ROOT / 'logs' / 'action_debug'
    if debug_dir.is_dir():
        for p in sorted(debug_dir.glob('*.json'))[:2000]:
            try:
                d = _json.loads(p.read_text())
            except Exception:
                continue
            ov = d.get('obs_vector') or []
            if len(ov) < sch.STANDARD_OBS_DIM:
                continue
            gs = d.get('game_state') or {}
            me = next((x for x in (gs.get('players') or [])
                       if x.get('seat') == gs.get('my_seat')), {}) or {}
            out.append({
                'name': p.name,
                'street': gs.get('street', '?'),
                'hole': [f"{c['rank']}{c['suit']}"
                         for c in (me.get('hole_cards') or [])],
                'board': [f"{c['rank']}{c['suit']}"
                          for c in (gs.get('community_cards') or [])],
                'right_action': d.get('right_action'),
                'obs': [float(x) for x in ov[:sch.STANDARD_OBS_DIM]],
                'legal': gs.get('legal_actions') or [0, 1, 2, 3, 4],
            })
            if len(out) >= limit:
                break
    return {'samples': out}


@app.post('/api/vector/predict')
def vector_predict(req: VectorReq) -> Dict[str, Any]:
    from src import obs_schema as sch
    if len(req.obs) != sch.STANDARD_OBS_DIM:
        raise HTTPException(
            400, f'expected {sch.STANDARD_OBS_DIM} values, got {len(req.obs)}')
    ev = _get_evaluator(req.model)
    strat = ev.strategy_for_vector(req.obs, req.legal)
    return {'strategy': [round(x, 5) for x in strat]}


@app.post('/api/stop')
def stop() -> Dict[str, Any]:
    return {'ok': MANAGER.stop()}


@app.get('/api/status')
def status() -> Dict[str, Any]:
    job = MANAGER.job.public() if MANAGER.job else None
    return {
        'running': MANAGER.is_running(),
        'job': job,
        'metrics': MANAGER.metrics,
    }


@app.get('/api/history')
def history() -> Dict[str, Any]:
    return {'jobs': MANAGER.history()}


@app.get('/api/checkpoints')
def checkpoints() -> Dict[str, Any]:
    return {'checkpoints': _list_checkpoints()}


@app.get('/api/disk')
def disk() -> Dict[str, Any]:
    try:
        u = shutil.disk_usage(str(REPO_ROOT))
        return {'total_gb': round(u.total / 1e9, 1),
                'free_gb': round(u.free / 1e9, 1),
                'used_pct': round(100 * u.used / u.total)}
    except Exception:
        return {}


@app.get('/api/stream')
async def stream() -> StreamingResponse:
    """Server-sent events: backlog first, then live lines."""
    q = MANAGER.subscribe()
    backlog = MANAGER.backlog()

    async def gen():
        try:
            for rec in backlog:
                yield f'data: {json.dumps(rec)}\n\n'
            loop = asyncio.get_event_loop()
            while True:
                try:
                    rec = await loop.run_in_executor(None, q.get, True, 15)
                except Exception:
                    yield ': keepalive\n\n'      # keep proxies from timing out
                    continue
                yield f'data: {json.dumps(rec)}\n\n'
        finally:
            MANAGER.unsubscribe(q)

    return StreamingResponse(gen(), media_type='text/event-stream',
                             headers={'Cache-Control': 'no-cache',
                                      'X-Accel-Buffering': 'no'})


@app.get('/')
def index() -> FileResponse:
    return FileResponse(STATIC / 'index.html')


app.mount('/static', StaticFiles(directory=str(STATIC)), name='static')


def main() -> None:
    ap = argparse.ArgumentParser(description='Poker script runner web UI')
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8000)
    args = ap.parse_args()
    import uvicorn
    print(f'Poker Runner → http://{args.host}:{args.port}  (no auth — keep it private)')
    uvicorn.run(app, host=args.host, port=args.port, log_level='warning')


if __name__ == '__main__':
    main()

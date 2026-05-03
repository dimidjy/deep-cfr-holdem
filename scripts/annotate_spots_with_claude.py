"""Annotate postflop spots in logs/action_debug/ with Claude via the
Anthropic API, restricted to four actions: fold, call/check, bet_half,
bet_pot. Preflop spots are skipped.

Pipeline:
  1. Scan logs/action_debug/*.json
  2. Filter to postflop (flop / turn / river) spots
  3. Skip already-annotated spots (idempotent — re-runnable)
  4. Build per-spot text description from game_state
  5. Submit a Message Batch (50% cheaper than single calls)
  6. Poll until the batch finishes
  7. Parse results into structured JSON (validated against schema)
  8. Write right_action + claude_reasoning + claude_confidence into each JSON

Cost estimates (1300 postflop spots, with prompt caching + Batch API):
  - Claude Haiku 4.5  : ~$0.50 total
  - Claude Sonnet 4.6 : ~$1.50 total
  - Claude Opus 4.7   : ~$5     total

Setup:
  1. Sign up at https://console.anthropic.com (free $5 credit on first signup)
  2. Get an API key from the console
  3. Export the key:
       export ANTHROPIC_API_KEY=sk-ant-...
  4. Install the SDK:
       pip install anthropic
  5. Dry-run to confirm the spots and cost:
       python3 scripts/annotate_spots_with_claude.py --dry-run
  6. Actual run:
       python3 scripts/annotate_spots_with_claude.py

Usage:
  --model            : claude-opus-4-7 (default) | claude-sonnet-4-6 | claude-haiku-4-5
  --mode             : batch (default, cheaper, slower) | single (testing)
  --dry-run          : estimate cost, list candidate spots, no API calls
  --max-spots N      : cap the number of spots (testing/cost control)
  --since YYYYMMDD   : only spots whose filename is >= this prefix
  --re-annotate      : overwrite existing right_action; default skips them
  --strategy-file PATH : load custom system prompt from a file
"""
import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ── Strategy / system prompt ─────────────────────────────────────────────
# Edit this OR pass --strategy-file to override.
DEFAULT_STRATEGY = """You are reviewing postflop poker spots from a 6-max NL5
cash game. Your job: pick the action you would take given the strategy
guidelines below. Output ONLY ONE of four actions: fold, call, bet_half,
bet_pot. (Treat "call" as "call when there's a bet to face, check when
there isn't" — they share the same action label.)

STRATEGY (exploit NL5 population, lean tight, value-focused):

GENERAL
- AF target: 2.0-3.5 (don't be a calling station, don't be a maniac)
- Don't bluff multiway (3+ players to act) without strong equity
- "Vs big bets, lean tight" — NL5 reg overbets are usually value
- When facing a bet, calling needs ~30% equity vs the opp's range or
  better (depending on pot odds and implied odds)

CBET (when hero has initiative on flop)
- Heads-up: CBet 55-70% on most boards; lean toward smaller (bet_half)
  for protection-bet or thin-value, larger (bet_pot) on dry boards
  with strong made hands
- Multiway: CBet 35-50%, only with showdown value or strong draws
- Skip CBets with pure air on wet boards

FACING BETS
- Don't bluff-raise without blockers
- Don't call-down weak bluff-catchers in big pots
- Top pair is usually a call once, but folds vs heavy aggression on
  scary turn/river cards
- Sets, two pair, straights, flushes: value-bet or call-down
- Underpairs to board: usually fold facing bets
- Drawing hands: semi-bluff with strong draws (FD + OESD), call with
  pot odds for one street with weaker draws

POSITION
- Out of position: lean check-call vs IP bets, donk-leads only with
  strong made hands or semi-bluffs that benefit from initiative
- In position: check back marginal hands behind for pot control,
  bet for value/protection with strong hands

THINKING
- Output ACTION + brief 1-sentence REASONING + CONFIDENCE (0.0-1.0)
- Confidence 0.95+ = clear-cut spots (folding air vs big bet, raising
  the nuts)
- Confidence 0.5-0.7 = judgment calls where multiple actions are
  reasonable
"""

# JSON schema enforced on Claude's response — guarantees we get a parseable
# object every time. The action enum is the limited 4-action set.
# NOTE: Anthropic structured-outputs schema does NOT support numerical
# constraints (minimum, maximum, multipleOf) — they 400 the request.
# We enforce the [0.0, 1.0] range client-side via clamp_confidence().
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["fold", "call", "bet_half", "bet_pot"],
            "description": (
                "Pick exactly one. 'call' means call/check (same label "
                "regardless of whether there's a bet to face)."
            ),
        },
        "confidence": {
            "type": "number",
            "description": (
                "How confident in the action choice, between 0.0 (pure "
                "guess) and 1.0 (clear-cut). Use 0.5-0.65 for judgment "
                "calls, 0.95+ only for spots with one obviously correct "
                "answer."
            ),
        },
        "reasoning": {
            "type": "string",
            "description": "One brief sentence justifying the action.",
        },
    },
    "required": ["action", "confidence", "reasoning"],
    "additionalProperties": False,
}


def clamp_confidence(v) -> float:
    """Clip confidence to [0.0, 1.0]. Schema can't enforce this server-side
    for structured outputs, so we clamp here."""
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.5

# Map Claude's action label → integer action_idx used in your codebase.
ACTION_TO_IDX = {
    "fold":     0,   # ACTION_FOLD
    "call":     1,   # ACTION_CALL (also covers check when call_amount=0)
    "bet_half": 2,   # ACTION_RAISE_HALF
    "bet_pot":  3,   # ACTION_RAISE_POT
}

# ── Per-spot text description ────────────────────────────────────────────

_POS_LABELS = ['BTN', 'SB', 'BB', 'UTG', 'MP', 'CO']

def _pot_history_label(num_raises_norm: float) -> str:
    """Decode hand-level raise count → human-readable pot type.
    num_raises_norm = raise_count / 6.0 (post-fix tracker output)."""
    raises = round(num_raises_norm * 6)
    return {
        0: "limped pot (no preflop raise)",
        1: "single-raised pot",
        2: "3-bet pot",
        3: "4-bet pot",
        4: "5-bet pot",
    }.get(raises, f"{raises}-bet pot (multi-raise war)")


def describe_spot(d: dict) -> str:
    """Build a clean human-readable description of one spot for Claude."""
    gs = d['game_state']
    f = d.get('features', {})
    me = next((p for p in gs['players'] if p['seat'] == gs['my_seat']), {})
    hole = ' '.join(c['rank']+c['suit'] for c in me.get('hole_cards', []))
    board = ' '.join(c['rank']+c['suit'] for c in gs.get('community_cards', [])) or '-'
    pos = next((label for label in _POS_LABELS if f.get(f'pos_{label.lower()}')), '?')

    facing = round(f.get('facing_bet', 0)) == 1
    init = round(f.get('hero_has_initiative', 0)) == 1
    n_active = gs.get('num_active_players', 0)

    # Stacks & money
    pot = float(gs.get('pot_total', 0))
    call_amt = float(gs.get('call_amount', 0))
    hero_stack = float(me.get('stack', 0))

    # Effective stack: min(hero_stack, max active-opp stack). What's
    # actually "in play" for stack-commitment decisions.
    active_opps = [p for p in gs['players']
                   if p['seat'] != gs['my_seat'] and p['is_active']]
    max_opp_stack = max((float(p['stack']) for p in active_opps), default=0)
    eff_stack = min(hero_stack, max_opp_stack) if max_opp_stack > 0 else hero_stack

    # SPR: effective stack / pot (capped at 99 to avoid div-by-tiny noise)
    spr = round(eff_stack / max(pot, 0.5), 1) if pot > 0 else None
    spr_str = f"{spr}" if spr is not None and spr < 99 else "very deep"

    # Pot odds & required equity (only when facing a bet)
    pot_odds_str = ""
    if call_amt > 0.01:
        # required equity = call / (call + pot) where pot already includes opp's bet
        req_eq = call_amt / (call_amt + pot)
        pot_odds_str = (f"  |  POT ODDS: need {req_eq*100:.0f}% equity "
                        f"to call (call {call_amt:.1f} into {pot:.1f})")

    # Pot history (single-raised, 3-bet, etc.)
    pot_history = _pot_history_label(f.get('num_raises_norm', 0))

    # Opponent action profile (the new behavioral features we wired up)
    opp_agg = f.get('opp_aggression_ratio', 0)
    opp_fold = f.get('opp_fold_ratio', 0)
    opp_profile = (
        f"opp_aggression={opp_agg:.2f} (fraction of opp actions that "
        f"were aggressive), opp_fold_rate={opp_fold:.2f} (fraction of "
        f"opp actions that were folds)")

    # Per-opp summary
    opp_lines = ', '.join(
        f"opp_{i}(stack {p['stack']:.0f}BB, bet_this_street {p['bet_current_round']:.1f}BB)"
        for i, p in enumerate(active_opps[:5]))

    # Board texture flags
    tex = []
    if f.get('top_pair', 0) > 0.5:        tex.append("top pair")
    if f.get('hero_flush_participation', 0) > 0.4:    tex.append("flush draw/made")
    if f.get('hero_straight_participation', 0) > 0.4: tex.append("straight draw/made")
    if f.get('board_pairing_level', 0) > 0.3:         tex.append("board paired")
    if f.get('board_flush_texture', 0) > 0.5:         tex.append("flush threat on board")
    if f.get('board_straight_texture', 0) > 0.5:      tex.append("straight threat on board")
    tex_str = ", ".join(tex) if tex else "no notable board texture"

    legal = gs.get('legal_actions', [])
    legal_lbl = []
    for a in legal:
        legal_lbl.append({
            0:'fold', 1:'call_or_check', 2:'bet_half', 3:'bet_pot',
            4:'all_in', 5:'bet_min', 7:'bet_third'}.get(a, str(a)))

    return f"""STREET: {gs['street']}
POSITION: {pos}  |  PLAYERS ACTIVE: {n_active}  |  FACING BET: {facing}  |  HERO IS PFR: {init}
POT TYPE: {pot_history}

HAND: {hole}
BOARD: [{board}]
HAND/BOARD FEATURES: {tex_str}
HAND STRENGTH (vs random, 0-1): {f.get('hand_strength', 0):.2f}

POT: {pot:.1f}BB  |  CALL AMOUNT: {call_amt:.1f}BB  |  HERO STACK: {hero_stack:.1f}BB
EFFECTIVE STACK: {eff_stack:.1f}BB  |  SPR: {spr_str}{pot_odds_str}

OPPONENT ACTION PROFILE (this hand): {opp_profile}
OPPONENTS: {opp_lines}

LEGAL ACTIONS available in client: {legal_lbl}

Pick one of: fold, call, bet_half, bet_pot.
Note: 'call' covers both 'call vs bet' and 'check when no bet'."""


# ── Filtering ────────────────────────────────────────────────────────────

@dataclass
class SpotCandidate:
    path: Path
    json_data: dict
    description: str

def is_postflop(d: dict) -> bool:
    return d.get('game_state', {}).get('street') in ('flop', 'turn', 'river')

def is_already_annotated(d: dict) -> bool:
    return isinstance(d.get('right_action'), int) \
        and 'claude_reasoning' in d

def collect_spots(spots_dir: Path, since: Optional[str], max_spots: Optional[int],
                  re_annotate: bool) -> list:
    """Scan spots_dir and return list of SpotCandidate to annotate."""
    paths = sorted(spots_dir.glob('*.json'))
    if since:
        paths = [p for p in paths if p.name >= since]

    out = []
    for p in paths:
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if 'game_state' not in d:
            continue
        if not is_postflop(d):
            continue
        if not re_annotate and is_already_annotated(d):
            continue
        if d.get('deleted'):
            continue
        try:
            desc = describe_spot(d)
        except Exception as e:
            print(f"  skip {p.name}: describe failed ({e})", file=sys.stderr)
            continue
        out.append(SpotCandidate(p, d, desc))
        if max_spots and len(out) >= max_spots:
            break
    return out


# ── Cost estimation ──────────────────────────────────────────────────────

# Per-1M-token pricing (USD), Batch API gets 50% discount.
PRICING = {
    'claude-opus-4-7':   {'input': 5.00, 'output': 25.00},
    'claude-sonnet-4-6': {'input': 3.00, 'output': 15.00},
    'claude-haiku-4-5':  {'input': 1.00, 'output':  5.00},
}

def estimate_cost(spots: list, system_prompt: str, model: str,
                  use_batch: bool) -> dict:
    n = len(spots)
    avg_user_tokens = sum(len(s.description) for s in spots) // max(n, 1) // 4
    sys_tokens = len(system_prompt) // 4
    output_tokens_per = 100   # rough, with concise reasoning

    # With caching: system tokens are written once, read N-1 times.
    # Cache writes cost 1.25x base; reads cost 0.1x base.
    p = PRICING.get(model, PRICING['claude-sonnet-4-6'])
    in_per_M  = p['input'] * (0.5 if use_batch else 1.0)
    out_per_M = p['output'] * (0.5 if use_batch else 1.0)

    # Input cost: first request pays full sys + user; subsequent pay
    # cached_read on sys + full user.
    if n == 0:
        return {'n': 0, 'total_usd': 0.0}
    first_input    = sys_tokens + avg_user_tokens
    cache_write    = sys_tokens                          # paid once at 1.25x
    subsequent_in  = avg_user_tokens                     # per call
    cache_reads    = sys_tokens                          # per call after first
    total_in_tokens = first_input + (n-1) * subsequent_in
    total_cache_read_tokens = (n-1) * cache_reads
    total_out_tokens = n * output_tokens_per

    cost_input_full   = total_in_tokens * in_per_M / 1_000_000
    cost_cache_write  = cache_write * (in_per_M * 1.25) / 1_000_000
    cost_cache_read   = total_cache_read_tokens * (in_per_M * 0.10) / 1_000_000
    cost_output       = total_out_tokens * out_per_M / 1_000_000

    return {
        'n': n,
        'avg_user_tokens': avg_user_tokens,
        'sys_tokens': sys_tokens,
        'cost_input_uncached': round(cost_input_full, 4),
        'cost_cache_write':    round(cost_cache_write, 4),
        'cost_cache_read':     round(cost_cache_read, 4),
        'cost_output':         round(cost_output, 4),
        'total_usd': round(
            cost_input_full + cost_cache_write + cost_cache_read + cost_output, 4),
    }


# ── Annotate / write back ────────────────────────────────────────────────

def parse_response_obj(obj: dict) -> tuple:
    """Validate Claude's structured response → (action_idx, confidence, reasoning)."""
    action = obj['action']
    if action not in ACTION_TO_IDX:
        raise ValueError(f"unexpected action {action!r}")
    return ACTION_TO_IDX[action], clamp_confidence(obj['confidence']), str(obj['reasoning'])

def write_annotation(path: Path, action_idx: int, confidence: float,
                     reasoning: str, model: str) -> None:
    d = json.loads(path.read_text())
    d['right_action']        = action_idx
    d['claude_reasoning']    = reasoning
    d['claude_confidence']   = confidence
    d['claude_annotated_at'] = time.strftime('%Y-%m-%d')
    d['claude_model']        = model
    path.write_text(json.dumps(d, indent=2, default=str))


# ── Single-request mode (for testing / small runs) ───────────────────────

def run_single_mode(client, spots, model, system_prompt) -> int:
    """Submit one request at a time. Slower and 2x more expensive than
    batch, but useful for testing the prompt + verifying output shape."""
    written = 0
    for i, s in enumerate(spots, 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=512,
                system=[{
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},  # cache the strategy
                }],
                output_config={
                    "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA},
                },
                messages=[{"role": "user", "content": s.description}],
            )
        except Exception as e:
            print(f"  [{i}/{len(spots)}] {s.path.name}: API error {e}", file=sys.stderr)
            continue

        text = next((b.text for b in response.content if b.type == 'text'), '')
        try:
            parsed = json.loads(text)
            action_idx, conf, reason = parse_response_obj(parsed)
        except Exception as e:
            print(f"  [{i}/{len(spots)}] {s.path.name}: parse error {e}",
                  file=sys.stderr)
            continue

        write_annotation(s.path, action_idx, conf, reason, model)
        written += 1
        action_label = next(k for k, v in ACTION_TO_IDX.items() if v == action_idx)
        print(f"  [{i}/{len(spots)}] {s.path.name} → {action_label} "
              f"(conf {conf:.2f})")
    return written


# ── Batch mode (cheap + scalable) ────────────────────────────────────────

def _safe_custom_id(filename: str) -> str:
    """custom_id must match '^[a-zA-Z0-9_-]{1,64}$' — strip '.json',
    replace dots/anything-else with underscores, truncate to 64 chars.
    Filenames like '20260502_113020_781_raise_custom_22.00bb.json'
    have dots and run long; this normalizes them."""
    import re
    stem = filename[:-5] if filename.endswith('.json') else filename
    safe = re.sub(r'[^a-zA-Z0-9_-]', '_', stem)
    return safe[:64]


def run_batch_mode(client, spots, model, system_prompt) -> int:
    from anthropic.types.message_create_params import (
        MessageCreateParamsNonStreaming)
    from anthropic.types.messages.batch_create_params import Request

    print(f"\nBuilding batch of {len(spots)} requests...")
    # Build a safe_id → spot map so we can match results back.
    # Detect collisions defensively (truncation could in theory map two
    # different filenames to the same safe_id).
    by_id = {}
    for s in spots:
        cid = _safe_custom_id(s.path.name)
        if cid in by_id:
            raise RuntimeError(
                f"custom_id collision: {by_id[cid].path.name!r} and "
                f"{s.path.name!r} both map to {cid!r}. Rename one of "
                f"the source files or shorten the safe_id derivation.")
        by_id[cid] = s
    requests = [
        Request(
            custom_id=_safe_custom_id(s.path.name),
            params=MessageCreateParamsNonStreaming(
                model=model,
                max_tokens=512,
                system=[{
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }],
                output_config={
                    "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA},
                },
                messages=[{"role": "user", "content": s.description}],
            ),
        )
        for s in spots
    ]

    print(f"Submitting batch...")
    batch = client.messages.batches.create(requests=requests)
    print(f"  batch id: {batch.id}")
    print(f"  status:   {batch.processing_status}")

    print(f"\nPolling for completion (typically 1-30 minutes)...")
    poll_interval = 30
    while True:
        batch = client.messages.batches.retrieve(batch.id)
        rc = batch.request_counts
        print(f"  [{time.strftime('%H:%M:%S')}] status={batch.processing_status} "
              f"processing={rc.processing} succeeded={rc.succeeded} "
              f"errored={rc.errored}")
        if batch.processing_status == 'ended':
            break
        time.sleep(poll_interval)

    print(f"\nBatch finished: succeeded={batch.request_counts.succeeded}, "
          f"errored={batch.request_counts.errored}")
    print(f"Writing annotations...")

    written = 0
    errored = 0
    for result in client.messages.batches.results(batch.id):
        cid = result.custom_id
        spot = by_id.get(cid)
        if spot is None:
            continue
        if result.result.type != 'succeeded':
            errored += 1
            err_msg = getattr(result.result, 'error', None)
            print(f"  {cid}: {result.result.type} {err_msg}", file=sys.stderr)
            continue
        msg = result.result.message
        text = next((b.text for b in msg.content if b.type == 'text'), '')
        try:
            parsed = json.loads(text)
            action_idx, conf, reason = parse_response_obj(parsed)
        except Exception as e:
            errored += 1
            print(f"  {cid}: parse error {e}", file=sys.stderr)
            continue
        write_annotation(spot.path, action_idx, conf, reason, model)
        written += 1

    print(f"\nWrote {written} annotations; {errored} errors.")
    return written


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--spots-dir', default='logs/action_debug')
    ap.add_argument('--model', default='claude-opus-4-7',
                    choices=['claude-opus-4-7', 'claude-sonnet-4-6',
                             'claude-haiku-4-5'])
    ap.add_argument('--mode', default='batch', choices=['batch', 'single'])
    ap.add_argument('--dry-run', action='store_true',
                    help='Show counts and cost; no API calls')
    ap.add_argument('--max-spots', type=int, default=None)
    ap.add_argument('--since', default=None,
                    help='Filename prefix filter (e.g. 20260501)')
    ap.add_argument('--re-annotate', action='store_true',
                    help='Overwrite existing right_action annotations')
    ap.add_argument('--strategy-file', default=None,
                    help='Load custom strategy/system prompt from file')
    args = ap.parse_args()

    spots_dir = Path(args.spots_dir)
    if not spots_dir.is_dir():
        print(f'Spots dir not found: {spots_dir}', file=sys.stderr)
        return 1

    system_prompt = DEFAULT_STRATEGY
    if args.strategy_file:
        system_prompt = Path(args.strategy_file).read_text()
        print(f"Loaded strategy from {args.strategy_file}")

    print(f"Scanning {spots_dir}...")
    spots = collect_spots(
        spots_dir, args.since, args.max_spots, args.re_annotate)
    print(f"  {len(spots)} postflop spots to annotate")
    if not spots:
        print("Nothing to do (try --re-annotate or check --since).")
        return 0

    by_street = {}
    for s in spots:
        st = s.json_data['game_state'].get('street', '?')
        by_street[st] = by_street.get(st, 0) + 1
    print(f"  by street: {by_street}")

    cost = estimate_cost(spots, system_prompt, args.model,
                         use_batch=(args.mode == 'batch'))
    print(f"\nEstimated cost ({args.model}, {args.mode} mode):")
    print(f"  ~{cost['avg_user_tokens']} user tokens/spot, "
          f"~{cost['sys_tokens']} system tokens (cached)")
    print(f"  total estimated: ${cost['total_usd']:.2f} USD")

    if args.dry_run:
        print('\n[DRY RUN] Stopping. Re-run without --dry-run to submit.')
        return 0

    # Lazy-import anthropic only when actually running
    try:
        import anthropic
    except ImportError:
        print('\nAnthropic SDK not installed.\n  pip install anthropic',
              file=sys.stderr)
        return 1

    import os
    if not os.environ.get('ANTHROPIC_API_KEY'):
        print('\nANTHROPIC_API_KEY env var not set. Get a key at '
              'console.anthropic.com and:\n'
              '  export ANTHROPIC_API_KEY=sk-ant-...', file=sys.stderr)
        return 1

    client = anthropic.Anthropic()
    print(f"\nRunning in {args.mode.upper()} mode with model {args.model}...")

    if args.mode == 'single':
        written = run_single_mode(client, spots, args.model, system_prompt)
    else:
        written = run_batch_mode(client, spots, args.model, system_prompt)

    print(f"\nDone — annotated {written} spots.")
    return 0


if __name__ == '__main__':
    sys.exit(main())

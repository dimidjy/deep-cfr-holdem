"""Train a fast GPU equity predictor to replace Monte Carlo simulation.

Generates millions of (hole_cards, board) → equity samples using eval7,
then trains a neural network to predict equity features.

v2 improvements over v1:
  - Rich card encoding: 52-dim one-hot per card + derived features = 82 dims
  - Bigger network: [256, 256, 128] with residual block
  - 10M samples with 300 MC samples for better ground truth
  - Separate flush/straight draw detection from raw cards

Usage:
    python3 scripts/train_equity_model.py --samples 10000000 --epochs 80
    python3 scripts/train_equity_model.py --train-only  # skip data gen, use cached
"""

import sys
import os
import argparse
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

import eval7

# ── Card encoding ─────────────────────────────────────────────────────────────

RANKS = '23456789TJQKA'
SUITS = 'shdc'
RANK_MAP = {r: i for i, r in enumerate(RANKS)}
SUIT_MAP = {s: i for i, s in enumerate(SUITS)}

# Total input dims: 7 cards × 6 features + 40 derived + 1 n_opponents = 83
INPUT_SIZE = 83


def encode_card_rich(card_str: str) -> np.ndarray:
    """Encode one card as 6 features: rank_norm, 4 suit one-hot, is_present."""
    r = RANK_MAP.get(card_str[0].upper(), 0)
    s = SUIT_MAP.get(card_str[1].lower(), 0)
    feat = np.zeros(6, dtype=np.float32)
    feat[0] = r / 12.0          # rank normalized
    feat[1 + s] = 1.0           # suit one-hot (4 dims)
    feat[5] = 1.0               # card is present
    return feat


def encode_hand_board(hole: list, board: list, n_opponents: int = 1) -> np.ndarray:
    """Encode hole cards + board + n_opponents into 83-dim feature vector.

    Layout:
      7 cards × 6 features each = 42 (hole[0], hole[1], board[0-4])
      + 40 derived features
      + 1 n_opponents (normalized by /5)
    """
    features = np.zeros(INPUT_SIZE, dtype=np.float32)

    # Card features: 7 slots × 6 dims = 42
    for i, c in enumerate(hole[:2]):
        features[i*6:(i+1)*6] = encode_card_rich(str(c))
    for i, c in enumerate(board[:5]):
        features[(i+2)*6:(i+3)*6] = encode_card_rich(str(c))

    # Derived features start at index 42
    d = 42
    h_ranks = [RANK_MAP[str(c)[0].upper()] for c in hole[:2]]
    h_suits = [SUIT_MAP[str(c)[1].lower()] for c in hole[:2]]

    # Hole card features
    features[d+0] = 1.0 if h_ranks[0] == h_ranks[1] else 0.0       # is_pair
    features[d+1] = 1.0 if h_suits[0] == h_suits[1] else 0.0       # is_suited
    features[d+2] = max(h_ranks) / 12.0                              # high_rank
    features[d+3] = min(h_ranks) / 12.0                              # low_rank
    features[d+4] = abs(h_ranks[0] - h_ranks[1]) / 12.0             # gap
    features[d+5] = len(board) / 5.0                                  # n_board_cards

    if board:
        b_ranks = [RANK_MAP[str(c)[0].upper()] for c in board]
        b_suits = [SUIT_MAP[str(c)[1].lower()] for c in board]
        all_ranks = h_ranks + b_ranks
        all_suits = h_suits + b_suits

        # Board rank features
        features[d+6] = max(b_ranks) / 12.0                          # board_max
        features[d+7] = min(b_ranks) / 12.0                          # board_min
        features[d+8] = (max(b_ranks) - min(b_ranks)) / 12.0        # board_span

        # Board pairing
        brc = Counter(b_ranks)
        max_count = max(brc.values())
        num_pairs = sum(1 for v in brc.values() if v >= 2)
        features[d+9] = max_count / 4.0                              # max_same_rank
        features[d+10] = num_pairs / 3.0                             # num_pairs

        # Suit counts (board)
        bsc = Counter(b_suits)
        features[d+11] = max(bsc.values()) / 5.0                    # max_board_suited
        dom_suit = bsc.most_common(1)[0][0]

        # Hero flush participation
        hero_in_dom = sum(1 for s in h_suits if s == dom_suit)
        features[d+12] = hero_in_dom / 2.0                          # hero_flush_part

        # All suits combined flush potential
        asc = Counter(all_suits)
        features[d+13] = max(asc.values()) / 7.0                    # max_all_suited
        features[d+14] = 1.0 if max(asc.values()) >= 4 else 0.0    # has_flush_draw
        features[d+15] = 1.0 if max(asc.values()) >= 5 else 0.0    # has_flush

        # Straight potential
        unique_all = sorted(set(all_ranks))
        if 12 in unique_all:
            unique_all = sorted(set([-1] + unique_all))  # ace low
        unique_board = sorted(set(b_ranks))
        if 12 in unique_board:
            unique_board = sorted(set([-1] + unique_board))

        # Count max consecutive in all cards and board
        def max_consecutive(ranks):
            if not ranks:
                return 0
            mc, c = 1, 1
            for i in range(1, len(ranks)):
                if ranks[i] - ranks[i-1] == 1:
                    c += 1; mc = max(mc, c)
                else:
                    c = 1
            return mc

        features[d+16] = max_consecutive(unique_all) / 5.0          # all_consecutive
        features[d+17] = max_consecutive(unique_board) / 5.0        # board_consecutive

        # Hero straight participation
        board_set = set(b_ranks)
        if 12 in board_set: board_set.add(-1)
        hero_set = set(h_ranks)
        if 12 in hero_set: hero_set.add(-1)
        hero_str = 0.0
        for start in range(-1, 10):
            window = set(range(start, start + 5))
            combined = len(window & (board_set | hero_set))
            hero_adds = combined - len(window & board_set)
            if combined >= 4 and hero_adds > 0:
                hero_str = 1.0 if combined == 5 else 0.5
                if combined == 5: break
        features[d+18] = hero_str                                    # hero_straight_part

        # Top pair, overpair
        features[d+19] = 1.0 if max(b_ranks) in h_ranks else 0.0   # top_pair
        features[d+20] = 1.0 if (h_ranks[0] == h_ranks[1] and
                                   h_ranks[0] > max(b_ranks)) else 0.0  # overpair

        # How many hero cards connect to board
        hero_on_board = sum(1 for r in h_ranks if r in b_ranks)
        features[d+21] = hero_on_board / 2.0                        # hero_board_connect

        # Number of overcards hero has
        overcards = sum(1 for r in h_ranks if r > max(b_ranks))
        features[d+22] = overcards / 2.0                            # overcards

        # Board monotone (all same suit)
        features[d+23] = 1.0 if len(set(b_suits)) == 1 else 0.0    # monotone

        # Number of board cards
        features[d+24] = len(board) / 5.0                           # n_board (redundant but helps)

        # Hero suit matches per board suit
        for si in range(4):
            hero_has = sum(1 for s in h_suits if s == si)
            board_has = bsc.get(si, 0)
            features[d+25+si] = (hero_has + board_has) / 7.0        # per_suit_total

        # Combined rank histogram features (how many of each rank in all 7 cards)
        arc = Counter(all_ranks)
        features[d+29] = sum(1 for v in arc.values() if v >= 2) / 6.0  # num_pairs_all
        features[d+30] = sum(1 for v in arc.values() if v >= 3) / 3.0  # num_trips_all
        features[d+31] = sum(1 for v in arc.values() if v >= 4) / 2.0  # num_quads_all

        # Board texture: low/mid/high card count
        features[d+32] = sum(1 for r in b_ranks if r <= 4) / max(len(b_ranks), 1)  # low_cards
        features[d+33] = sum(1 for r in b_ranks if 5 <= r <= 8) / max(len(b_ranks), 1)  # mid_cards
        features[d+34] = sum(1 for r in b_ranks if r >= 9) / max(len(b_ranks), 1)  # high_cards

        # Kicker strength (when hero has one card matching board)
        if hero_on_board == 1:
            kicker = [r for r in h_ranks if r not in b_ranks]
            features[d+35] = kicker[0] / 12.0 if kicker else 0.0   # kicker_rank
        elif hero_on_board == 0:
            features[d+35] = max(h_ranks) / 12.0                    # best card as kicker

        # Street-specific features
        features[d+36] = 1.0 if len(board) == 3 else 0.0           # is_flop
        features[d+37] = 1.0 if len(board) == 4 else 0.0           # is_turn
        features[d+38] = 1.0 if len(board) == 5 else 0.0           # is_river
        features[d+39] = 1.0 if len(board) == 0 else 0.0           # is_preflop

    # n_opponents feature (index 82)
    features[82] = n_opponents / 5.0

    return features


# ── Target computation ────────────────────────────────────────────────────────

_ALL_COMBOS = []
for i, r1 in enumerate(RANKS):
    for j, r2 in enumerate(RANKS):
        if i < j:
            _ALL_COMBOS.append(f'{r2}{r1}s')
            _ALL_COMBOS.append(f'{r2}{r1}o')
        elif i == j:
            _ALL_COMBOS.append(f'{r1}{r2}')
_RANDOM_RANGE = eval7.HandRange(','.join(_ALL_COMBOS))

try:
    from treys import Card as TreysCard, Evaluator as TreysEvaluator
    _TREYS_EVAL = TreysEvaluator()
    _TREYS_OK = True
except ImportError:
    _TREYS_OK = False

_TREYS_RANK_BOUNDS = {
    1: (1, 10), 2: (11, 166), 3: (167, 322), 4: (323, 1599),
    5: (1600, 1609), 6: (1610, 2467), 7: (2468, 3325),
    8: (3326, 6185), 9: (6186, 7462),
}


def _multiway_equity_mc(hole, board, n_opponents, n_samples=300):
    """Monte Carlo multiway equity: win rate vs N random opponents."""
    import random as _rng
    deck = [eval7.Card(r + s) for r in RANKS for s in SUITS]
    used = set(str(c) for c in list(hole) + list(board))
    available = [c for c in deck if str(c) not in used]
    cards_needed = n_opponents * 2 + (5 - len(board))
    if len(available) < cards_needed:
        return 0.5
    wins = 0
    for _ in range(n_samples):
        _rng.shuffle(available)
        idx = 0
        full_board = list(board) + available[idx:idx + (5 - len(board))]
        idx += 5 - len(board)
        my_score = eval7.evaluate(list(hole) + full_board)
        beat_all = True
        for _ in range(n_opponents):
            opp = [available[idx], available[idx + 1]]
            idx += 2
            if eval7.evaluate(opp + full_board) > my_score:
                beat_all = False
                break
        if beat_all:
            wins += 1
    return wins / max(n_samples, 1)


def compute_targets(hole, board, n_equity_samples=300, n_opponents=1):
    """Compute the 9-dim equity feature targets using eval7 + treys."""
    if n_opponents <= 1:
        equity = eval7.py_hand_vs_range_monte_carlo(
            hole, _RANDOM_RANGE, board, n_equity_samples)
    else:
        equity = _multiway_equity_mc(hole, board, n_opponents, n_equity_samples)

    if not board or not _TREYS_OK:
        return np.array([equity, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

    h_strs = [str(c) for c in hole]
    b_strs = [str(c) for c in board]
    try:
        treys_hole = [TreysCard.new(s[0] + s[1].lower()) for s in h_strs]
        treys_board = [TreysCard.new(s[0] + s[1].lower()) for s in b_strs]
        rank_int = _TREYS_EVAL.evaluate(treys_board, treys_hole)
        rank_class = _TREYS_EVAL.get_rank_class(rank_int)
        made_hand_cat = (9 - rank_class) / 8.0
        lower, upper = _TREYS_RANK_BOUNDS.get(rank_class, (1, 7462))
        rank_within = float((upper - rank_int) / (upper - lower)) if upper > lower else 1.0
    except Exception:
        return np.array([equity, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

    hole_ranks = [RANK_MAP[str(c)[0].upper()] for c in hole]
    board_ranks = [RANK_MAP[str(c)[0].upper()] for c in board]
    hole_suits = [SUIT_MAP[str(c)[1].lower()] for c in hole]
    board_suits = [SUIT_MAP[str(c)[1].lower()] for c in board]

    # Hero flush participation
    bsc = Counter(board_suits)
    dom_suit = bsc.most_common(1)[0][0]
    max_board_suited = bsc[dom_suit]
    hero_flush = sum(1 for s in hole_suits if s == dom_suit) / 2.0 if max_board_suited >= 2 else 0.0

    # Hero straight participation
    board_set = set(board_ranks)
    if 12 in board_set: board_set.add(-1)
    hero_set = set(hole_ranks)
    if 12 in hero_set: hero_set.add(-1)
    hero_straight = 0.0
    for start in range(-1, 10):
        window = set(range(start, start + 5))
        combined = len(window & (board_set | hero_set))
        hero_adds = combined - len(window & board_set)
        if combined >= 4 and hero_adds > 0:
            hero_straight = 1.0 if combined == 5 else 0.5
            if combined == 5: break

    top_pair = 1.0 if max(board_ranks) in hole_ranks else 0.0

    brc = Counter(board_ranks)
    max_rc = max(brc.values())
    num_pairs = sum(1 for c in brc.values() if c >= 2)
    if max_rc >= 3: board_pairing = 1.0
    elif num_pairs >= 2: board_pairing = 0.67
    elif num_pairs == 1: board_pairing = 0.33
    else: board_pairing = 0.0

    board_flush_tex = float(np.clip((max_board_suited - 1) / 3.0, 0, 1))

    unique_br = sorted(set(board_ranks))
    if 12 in unique_br: unique_br = sorted(set([-1] + unique_br))
    consec = sum(1 for i in range(len(unique_br)-1) if unique_br[i+1] - unique_br[i] == 1)
    board_str_tex = float(consec / max(len(unique_br)-1, 1))

    return np.array([
        equity, made_hand_cat, hero_flush, hero_straight,
        top_pair, board_pairing, board_flush_tex, board_str_tex, rank_within
    ], dtype=np.float32)


# ── Neural network ────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, size, dropout=0.05):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(size, size), nn.LayerNorm(size), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(size, size), nn.LayerNorm(size),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class EquityNetwork(nn.Module):
    """Fast equity predictor: card features → equity features."""

    def __init__(self, input_size=INPUT_SIZE, output_size=9, hidden=None):
        super().__init__()
        if hidden is None:
            hidden = [256, 256, 128]

        layers = [nn.Linear(input_size, hidden[0]), nn.LayerNorm(hidden[0]), nn.GELU()]

        for i in range(len(hidden) - 1):
            if hidden[i] == hidden[i+1]:
                layers.append(ResBlock(hidden[i]))
            else:
                layers.extend([
                    nn.Linear(hidden[i], hidden[i+1]),
                    nn.LayerNorm(hidden[i+1]),
                    nn.GELU(),
                    nn.Dropout(0.05),
                ])

        layers.append(nn.Linear(hidden[-1], output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── Data generation ───────────────────────────────────────────────────────────

def _generate_chunk(args):
    """Generate a chunk of training samples (for multiprocessing)."""
    chunk_size, n_equity_samples, seed = args
    np.random.seed(seed)

    deck_cards = [eval7.Card(r+s) for r in RANKS for s in SUITS]
    chunk_inputs = np.zeros((chunk_size, INPUT_SIZE), dtype=np.float32)
    chunk_targets = np.zeros((chunk_size, 9), dtype=np.float32)

    for i in range(chunk_size):
        deck = list(deck_cards)
        np.random.shuffle(deck)
        hole = deck[:2]

        r = np.random.random()
        if r < 0.15:
            board = []
        elif r < 0.45:
            board = deck[2:5]
        elif r < 0.70:
            board = deck[2:6]
        else:
            board = deck[2:7]

        opp_r = np.random.random()
        if opp_r < 0.35:
            n_opp = 1
        elif opp_r < 0.65:
            n_opp = 2
        elif opp_r < 0.80:
            n_opp = 3
        elif opp_r < 0.92:
            n_opp = 4
        else:
            n_opp = 5

        chunk_inputs[i] = encode_hand_board(hole, board, n_opp)
        chunk_targets[i] = compute_targets(hole, board, n_equity_samples, n_opp)

    return chunk_inputs, chunk_targets


def generate_data(n_samples, output_path, n_equity_samples=300, n_workers=None):
    """Generate training data using eval7 ground truth (multiprocessing)."""
    import multiprocessing as mp

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    # Split into chunks
    chunk_size = max(1000, n_samples // (n_workers * 4))
    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    chunks = []
    remaining = n_samples
    for i in range(n_chunks):
        sz = min(chunk_size, remaining)
        chunks.append((sz, n_equity_samples, i * 1000 + 42))
        remaining -= sz

    print(f'  Using {n_workers} workers, {n_chunks} chunks of ~{chunk_size:,}')

    inputs = np.zeros((n_samples, INPUT_SIZE), dtype=np.float32)
    targets = np.zeros((n_samples, 9), dtype=np.float32)

    t0 = time.time()
    done = 0

    with mp.Pool(n_workers) as pool:
        for chunk_in, chunk_tgt in pool.imap_unordered(_generate_chunk, chunks):
            n = len(chunk_in)
            if done + n > n_samples:
                n = n_samples - done
            inputs[done:done+n] = chunk_in[:n]
            targets[done:done+n] = chunk_tgt[:n]
            done += n

            if done % 500000 < chunk_size:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (n_samples - done) / rate if rate > 0 else 0
                print(f'  [{done:,}/{n_samples:,}] {rate:.0f}/s, ETA {eta/60:.1f}min')

    np.savez_compressed(output_path, inputs=inputs, targets=targets)
    elapsed = time.time() - t0
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f'  Saved {n_samples:,} samples to {output_path} ({size_mb:.0f}MB) in {elapsed/60:.1f}min')


# ── Training ──────────────────────────────────────────────────────────────────

def train(data_path, output_model, epochs=80, batch_size=8192, lr=0.001):
    """Train equity network on generated data."""
    data = np.load(data_path)
    X = torch.from_numpy(data['inputs'])
    Y = torch.from_numpy(data['targets'])

    n = len(X)
    split = int(n * 0.95)
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EquityNetwork(input_size=INPUT_SIZE).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/20)

    params = sum(p.numel() for p in model.parameters())
    print(f'  Model params: {params:,}')
    print(f'  Input size: {INPUT_SIZE}')
    print(f'  Train: {split:,}, Val: {n-split:,}')
    print(f'  Device: {device}')

    best_val = float('inf')
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(split)
        total_loss = 0
        batches = 0
        for start in range(0, split, batch_size):
            idx = perm[start:start+batch_size]
            xb = X_train[idx].to(device)
            yb = Y_train[idx].to(device)

            pred = model(xb)

            # Weighted loss: equity gets 3x weight (most important for gameplay)
            weights = torch.ones(9, device=device)
            weights[0] = 3.0  # equity
            weights[1] = 2.0  # made_hand_cat
            weights[8] = 2.0  # rank_within_cat
            loss = ((pred - yb) ** 2 * weights).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val.to(device))
            val_loss = nn.MSELoss()(val_pred, Y_val.to(device)).item()
            mae = (val_pred - Y_val.to(device)).abs().mean(dim=0).cpu().numpy()

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_size': INPUT_SIZE,
                'output_size': 9,
                'hidden': [256, 256, 128],
                'val_loss': val_loss,
                'epoch': epoch,
            }, output_model)

        labels = ['equity', 'made_hand', 'flush_p', 'str_p', 'top_pair',
                  'board_pair', 'flush_tex', 'str_tex', 'rank_cat']
        print(f'  Epoch {epoch+1:3d}/{epochs}  train={total_loss/batches:.4f}  '
              f'val={val_loss:.4f}  {"*best" if improved else ""}')
        if (epoch + 1) % 10 == 0 or epoch == 0:
            mae_str = ' '.join(f'{l}={v:.3f}' for l, v in zip(labels, mae))
            print(f'    MAE: {mae_str}')

    print(f'\n  Best val loss: {best_val:.4f}')
    print(f'  Model saved to: {output_model}')


def main():
    parser = argparse.ArgumentParser(description='Train equity prediction network (v2)')
    parser.add_argument('--samples', type=int, default=10000000)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--train-only', action='store_true')
    parser.add_argument('--data-path', type=str, default='data/equity_training_data_v3.npz')
    parser.add_argument('--output', type=str, default='models/equity_model_v3.pt')
    parser.add_argument('--workers', type=int, default=None, help='CPU workers for data generation')
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.data_path).parent.mkdir(parents=True, exist_ok=True)

    if not args.train_only:
        print(f'Generating {args.samples:,} equity samples (v3 encoding, multiway)...')
        generate_data(args.samples, args.data_path, n_workers=args.workers)

    print(f'\nTraining equity model v2...')
    train(args.data_path, args.output, epochs=args.epochs)

    # Benchmark
    print(f'\nBenchmark:')
    model = EquityNetwork(input_size=INPUT_SIZE)
    ckpt = torch.load(args.output, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        x = torch.randn(10000, INPUT_SIZE).cuda()
        for _ in range(10): model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100): model(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        per_sample = elapsed / (100 * 10000) * 1e6
        print(f'  GPU: {per_sample:.3f} μs/sample ({10000*100/elapsed:,.0f} samples/s)')
        print(f'  vs Monte Carlo: ~300 μs/sample')
        print(f'  Speedup: ~{300/per_sample:.0f}x')

    # Accuracy test on known hands (single + multiway)
    print(f'\nAccuracy test:')
    test_hands = [
        # (name, hole, board, n_opponents)
        ('AA on K72 (1v1)',     ['As','Ah'], ['Kd','7c','2s'], 1),
        ('AA on K72 (1v2)',     ['As','Ah'], ['Kd','7c','2s'], 2),
        ('72o on KQJ (1v1)',   ['7d','2c'], ['Ks','Qh','Jd'], 1),
        ('AKs flush (1v1)',    ['Ah','Kh'], ['Qh','7h','2d'], 1),
        ('AKs flush (1v3)',    ['Ah','Kh'], ['Qh','7h','2d'], 3),
        ('55 on 478 (1v1)',    ['5c','5s'], ['4c','7c','8s'], 1),
        ('55 on 478 (1v2)',    ['5c','5s'], ['4c','7c','8s'], 2),
        ('55 on 478 (1v4)',    ['5c','5s'], ['4c','7c','8s'], 4),
        ('AA preflop (1v1)',   ['As','Ah'], [], 1),
    ]
    for name, h, b, n_opp in test_hands:
        hole = [eval7.Card(c) for c in h]
        board = [eval7.Card(c) for c in b] if b else []
        feat = encode_hand_board(hole, board, n_opp)
        with torch.no_grad():
            inp = torch.from_numpy(feat).unsqueeze(0)
            if torch.cuda.is_available(): inp = inp.cuda()
            pred = model(inp).cpu().numpy()[0]
        if n_opp <= 1:
            real = eval7.py_hand_vs_range_monte_carlo(hole, _RANDOM_RANGE, board, 5000)
        else:
            real = _multiway_equity_mc(hole, board, n_opp, 5000)
        err = abs(pred[0] - real)
        print(f'  {name:22s}  pred={pred[0]:.3f}  real={real:.3f}  err={err:.3f}')


if __name__ == '__main__':
    main()

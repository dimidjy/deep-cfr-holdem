"""Range Estimator model.

Estimates opponent hand strength from the sequence of actions taken
in previous streets. This gives the strategy model information about
HOW the pot was built, not just its current size.

Input: per-street action sequence (variable length, padded)
       + target player's relative position (fed to classifier head)
Output: opponent hand category distribution (9 categories)

The output is meant to be concatenated with the existing 38-dim observation
before feeding into the strategy network.

Labels use actual hand categories (via eval7) instead of equity-vs-random,
giving unambiguous ground truth that doesn't depend on assumed ranges.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# Action encoding dimensions per action step:
#   action_type (6): fold/check/call/bet/raise/all_in  one-hot
#   position (6): one-hot relative position of actor
#   sizing (1): bet/raise size as fraction of pot (0 for non-bet actions)
#   pot_ratio (1): current pot / starting stacks (game progression)
#   stack_ratio (1): actor's remaining stack / starting stack
#   street (4): one-hot preflop/flop/turn/river
#   board texture (7):
#     num_board_cards (1): normalized 0-1 (0/3/4/5 -> 0/0.6/0.8/1.0)
#     max_suit_count (1): most common suit count / board size (flush texture)
#     has_flush_possible (1): 3+ of same suit on board
#     board_paired (1): any rank appears 2+ times
#     board_trips (1): any rank appears 3+ times
#     max_straight_run (1): longest connected rank sequence / 5
#     high_card_norm (1): highest board rank normalized (2=0, A=1)
# Total: 26 dims per action step
ACTION_STEP_DIM = 26

# Maximum actions per hand (preflop can have many raises in multiway)
MAX_ACTIONS = 30

# Output: merged hand categories (from eval7.handtype)
#   0 = High Card
#   1 = Pair
#   2 = Two Pair
#   3 = Trips+ (trips, quads — set-like hands)
#   4 = Straight (straight, straight flush — straight board)
#   5 = Flush
#   6 = Full House
NUM_HAND_CATEGORIES = 7

# Keep old name for backward compatibility in imports
NUM_STRENGTH_BUCKETS = NUM_HAND_CATEGORIES

# Map eval7 handtype strings to category indices
HANDTYPE_TO_CATEGORY = {
    'High Card': 0,
    'Pair': 1,
    'Two Pair': 2,
    'Trips': 3,
    'Straight': 4,
    'Flush': 5,
    'Full House': 6,
    'Quads': 3,            # merged with trips
    'Straight Flush': 4,   # merged with straight
}

CATEGORY_NAMES = [
    'high_card', 'pair', 'two_pair', 'trips+',
    'straight+', 'flush', 'full_house',
]


class RangeEstimator(nn.Module):
    """GRU-based model that reads action sequences and estimates opponent range.

    Architecture:
        Action sequence -> GRU -> concat(hidden, target_position) -> FC -> hand category

    The target player's position is fed to the FC head (not per-action),
    so the GRU must learn from ALL players' actions, not just the target's.
    """

    def __init__(self, input_dim: int = ACTION_STEP_DIM,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 output_dim: int = NUM_HAND_CATEGORIES,
                 dropout: float = 0.1,
                 num_positions: int = 6):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_positions = num_positions

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # FC head receives GRU hidden + target position one-hot
        fc_input_dim = hidden_dim + num_positions
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, action_seq: torch.Tensor,
                seq_lengths: torch.Tensor = None,
                target_positions: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            action_seq: (batch, max_seq_len, input_dim) padded action sequences
            seq_lengths: (batch,) actual lengths (optional, for packed sequences)
            target_positions: (batch,) relative position of target opponent [0, num_positions)

        Returns:
            (batch, output_dim) -- logits (apply softmax externally for probabilities)
        """
        if seq_lengths is not None and seq_lengths.min() > 0:
            packed = nn.utils.rnn.pack_padded_sequence(
                action_seq, seq_lengths.cpu(), batch_first=True,
                enforce_sorted=False)
            _, hidden = self.gru(packed)
        else:
            _, hidden = self.gru(action_seq)

        # Use last layer's hidden state
        last_hidden = hidden[-1]  # (batch, hidden_dim)

        # Concat target position one-hot
        if target_positions is not None:
            pos_oh = F.one_hot(target_positions.long(), self.num_positions).float()
            last_hidden = torch.cat([last_hidden, pos_oh], dim=-1)
        else:
            # Fallback: zero position (shouldn't happen in normal use)
            pad = torch.zeros(last_hidden.size(0), self.num_positions,
                              device=last_hidden.device)
            last_hidden = torch.cat([last_hidden, pad], dim=-1)

        logits = self.fc(last_hidden)
        return logits

    def predict_probs(self, action_seq: torch.Tensor,
                      seq_lengths: torch.Tensor = None,
                      target_positions: torch.Tensor = None) -> torch.Tensor:
        """Returns softmax probabilities instead of logits."""
        logits = self.forward(action_seq, seq_lengths, target_positions)
        return F.softmax(logits, dim=-1)


_RANK_MAP = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
             '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}


def encode_board_texture(community_cards: List[str]) -> List[float]:
    """Encode board texture into 7 floats.

    Args:
        community_cards: e.g. ['9h', '3c', '7d'] — card format is RankSuit

    Returns:
        7 floats: [num_cards_norm, max_suit_frac, flush_possible,
                   board_paired, board_trips, straight_run_norm, high_card_norm]
    """
    n = len(community_cards)
    if n == 0:
        return [0.0] * 7

    # Parse ranks and suits
    ranks = []
    suits = []
    for card in community_cards:
        if len(card) >= 2:
            ranks.append(_RANK_MAP.get(card[0], 0))
            suits.append(card[1])

    if not ranks:
        return [0.0] * 7

    # num_board_cards normalized (3=0.6, 4=0.8, 5=1.0)
    num_norm = min(n, 5) / 5.0

    # Suit counts -> flush texture
    from collections import Counter
    suit_counts = Counter(suits)
    max_suit = max(suit_counts.values())
    max_suit_frac = max_suit / max(n, 1)
    flush_possible = 1.0 if max_suit >= 3 else 0.0

    # Rank counts -> paired/trips
    rank_counts = Counter(ranks)
    max_rank_count = max(rank_counts.values())
    board_paired = 1.0 if max_rank_count >= 2 else 0.0
    board_trips = 1.0 if max_rank_count >= 3 else 0.0

    # Straight texture: longest run of connected ranks
    unique_ranks = sorted(set(ranks))
    # Add low ace for wheel straights
    if 14 in unique_ranks:
        unique_ranks = [1] + unique_ranks
    max_run = 1
    current_run = 1
    for i in range(1, len(unique_ranks)):
        if unique_ranks[i] - unique_ranks[i - 1] == 1:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    straight_run_norm = min(max_run, 5) / 5.0

    # High card normalized (2=0.0, A=1.0)
    high_card_norm = (max(ranks) - 2) / 12.0

    return [num_norm, max_suit_frac, flush_possible,
            board_paired, board_trips, straight_run_norm, high_card_norm]


def encode_action(action_type: str, rel_position: int, sizing: float,
                  street: str, pot_ratio: float = 0.0,
                  stack_ratio: float = 1.0,
                  num_players: int = 6,
                  community_cards: Optional[List[str]] = None) -> List[float]:
    """Encode a single action into ACTION_STEP_DIM floats.

    Args:
        action_type: fold/check/call/bet/raise/all_in
        rel_position: relative position of actor (0=BTN, 1=SB, ...)
        sizing: bet/raise amount as fraction of pot (0 for non-bet)
        street: preflop/flop/turn/river
        pot_ratio: current pot / starting stacks (game progression context)
        stack_ratio: actor's remaining stack / starting stack
        num_players: table size
        community_cards: board cards at this point (e.g. ['9h', '3c', '7d'])

    Returns:
        List of ACTION_STEP_DIM floats
    """
    # Action type one-hot (6)
    action_types = ['fold', 'check', 'call', 'bet', 'raise', 'all_in']
    action_oh = [0.0] * 6
    idx = action_types.index(action_type) if action_type in action_types else 0
    action_oh[idx] = 1.0

    # Position one-hot (6)
    pos_oh = [0.0] * 6
    if 0 <= rel_position < 6:
        pos_oh[rel_position] = 1.0

    # Sizing -- clipped to [0, 5] then normalized
    sizing_norm = min(sizing, 5.0) / 5.0

    # Pot ratio -- clipped to [0, 5] then normalized
    pot_ratio_norm = min(max(pot_ratio, 0.0), 5.0) / 5.0

    # Stack ratio -- clipped to [0, 2] then normalized
    stack_ratio_norm = min(max(stack_ratio, 0.0), 2.0) / 2.0

    # Street one-hot (4)
    street_map = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
    street_oh = [0.0] * 4
    s_idx = street_map.get(street, 0)
    street_oh[s_idx] = 1.0

    # Board texture (7)
    board = encode_board_texture(community_cards or [])

    return action_oh + pos_oh + [sizing_norm, pot_ratio_norm, stack_ratio_norm] + street_oh + board


try:
    import eval7 as _eval7
    _EVAL7_AVAILABLE = True
except ImportError:
    _EVAL7_AVAILABLE = False


_hand_category_errors = 0


def hand_category_from_cards(hole_cards: List[str],
                             community_cards: List[str]) -> Optional[int]:
    """Determine hand category using eval7.

    Args:
        hole_cards: e.g. ['As', 'Kd']
        community_cards: e.g. ['9h', '3c', '7d'] (3-5 cards)

    Returns:
        Category index 0-8, or None if evaluation fails
    """
    global _hand_category_errors

    if not _EVAL7_AVAILABLE:
        if _hand_category_errors == 0:
            import sys
            print('ERROR: eval7 not available — hand_category_from_cards will return None',
                  file=sys.stderr)
        _hand_category_errors += 1
        return None

    try:
        all_cards = [_eval7.Card(c) for c in hole_cards[:2]]
        all_cards += [_eval7.Card(c) for c in community_cards]
        if len(all_cards) < 5:
            return None
        score = _eval7.evaluate(all_cards)
        handtype_str = _eval7.handtype(score)
        return HANDTYPE_TO_CATEGORY.get(handtype_str, None)
    except Exception as e:
        _hand_category_errors += 1
        if _hand_category_errors <= 5:
            import sys
            print(f'WARNING: hand_category_from_cards failed: {e} '
                  f'(hole={hole_cards}, community={community_cards})', file=sys.stderr)
        return None


# Keep old function name for any external references, but it's no longer the primary path
def hand_strength_to_bucket(equity: float) -> int:
    """Legacy: map equity to bucket. Prefer hand_category_from_cards()."""
    if equity < 0.25:
        return 0
    elif equity < 0.40:
        return 1
    elif equity < 0.60:
        return 2
    elif equity < 0.80:
        return 3
    else:
        return min(4, NUM_HAND_CATEGORIES - 1)

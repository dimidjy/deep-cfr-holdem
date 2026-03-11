"""Parse PokerStars hand history files into structured ParsedHand objects."""

import re
from pathlib import Path
from typing import List, Optional

from .models import ParsedPlayer, ParsedAction, ParsedHand, HandResult

# ── Regex patterns ────────────────────────────────────────────────────────────

_RE_HEADER = re.compile(
    r"PokerStars Hand #(\d+):\s+Hold'em No Limit "
    r"\(\$([0-9.]+)/\$([0-9.]+) USD\) - (.+)"
)
_RE_TABLE = re.compile(
    r"Table '(.+?)' (\d+)-max Seat #(\d+) is the button"
)
_RE_SEAT = re.compile(
    r"Seat (\d+): (.+?) \(\$([0-9.]+) in chips\)(.*)"
)
_RE_BLIND = re.compile(
    r"^(.+?): posts (small blind|big blind|small & big blinds) \$([0-9.]+)"
)
_RE_FOLD = re.compile(r"^(.+?): folds\s*$")
_RE_CHECK = re.compile(r"^(.+?): checks\s*$")
_RE_CALL = re.compile(r"^(.+?): calls \$([0-9.]+)(.*)")
_RE_BET = re.compile(r"^(.+?): bets \$([0-9.]+)(.*)")
_RE_RAISE = re.compile(r"^(.+?): raises \$[0-9.]+ to \$([0-9.]+)(.*)")
_RE_SHOW = re.compile(r"^(.+?): shows \[(.+?)\]")
_RE_MUCK = re.compile(r"^(.+?): mucks hand")
_RE_COLLECTED = re.compile(r"^(.+?) collected \$([0-9.]+) from")
_RE_SUMMARY_POT = re.compile(r"Total pot \$([0-9.]+) \| Rake \$([0-9.]+)")
_RE_COMMUNITY = re.compile(r"\*\*\* (FLOP|TURN|RIVER) \*\*\* (.+)")
_RE_UNCALLED = re.compile(r"Uncalled bet \(\$([0-9.]+)\) returned to (.+)")

_STREETS = {'*** HOLE CARDS ***': 'preflop',
            '*** FLOP ***': 'flop',
            '*** TURN ***': 'turn',
            '*** RIVER ***': 'river'}


def _parse_card_list(text: str) -> List[str]:
    """Parse '[9h 9d Qs]' or '[9h 9d Qs] [Kc]' into card strings."""
    cards = re.findall(r'[2-9TJQKA][cdhs]', text)
    return cards


def parse_hand(hand_text: str) -> Optional[ParsedHand]:
    """Parse a single hand block into a ParsedHand."""
    lines = hand_text.strip().split('\n')
    if not lines:
        return None

    hand = ParsedHand()
    current_street = 'preflop'
    section = 'header'  # header, action, showdown, summary

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # ── Header ────────────────────────────────────────────────────
        m = _RE_HEADER.match(line)
        if m:
            hand.hand_id = m.group(1)
            hand.small_blind_dollars = float(m.group(2))
            hand.big_blind_dollars = float(m.group(3))
            hand.timestamp = m.group(4)
            continue

        m = _RE_TABLE.match(line)
        if m:
            hand.table_name = m.group(1)
            hand.max_seats = int(m.group(2))
            hand.button_seat = int(m.group(3))
            continue

        # ── Seats ─────────────────────────────────────────────────────
        m = _RE_SEAT.match(line)
        if m:
            seat = int(m.group(1))
            name = m.group(2)
            stack = float(m.group(3))
            rest = m.group(4).strip()
            sitting_out = 'is sitting out' in rest
            hand.players.append(ParsedPlayer(
                name=name, seat=seat, stack_dollars=stack,
                is_sitting_out=sitting_out
            ))
            continue

        # ── Section markers ───────────────────────────────────────────
        if line.startswith('*** HOLE CARDS ***'):
            section = 'action'
            current_street = 'preflop'
            continue

        if line.startswith('*** SHOW DOWN ***'):
            section = 'showdown'
            continue

        if line.startswith('*** SUMMARY ***'):
            section = 'summary'
            continue

        # ── Community cards ───────────────────────────────────────────
        m = _RE_COMMUNITY.match(line)
        if m:
            street_name = m.group(1)
            cards_text = m.group(2)
            cards = _parse_card_list(cards_text)

            if street_name == 'FLOP':
                current_street = 'flop'
                hand.flop = cards[:3]
            elif street_name == 'TURN':
                current_street = 'turn'
                if cards:
                    hand.turn = cards[-1]
            elif street_name == 'RIVER':
                current_street = 'river'
                if cards:
                    hand.river = cards[-1]
            continue

        # ── Ignore non-action lines ───────────────────────────────────
        if ('joins the table' in line or 'leaves the table' in line
                or 'is disconnected' in line or 'is connected' in line
                or 'sits out' in line or "doesn't show hand" in line
                or 'said,' in line or 'was removed' in line
                or 'has timed out' in line or 'has returned' in line
                or 'will be allowed to play' in line
                or line.startswith('Dealt to ')):
            continue

        # ── Blinds (before HOLE CARDS or in action) ──────────────────
        m = _RE_BLIND.match(line)
        if m:
            name = m.group(1)
            blind_type = m.group(2)
            amount = float(m.group(3))
            action_type = 'blind_sb' if blind_type == 'small blind' else 'blind_bb'
            hand.actions.append(ParsedAction(
                player_name=name, action_type=action_type,
                amount_dollars=amount, street='preflop', is_blind=True
            ))
            continue

        # ── Actions ───────────────────────────────────────────────────
        if section == 'action':
            m = _RE_FOLD.match(line)
            if m:
                hand.actions.append(ParsedAction(
                    player_name=m.group(1), action_type='fold',
                    street=current_street
                ))
                continue

            m = _RE_CHECK.match(line)
            if m:
                hand.actions.append(ParsedAction(
                    player_name=m.group(1), action_type='check',
                    street=current_street
                ))
                continue

            m = _RE_CALL.match(line)
            if m:
                is_allin = 'all-in' in m.group(3)
                hand.actions.append(ParsedAction(
                    player_name=m.group(1), action_type='call',
                    amount_dollars=float(m.group(2)),
                    street=current_street, is_all_in=is_allin
                ))
                continue

            m = _RE_RAISE.match(line)
            if m:
                is_allin = 'all-in' in m.group(3)
                hand.actions.append(ParsedAction(
                    player_name=m.group(1), action_type='raise',
                    amount_dollars=float(m.group(2)),  # "to" amount
                    street=current_street, is_all_in=is_allin
                ))
                continue

            m = _RE_BET.match(line)
            if m:
                is_allin = 'all-in' in m.group(3)
                hand.actions.append(ParsedAction(
                    player_name=m.group(1), action_type='bet',
                    amount_dollars=float(m.group(2)),
                    street=current_street, is_all_in=is_allin
                ))
                continue

        # ── Showdown ──────────────────────────────────────────────────
        if section == 'showdown':
            m = _RE_SHOW.match(line)
            if m:
                name = m.group(1)
                cards = _parse_card_list(m.group(2))
                hand.shown_hands[name] = cards
                continue

        # ── Summary ───────────────────────────────────────────────────
        if section == 'summary':
            m = _RE_SUMMARY_POT.match(line)
            if m:
                hand.result.total_pot_dollars = float(m.group(1))
                hand.result.rake_dollars = float(m.group(2))
                continue

            # Also pick up shown hands from summary seat lines
            seat_show = re.match(
                r'Seat \d+: .+? showed \[(.+?)\] and (won|lost)', line)
            if seat_show:
                cards = _parse_card_list(seat_show.group(1))
                # Extract player name from between "Seat N: " and " ("
                name_match = re.match(r'Seat \d+: (.+?) (?:\(|showed)', line)
                if name_match and cards:
                    name = name_match.group(1).strip()
                    if name not in hand.shown_hands:
                        hand.shown_hands[name] = cards

        # ── Collected (can appear in action or summary) ──────────────
        m = _RE_COLLECTED.match(line)
        if m:
            hand.result.winners.append({
                'name': m.group(1),
                'amount_dollars': float(m.group(2))
            })
            continue

        # Uncalled bet — just ignore (tracked implicitly)
        if _RE_UNCALLED.match(line):
            continue

    if not hand.hand_id:
        return None

    # Assign shown hole cards to players
    for player in hand.players:
        if player.name in hand.shown_hands:
            player.hole_cards = hand.shown_hands[player.name]

    return hand


def parse_file(filepath: str) -> List[ParsedHand]:
    """Parse an entire PokerStars HH file into a list of ParsedHand objects."""
    path = Path(filepath)
    text = path.read_text(encoding='utf-8-sig', errors='replace')

    # Split on hand boundaries
    blocks = re.split(r'\n(?=PokerStars Hand #)', text)

    hands = []
    for block in blocks:
        block = block.strip()
        if not block or not block.startswith('PokerStars Hand #'):
            continue
        hand = parse_hand(block)
        if hand is not None:
            hands.append(hand)

    return hands


def parse_directory(dir_path: str,
                    require_6_active: bool = True,
                    require_showdown: bool = True,
                    require_nl5: bool = True) -> List[ParsedHand]:
    """Parse all HH files in a directory (recursively) with optional filtering."""
    dirp = Path(dir_path)
    files = sorted(dirp.rglob('*.txt'))

    all_hands = []
    for f in files:
        all_hands.extend(parse_file(str(f)))

    total = len(all_hands)

    filtered = all_hands
    if require_nl5:
        filtered = [h for h in filtered if abs(h.big_blind_dollars - 0.05) < 0.001]
    if require_6_active:
        filtered = [h for h in filtered if h.num_active_players == 6]
    if require_showdown:
        filtered = [h for h in filtered if h.has_showdown]

    return filtered

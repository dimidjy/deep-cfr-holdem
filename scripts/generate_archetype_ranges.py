"""Generate preflop range JSON files for each of the 5 archetypes.

Each archetype's ranges target specific VPIP/PFR stats:
  Nit:            VPIP 12%, PFR 10%
  Weak-Tight:     VPIP 18%, PFR 14%
  Calling Station: VPIP 55%, PFR 10%
  LAG:            VPIP 32%, PFR 25%
  Maniac:         VPIP 60%, PFR 45%

Output: configs/archetype_ranges/{archetype}.json for each archetype.

Run validation after generating:
    python3 scripts/generate_archetype_ranges.py
    python3 scripts/validate_archetype_ranges.py
"""
import json
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / 'configs' / 'archetype_ranges'

RANKS = '23456789TJQKA'


# ── Combo helpers ────────────────────────────────────────────────────────────

def all_pairs_above(min_rank: str) -> List[str]:
    """Return all pair combos rank+ (e.g. '77'+ returns 77,88,...,AA).

    Accepts either single-char rank ('7') or pair string ('77').
    """
    r = min_rank[0]  # handle both '7' and '77'
    idx = RANKS.index(r)
    return [f'{x}{x}' for x in RANKS[idx:]]


def suited_connectors(min_low: str = '5', max_low: str = 'T') -> List[str]:
    """Suited connectors like 54s, 65s, ..., T9s."""
    out = []
    for i, r in enumerate(RANKS):
        if r < min_low or r > max_low:
            continue
        next_r = RANKS[i + 1] if i + 1 < len(RANKS) else None
        if next_r:
            out.append(f'{next_r}{r}S')
    return out


def suited_one_gappers(min_low: str = '5', max_low: str = '9') -> List[str]:
    """Suited one-gappers like 53s, 64s, 75s, 86s, 97s."""
    out = []
    for i, r in enumerate(RANKS):
        if r < min_low or r > max_low:
            continue
        next_r = RANKS[i + 2] if i + 2 < len(RANKS) else None
        if next_r:
            out.append(f'{next_r}{r}S')
    return out


def suited_axs(min_low: str = '2') -> List[str]:
    """All Axs like A2s, A3s, ... AKs."""
    idx = RANKS.index(min_low)
    return [f'A{r}S' for r in RANKS[idx:-1]]


def offsuit_ax(min_low: str = 'T') -> List[str]:
    """Offsuit Ax like ATo, AJo, AQo, AKo."""
    idx = RANKS.index(min_low)
    return [f'A{r}O' for r in RANKS[idx:-1]]


def suited_kx(min_low: str = 'T') -> List[str]:
    """Suited Kx like KTs, KJs, KQs."""
    idx = RANKS.index(min_low)
    return [f'K{r}S' for r in RANKS[idx:RANKS.index('K')]]


def offsuit_kx(min_low: str = 'J') -> List[str]:
    """Offsuit Kx."""
    idx = RANKS.index(min_low)
    return [f'K{r}O' for r in RANKS[idx:RANKS.index('K')]]


def suited_qx(min_low: str = 'T') -> List[str]:
    """Suited Qx."""
    idx = RANKS.index(min_low)
    return [f'Q{r}S' for r in RANKS[idx:RANKS.index('Q')]]


def offsuit_qx(min_low: str = 'J') -> List[str]:
    """Offsuit Qx."""
    idx = RANKS.index(min_low)
    return [f'Q{r}O' for r in RANKS[idx:RANKS.index('Q')]]


def suited_jx(min_low: str = 'T') -> List[str]:
    """Suited Jx."""
    idx = RANKS.index(min_low)
    return [f'J{r}S' for r in RANKS[idx:RANKS.index('J')]]


def to_range_dict(combos: List[str], default_value: str = '1') -> Dict[str, str]:
    """Convert list of combos to dict format expected by preflop_ranges.json.
    Pair combos prefix with ':' (matching existing file convention)."""
    out = {}
    for combo in combos:
        if len(combo) == 2 and combo[0] == combo[1]:  # pair
            out[combo] = f':{default_value}' if default_value.startswith('0') else ':0.99'
        else:
            out[combo] = default_value if '.' in default_value else '0.99'
    return out


# ── Archetype range definitions ──────────────────────────────────────────────

def build_nit() -> Dict:
    """Tight ABC — top 8-20% by position. Folds to aggression."""
    very_premium = ['KK', 'AA', 'AKS']

    return {
        'rfi': {
            # LJ/HJ/CO: open ~5-10%
            'LJ': to_range_dict(
                all_pairs_above('88')
                + ['AKS', 'AKO', 'AQS', 'AJS', 'KQS']
            ),
            'HJ': to_range_dict(
                all_pairs_above('77')
                + ['AKS', 'AKO', 'AQS', 'AJS', 'ATS', 'KQS', 'KJS']
            ),
            'CO': to_range_dict(
                all_pairs_above('55')
                + suited_axs('9')
                + ['AJO', 'AQO', 'AKO', 'KQS', 'KJS', 'KTS', 'QJS', 'QTS', 'JTS']
            ),
            'BTN': to_range_dict(
                all_pairs_above('22')
                + suited_axs('2')
                + offsuit_ax('9')
                + ['KQS', 'KJS', 'KTS', 'K9S', 'KQO', 'KJO',
                   'QJS', 'QTS', 'Q9S', 'QJO',
                   'JTS', 'J9S', 'T9S', '98S']
            ),
            'SB': to_range_dict(
                all_pairs_above('44')
                + suited_axs('5')
                + ['ATO', 'AJO', 'AQO', 'AKO',
                   'KQS', 'KJS', 'KTS', 'K9S', 'KQO',
                   'QJS', 'QTS', 'JTS', 'T9S']
            ),
        },
        'rfi_call': {
            'LJ': to_range_dict(['QQ', 'KK', 'AA', 'AKS', 'AKO', 'AQS', 'JJ']),
            'HJ': to_range_dict(['JJ', 'QQ', 'KK', 'AA', 'AKS', 'AKO', 'AQS']),
            'CO': to_range_dict(['TT', 'JJ', 'QQ', 'KK', 'AA', 'AKS', 'AKO', 'AQS']),
            'BTN': to_range_dict(['99', 'TT', 'JJ', 'QQ', 'KK', 'AA',
                                   'AKS', 'AKO', 'AQS', 'AJS', 'KQS']),
            'SB': to_range_dict(['TT', 'JJ', 'QQ', 'KK', 'AA', 'AKS', 'AKO', 'AQS', 'AJS']),
            'BB': to_range_dict(
                all_pairs_above('66')
                + ['AKS', 'AKO', 'AQS', 'AJS', 'ATS', 'KQS', 'KJS', 'QJS', 'JTS']
            ),
        },
        '3b': {
            f'{pos}_3b': to_range_dict(very_premium + ['QQ'])
            for pos in ['LJ', 'HJ', 'CO', 'BTN', 'SB', 'BB']
        },
        '3b_call': {
            f'{pos}_call': to_range_dict(['JJ', 'AKS', 'AKO'])
            for pos in ['LJ', 'HJ', 'CO', 'BTN', 'SB', 'BB']
        },
        '4b': {
            f'{pos}_4b': to_range_dict(['KK', 'AA', 'AKS'])
            for pos in ['LJ', 'HJ', 'CO', 'BTN', 'SB', 'BB']
        },
        '4b_call': {
            f'{pos}_call': to_range_dict(['KK', 'AA'])
            for pos in ['LJ', 'HJ', 'CO', 'BTN', 'SB', 'BB']
        },
        'Allin': {
            'allin_base': to_range_dict(['KK', 'AA', 'AKS']),
        },
        'SB_DEF': {
            'SB_DEF_call': to_range_dict(
                all_pairs_above('55')
                + ['AKS', 'AKO', 'AQS', 'AJS', 'ATS',
                   'KQS', 'KJS', 'QJS', 'JTS']
            ),
        },
    }


def build_weak_tight() -> Dict:
    """Tight but passive — slightly wider than nit, calls more than raises.
    Target: VPIP 18%, PFR 14% — still tight, just slightly wider."""
    return {
        'rfi': {
            'LJ': to_range_dict(
                all_pairs_above('66')
                + ['AKS', 'AKO', 'AQS', 'AJS', 'ATS',
                   'KQS', 'KJS']
            ),
            'HJ': to_range_dict(
                all_pairs_above('55')
                + ['AKS', 'AKO', 'AQS', 'AJS', 'ATS', 'AQO', 'AJO',
                   'KQS', 'KJS', 'KTS', 'QJS']
            ),
            'CO': to_range_dict(
                all_pairs_above('33')
                + suited_axs('8')
                + ['AQO', 'AJO', 'AKO', 'ATO',
                   'KQS', 'KJS', 'KTS', 'K9S',
                   'QJS', 'QTS', 'Q9S',
                   'JTS', 'J9S', 'T9S', '98S']
            ),
            'BTN': to_range_dict(
                all_pairs_above('22')
                + suited_axs('2')
                + offsuit_ax('9')
                + ['KQO', 'KJO', 'KTO',
                   'KQS', 'KJS', 'KTS', 'K9S', 'K8S',
                   'QJS', 'QTS', 'Q9S', 'QJO', 'QTO',
                   'JTS', 'J9S', 'J8S', 'JTO',
                   'T9S', 'T8S', '98S', '87S', '76S']
            ),
            'SB': to_range_dict(
                all_pairs_above('44')
                + suited_axs('7')
                + ['AKO', 'AQO', 'AJO', 'ATO',
                   'KQS', 'KJS', 'KTS', 'K9S', 'KQO', 'KJO',
                   'QJS', 'QTS', 'Q9S', 'QJO',
                   'JTS', 'J9S', 'T9S', '98S']
            ),
        },
        'rfi_call': {
            'LJ': to_range_dict(
                all_pairs_above('66')
                + ['AKS', 'AKO', 'AQS', 'AQO', 'AJS', 'KQS']
            ),
            'HJ': to_range_dict(
                all_pairs_above('55')
                + ['AKS', 'AKO', 'AQS', 'AQO', 'AJS', 'AJO', 'KQS', 'QJS']
            ),
            'CO': to_range_dict(
                all_pairs_above('33')
                + suited_axs('7')
                + ['AJO', 'AQO', 'AKO',
                   'KQS', 'KJS', 'KTS', 'KQO', 'QJS', 'QTS', 'JTS', 'T9S']
            ),
            'BTN': to_range_dict(
                all_pairs_above('22')
                + suited_axs('2')
                + ['ATO', 'AJO', 'AQO', 'AKO', 'KQO',
                   'KQS', 'KJS', 'KTS', 'K9S', 'K8S',
                   'QJS', 'QTS', 'Q9S',
                   'JTS', 'J9S', 'T9S', '98S', '87S']
            ),
            'SB': to_range_dict(
                all_pairs_above('44')
                + suited_axs('2')
                + ['AJO', 'AQO', 'AKO',
                   'KQS', 'KJS', 'KTS', 'K9S', 'KQO',
                   'QJS', 'QTS', 'JTS', 'T9S', '98S']
            ),
            'BB': to_range_dict(
                all_pairs_above('22')
                + suited_axs('2')
                + ['A9O', 'ATO', 'AJO', 'AQO', 'AKO',
                   'K9O', 'KTO', 'KJO', 'KQO',
                   'KQS', 'KJS', 'KTS', 'K9S', 'K8S', 'K7S',
                   'QTO', 'QJO',
                   'QJS', 'QTS', 'Q9S', 'Q8S',
                   'JTS', 'J9S', 'J8S',
                   'T9S', 'T8S', '98S', '87S', '76S']
            ),
        },
        '3b': {
            f'{pos}_3b': to_range_dict(['QQ', 'KK', 'AA', 'AKS', 'AKO'])
            for pos in ['LJ', 'HJ', 'CO', 'BTN', 'SB', 'BB']
        },
        '3b_call': {
            f'{pos}_call': to_range_dict(
                ['99', 'TT', 'JJ', 'AKS', 'AKO', 'AQS', 'AQO', 'KQS']
            )
            for pos in ['LJ', 'HJ', 'CO', 'BTN', 'SB', 'BB']
        },
        '4b': {
            f'{pos}_4b': to_range_dict(['KK', 'AA', 'AKS', 'AKO'])
            for pos in ['LJ', 'HJ', 'CO', 'BTN', 'SB', 'BB']
        },
        '4b_call': {
            f'{pos}_call': to_range_dict(['QQ', 'KK', 'AA', 'AKS', 'AKO'])
            for pos in ['LJ', 'HJ', 'CO', 'BTN', 'SB', 'BB']
        },
        'Allin': {
            'allin_base': to_range_dict(['KK', 'AA', 'AKS', 'AKO']),
        },
        'SB_DEF': {
            'SB_DEF_call': to_range_dict(
                all_pairs_above('22')
                + suited_axs('2')
                + ['AJO', 'AQO', 'AKO',
                   'KQS', 'KJS', 'KTS', 'K9S', 'KQO', 'KJO',
                   'QJS', 'QTS', 'Q9S', 'QJO',
                   'JTS', 'J9S', 'T9S', '98S', '87S', '76S']
            ),
        },
    }


def build_calling_station() -> Dict:
    """Loose-passive — narrow RFI, ultra-wide calling ranges (pays everything off)."""
    # Super wide — roughly 65-70% of hands
    wide_calling_range = (
        all_pairs_above('22')
        + suited_axs('2')
        + ['A2O', 'A3O', 'A4O', 'A5O', 'A6O', 'A7O', 'A8O', 'A9O',
           'ATO', 'AJO', 'AQO', 'AKO']
        # Kings — everything
        + ['K2S', 'K3S', 'K4S', 'K5S', 'K6S', 'K7S', 'K8S', 'K9S',
           'KTS', 'KJS', 'KQS',
           'K2O', 'K3O', 'K4O', 'K5O', 'K6O', 'K7O', 'K8O', 'K9O',
           'KTO', 'KJO', 'KQO']
        # Queens — wide
        + ['Q2S', 'Q3S', 'Q4S', 'Q5S', 'Q6S', 'Q7S', 'Q8S', 'Q9S',
           'QTS', 'QJS',
           'Q5O', 'Q6O', 'Q7O', 'Q8O', 'Q9O', 'QTO', 'QJO']
        # Jacks — wide
        + ['J2S', 'J3S', 'J4S', 'J5S', 'J6S', 'J7S', 'J8S', 'J9S', 'JTS',
           'J7O', 'J8O', 'J9O', 'JTO']
        # Tens — wide
        + ['T2S', 'T3S', 'T4S', 'T5S', 'T6S', 'T7S', 'T8S', 'T9S',
           'T7O', 'T8O', 'T9O']
        # 9x
        + ['92S', '93S', '94S', '95S', '96S', '97S', '98S', '97O', '98O']
        # 8x
        + ['82S', '83S', '84S', '85S', '86S', '87S', '87O']
        # 7x
        + ['73S', '74S', '75S', '76S', '76O']
        # 6x
        + ['63S', '64S', '65S', '65O']
        # 5x
        + ['53S', '54S', '54O']
        # 4x
        + ['43S']
        # 3x
        + ['32S']
    )

    return {
        'rfi': {
            'LJ': to_range_dict(['JJ', 'QQ', 'KK', 'AA', 'AKS', 'AKO']),
            'HJ': to_range_dict(['TT', 'JJ', 'QQ', 'KK', 'AA',
                                  'AKS', 'AKO', 'AQS']),
            'CO': to_range_dict(['99', 'TT', 'JJ', 'QQ', 'KK', 'AA',
                                  'AKS', 'AKO', 'AQS']),
            'BTN': to_range_dict(['77', '88', '99', 'TT', 'JJ', 'QQ', 'KK', 'AA',
                                   'AKS', 'AKO', 'AQS', 'AQO', 'AJS']),
            'SB': to_range_dict(['99', 'TT', 'JJ', 'QQ', 'KK', 'AA',
                                  'AKS', 'AKO', 'AQS']),
        },
        'rfi_call': {
            'LJ': to_range_dict(wide_calling_range),
            'HJ': to_range_dict(wide_calling_range),
            'CO': to_range_dict(wide_calling_range),
            'BTN': to_range_dict(wide_calling_range),
            'SB': to_range_dict(wide_calling_range),
            'BB': to_range_dict(wide_calling_range),
        },
        '3b': {
            f'{pos}_3b': to_range_dict(['KK', 'AA'])
            for pos in ['LJ', 'HJ', 'CO', 'BTN', 'SB', 'BB']
        },
        '3b_call': {
            f'{pos}_call': to_range_dict(
                all_pairs_above('22')
                + ['AKS', 'AKO', 'AQS', 'AQO', 'AJS', 'AJO',
                   'KQS', 'KQO', 'KJS', 'QJS']
            )
            for pos in ['LJ', 'HJ', 'CO', 'BTN', 'SB', 'BB']
        },
        '4b': {
            f'{pos}_4b': to_range_dict(['AA'])
            for pos in ['LJ', 'HJ', 'CO', 'BTN', 'SB', 'BB']
        },
        '4b_call': {
            f'{pos}_call': to_range_dict(
                ['JJ', 'QQ', 'KK', 'AA', 'AKS', 'AKO', 'AQS']
            )
            for pos in ['LJ', 'HJ', 'CO', 'BTN', 'SB', 'BB']
        },
        'Allin': {
            'allin_base': to_range_dict(['KK', 'AA']),
        },
        'SB_DEF': {
            'SB_DEF_call': to_range_dict(wide_calling_range),
        },
    }


def build_lag() -> Dict:
    """Loose-aggressive winning style — wide RFI, aggressive 3-bets.
    Target: VPIP 32%, PFR 25% — opens ~25-30% EP, wider late."""
    return {
        'rfi': {
            # LJ: 15-18%
            'LJ': to_range_dict(
                all_pairs_above('55')
                + suited_axs('5')
                + ['AJO', 'AQO', 'AKO',
                   'K9S', 'KTS', 'KJS', 'KQS', 'KJO', 'KQO',
                   'QTS', 'QJS', 'QJO',
                   'JTS', 'T9S', '98S', '87S']
            ),
            # HJ: 22-25%
            'HJ': to_range_dict(
                all_pairs_above('33')
                + suited_axs('2')
                + offsuit_ax('9')
                + ['KTS', 'KJS', 'KQS', 'K9S', 'K8S',
                   'KJO', 'KQO', 'KTO',
                   'QTS', 'QJS', 'Q9S', 'Q8S',
                   'QTO', 'QJO',
                   'JTS', 'J9S', 'J8S', 'JTO',
                   'T9S', 'T8S', '98S', '87S', '76S']
            ),
            'CO': to_range_dict(
                all_pairs_above('22')
                + suited_axs('2')
                + offsuit_ax('8')
                + ['K8S', 'K9S', 'KTS', 'KJS', 'KQS',
                   'K9O', 'KTO', 'KJO', 'KQO',
                   'Q8S', 'Q9S', 'QTS', 'QJS',
                   'Q9O', 'QTO', 'QJO',
                   'J8S', 'J9S', 'JTS', 'JTO',
                   'T8S', 'T9S', '97S', '98S',
                   '86S', '87S', '76S', '65S']
            ),
            'BTN': to_range_dict(
                all_pairs_above('22')
                + suited_axs('2')
                + offsuit_ax('2')
                + ['K2S', 'K3S', 'K4S', 'K5S', 'K6S', 'K7S', 'K8S', 'K9S',
                   'KTS', 'KJS', 'KQS',
                   'K7O', 'K8O', 'K9O', 'KTO', 'KJO', 'KQO',
                   'Q5S', 'Q6S', 'Q7S', 'Q8S', 'Q9S', 'QTS', 'QJS',
                   'Q8O', 'Q9O', 'QTO', 'QJO',
                   'J7S', 'J8S', 'J9S', 'JTS',
                   'J8O', 'J9O', 'JTO',
                   'T7S', 'T8S', 'T9S', 'T9O',
                   '96S', '97S', '98S',
                   '85S', '86S', '87S',
                   '75S', '76S',
                   '64S', '65S',
                   '54S', '53S']
            ),
            'SB': to_range_dict(
                all_pairs_above('22')
                + suited_axs('2')
                + offsuit_ax('8')
                + ['K8S', 'K9S', 'KTS', 'KJS', 'KQS',
                   'KTO', 'KJO', 'KQO',
                   'Q9S', 'QTS', 'QJS',
                   'QTO', 'QJO',
                   'J9S', 'JTS', 'JTO',
                   'T9S', '98S', '87S', '76S', '65S']
            ),
        },
        'rfi_call': {
            'LJ': to_range_dict(
                all_pairs_above('55')
                + ['AKS', 'AKO', 'AQS', 'AQO', 'AJS', 'KQS', 'QJS', 'JTS']
            ),
            'HJ': to_range_dict(
                all_pairs_above('33')
                + suited_axs('9')
                + ['AJO', 'AQO', 'AKO',
                   'KQS', 'KJS', 'KTS', 'QJS', 'QTS', 'JTS', 'T9S']
            ),
            'CO': to_range_dict(
                all_pairs_above('22')
                + suited_axs('5')
                + ['ATO', 'AJO', 'AQO', 'AKO',
                   'KQS', 'KJS', 'KTS', 'K9S',
                   'QJS', 'QTS', 'Q9S',
                   'JTS', 'J9S', 'T9S', '98S', '87S']
            ),
            'BTN': to_range_dict(
                all_pairs_above('22')
                + suited_axs('2')
                + offsuit_ax('9')
                + ['K8S', 'K9S', 'KTS', 'KJS', 'KQS',
                   'KTO', 'KJO', 'KQO',
                   'Q8S', 'Q9S', 'QTS', 'QJS',
                   'QTO', 'QJO',
                   'J8S', 'J9S', 'JTS', 'JTO',
                   'T8S', 'T9S', '97S', '98S',
                   '86S', '87S', '75S', '76S', '65S', '54S']
            ),
            'SB': to_range_dict(
                all_pairs_above('33')
                + suited_axs('3')
                + ['AJO', 'AQO', 'AKO',
                   'KQS', 'KJS', 'KTS', 'K9S', 'KQO',
                   'QJS', 'QTS', 'Q9S',
                   'JTS', 'J9S', 'T9S', '98S', '87S']
            ),
            'BB': to_range_dict(
                all_pairs_above('22')
                + suited_axs('2')
                + offsuit_ax('2')
                + ['K5S', 'K6S', 'K7S', 'K8S', 'K9S', 'KTS', 'KJS', 'KQS',
                   'K9O', 'KTO', 'KJO', 'KQO',
                   'Q7S', 'Q8S', 'Q9S', 'QTS', 'QJS',
                   'QTO', 'QJO',
                   'J7S', 'J8S', 'J9S', 'JTS',
                   'JTO',
                   'T7S', 'T8S', 'T9S',
                   '96S', '97S', '98S',
                   '85S', '86S', '87S',
                   '75S', '76S', '64S', '65S', '54S']
            ),
        },
        '3b': {
            'LJ_3b': to_range_dict(['JJ', 'QQ', 'KK', 'AA', 'AKS', 'AKO', 'AQS']),
            'HJ_3b': to_range_dict(['JJ', 'QQ', 'KK', 'AA', 'AKS', 'AKO', 'AQS',
                                     'A5S']),
            'CO_3b': to_range_dict(['TT', 'JJ', 'QQ', 'KK', 'AA',
                                     'AKS', 'AKO', 'AQS', 'AQO',
                                     'A5S', 'A4S', 'KQS']),
            'BTN_3b': to_range_dict(['99', 'TT', 'JJ', 'QQ', 'KK', 'AA',
                                      'AKS', 'AKO', 'AQS', 'AQO', 'AJS',
                                      'A5S', 'A4S', 'A3S', 'KQS', 'KJS']),
            'SB_3b': to_range_dict(['99', 'TT', 'JJ', 'QQ', 'KK', 'AA',
                                     'AKS', 'AKO', 'AQS', 'AQO', 'AJS',
                                     'A5S', 'A4S', 'KQS', 'KJS']),
            'BB_3b': to_range_dict(['99', 'TT', 'JJ', 'QQ', 'KK', 'AA',
                                     'AKS', 'AKO', 'AQS', 'AQO', 'AJS', 'ATS',
                                     'A5S', 'A4S', 'A3S', 'KQS', 'KJS', 'KQO']),
        },
        '3b_call': {
            f'{pos}_call': to_range_dict(
                all_pairs_above('77')
                + ['AKS', 'AKO', 'AQS', 'AQO', 'AJS', 'KQS']
            )
            for pos in ['LJ', 'HJ', 'CO', 'BTN', 'SB', 'BB']
        },
        '4b': {
            f'{pos}_4b': to_range_dict(['QQ', 'KK', 'AA', 'AKS', 'AKO', 'A5S'])
            for pos in ['LJ', 'HJ', 'CO', 'BTN', 'SB', 'BB']
        },
        '4b_call': {
            f'{pos}_call': to_range_dict(['JJ', 'QQ', 'KK', 'AA', 'AKS', 'AKO'])
            for pos in ['LJ', 'HJ', 'CO', 'BTN', 'SB', 'BB']
        },
        'Allin': {
            'allin_base': to_range_dict(['QQ', 'KK', 'AA', 'AKS', 'AKO']),
        },
        'SB_DEF': {
            'SB_DEF_call': to_range_dict(
                all_pairs_above('22')
                + suited_axs('2')
                + offsuit_ax('7')
                + ['K6S', 'K7S', 'K8S', 'K9S', 'KTS', 'KJS', 'KQS',
                   'K9O', 'KTO', 'KJO', 'KQO',
                   'Q8S', 'Q9S', 'QTS', 'QJS', 'QJO',
                   'J8S', 'J9S', 'JTS', 'JTO',
                   'T8S', 'T9S', '97S', '98S',
                   '86S', '87S', '75S', '76S', '64S', '65S', '54S']
            ),
        },
    }


def build_maniac() -> Dict:
    """Extreme loose-aggressive — raises constantly, plays any two.
    Target: VPIP 60%, PFR 45% — needs MUCH wider ranges than TAG."""
    return {
        'rfi': {
            # Maniac opens 35-40% in EP, 70%+ in late position
            'LJ': to_range_dict(
                all_pairs_above('22')
                + suited_axs('2')
                + offsuit_ax('5')
                + ['K5S', 'K6S', 'K7S', 'K8S', 'K9S', 'KTS', 'KJS', 'KQS',
                   'K9O', 'KTO', 'KJO', 'KQO',
                   'Q7S', 'Q8S', 'Q9S', 'QTS', 'QJS',
                   'Q9O', 'QTO', 'QJO',
                   'J7S', 'J8S', 'J9S', 'JTS',
                   'J9O', 'JTO',
                   'T7S', 'T8S', 'T9S', 'T9O',
                   '96S', '97S', '98S',
                   '85S', '86S', '87S',
                   '74S', '75S', '76S',
                   '64S', '65S',
                   '53S', '54S', '43S']
            ),
            'HJ': to_range_dict(
                all_pairs_above('22')
                + suited_axs('2')
                + offsuit_ax('4')
                + ['K4S', 'K5S', 'K6S', 'K7S', 'K8S', 'K9S', 'KTS', 'KJS', 'KQS',
                   'K7O', 'K8O', 'K9O', 'KTO', 'KJO', 'KQO',
                   'Q6S', 'Q7S', 'Q8S', 'Q9S', 'QTS', 'QJS',
                   'Q8O', 'Q9O', 'QTO', 'QJO',
                   'J6S', 'J7S', 'J8S', 'J9S', 'JTS',
                   'J8O', 'J9O', 'JTO',
                   'T6S', 'T7S', 'T8S', 'T9S', 'T9O',
                   '95S', '96S', '97S', '98S', '98O',
                   '85S', '86S', '87S', '87O',
                   '74S', '75S', '76S', '76O',
                   '64S', '65S',
                   '53S', '54S', '43S', '32S']
            ),
            'CO': to_range_dict(
                all_pairs_above('22')
                + suited_axs('2')
                + offsuit_ax('2')
                + ['K2S', 'K3S', 'K4S', 'K5S', 'K6S', 'K7S', 'K8S', 'K9S',
                   'KTS', 'KJS', 'KQS',
                   'K2O', 'K3O', 'K4O', 'K5O', 'K6O', 'K7O', 'K8O', 'K9O',
                   'KTO', 'KJO', 'KQO',
                   'Q2S', 'Q3S', 'Q4S', 'Q5S', 'Q6S', 'Q7S', 'Q8S', 'Q9S',
                   'QTS', 'QJS',
                   'Q5O', 'Q6O', 'Q7O', 'Q8O', 'Q9O', 'QTO', 'QJO',
                   'J4S', 'J5S', 'J6S', 'J7S', 'J8S', 'J9S', 'JTS',
                   'J6O', 'J7O', 'J8O', 'J9O', 'JTO',
                   'T5S', 'T6S', 'T7S', 'T8S', 'T9S',
                   'T7O', 'T8O', 'T9O',
                   '94S', '95S', '96S', '97S', '98S',
                   '97O', '98O',
                   '84S', '85S', '86S', '87S',
                   '86O', '87O',
                   '73S', '74S', '75S', '76S',
                   '75O', '76O',
                   '63S', '64S', '65S',
                   '65O',
                   '52S', '53S', '54S', '54O',
                   '42S', '43S', '32S']
            ),
            # BTN: any two except absolute trash (~85% of hands)
            'BTN': to_range_dict(
                all_pairs_above('22')
                + suited_axs('2')
                + offsuit_ax('2')
                # All K
                + ['K2S', 'K3S', 'K4S', 'K5S', 'K6S', 'K7S', 'K8S', 'K9S',
                   'KTS', 'KJS', 'KQS',
                   'K2O', 'K3O', 'K4O', 'K5O', 'K6O', 'K7O', 'K8O', 'K9O',
                   'KTO', 'KJO', 'KQO']
                # All Q
                + ['Q2S', 'Q3S', 'Q4S', 'Q5S', 'Q6S', 'Q7S', 'Q8S', 'Q9S',
                   'QTS', 'QJS',
                   'Q3O', 'Q4O', 'Q5O', 'Q6O', 'Q7O', 'Q8O', 'Q9O', 'QTO', 'QJO']
                # All J
                + ['J2S', 'J3S', 'J4S', 'J5S', 'J6S', 'J7S', 'J8S', 'J9S', 'JTS',
                   'J4O', 'J5O', 'J6O', 'J7O', 'J8O', 'J9O', 'JTO']
                # All T
                + ['T2S', 'T3S', 'T4S', 'T5S', 'T6S', 'T7S', 'T8S', 'T9S',
                   'T6O', 'T7O', 'T8O', 'T9O']
                # 9x
                + ['92S', '93S', '94S', '95S', '96S', '97S', '98S',
                   '96O', '97O', '98O']
                # 8x
                + ['82S', '83S', '84S', '85S', '86S', '87S', '86O', '87O']
                # 7x
                + ['72S', '73S', '74S', '75S', '76S', '75O', '76O']
                # 6x
                + ['63S', '64S', '65S', '64O', '65O']
                # 5x
                + ['52S', '53S', '54S', '53O', '54O']
                # Low
                + ['42S', '43S', '32S', '43O']
            ),
            # SB: very wide (~65%)
            'SB': to_range_dict(
                all_pairs_above('22')
                + suited_axs('2')
                + offsuit_ax('2')
                + ['K2S', 'K3S', 'K4S', 'K5S', 'K6S', 'K7S', 'K8S', 'K9S',
                   'KTS', 'KJS', 'KQS',
                   'K4O', 'K5O', 'K6O', 'K7O', 'K8O', 'K9O', 'KTO', 'KJO', 'KQO',
                   'Q3S', 'Q4S', 'Q5S', 'Q6S', 'Q7S', 'Q8S', 'Q9S', 'QTS', 'QJS',
                   'Q7O', 'Q8O', 'Q9O', 'QTO', 'QJO',
                   'J4S', 'J5S', 'J6S', 'J7S', 'J8S', 'J9S', 'JTS',
                   'J7O', 'J8O', 'J9O', 'JTO',
                   'T5S', 'T6S', 'T7S', 'T8S', 'T9S',
                   'T7O', 'T8O', 'T9O',
                   '94S', '95S', '96S', '97S', '98S',
                   '97O', '98O',
                   '84S', '85S', '86S', '87S', '86O', '87O',
                   '73S', '74S', '75S', '76S', '76O',
                   '63S', '64S', '65S', '65O',
                   '53S', '54S', '43S', '32S']
            ),
        },
        'rfi_call': {
            'LJ': to_range_dict(
                all_pairs_above('22')
                + suited_axs('2')
                + offsuit_ax('9')
                + ['KJS', 'KQS', 'KJO', 'KQO',
                   'QJS', 'QTS', 'JTS', 'T9S', '98S']
            ),
            'HJ': to_range_dict(
                all_pairs_above('22')
                + suited_axs('2')
                + offsuit_ax('8')
                + ['KTS', 'KJS', 'KQS', 'KJO', 'KQO',
                   'QTS', 'QJS', 'QJO',
                   'JTS', 'T9S', '98S', '87S']
            ),
            'CO': to_range_dict(
                all_pairs_above('22')
                + suited_axs('2')
                + offsuit_ax('5')
                + ['K8S', 'K9S', 'KTS', 'KJS', 'KQS',
                   'KTO', 'KJO', 'KQO',
                   'Q8S', 'Q9S', 'QTS', 'QJS', 'QTO', 'QJO',
                   'J8S', 'J9S', 'JTS', 'JTO',
                   'T8S', 'T9S', '97S', '98S',
                   '87S', '76S', '65S']
            ),
            'BTN': to_range_dict(
                all_pairs_above('22')
                + suited_axs('2')
                + offsuit_ax('2')
                + ['K2S', 'K3S', 'K4S', 'K5S', 'K6S', 'K7S', 'K8S', 'K9S',
                   'KTS', 'KJS', 'KQS',
                   'K7O', 'K8O', 'K9O', 'KTO', 'KJO', 'KQO',
                   'Q5S', 'Q6S', 'Q7S', 'Q8S', 'Q9S', 'QTS', 'QJS',
                   'Q9O', 'QTO', 'QJO',
                   'J6S', 'J7S', 'J8S', 'J9S', 'JTS',
                   'J8O', 'J9O', 'JTO',
                   'T6S', 'T7S', 'T8S', 'T9S', 'T9O',
                   '95S', '96S', '97S', '98S',
                   '85S', '86S', '87S',
                   '74S', '75S', '76S',
                   '64S', '65S',
                   '53S', '54S', '43S']
            ),
            'SB': to_range_dict(
                all_pairs_above('22')
                + suited_axs('2')
                + offsuit_ax('2')
                + ['K5S', 'K6S', 'K7S', 'K8S', 'K9S',
                   'KTS', 'KJS', 'KQS',
                   'K8O', 'K9O', 'KTO', 'KJO', 'KQO',
                   'Q7S', 'Q8S', 'Q9S', 'QTS', 'QJS',
                   'Q9O', 'QTO', 'QJO',
                   'J7S', 'J8S', 'J9S', 'JTS', 'JTO',
                   'T8S', 'T9S', 'T9O',
                   '96S', '97S', '98S',
                   '86S', '87S', '75S', '76S', '65S', '54S']
            ),
            'BB': to_range_dict(
                all_pairs_above('22')
                + suited_axs('2')
                + offsuit_ax('2')
                + ['K2S', 'K3S', 'K4S', 'K5S', 'K6S', 'K7S', 'K8S', 'K9S',
                   'KTS', 'KJS', 'KQS',
                   'K5O', 'K6O', 'K7O', 'K8O', 'K9O', 'KTO', 'KJO', 'KQO',
                   'Q4S', 'Q5S', 'Q6S', 'Q7S', 'Q8S', 'Q9S', 'QTS', 'QJS',
                   'Q7O', 'Q8O', 'Q9O', 'QTO', 'QJO',
                   'J5S', 'J6S', 'J7S', 'J8S', 'J9S', 'JTS',
                   'J7O', 'J8O', 'J9O', 'JTO',
                   'T5S', 'T6S', 'T7S', 'T8S', 'T9S',
                   'T8O', 'T9O',
                   '94S', '95S', '96S', '97S', '98S',
                   '97O', '98O',
                   '84S', '85S', '86S', '87S',
                   '87O',
                   '74S', '75S', '76S',
                   '64S', '65S',
                   '53S', '54S', '43S', '32S']
            ),
        },
        '3b': {
            f'{pos}_3b': to_range_dict(
                all_pairs_above('77')
                + suited_axs('2')
                + ['AKO', 'AQO', 'AJO',
                   'KQS', 'KJS', 'KQO', 'QJS']
            )
            for pos in ['LJ', 'HJ', 'CO', 'BTN', 'SB', 'BB']
        },
        '3b_call': {
            f'{pos}_call': to_range_dict(
                all_pairs_above('22')
                + suited_axs('2')
                + ['AKO', 'AQO', 'AJO', 'ATO',
                   'KQS', 'KJS', 'KTS', 'QJS', 'QTS', 'JTS']
            )
            for pos in ['LJ', 'HJ', 'CO', 'BTN', 'SB', 'BB']
        },
        '4b': {
            f'{pos}_4b': to_range_dict(
                ['TT', 'JJ', 'QQ', 'KK', 'AA',
                 'AKS', 'AKO', 'AQS', 'AQO',
                 'A5S', 'A4S', 'KQS']
            )
            for pos in ['LJ', 'HJ', 'CO', 'BTN', 'SB', 'BB']
        },
        '4b_call': {
            f'{pos}_call': to_range_dict(
                ['TT', 'JJ', 'QQ', 'KK', 'AA', 'AKS', 'AKO', 'AQS']
            )
            for pos in ['LJ', 'HJ', 'CO', 'BTN', 'SB', 'BB']
        },
        'Allin': {
            'allin_base': to_range_dict(
                ['TT', 'JJ', 'QQ', 'KK', 'AA', 'AKS', 'AKO', 'AQS']
            ),
        },
        'SB_DEF': {
            'SB_DEF_call': to_range_dict(
                all_pairs_above('22')
                + suited_axs('2')
                + offsuit_ax('2')
                + ['K2S', 'K3S', 'K4S', 'K5S', 'K6S', 'K7S', 'K8S', 'K9S',
                   'KTS', 'KJS', 'KQS',
                   'K6O', 'K7O', 'K8O', 'K9O', 'KTO', 'KJO', 'KQO',
                   'Q5S', 'Q6S', 'Q7S', 'Q8S', 'Q9S', 'QTS', 'QJS',
                   'Q9O', 'QTO', 'QJO',
                   'J6S', 'J7S', 'J8S', 'J9S', 'JTS',
                   'J8O', 'J9O', 'JTO',
                   'T6S', 'T7S', 'T8S', 'T9S',
                   'T9O',
                   '95S', '96S', '97S', '98S',
                   '85S', '86S', '87S',
                   '74S', '75S', '76S',
                   '64S', '65S',
                   '53S', '54S', '43S', '32S']
            ),
        },
    }


# ── Main ─────────────────────────────────────────────────────────────────────

ARCHETYPES = {
    'nit': build_nit,
    'weak_tight': build_weak_tight,
    'calling_station': build_calling_station,
    'lag': build_lag,
    'maniac': build_maniac,
}


def count_combos_in_hand_set(hands: Set[str]) -> int:
    """Count actual 2-card combos represented by a set of hand strings."""
    total = 0
    for h in hands:
        if len(h) == 2:  # pair like 'AA'
            total += 6
        elif h.endswith('S'):  # suited
            total += 4
        elif h.endswith('O'):  # offsuit
            total += 12
    return total


def summarize_range(range_dict: Dict[str, str]) -> tuple:
    """Return (num_hand_types, num_combos, pct_of_deck)."""
    hand_types = set(range_dict.keys())
    combos = count_combos_in_hand_set(hand_types)
    return len(hand_types), combos, 100.0 * combos / 1326


def print_summary(name: str, ranges: Dict) -> None:
    """Print per-position combo counts for an archetype."""
    print(f'\n── {name.upper()} ──')
    print(f'  {"Section":<12} {"Position":<15} {"Types":>6} {"Combos":>8} {"% of deck":>10}')
    for section in ['rfi', 'rfi_call', '3b', '3b_call', '4b', '4b_call', 'SB_DEF', 'Allin']:
        if section not in ranges:
            continue
        for key, rng in ranges[section].items():
            types, combos, pct = summarize_range(rng)
            print(f'  {section:<12} {key:<15} {types:>6} {combos:>8} {pct:>9.2f}%')


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, builder in ARCHETYPES.items():
        ranges = builder()
        output_path = OUTPUT_DIR / f'{name}.json'
        with open(output_path, 'w') as f:
            json.dump(ranges, f, indent=2)
        print(f'Wrote {output_path}')
        print_summary(name, ranges)

    print(f'\nDone. {len(ARCHETYPES)} range files written to {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()

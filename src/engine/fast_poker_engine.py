"""
Fast No-Limit Hold'em engine — drop-in replacement for PokerEngine.

Uses the Rust-backed `pokers` library (pip install pokers) for game logic.
~30x faster than PettingZoo/RLCard.  Immutable state model simplifies
MCCFR branching: no RNG save/restore needed — just save the initial state.
"""

import os
import sys
import numpy as np

try:
    import pokers as pkrs
except ImportError:
    raise ImportError(
        "pokers library required for FastPokerEngine. Install with: pip install pokers"
    )


# ── Suppress C-level stdout from pokers Rust library permanently ─────────────
# The Rust library prints "Winner id" / "Ranks" on showdown via C stdout (fd 1).
# We redirect fd 1 to /dev/null once at import time, then re-point Python's
# sys.stdout to a saved copy of the original fd so print() still works.

_saved_stdout_fd = os.dup(1)                       # save original fd 1
os.dup2(os.open(os.devnull, os.O_WRONLY), 1)       # fd 1 → /dev/null
sys.stdout = os.fdopen(_saved_stdout_fd, 'w', 1)   # Python print → saved fd


def _suppress_stdout():
    """No-op — stdout is permanently redirected at import time."""
    return None


def _restore_stdout(_saved):
    """No-op — stdout is permanently redirected at import time."""
    pass


# ── Card conversion helpers ──────────────────────────────────────────────────

_RANK_MAP = {
    'R2': '2', 'R3': '3', 'R4': '4', 'R5': '5', 'R6': '6',
    'R7': '7', 'R8': '8', 'R9': '9', 'RT': 'T', 'RJ': 'J',
    'RQ': 'Q', 'RK': 'K', 'RA': 'A',
}

_SUIT_MAP = {
    'Spades': 'S', 'Hearts': 'H', 'Diamonds': 'D', 'Clubs': 'C',
}


def _pkrs_card_rank(card) -> str:
    """Convert pokers CardRank to single-char string ('2'-'9','T','J','Q','K','A')."""
    raw = str(card.rank).split('.')[-1]          # e.g. 'RQ'
    return _RANK_MAP.get(raw, raw)


def _pkrs_card_suit(card) -> str:
    """Convert pokers CardSuit to single-char string ('S','H','D','C')."""
    raw = str(card.suit).split('.')[-1]          # e.g. 'Spades'
    return _SUIT_MAP.get(raw, raw[0])


# ── Lightweight card/player/round objects for EnhancedPokerEngine compat ─────

class _Card:
    """Matches RLCard Card interface (rank, suit strings)."""
    __slots__ = ('rank', 'suit')

    def __init__(self, rank: str, suit: str):
        self.rank = rank
        self.suit = suit

    def __repr__(self):
        return f'{self.rank}{self.suit}'


class _PlayerStatus:
    ALIVE = 0
    FOLDED = 1
    ALLIN = 2


class _Player:
    __slots__ = ('hand', 'remained_chips', 'in_chips', 'status')

    def __init__(self):
        self.hand = []
        self.remained_chips = 0
        self.in_chips = 0
        self.status = _PlayerStatus.ALIVE


class _Round:
    __slots__ = ('raised', 'not_raise_num')

    def __init__(self, num_players):
        self.raised = [0] * num_players
        self.not_raise_num = 0


class _FakeRNG:
    """Emulates np.random.RandomState.get_state / set_state for training code.

    The 'state' is actually the initial pokers State object (immutable).
    Saving and restoring it enables deterministic game replay without RNG.
    """

    def __init__(self, engine: 'FastPokerEngine'):
        self._engine = engine

    def get_state(self):
        return self._engine._initial_state

    def set_state(self, state):
        self._engine._pending_state = state


class _GameProxy:
    """Facade compatible with RLCard game object for training code & features."""

    def __init__(self, engine: 'FastPokerEngine'):
        self._engine = engine
        self.np_random = _FakeRNG(engine)

    @property
    def dealer_id(self):
        return self._engine._button

    @dealer_id.setter
    def dealer_id(self, value):
        pass  # no-op: button is baked into the pokers State

    @property
    def big_blind(self):
        return 2

    @property
    def stage(self):
        return self._engine._stage_obj

    @property
    def players(self):
        return self._engine._build_players()

    @property
    def public_cards(self):
        return self._engine._build_public_cards()

    @property
    def round(self):
        return self._engine._build_round()


# ── Stage enum compatible with RLCard ────────────────────────────────────────

class _Stage:
    """Matches rlcard Stage enum for == comparisons in feature extraction."""
    pass


class _StagePreflop(_Stage):
    pass


class _StageFlop(_Stage):
    pass


class _StageTurn(_Stage):
    pass


class _StageRiver(_Stage):
    pass


# Singletons for identity comparison
STAGE_PREFLOP = _StagePreflop()
STAGE_FLOP = _StageFlop()
STAGE_TURN = _StageTurn()
STAGE_RIVER = _StageRiver()

def _map_pkrs_stage(stage):
    """Map pokers Stage to our stage singleton (string comparison fallback)."""
    s = str(stage)
    if 'Preflop' in s:
        return STAGE_PREFLOP
    if 'Flop' in s:
        return STAGE_FLOP
    if 'Turn' in s:
        return STAGE_TURN
    if 'River' in s:
        return STAGE_RIVER
    return STAGE_PREFLOP


# ── Main engine ─────────────────────────────────────────────────────────────

class FastPokerEngine:
    """
    Fast NLHE engine wrapping the Rust-backed `pokers` library.
    Same public API as PokerEngine for drop-in replacement.

    Action space (matching PettingZoo texas_holdem_no_limit_v6):
        0 = Fold
        1 = Check / Call
        2 = Raise half pot
        3 = Raise pot
        4 = All-in
    """

    def __init__(self, num_players: int = 6):
        self.num_players = num_players
        self.agents = [f'player_{i}' for i in range(num_players)]
        self.current_agent = self.agents[0]

        self._state = None             # current pokers State
        self._initial_state = None     # saved for replay (immutable)
        self._pending_state = None     # set via _get_game().np_random.set_state()
        self._button = 0
        self._done = False
        self._rewards = {a: 0.0 for a in self.agents}
        self._proxy = _GameProxy(self)
        self._action_history = []      # for tracking raise counts etc.

        self.reset()

    # ── Public API (matching PokerEngine) ────────────────────────────────────

    def reset(self):
        if self._pending_state is not None:
            # Replay mode: use saved initial state
            self._state = self._pending_state
            self._initial_state = self._pending_state
            self._pending_state = None
        else:
            # New game: random seed and button
            seed = int(np.random.randint(0, 2**31))
            self._button = int(np.random.randint(0, self.num_players))
            self._state = pkrs.State.from_seed(
                n_players=self.num_players,
                button=self._button,
                sb=1, bb=2, stake=100,
                seed=seed,
            )
            self._initial_state = self._state

        self._button = self._state.button
        self._done = False
        self._rewards = {a: 0.0 for a in self.agents}
        self._action_history = []

        if self._state.legal_actions:
            self.current_agent = self.agents[self._state.current_player]
        else:
            self._finish()

        return {
            'observation': self.get_observation(),
            'legal_actions': self.get_legal_actions(),
            'current_player': self.current_agent,
        }

    def step(self, action: int):
        if self._done:
            return self._make_result()

        pkrs_action = self._map_action(action)
        saved = _suppress_stdout()
        try:
            self._state = self._state.apply_action(pkrs_action)
        finally:
            _restore_stdout(saved)
        self._action_history.append(action)

        if not self._state.legal_actions:
            self._finish()
        else:
            self.current_agent = self.agents[self._state.current_player]

        return self._make_result()

    def get_legal_actions(self):
        if self._done:
            return []
        return self._compute_legal_actions()

    def get_observation(self):
        # Minimal obs — overridden by FastEnhancedPokerEngine
        return np.zeros(54, dtype=np.float32)

    def get_rewards(self):
        if not self._done:
            return {a: 0.0 for a in self.agents}
        return dict(self._rewards)

    def is_done(self):
        return self._done

    def get_state(self):
        return {
            'observation': self.get_observation(),
            'current_player': self.current_agent,
            'legal_actions': self.get_legal_actions(),
            'agents': list(self.agents),
        }

    def __deepcopy__(self, memo):
        """Custom deepcopy — pokers State objects are immutable, so shallow-copy them."""
        import copy as _copy
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k in ('_state', '_initial_state', '_pending_state'):
                # Immutable Rust objects — safe to share references
                setattr(new, k, v)
            elif k == '_proxy':
                # Rebuild proxy pointing to new engine
                setattr(new, k, _GameProxy(new))
            else:
                setattr(new, k, _copy.deepcopy(v, memo))
        return new

    def close(self):
        pass  # nothing to clean up

    def get_action_space_size(self):
        return 4  # fold, call, raise_half, raise_pot (no all-in)

    def get_observation_space_size(self):
        return 54  # overridden by enhanced engine

    def _get_game(self):
        return self._proxy

    def get_player_cards(self, player):
        if player not in self.agents:
            return []
        idx = self.agents.index(player)
        ps = self._state.players_state[idx]
        return [f'{_pkrs_card_rank(c)}{_pkrs_card_suit(c)}' for c in ps.hand]

    def get_community_cards(self):
        return [f'{_pkrs_card_rank(c)}{_pkrs_card_suit(c)}'
                for c in self._state.public_cards]

    def get_pot_size(self):
        return int(self._state.pot)

    def get_player_chips(self, player):
        if player not in self.agents:
            return 0
        idx = self.agents.index(player)
        return int(self._state.players_state[idx].stake)

    def get_stage(self):
        return str(self._state.stage).split('.')[-1].capitalize()

    def _card_to_string(self, card):
        return f"{card.rank}{card.suit}"

    # ── Internal helpers ─────────────────────────────────────────────────────

    @property
    def _stage_obj(self):
        return _map_pkrs_stage(self._state.stage)

    def _finish(self):
        """Mark hand as done and collect rewards."""
        self._done = True
        saved = _suppress_stdout()
        try:
            for i, ps in enumerate(self._state.players_state):
                self._rewards[self.agents[i]] = float(ps.reward)
        finally:
            _restore_stdout(saved)

    def _map_action(self, action: int):
        """Map our 5-action space to a pokers Action."""
        la = self._state.legal_actions

        if action == 0:  # Fold
            return pkrs.Action(pkrs.ActionEnum.Fold, 0)

        if action == 1:  # Check / Call
            if pkrs.ActionEnum.Check in la:
                return pkrs.Action(pkrs.ActionEnum.Check, 0)
            return pkrs.Action(pkrs.ActionEnum.Call, 0)

        # For raises, compute the increment amount
        pid = self._state.current_player
        ps = self._state.players_state[pid]
        my_bet = ps.bet_chips
        max_bet = max(p.bet_chips for p in self._state.players_state)
        call_cost = max_bet - my_bet
        pot = self._state.pot
        stake = ps.stake
        max_raise = stake - call_cost  # max affordable raise increment
        min_raise = self._state.min_bet

        if action == 2:  # Raise half pot
            # RLCard-style: int(pot / 2)
            amount = int(pot / 2)
            amount = max(amount, min_raise)
            amount = min(amount, max_raise)
            return pkrs.Action(pkrs.ActionEnum.Raise, amount)

        if action == 3:  # Raise pot
            amount = int(pot)
            amount = max(amount, min_raise)
            amount = min(amount, max_raise)
            return pkrs.Action(pkrs.ActionEnum.Raise, amount)

        if action == 4:  # All-in
            amount = max_raise
            if amount < min_raise:
                # Can't raise enough — just call (forced all-in)
                if pkrs.ActionEnum.Check in la:
                    return pkrs.Action(pkrs.ActionEnum.Check, 0)
                return pkrs.Action(pkrs.ActionEnum.Call, 0)
            return pkrs.Action(pkrs.ActionEnum.Raise, amount)

        # Fallback
        return pkrs.Action(pkrs.ActionEnum.Fold, 0)

    def _compute_legal_actions(self):
        """Compute which of our 5 discrete actions are legal."""
        la = self._state.legal_actions
        if not la:
            return []

        legal = []

        # 0: Fold — always legal if Fold in legal_actions
        if pkrs.ActionEnum.Fold in la:
            legal.append(0)

        # 1: Check/Call — always legal
        if pkrs.ActionEnum.Check in la or pkrs.ActionEnum.Call in la:
            legal.append(1)

        # Raise-related
        if pkrs.ActionEnum.Raise not in la:
            return legal

        pid = self._state.current_player
        ps = self._state.players_state[pid]
        my_bet = ps.bet_chips
        max_bet = max(p.bet_chips for p in self._state.players_state)
        call_cost = max_bet - my_bet
        pot = self._state.pot
        stake = ps.stake
        max_raise = stake - call_cost
        min_raise = self._state.min_bet

        if max_raise < min_raise:
            # Can't afford any raise — only fold/call
            return legal

        # 2: Raise half pot
        half_pot = int(pot / 2)
        half_pot_clamped = max(half_pot, min_raise)
        if half_pot_clamped <= max_raise and half_pot_clamped > 0:
            legal.append(2)

        # 3: Raise pot
        pot_raise = int(pot)
        pot_raise_clamped = max(pot_raise, min_raise)
        if pot_raise_clamped <= max_raise:
            legal.append(3)

        # 4: All-in — DISABLED for faster convergence (4-action space)
        # With only fold/call/raise½/raisePot, the model converges faster
        # because there's no extreme action to exploit in self-play.
        # if max_raise >= min_raise:
        #     legal.append(4)

        return legal

    def _build_players(self):
        """Build list of _Player objects compatible with EnhancedPokerEngine."""
        players = []
        for i in range(self.num_players):
            ps = self._state.players_state[i]
            p = _Player()
            p.hand = [_Card(_pkrs_card_rank(c), _pkrs_card_suit(c)) for c in ps.hand]
            p.remained_chips = int(ps.stake)
            p.in_chips = int(ps.pot_chips + ps.bet_chips)
            if not ps.active:
                # Determine if folded or all-in
                if ps.stake == 0 and (ps.pot_chips + ps.bet_chips) > 0:
                    p.status = _PlayerStatus.ALLIN
                else:
                    p.status = _PlayerStatus.FOLDED
            else:
                if ps.stake == 0:
                    p.status = _PlayerStatus.ALLIN
                else:
                    p.status = _PlayerStatus.ALIVE
            players.append(p)
        return players

    def _build_public_cards(self):
        """Build list of _Card objects for community cards."""
        return [_Card(_pkrs_card_rank(c), _pkrs_card_suit(c))
                for c in self._state.public_cards]

    def _build_round(self):
        """Build _Round object compatible with EnhancedPokerEngine."""
        r = _Round(self.num_players)
        # raised = per-player current-round bet amounts
        for i in range(self.num_players):
            r.raised[i] = int(self._state.players_state[i].bet_chips)
        # not_raise_num: approximate from active non-raising players
        # This is used in feature extraction — compute as players who checked/called
        active = sum(1 for ps in self._state.players_state if ps.active)
        r.not_raise_num = max(0, active - 1)  # approximate
        return r

    def _make_result(self):
        return {
            'observation': None,
            'reward': self._rewards.get(self.current_agent, 0.0),
            'done': self._done,
            'legal_actions': self.get_legal_actions() if not self._done else [],
            'current_player': self.current_agent,
        }

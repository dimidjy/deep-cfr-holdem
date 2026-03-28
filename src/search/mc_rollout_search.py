"""
Monte Carlo rollout search for play-time decision refinement.

Works with ANY model (supervised or Deep CFR) — only needs get_strategy().

At each AI decision point:
  1. For each legal action, run N rollout games to completion
  2. All players (including ourselves) use the policy network for subsequent decisions
  3. Average the reward across rollouts for each action
  4. Return a strategy that concentrates probability on the highest-EV action,
     blended with the raw network output (to preserve mixed-strategy play)

This is simpler than DepthLimitedSearch (no CFR iterations needed) but still
significantly improves play by looking ahead at consequences of each action.
"""

import copy
import numpy as np
from typing import List, Optional


class MCRolloutSearch:

    def __init__(self, model, num_rollouts: int = 50,
                 blend: float = 0.3, temperature: float = 1.0,
                 obs_pad: int = 0):
        """
        Args:
            model:        any model with get_strategy(obs, player_idx, legal_actions)
            num_rollouts: rollouts per legal action (total = num_rollouts * len(legal_actions))
            blend:        0.0 = pure network, 1.0 = pure rollout EV (default 0.3)
            temperature:  softmax temperature on normalized EVs (default 1.0)
            obs_pad:      extra dims to zero-pad onto raw engine obs during rollouts
                          (e.g. 8 for behavioral stats models — rollout engines don't
                          have behavioral tracking, so we pad with zeros)
        """
        self.model = model
        self.num_rollouts = num_rollouts
        self.blend = max(0.0, min(1.0, blend))
        self.temperature = max(0.01, temperature)
        self.obs_pad = obs_pad

    def _raw_obs(self, engine):
        """Get the real engine observation (strip any monkey-patches) + zero-pad."""
        # Use the class method to bypass any instance-level monkey-patches
        obs = engine.__class__.get_observation(engine)
        if self.obs_pad > 0:
            obs = np.concatenate([obs, np.zeros(self.obs_pad, dtype=np.float32)])
        return obs

    def search(self, engine, player_idx: int,
               legal_actions: List[int]) -> np.ndarray:
        """
        Run MC rollout search and return a refined strategy.

        Args:
            engine:        live game engine (NOT modified — deepcopied per rollout)
            player_idx:    index of the player we're searching for
            legal_actions: legal actions at the current node

        Returns:
            strategy: np.ndarray of shape (action_size,) — probability over all actions
        """
        action_size = self.model.action_size
        if len(legal_actions) <= 1:
            strategy = np.zeros(action_size, dtype=np.float32)
            if legal_actions:
                strategy[legal_actions[0]] = 1.0
            return strategy

        # Get raw network strategy for blending
        # Use engine.get_observation() here — may include behavioral stats from GUI patch
        obs = engine.get_observation()
        net_strategy = self.model.get_strategy(obs, player_idx, legal_actions)

        # Run rollouts for each action
        action_values = {}
        agent_name = engine.agents[player_idx]

        for action in legal_actions:
            total_reward = 0.0
            for _ in range(self.num_rollouts):
                engine_copy = copy.deepcopy(engine)
                # Remove any monkey-patched methods from the copy
                engine_copy.__dict__.pop('get_observation', None)
                engine_copy.__dict__.pop('step', None)
                engine_copy.step(action)
                reward = self._rollout(engine_copy, agent_name)
                total_reward += reward
            action_values[action] = total_reward / self.num_rollouts

        # Convert rollout EVs to a probability distribution
        rollout_strategy = self._ev_to_strategy(action_values, legal_actions, action_size)

        # Blend network + rollout
        final = (1.0 - self.blend) * net_strategy + self.blend * rollout_strategy

        # Re-normalize over legal actions
        legal_mask = np.zeros(action_size, dtype=np.float32)
        legal_mask[legal_actions] = 1.0
        final *= legal_mask
        total = final.sum()
        if total > 0:
            final /= total
        else:
            final[legal_actions] = 1.0 / len(legal_actions)

        return final

    def _rollout(self, engine, target_agent: str) -> float:
        """Play game to completion using the policy network for all decisions."""
        max_steps = 50  # safety limit
        for _ in range(max_steps):
            if engine.is_done():
                break

            current = engine.current_agent
            if current not in engine.agents:
                break

            current_idx = engine.agents.index(current)
            la = engine.get_legal_actions()
            if not la:
                break

            obs = self._raw_obs(engine)
            strategy = self.model.get_strategy(obs, current_idx, la)

            # Sample action from strategy
            probs = strategy[la]
            total = probs.sum()
            if total > 0:
                probs /= total
            else:
                probs = np.ones(len(la), dtype=np.float32) / len(la)
            action = np.random.choice(la, p=probs)
            engine.step(action)

        rewards = engine.get_rewards()
        return float(rewards.get(target_agent, 0.0))

    def _ev_to_strategy(self, action_values: dict, legal_actions: List[int],
                        action_size: int) -> np.ndarray:
        """Convert action EVs to a strategy via softmax with temperature.

        EVs are normalized to zero-mean unit-variance before softmax so that
        the temperature parameter has consistent meaning regardless of the
        chip scale (rewards can range from -200 to +200).
        """
        strategy = np.zeros(action_size, dtype=np.float32)

        evs = np.array([action_values[a] for a in legal_actions], dtype=np.float64)

        # Normalize EVs so temperature is scale-independent
        ev_std = evs.std()
        if ev_std > 1e-8:
            evs_norm = (evs - evs.mean()) / ev_std
        else:
            # All EVs nearly equal — uniform over legal actions
            for a in legal_actions:
                strategy[a] = 1.0 / len(legal_actions)
            return strategy

        # Softmax with temperature on normalized EVs
        evs_scaled = evs_norm / max(self.temperature, 0.01)
        evs_scaled -= evs_scaled.max()  # numerical stability
        exp_evs = np.exp(evs_scaled)
        probs = exp_evs / exp_evs.sum()

        for i, a in enumerate(legal_actions):
            strategy[a] = probs[i]

        return strategy

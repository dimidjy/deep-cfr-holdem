"""
Depth-limited CFR search for play-time strategy refinement.

At each AI decision point, builds a small game subtree (up to `depth` plies),
runs several CFR iterations within it using the trained networks as leaf
evaluators, and returns a refined strategy that is stronger than the raw
network output alone.

Algorithm:
  - Traverser nodes (our player): enumerate ALL legal actions, compute regrets.
  - Opponent nodes: sample ONE action from the strategy network (external sampling).
  - Leaf / depth-limit nodes: value estimated by the advantage network.
  - Multiple CFR iterations accumulate regrets; the average strategy is returned.
"""

import copy
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple


class DepthLimitedSearch:

    def __init__(self, deep_cfr, num_rollouts: int = 10,
                 cfr_iters: int = 50, max_depth: int = 2):
        """
        Args:
            deep_cfr:      trained DeepCFR instance (has advantage/strategy networks)
            num_rollouts:  rollouts per opponent sample to reduce chance variance
            cfr_iters:     CFR iterations within the subtree
            max_depth:     max traverser decision plies to expand
        """
        self.deep_cfr = deep_cfr
        self.num_rollouts = num_rollouts
        self.cfr_iters = cfr_iters
        self.max_depth = max_depth

    def search(self, engine, player_idx: int,
               legal_actions: List[int]) -> np.ndarray:
        """
        Run depth-limited CFR search and return a refined strategy.

        Args:
            engine:        live game engine (will NOT be modified — deepcopied internally)
            player_idx:    index of the player we're searching for
            legal_actions: legal actions at the current node

        Returns:
            strategy: np.ndarray of shape (action_size,) — probability over all actions
        """
        action_size = self.deep_cfr.action_size
        if len(legal_actions) <= 1:
            strategy = np.zeros(action_size, dtype=np.float32)
            if legal_actions:
                strategy[legal_actions[0]] = 1.0
            return strategy

        # Accumulated regrets and strategy sums across CFR iterations
        cumulative_regrets = np.zeros(action_size, dtype=np.float64)
        strategy_sum = np.zeros(action_size, dtype=np.float64)

        for _ in range(self.cfr_iters):
            # Current strategy from regret matching
            current_strategy = self._regret_match(cumulative_regrets, legal_actions)
            strategy_sum += current_strategy

            # Compute value of each action via rollouts
            action_values = np.zeros(action_size, dtype=np.float64)
            for action in legal_actions:
                val_sum = 0.0
                for _ in range(self.num_rollouts):
                    engine_copy = copy.deepcopy(engine)
                    engine_copy.step(action)
                    val_sum += self._traverse(
                        engine_copy, player_idx,
                        depth=self.max_depth - 1
                    )
                action_values[action] = val_sum / self.num_rollouts

            # Node value under current strategy
            node_value = sum(
                current_strategy[a] * action_values[a] for a in legal_actions
            )

            # Update regrets (CFR+: floor at 0)
            for a in legal_actions:
                cumulative_regrets[a] = max(
                    cumulative_regrets[a] + action_values[a] - node_value, 0.0
                )

        # Average strategy across all iterations
        total = strategy_sum.sum()
        if total > 0:
            avg_strategy = (strategy_sum / total).astype(np.float32)
        else:
            avg_strategy = np.zeros(action_size, dtype=np.float32)
            avg_strategy[legal_actions] = 1.0 / len(legal_actions)

        return avg_strategy

    def _traverse(self, engine, traverser_idx: int, depth: int) -> float:
        """
        Recursive traversal within the subtree.

        At terminal states or depth=0, returns the estimated value.
        At traverser nodes within remaining depth, enumerates actions.
        At opponent nodes, samples one action from the strategy network.
        """
        if engine.is_done():
            rewards = engine.get_rewards()
            agent_name = engine.agents[traverser_idx]
            return float(rewards.get(agent_name, 0.0))

        current_agent = engine.current_agent
        if current_agent not in engine.agents:
            return 0.0
        current_idx = engine.agents.index(current_agent)
        legal_actions = engine.get_legal_actions()

        if not legal_actions:
            return 0.0

        obs = engine.get_observation()

        # Leaf node: estimate value from advantage network
        if depth <= 0:
            return self._evaluate_leaf(obs, current_idx, traverser_idx, legal_actions)

        if current_idx == traverser_idx:
            # Traverser node: enumerate all actions, pick best weighted value
            # Use strategy network to weight (approximates the average strategy)
            strategy = self._get_network_strategy(obs, current_idx, legal_actions)
            node_value = 0.0
            for action in legal_actions:
                engine_copy = copy.deepcopy(engine)
                engine_copy.step(action)
                child_value = self._traverse(engine_copy, traverser_idx, depth - 1)
                node_value += strategy[action] * child_value
            return node_value
        else:
            # Opponent node: sample one action from strategy network
            strategy = self._get_network_strategy(obs, current_idx, legal_actions)
            legal_probs = strategy[legal_actions]
            total = legal_probs.sum()
            if total > 0:
                legal_probs /= total
            else:
                legal_probs = np.ones(len(legal_actions)) / len(legal_actions)
            action = np.random.choice(legal_actions, p=legal_probs)
            engine.step(action)
            return self._traverse(engine, traverser_idx, depth)

    def _evaluate_leaf(self, obs: np.ndarray, current_idx: int,
                       traverser_idx: int, legal_actions: List[int]) -> float:
        """
        Estimate node value using the advantage network.
        Returns the expected value under the current strategy.
        """
        advantages = self.deep_cfr.compute_counterfactual_values(
            obs, current_idx, legal_actions
        )
        strategy = self._get_network_strategy(obs, current_idx, legal_actions)

        # Value = sum of strategy[a] * advantage[a] for legal actions
        value = sum(strategy[a] * advantages[a] for a in legal_actions)
        return float(value)

    def _get_network_strategy(self, obs: np.ndarray, player_idx: int,
                              legal_actions: List[int]) -> np.ndarray:
        """Get strategy from the strategy network."""
        return self.deep_cfr.get_strategy(obs, player_idx, legal_actions)

    def _regret_match(self, cumulative_regrets: np.ndarray,
                      legal_actions: List[int]) -> np.ndarray:
        """Standard regret matching over legal actions."""
        action_size = len(cumulative_regrets)
        strategy = np.zeros(action_size, dtype=np.float64)
        positive = np.maximum(cumulative_regrets, 0)
        total = sum(positive[a] for a in legal_actions)
        if total > 0:
            for a in legal_actions:
                strategy[a] = positive[a] / total
        else:
            for a in legal_actions:
                strategy[a] = 1.0 / len(legal_actions)
        return strategy

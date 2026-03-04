import numpy as np
from typing import Dict, List, Optional
from src.engine import PokerEngine
from src.training.deep_cfr import DeepCFR
import json
from pathlib import Path


class CFRTrainer:

    def __init__(self,
                 engine: PokerEngine,
                 deep_cfr: DeepCFR,
                 config: Optional[Dict] = None):

        self.engine = engine
        self.deep_cfr = deep_cfr
        self.config = config or {}

        self.iteration = 0

    def run_traversal(self, player_idx: int) -> float:
        self.engine.reset()

        episode_data = []
        total_reward = 0

        while not self.engine.is_done():
            current_agent = self.engine.current_agent
            agent_idx = self.engine.agents.index(current_agent)

            observation = self.engine.get_observation()
            legal_actions = self.engine.get_legal_actions()

            if not legal_actions:
                break

            if agent_idx == player_idx:
                advantages = self.deep_cfr.compute_counterfactual_values(
                    observation, agent_idx, legal_actions
                )
                strategy = self.deep_cfr.get_regret_matching_strategy(advantages, legal_actions)
            else:
                advantages = None
                strategy = self.deep_cfr.get_strategy(observation, agent_idx, legal_actions)

            legal_probs = strategy[legal_actions]
            legal_probs = legal_probs / legal_probs.sum()
            action = np.random.choice(legal_actions, p=legal_probs)

            episode_data.append({
                'player': agent_idx,
                'observation': observation.copy(),
                'action': action,
                'strategy': strategy.copy(),
                'legal_actions': list(legal_actions),
                'advantages': advantages,
            })

            result = self.engine.step(action)

            if agent_idx == player_idx:
                total_reward += result['reward']

        self._update_regrets(episode_data, player_idx, total_reward)

        return total_reward

    def _update_regrets(self, episode_data: List[Dict], player_idx: int, final_reward: float):
        """Compute per-action counterfactual regrets and add to replay buffer.

        For the sampled action: ground the regret with the actual terminal value.
        For unsampled actions: bootstrap using the advantage network estimate.
        """
        for step_data in episode_data:
            if step_data['player'] != player_idx:
                continue

            observation = step_data['observation']
            action = step_data['action']
            strategy = step_data['strategy']
            legal_actions = step_data['legal_actions']
            advantages = step_data['advantages']

            # Expected value under the current strategy
            expected_value = float(np.sum(strategy[legal_actions] * advantages[legal_actions]))

            # Store a regret entry for every legal action
            for a in legal_actions:
                if a == action:
                    # Sampled action: use actual terminal outcome (grounded)
                    regret = final_reward - expected_value
                else:
                    # Unsampled actions: use network advantage estimate (bootstrapped)
                    regret = float(advantages[a]) - expected_value

                self.deep_cfr.replay_buffer.add(
                    observation, a, regret, strategy, player_idx,
                    iteration=self.iteration
                )

    def train_iteration(self, traversals_per_iteration: int = 1000):
        self.iteration += 1

        for player_idx in range(self.deep_cfr.num_players):
            for _ in range(traversals_per_iteration):
                self.run_traversal(player_idx)

        batch_size = self.config.get('batch_size', 256)
        train_passes = self.config.get('train_passes', 20)

        adv_losses = []
        strat_losses = []
        for _ in range(train_passes):
            adv_losses.append(self.deep_cfr.train_advantage_network(batch_size))
            strat_losses.append(self.deep_cfr.train_strategy_network(batch_size))

        self.deep_cfr.iteration = self.iteration

        return (
            float(np.mean(adv_losses)) if adv_losses else 0.0,
            float(np.mean(strat_losses)) if strat_losses else 0.0,
        )

    def train(self, num_iterations: int, checkpoint_interval: int = 100):
        traversals = self.config.get('traversals_per_iteration', 1000)

        for iteration in range(num_iterations):
            adv_loss, strat_loss = self.train_iteration(traversals)

            if (iteration + 1) % checkpoint_interval == 0:
                self.save_checkpoint(f"models/checkpoint_iter_{iteration + 1}.pt")
                print(
                    f"Iter {iteration + 1:4d}/{num_iterations}"
                    f"  buf={len(self.deep_cfr.replay_buffer)}"
                    f"  adv_loss={adv_loss:.4f}"
                    f"  strat_loss={strat_loss:.4f}"
                )

    def save_checkpoint(self, filepath: str):
        self.deep_cfr.save_models(filepath)

        metadata = {
            'iteration': self.iteration,
            'buffer_size': len(self.deep_cfr.replay_buffer),
            'config': self.config
        }

        metadata_path = filepath.replace('.pt', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def load_checkpoint(self, filepath: str):
        self.deep_cfr.load_models(filepath)

        metadata_path = filepath.replace('.pt', '_metadata.json')
        if Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.iteration = metadata.get('iteration', 0)

    def evaluate(self, num_games: int = 1000) -> Dict:
        results = {
            'total_games': num_games,
            'player_rewards': {i: [] for i in range(self.deep_cfr.num_players)}
        }

        for _ in range(num_games):
            self.engine.reset()

            while not self.engine.is_done():
                current_agent = self.engine.current_agent
                agent_idx = self.engine.agents.index(current_agent)

                observation = self.engine.get_observation()
                legal_actions = self.engine.get_legal_actions()

                if not legal_actions:
                    break

                strategy = self.deep_cfr.get_strategy(observation, agent_idx, legal_actions)
                legal_probs = strategy[legal_actions]
                legal_probs = legal_probs / legal_probs.sum()
                action = np.random.choice(legal_actions, p=legal_probs)

                self.engine.step(action)

            rewards = self.engine.get_rewards()
            for idx, agent in enumerate(self.engine.agents):
                results['player_rewards'][idx].append(rewards[agent])

        for player_idx in range(self.deep_cfr.num_players):
            rewards = results['player_rewards'][player_idx]
            results[f'player_{player_idx}_avg_reward'] = float(np.mean(rewards))
            results[f'player_{player_idx}_std_reward'] = float(np.std(rewards))

        return results

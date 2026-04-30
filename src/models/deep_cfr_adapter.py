"""Deep CFR model adapter."""

from typing import List

import numpy as np
import torch

from .base import ModelAdapter


class DeepCFRAdapter(ModelAdapter):
    """Wraps Deep CFR strategy networks for inference.

    Deep CFR has per-player strategy networks. The average strategy
    across all players approximates the Nash equilibrium.
    """

    def __init__(self, deep_cfr, obs_size: int, action_size: int):
        self.deep_cfr = deep_cfr
        self.obs_size = obs_size
        self.action_size = action_size
        self.iteration = deep_cfr.iteration

    def get_strategy(self, obs: np.ndarray, player_idx: int,
                     legal_actions: List[int]) -> np.ndarray:
        obs_padded = self._pad_obs(obs)
        pidx = player_idx % len(self.deep_cfr.strategy_networks)
        with torch.no_grad():
            obs_t = torch.from_numpy(obs_padded).unsqueeze(0)
            logits = self.deep_cfr.strategy_networks[pidx](obs_t).numpy()[0]
        return self._normalize(logits, legal_actions)

    def _pad_obs(self, obs: np.ndarray) -> np.ndarray:
        if len(obs) == self.obs_size:
            return obs.astype(np.float32)
        if len(obs) > self.obs_size:
            return obs[:self.obs_size].astype(np.float32)
        padded = np.zeros(self.obs_size, dtype=np.float32)
        padded[:len(obs)] = obs
        return padded

    @classmethod
    def from_checkpoint(cls, path: str, **kwargs) -> 'DeepCFRAdapter':
        from src.training import DeepCFR
        from pathlib import Path
        import json

        checkpoint = torch.load(path, map_location='cpu')

        # Infer obs_size and action_size from first strategy network weights
        strat_nets = checkpoint.get('strategy_networks', [])
        if strat_nets and isinstance(strat_nets[0], dict):
            state_dict = strat_nets[0]
            # First linear layer: network.0.weight = [hidden_0, obs_size]
            first_w = state_dict.get('network.0.weight')
            # Last linear layer: find the one with smallest output dim
            last_keys = sorted([k for k in state_dict if '.weight' in k
                                and state_dict[k].dim() == 2],
                               key=lambda k: int(k.split('.')[1].split('.')[0]
                                                  if k.split('.')[1].isdigit()
                                                  else '999'))
            last_w = state_dict[last_keys[-1]]
            obs_size = first_w.shape[1] if first_w is not None else 50
            action_size = last_w.shape[0]
            num_players = len(strat_nets)
        else:
            obs_size = kwargs.get('obs_size', 50)
            action_size = kwargs.get('action_size', 5)
            num_players = 6

        # Infer hidden_layers from the state_dict directly. Layers 0/5/9 of
        # the AdvantageNetwork are the three Linear layers between residual
        # blocks; their output dims = hidden[0], hidden[1], hidden[2].
        hidden = kwargs.get('hidden_layers', None)
        if hidden is None and strat_nets and isinstance(strat_nets[0], dict):
            sd = strat_nets[0]
            try:
                h0 = sd['network.0.weight'].shape[0]
                h1 = sd['network.5.weight'].shape[0]
                h2 = sd['network.9.weight'].shape[0]
                hidden = [h0, h1, h2]
            except (KeyError, IndexError):
                hidden = None
        # Fall back to config-file lookup if inference failed
        if hidden is None:
            config_path = Path(path).parent.parent / 'configs' / 'training_config.json'
            if not config_path.exists():
                config_path = Path('configs/training_config.json')
            if config_path.exists():
                with open(config_path) as f:
                    cfg = json.load(f)
                hidden = cfg.get('network', {}).get('hidden_layers', [256, 256, 128])
            else:
                hidden = [256, 256, 128]

        deep_cfr = DeepCFR(
            observation_size=obs_size, action_size=action_size,
            num_players=num_players, hidden_layers=hidden,
            learning_rate=0.001, buffer_size=1000, device='cpu',
        )
        deep_cfr.load_models(path)

        for net in deep_cfr.strategy_networks + deep_cfr.advantage_networks:
            net.eval()

        return cls(deep_cfr, obs_size, action_size)

import numpy as np
import torch
import torch.multiprocessing as mp
from multiprocessing import Process, Queue, Manager
from queue import Empty
from typing import Dict, List, Optional
from pathlib import Path
import json
import time


class AsyncGPUCFRTrainer:
    
    def __init__(self, engine, deep_cfr, config: Optional[Dict] = None, num_workers: int = None):
        self.engine = engine
        self.deep_cfr = deep_cfr
        self.config = config or {}
        self.num_workers = num_workers or max(1, mp.cpu_count() - 2)
        self.iteration = 0
        
        for net in self.deep_cfr.advantage_networks + self.deep_cfr.strategy_networks:
            net.share_memory()
        
        print(f"Async GPU trainer initialized with {self.num_workers} workers")
    
    def gpu_inference_worker(self, inference_queue, result_queue, stop_event):
        while not stop_event.is_set():
            batch_requests = []
            
            try:
                for _ in range(32):
                    try:
                        item = inference_queue.get(timeout=0.001)
                        batch_requests.append(item)
                    except Empty:
                        break
                
                if not batch_requests:
                    time.sleep(0.001)
                    continue
                
                observations = np.array([req['obs'] for req in batch_requests])
                player_indices = [req['player'] for req in batch_requests]
                legal_actions_list = [req['legal'] for req in batch_requests]
                request_types = [req['type'] for req in batch_requests]
                
                advantage_requests = [(i, req) for i, req in enumerate(batch_requests) if req['type'] == 'advantage']
                strategy_requests = [(i, req) for i, req in enumerate(batch_requests) if req['type'] == 'strategy']
                
                results = [None] * len(batch_requests)
                
                if advantage_requests:
                    adv_obs = np.array([batch_requests[i]['obs'] for i, _ in advantage_requests])
                    adv_players = [batch_requests[i]['player'] for i, _ in advantage_requests]
                    adv_legal = [batch_requests[i]['legal'] for i, _ in advantage_requests]
                    
                    advantages = self._compute_advantages_batch(adv_obs, adv_players, adv_legal)
                    
                    for (idx, _), advantage in zip(advantage_requests, advantages):
                        results[idx] = advantage
                
                if strategy_requests:
                    strat_obs = np.array([batch_requests[i]['obs'] for i, _ in strategy_requests])
                    strat_players = [batch_requests[i]['player'] for i, _ in strategy_requests]
                    strat_legal = [batch_requests[i]['legal'] for i, _ in strategy_requests]
                    
                    strategies = self._compute_strategies_batch(strat_obs, strat_players, strat_legal)
                    
                    for (idx, _), strategy in zip(strategy_requests, strategies):
                        results[idx] = strategy
                
                for req, result in zip(batch_requests, results):
                    result_queue.put({
                        'id': req['id'],
                        'result': result
                    })
                
            except Exception as e:
                print(f"GPU worker error: {e}")
                import traceback
                traceback.print_exc()
    
    def _compute_advantages_batch(self, observations, player_indices, legal_actions_list):
        player_batches = {}
        for i, (obs, player_idx, legal_actions) in enumerate(
            zip(observations, player_indices, legal_actions_list)
        ):
            if player_idx not in player_batches:
                player_batches[player_idx] = {'obs': [], 'legal': [], 'indices': []}
            player_batches[player_idx]['obs'].append(obs)
            player_batches[player_idx]['legal'].append(legal_actions)
            player_batches[player_idx]['indices'].append(i)
        
        results = [None] * len(observations)
        
        for player_idx, batch_data in player_batches.items():
            obs_batch = torch.FloatTensor(np.array(batch_data['obs'])).to(self.deep_cfr.device)
            
            with torch.no_grad():
                advantages = self.deep_cfr.advantage_networks[player_idx](obs_batch).cpu().numpy()
            
            for advantage, legal_actions, orig_idx in zip(
                advantages, batch_data['legal'], batch_data['indices']
            ):
                legal_mask = np.zeros(self.deep_cfr.action_size)
                legal_mask[legal_actions] = 1
                advantage = advantage * legal_mask
                results[orig_idx] = advantage
        
        return results
    
    def _compute_strategies_batch(self, observations, player_indices, legal_actions_list):
        player_batches = {}
        for i, (obs, player_idx, legal_actions) in enumerate(
            zip(observations, player_indices, legal_actions_list)
        ):
            if player_idx not in player_batches:
                player_batches[player_idx] = {'obs': [], 'legal': [], 'indices': []}
            player_batches[player_idx]['obs'].append(obs)
            player_batches[player_idx]['legal'].append(legal_actions)
            player_batches[player_idx]['indices'].append(i)
        
        results = [None] * len(observations)
        
        for player_idx, batch_data in player_batches.items():
            obs_batch = torch.FloatTensor(np.array(batch_data['obs'])).to(self.deep_cfr.device)
            
            with torch.no_grad():
                strategies = self.deep_cfr.strategy_networks[player_idx](obs_batch).cpu().numpy()
            
            for strategy, legal_actions, orig_idx in zip(
                strategies, batch_data['legal'], batch_data['indices']
            ):
                legal_mask = np.zeros(self.deep_cfr.action_size)
                legal_mask[legal_actions] = 1
                strategy = strategy * legal_mask
                
                strategy_sum = strategy.sum()
                if strategy_sum > 0:
                    strategy = strategy / strategy_sum
                else:
                    strategy = legal_mask / legal_mask.sum()
                
                results[orig_idx] = strategy
        
        return results
    
    def traversal_worker(self, worker_id, task_queue, inference_queue, result_queue, replay_queue, stop_event):
        from src.engine import PokerEngine
        
        engine = PokerEngine(num_players=self.deep_cfr.num_players)
        pending_requests = {}
        request_counter = 0
        
        while not stop_event.is_set():
            try:
                task = task_queue.get(timeout=0.1)
            except Empty:
                continue
            
            if task is None:
                break
            
            player_idx = task['player_idx']
            
            episode_data = []
            engine.reset()
            
            while not engine.is_done():
                current_agent = engine.current_agent
                agent_idx = engine.agents.index(current_agent)
                
                observation = engine.get_observation()
                legal_actions = engine.get_legal_actions()
                
                if not legal_actions:
                    break
                
                request_id = f"{worker_id}_{request_counter}"
                request_counter += 1
                
                if agent_idx == player_idx:
                    inference_queue.put({
                        'id': request_id,
                        'obs': observation,
                        'player': agent_idx,
                        'legal': legal_actions,
                        'type': 'advantage'
                    })
                else:
                    inference_queue.put({
                        'id': request_id,
                        'obs': observation,
                        'player': agent_idx,
                        'legal': legal_actions,
                        'type': 'strategy'
                    })
                
                result = None
                while result is None:
                    try:
                        res = result_queue.get(timeout=0.01)
                        if res['id'] == request_id:
                            result = res['result']
                        else:
                            pending_requests[res['id']] = res['result']
                    except Empty:
                        if request_id in pending_requests:
                            result = pending_requests.pop(request_id)
                
                if agent_idx == player_idx:
                    advantages = result
                    strategy = self.deep_cfr.get_regret_matching_strategy(advantages, legal_actions)
                else:
                    strategy = result
                
                legal_strategy = strategy[legal_actions]
                legal_strategy_sum = legal_strategy.sum()
                if legal_strategy_sum > 0:
                    action = np.random.choice(legal_actions, p=legal_strategy / legal_strategy_sum)
                else:
                    action = np.random.choice(legal_actions)
                
                episode_data.append({
                    'player': agent_idx,
                    'observation': observation.copy(),
                    'action': action,
                    'strategy': strategy,
                    'legal_actions': legal_actions.copy()
                })
                
                engine.step(action)
            
            rewards = engine.get_rewards()
            final_reward = rewards[engine.agents[player_idx]]
            
            replay_queue.put((episode_data, final_reward, player_idx))
        
        engine.close()
    
    def train_iteration(self, traversals_per_iteration: int = 1000):
        self.iteration += 1
        
        manager = Manager()
        task_queue = manager.Queue()
        inference_queue = manager.Queue(maxsize=200)
        result_queue = manager.Queue(maxsize=200)
        replay_queue = manager.Queue(maxsize=1000)
        stop_event = manager.Event()
        
        for player_idx in range(self.deep_cfr.num_players):
            for _ in range(traversals_per_iteration):
                task_queue.put({'player_idx': player_idx})
        
        for _ in range(self.num_workers):
            task_queue.put(None)
        
        gpu_process = Process(
            target=self.gpu_inference_worker,
            args=(inference_queue, result_queue, stop_event)
        )
        gpu_process.start()
        
        workers = []
        for i in range(self.num_workers):
            p = Process(
                target=self.traversal_worker,
                args=(i, task_queue, inference_queue, result_queue, replay_queue, stop_event)
            )
            p.start()
            workers.append(p)
        
        for w in workers:
            w.join()
        
        stop_event.set()
        gpu_process.join(timeout=2)
        if gpu_process.is_alive():
            gpu_process.terminate()
        
        while not replay_queue.empty():
            try:
                episode_data, final_reward, player_idx = replay_queue.get_nowait()
                
                for step_data in episode_data:
                    if step_data['player'] != player_idx:
                        continue
                    
                    observation = step_data['observation']
                    action = step_data['action']
                    strategy = step_data['strategy']
                    
                    regret = final_reward
                    
                    self.deep_cfr.replay_buffer.add(
                        observation, action, regret, strategy, player_idx
                    )
            except Empty:
                break
        
        batch_size = self.config.get('batch_size', 128)
        
        for _ in range(10):
            self.deep_cfr.train_advantage_network(batch_size)
            self.deep_cfr.train_strategy_network(batch_size)
        
        self.deep_cfr.iteration = self.iteration
    
    def train(self, num_iterations: int, checkpoint_interval: int = 100):
        traversals = self.config.get('traversals_per_iteration', 1000)
        
        for iteration in range(num_iterations):
            self.train_iteration(traversals)
            
            if (iteration + 1) % checkpoint_interval == 0:
                self.save_checkpoint(f"models/checkpoint_iter_{iteration + 1}.pt")
                print(f"Iteration {iteration + 1}/{num_iterations} completed - Buffer: {len(self.deep_cfr.replay_buffer)}")
    
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
    
    def evaluate(self, num_games: int = 100) -> Dict[str, float]:
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
                action = np.random.choice(legal_actions, p=strategy[legal_actions] / strategy[legal_actions].sum())
                
                self.engine.step(action)
            
            rewards = self.engine.get_rewards()
            for idx, agent in enumerate(self.engine.agents):
                results['player_rewards'][idx].append(rewards[agent])
        
        for player_idx in range(self.deep_cfr.num_players):
            rewards = results['player_rewards'][player_idx]
            results[f'player_{player_idx}_avg_reward'] = np.mean(rewards)
            results[f'player_{player_idx}_std_reward'] = np.std(rewards)
        
        return results

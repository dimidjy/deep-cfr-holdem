import sys
import os
import argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.engine import PokerEngine
from src.training import DeepCFR
from src.search import DepthLimitedSearch, PlaytimeStrategy


MODEL_PATH = "models/final_model.pt"

ACTION_NAMES = {
    0: "Fold",
    1: "Check/Call",
    2: "Raise Half Pot",
    3: "Raise Full Pot",
    4: "All-In"
}


def get_human_action(legal_actions):
    print("\nLegal actions:")
    for idx, action in enumerate(legal_actions):
        print(f"  {idx}: {ACTION_NAMES.get(action, f'Action {action}')}")

    while True:
        try:
            choice = int(input("Choose action index: "))
            if 0 <= choice < len(legal_actions):
                return legal_actions[choice]
            else:
                print("Invalid choice. Try again.")
        except (ValueError, KeyboardInterrupt):
            print("\nExiting...")
            sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description='Play poker against a trained AI (CLI)')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Path to model file')
    parser.add_argument('--search', action='store_true', help='Enable depth-limited search')
    parser.add_argument('--search-depth', type=int, default=2, help='Search depth (default: 2)')
    parser.add_argument('--search-iters', type=int, default=50, help='CFR iterations per search (default: 50)')
    parser.add_argument('--search-rollouts', type=int, default=10, help='Rollouts per action (default: 10)')
    parser.add_argument('--no-refine', action='store_true', help='Disable strategy refinement')
    parser.add_argument('--temperature', type=float, default=0.5, help='Strategy temperature (default: 0.5)')
    args = parser.parse_args()

    print("="*60)
    print("PLAY VS AI")
    print("="*60)

    model_path = Path(__file__).parent.parent / args.model

    engine = PokerEngine(num_players=2)

    observation_size = engine.get_observation_space_size()
    action_size = engine.get_action_space_size()

    deep_cfr = DeepCFR(
        observation_size=observation_size,
        action_size=action_size,
        num_players=2
    )

    if model_path.exists():
        print(f"Loading AI model from: {model_path}")
        deep_cfr.load_models(str(model_path))
    else:
        print("No trained model found. AI will play randomly.")

    search = None
    if args.search:
        search = DepthLimitedSearch(
            deep_cfr,
            num_rollouts=args.search_rollouts,
            cfr_iters=args.search_iters,
            max_depth=args.search_depth,
        )
        print(f"  Search enabled: depth={args.search_depth}, "
              f"iters={args.search_iters}, rollouts={args.search_rollouts}")

    refiner = None
    if not args.no_refine and model_path.exists():
        refiner = PlaytimeStrategy(
            deep_cfr,
            num_players=2,
            temperature=args.temperature,
            allin_penalty=True,
        )
        print(f"  Strategy refinement: temperature={args.temperature}, all-in penalty=ON")

    print("\nYou are Player 0, AI is Player 1")
    print("="*60)

    state = engine.reset()

    while not engine.is_done():
        current_agent = state['current_player']
        agent_idx = engine.agents.index(current_agent)

        legal_actions = state['legal_actions']

        if not legal_actions:
            break

        print(f"\n--- {current_agent}'s turn ---")

        if agent_idx == 0:
            action = get_human_action(legal_actions)
        else:
            if model_path.exists():
                if search:
                    strategy = search.search(engine, agent_idx, legal_actions)
                elif refiner:
                    observation = engine.get_observation()
                    strategy = refiner.get_strategy(observation, agent_idx, legal_actions)
                else:
                    observation = engine.get_observation()
                    strategy = deep_cfr.get_strategy(observation, agent_idx, legal_actions)
                legal_strategy = strategy[legal_actions]
                legal_strategy /= legal_strategy.sum()
                action = np.random.choice(legal_actions, p=legal_strategy)
            else:
                action = np.random.choice(legal_actions)

            print(f"AI chooses: {ACTION_NAMES.get(action, f'Action {action}')}")

        state = engine.step(action)

    print("\n" + "="*60)
    print("GAME OVER")
    print("="*60)

    rewards = engine.get_rewards()
    for idx, agent in enumerate(engine.agents):
        result = "WON" if rewards[agent] > 0 else "LOST" if rewards[agent] < 0 else "TIED"
        print(f"{agent}: {result} (Reward: {rewards[agent]:.2f})")

    print("="*60)

    engine.close()


if __name__ == "__main__":
    main()

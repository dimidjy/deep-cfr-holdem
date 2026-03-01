import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.validation import SpotValidator
from src.engine import PokerEngine


SPOTS_FILE = "data/spots/sample_spots.json"


def main():
    spots_path = Path(__file__).parent.parent / SPOTS_FILE
    
    if not spots_path.exists():
        print(f"Spots file not found: {spots_path}")
        return
    
    validator = SpotValidator(str(spots_path))
    print(f"Loaded {len(validator.spots)} spots from {SPOTS_FILE}")
    
    engine = PokerEngine(num_players=2)
    game_states = []
    
    for game_idx in range(len(validator.spots)):
        engine.reset()
        state = engine.get_state()
        game_states.append(state)
    
    results = validator.validate_all_spots(game_states)
    
    print("\n" + "="*60)
    print("SPOT VALIDATION RESULTS")
    print("="*60)
    print(f"Total Spots: {results['total_spots']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Pass Rate: {results['pass_rate']*100:.1f}%")
    print("="*60)
    
    for result in results['results']:
        status = "PASS" if result['passed'] else "FAIL"
        print(f"\n[{status}] {result['spot_id']}")
        
        if not result['passed']:
            for error in result['errors']:
                print(f"  - {error['type']}")
                print(f"    Expected: {error.get('expected', 'N/A')}")
                print(f"    Actual: {error.get('actual', 'N/A')}")
    
    engine.close()


if __name__ == "__main__":
    main()

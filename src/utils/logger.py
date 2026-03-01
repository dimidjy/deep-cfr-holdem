import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List


class TrainingLogger:
    
    def __init__(self, log_dir: str = "data/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"training_{timestamp}.json"
        
        self.logs = {
            'start_time': timestamp,
            'iterations': [],
            'evaluations': []
        }
    
    def log_iteration(self, iteration: int, metrics: Dict[str, Any]):
        entry = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        self.logs['iterations'].append(entry)
        self._save()
    
    def log_evaluation(self, iteration: int, results: Dict[str, Any]):
        entry = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            **results
        }
        
        self.logs['evaluations'].append(entry)
        self._save()
    
    def _save(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)
    
    def get_logs(self) -> Dict[str, Any]:
        return self.logs
    
    def get_iteration_metrics(self, metric_name: str) -> List[float]:
        return [entry.get(metric_name, 0) for entry in self.logs['iterations']]

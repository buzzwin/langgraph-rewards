"""
Evaluation metrics for reward functions.
"""

from typing import Dict, List, Any
import numpy as np
from ..core.reward_function import RewardFunction, RewardContext


class RewardMetrics:
    """Collection of metrics for evaluating reward function performance."""
    
    def __init__(self):
        self.reward_history: List[float] = []
        self.context_history: List[RewardContext] = []
    
    def add_reward(self, reward: float, context: RewardContext):
        """Add a reward value and context to the history."""
        self.reward_history.append(reward)
        self.context_history.append(context)
    
    def get_average_reward(self) -> float:
        """Calculate the average reward value."""
        if not self.reward_history:
            return 0.0
        return np.mean(self.reward_history)
    
    def get_reward_std(self) -> float:
        """Calculate the standard deviation of reward values."""
        if len(self.reward_history) < 2:
            return 0.0
        return np.std(self.reward_history)
    
    def get_reward_trend(self) -> float:
        """Calculate the trend of reward values over time."""
        if len(self.reward_history) < 2:
            return 0.0
        
        x = np.arange(len(self.reward_history))
        slope = np.polyfit(x, self.reward_history, 1)[0]
        return slope
    
    def get_reward_distribution(self) -> Dict[str, float]:
        """Get distribution statistics of reward values."""
        if not self.reward_history:
            return {}
        
        rewards = np.array(self.reward_history)
        return {
            "min": float(np.min(rewards)),
            "max": float(np.max(rewards)),
            "mean": float(np.mean(rewards)),
            "median": float(np.median(rewards)),
            "std": float(np.std(rewards)),
            "q25": float(np.percentile(rewards, 25)),
            "q75": float(np.percentile(rewards, 75))
        }
    
    def get_improvement_rate(self) -> float:
        """Calculate the rate of improvement in rewards."""
        if len(self.reward_history) < 2:
            return 0.0
        
        improvements = 0
        for i in range(1, len(self.reward_history)):
            if self.reward_history[i] > self.reward_history[i-1]:
                improvements += 1
        
        return improvements / (len(self.reward_history) - 1)


class RewardEvaluator:
    """Evaluator for reward functions."""
    
    def __init__(self, reward_function: RewardFunction):
        self.reward_function = reward_function
        self.metrics = RewardMetrics()
    
    def evaluate_context(self, context: RewardContext) -> Dict[str, Any]:
        """Evaluate a single context and return metrics."""
        reward = self.reward_function(context)
        self.metrics.add_reward(reward, context)
        
        return {
            "reward": reward,
            "context_id": id(context),
            "timestamp": len(self.metrics.reward_history)
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a comprehensive performance summary."""
        return {
            "function_name": self.reward_function.name,
            "total_evaluations": len(self.metrics.reward_history),
            "current_reward": self.metrics.reward_history[-1] if self.metrics.reward_history else 0.0,
            "average_reward": self.metrics.get_average_reward(),
            "reward_std": self.metrics.get_reward_std(),
            "reward_trend": self.metrics.get_reward_trend(),
            "improvement_rate": self.metrics.get_improvement_rate(),
            "distribution": self.metrics.get_reward_distribution()
        }
    
    def compare_with_baseline(self, baseline_rewards: List[float]) -> Dict[str, Any]:
        """Compare performance with a baseline."""
        if not self.metrics.reward_history or not baseline_rewards:
            return {}
        
        current_avg = self.metrics.get_average_reward()
        baseline_avg = np.mean(baseline_rewards)
        
        return {
            "current_average": current_avg,
            "baseline_average": baseline_avg,
            "improvement": current_avg - baseline_avg,
            "improvement_percentage": ((current_avg - baseline_avg) / baseline_avg * 100) if baseline_avg != 0 else 0
        }

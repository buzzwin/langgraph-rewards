"""
Custom reward function template and utilities.
"""

from typing import Any, Callable, Dict, Optional
from pydantic import Field
from ..core.reward_function import ScalarRewardFunction, RewardContext


class CustomRewardFunction(ScalarRewardFunction):
    """Custom reward function that can be configured with custom logic."""
    
    name: str = "custom_reward"
    description: str = "Custom reward function with configurable logic"
    custom_function: Optional[Callable] = None
    
    def _calculate_raw_reward(self, context: RewardContext) -> float:
        """Calculate reward using custom function if provided."""
        if self.custom_function:
            try:
                result = self.custom_function(context)
                if isinstance(result, (int, float)):
                    return float(result)
                else:
                    return 0.5
            except Exception:
                return 0.5
        
        return 0.5


class LambdaRewardFunction(ScalarRewardFunction):
    """Reward function that uses a lambda function for calculation."""
    
    name: str = "lambda_reward"
    description: str = "Reward function using lambda function for calculation"
    lambda_func: Callable
    
    def _calculate_raw_reward(self, context: RewardContext) -> float:
        """Calculate reward using the provided lambda function."""
        try:
            result = self.lambda_func(context)
            if isinstance(result, (int, float)):
                return float(result)
            else:
                return 0.5
        except Exception:
            return 0.5


class ConditionalRewardFunction(ScalarRewardFunction):
    """Reward function with conditional logic."""
    
    name: str = "conditional_reward"
    description: str = "Reward function with conditional reward logic"
    conditions: Dict[str, float] = Field(default_factory=dict, description="Condition to reward mapping")
    default_reward: float = Field(default=0.0, description="Default reward value")
    
    def _calculate_raw_reward(self, context: RewardContext) -> float:
        """Calculate reward based on conditions."""
        for condition, reward in self.conditions.items():
            if self._evaluate_condition(condition, context):
                return reward
        
        return self.default_reward
    
    def _evaluate_condition(self, condition: str, context: RewardContext) -> bool:
        """Evaluate a condition string against the context."""
        try:
            # Simple condition evaluation - can be extended
            if "==" in condition:
                key, value = condition.split("==", 1)
                key = key.strip()
                value = value.strip()
                
                if key in context.agent_state:
                    return str(context.agent_state[key]) == value
                elif key in context.metadata:
                    return str(context.metadata[key]) == value
            
            elif "in" in condition:
                key, value = condition.split(" in ", 1)
                key = key.strip()
                value = value.strip()
                
                if key in context.agent_state:
                    return value in str(context.agent_state[key])
                elif key in context.metadata:
                    return value in str(context.metadata[key])
            
            return False
        except Exception:
            return False

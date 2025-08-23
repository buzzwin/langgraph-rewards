"""
Reward builder for creating and managing reward functions.
"""

from typing import Dict, List, Optional, Union, Callable
from pydantic import BaseModel, Field
from .reward_function import RewardFunction, RewardContext


class RewardBuilder(BaseModel):
    """Builder class for creating and managing reward functions."""
    
    reward_functions: Dict[str, RewardFunction] = Field(default_factory=dict, description="Registered reward functions")
    
    def add_reward_function(self, name: str, reward_function: RewardFunction) -> 'RewardBuilder':
        """Add a reward function to the builder."""
        self.reward_functions[name] = reward_function
        return self
    
    def remove_reward_function(self, name: str) -> 'RewardBuilder':
        """Remove a reward function from the builder."""
        if name in self.reward_functions:
            del self.reward_functions[name]
        return self
    
    def get_reward_function(self, name: str) -> Optional[RewardFunction]:
        """Get a reward function by name."""
        return self.reward_functions.get(name)
    
    def list_reward_functions(self) -> List[str]:
        """List all registered reward function names."""
        return list(self.reward_functions.keys())
    
    def create_composite_reward(self, 
                              function_names: List[str], 
                              weights: Optional[List[float]] = None) -> 'CompositeRewardFunction':
        """Create a composite reward function from multiple reward functions."""
        if weights is None:
            weights = [1.0] * len(function_names)
        
        if len(weights) != len(function_names):
            raise ValueError("Number of weights must match number of function names")
        
        functions = []
        for name, weight in zip(function_names, weights):
            if name not in self.reward_functions:
                raise ValueError(f"Reward function '{name}' not found")
            func = self.reward_functions[name]
            func.weight = weight
            functions.append(func)
        
        return CompositeRewardFunction(
            name="composite",
            description=f"Composite of {', '.join(function_names)}",
            reward_functions=functions
        )
    
    def build(self) -> Dict[str, RewardFunction]:
        """Build and return the reward functions dictionary."""
        return self.reward_functions.copy()


class CompositeRewardFunction(RewardFunction):
    """Composite reward function that combines multiple reward functions."""
    
    reward_functions: List[RewardFunction] = Field(description="List of reward functions to combine")
    combination_method: str = Field(default="weighted_sum", description="Method to combine rewards")
    
    def calculate_reward(self, context: RewardContext) -> float:
        """Calculate the combined reward value."""
        if not self.reward_functions:
            return 0.0
        
        rewards = [func.get_weighted_reward(context) for func in self.reward_functions]
        
        if self.combination_method == "weighted_sum":
            return sum(rewards)
        elif self.combination_method == "average":
            return sum(rewards) / len(rewards)
        elif self.combination_method == "min":
            return min(rewards)
        elif self.combination_method == "max":
            return max(rewards)
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")
    
    def get_individual_rewards(self, context: RewardContext) -> Dict[str, float]:
        """Get individual reward values for each function."""
        return {func.name: func.calculate_reward(context) for func in self.reward_functions}

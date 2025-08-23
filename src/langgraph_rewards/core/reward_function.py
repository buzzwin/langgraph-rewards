"""
Base reward function class for LangGraph agents.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field
import numpy as np


class RewardContext(BaseModel):
    """Context information for reward calculation."""
    agent_state: Dict[str, Any] = Field(description="Current agent state")
    action: Optional[Any] = Field(description="Action taken by agent")
    result: Optional[Any] = Field(description="Result of the action")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RewardFunction(ABC, BaseModel):
    """Abstract base class for reward functions."""
    
    name: str = Field(description="Name of the reward function")
    weight: float = Field(default=1.0, description="Weight for this reward function")
    description: str = Field(description="Description of what this reward function measures")
    
    class Config:
        arbitrary_types_allowed = True
    
    @abstractmethod
    def calculate_reward(self, context: RewardContext) -> float:
        """Calculate the reward value for the given context."""
        pass
    
    def __call__(self, context: RewardContext) -> float:
        """Make the reward function callable."""
        return self.calculate_reward(context)
    
    def get_weighted_reward(self, context: RewardContext) -> float:
        """Get the weighted reward value."""
        return self.weight * self.calculate_reward(context)


class ScalarRewardFunction(RewardFunction):
    """Base class for reward functions that return scalar values."""
    
    min_value: float = Field(default=0.0, description="Minimum possible reward value")
    max_value: float = Field(default=1.0, description="Maximum possible reward value")
    
    def normalize_reward(self, value: float) -> float:
        """Normalize reward value to [0, 1] range."""
        if self.max_value == self.min_value:
            return 0.5
        return np.clip((value - self.min_value) / (self.max_value - self.min_value), 0.0, 1.0)
    
    def calculate_reward(self, context: RewardContext) -> float:
        """Calculate and normalize the reward value."""
        raw_reward = self._calculate_raw_reward(context)
        return self.normalize_reward(raw_reward)
    
    @abstractmethod
    def _calculate_raw_reward(self, context: RewardContext) -> float:
        """Calculate the raw reward value (to be implemented by subclasses)."""
        pass

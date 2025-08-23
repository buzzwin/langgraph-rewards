"""
Tests for reward functions.
"""

import pytest
from langgraph_rewards import (
    RewardContext, CompletionReward, RelevanceReward,
    RewardBuilder, CompositeRewardFunction
)


class TestRewardFunctions:
    """Test cases for reward functions."""
    
    def test_completion_reward_success(self):
        """Test completion reward for successful completion."""
        reward_fn = CompletionReward()
        context = RewardContext(
            agent_state={"completed": True},
            action=None,
            result=None,
            metadata={"completion_status": "success"}
        )
        
        reward = reward_fn(context)
        assert reward == 1.0
    
    def test_completion_reward_partial(self):
        """Test completion reward for partial completion."""
        reward_fn = CompletionReward()
        context = RewardContext(
            agent_state={"completed": False},
            action=None,
            result=None,
            metadata={"completion_status": "partial"}
        )
        
        reward = reward_fn(context)
        assert reward == 0.5
    
    def test_relevance_reward(self):
        """Test relevance reward function."""
        reward_fn = RelevanceReward()
        context = RewardContext(
            agent_state={},
            action=None,
            result=None,
            metadata={"relevance_score": 0.8}
        )
        
        reward = reward_fn(context)
        assert reward == 0.8
    
    def test_reward_builder(self):
        """Test reward builder functionality."""
        builder = RewardBuilder()
        builder.add_reward_function("completion", CompletionReward())
        builder.add_reward_function("relevance", RelevanceReward())
        
        assert "completion" in builder.list_reward_functions()
        assert "relevance" in builder.list_reward_functions()
        
        completion_fn = builder.get_reward_function("completion")
        assert completion_fn is not None
        assert completion_fn.name == "completion_reward"
    
    def test_composite_reward(self):
        """Test composite reward function."""
        completion = CompletionReward()
        relevance = RelevanceReward()
        
        composite = CompositeRewardFunction(
            name="test_composite",
            description="Test composite",
            reward_functions=[completion, relevance]
        )
        
        context = RewardContext(
            agent_state={"completed": True},
            action=None,
            result=None,
            metadata={"relevance_score": 0.8}
        )
        
        reward = composite(context)
        assert reward > 0
        
        individual_rewards = composite.get_individual_rewards(context)
        assert "completion_reward" in individual_rewards
        assert "relevance_reward" in individual_rewards


if __name__ == "__main__":
    pytest.main([__file__])

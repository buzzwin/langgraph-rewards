"""
Basic usage example of LangGraph Rewards.
"""

from langgraph_rewards import (
    RewardBuilder, CompletionReward, RelevanceReward, 
    CompositeRewardFunction, RewardContext
)


def basic_reward_example():
    """Basic example of using reward functions."""
    
    # Create a reward builder
    builder = RewardBuilder()
    
    # Add reward functions
    builder.add_reward_function("completion", CompletionReward())
    builder.add_reward_function("relevance", RelevanceReward())
    
    # Create a composite reward function
    composite = builder.create_composite_reward(
        function_names=["completion", "relevance"],
        weights=[0.6, 0.4]
    )
    
    # Create a sample context
    context = RewardContext(
        agent_state={
            "completed": True,
            "goals": ["task1", "task2"],
            "achieved_goals": ["task1"]
        },
        action=None,
        result=None,
        metadata={
            "relevance_score": 0.8,
            "completion_status": "success"
        }
    )
    
    # Calculate rewards
    completion_reward = builder.get_reward_function("completion")(context)
    relevance_reward = builder.get_reward_function("relevance")(context)
    composite_reward = composite(context)
    
    print(f"Completion Reward: {completion_reward:.3f}")
    print(f"Relevance Reward: {relevance_reward:.3f}")
    print(f"Composite Reward: {composite_reward:.3f}")
    
    # Get individual rewards from composite
    individual_rewards = composite.get_individual_rewards(context)
    print(f"Individual Rewards: {individual_rewards}")


if __name__ == "__main__":
    basic_reward_example()

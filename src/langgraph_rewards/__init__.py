"""
LangGraph Rewards - A framework for building reward functions in LangGraph agents.
"""

from .core.reward_function import RewardFunction, RewardContext, ScalarRewardFunction
from .core.reward_builder import RewardBuilder, CompositeRewardFunction
from .functions.completion_reward import CompletionReward, StepCompletionReward, GoalAchievementReward
from .functions.relevance_reward import RelevanceReward, ContentRelevanceReward, ContextAwarenessReward
from .functions.custom_reward import CustomRewardFunction, LambdaRewardFunction, ConditionalRewardFunction
from .evaluation.metrics import RewardMetrics, RewardEvaluator

__version__ = "0.1.0"
__all__ = [
    "RewardFunction",
    "RewardContext", 
    "ScalarRewardFunction",
    "RewardBuilder",
    "CompositeRewardFunction",
    "CompletionReward",
    "StepCompletionReward",
    "GoalAchievementReward",
    "RelevanceReward",
    "ContentRelevanceReward",
    "ContextAwarenessReward",
    "CustomRewardFunction",
    "LambdaRewardFunction",
    "ConditionalRewardFunction",
    "RewardMetrics",
    "RewardEvaluator"
]

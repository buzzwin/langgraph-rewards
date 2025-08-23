"""
Completion-based reward functions for LangGraph agents.
"""

from typing import Any, Dict, Optional
from ..core.reward_function import ScalarRewardFunction, RewardContext


class CompletionReward(ScalarRewardFunction):
    """Reward function based on task completion."""
    
    name: str = "completion_reward"
    description: str = "Rewards agents for completing tasks successfully"
    
    def _calculate_raw_reward(self, context: RewardContext) -> float:
        """Calculate reward based on completion status."""
        # Check if the task is marked as completed
        if "completed" in context.agent_state:
            if context.agent_state["completed"]:
                return 1.0
        
        # Check for completion indicators in metadata
        if "completion_status" in context.metadata:
            status = context.metadata["completion_status"]
            if status == "success":
                return 1.0
            elif status == "partial":
                return 0.5
            elif status == "failed":
                return 0.0
        
        # Check for result quality
        if context.result is not None:
            if hasattr(context.result, "success") and context.result.success:
                return 1.0
            elif hasattr(context.result, "status"):
                if context.result.status == "completed":
                    return 1.0
                elif context.result.status == "in_progress":
                    return 0.3
        
        return 0.0


class StepCompletionReward(ScalarRewardFunction):
    """Reward function based on step-by-step completion."""
    
    name: str = "step_completion_reward"
    description: str = "Rewards agents for completing individual steps"
    required_steps: int = 1
    
    def _calculate_raw_reward(self, context: RewardContext) -> float:
        """Calculate reward based on completed steps."""
        completed_steps = context.agent_state.get("completed_steps", 0)
        total_steps = context.agent_state.get("total_steps", self.required_steps)
        
        if total_steps == 0:
            return 0.0
        
        completion_ratio = completed_steps / total_steps
        return completion_ratio


class GoalAchievementReward(ScalarRewardFunction):
    """Reward function based on goal achievement."""
    
    name: str = "goal_achievement_reward"
    description: str = "Rewards agents for achieving specific goals"
    
    def _calculate_raw_reward(self, context: RewardContext) -> float:
        """Calculate reward based on goal achievement."""
        goals = context.agent_state.get("goals", [])
        achieved_goals = context.agent_state.get("achieved_goals", [])
        
        if not goals:
            return 0.0
        
        achievement_ratio = len(achieved_goals) / len(goals)
        return achievement_ratio

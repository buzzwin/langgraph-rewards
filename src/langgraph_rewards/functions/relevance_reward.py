"""
Relevance-based reward functions for LangGraph agents.
"""

from typing import Any, Dict, List, Optional
from ..core.reward_function import ScalarRewardFunction, RewardContext
import numpy as np


class RelevanceReward(ScalarRewardFunction):
    """Reward function based on relevance of agent actions/responses."""
    
    name: str = "relevance_reward"
    description: str = "Rewards agents for relevant actions and responses"
    
    def _calculate_raw_reward(self, context: RewardContext) -> float:
        """Calculate reward based on relevance."""
        # Check for relevance score in metadata
        if "relevance_score" in context.metadata:
            return context.metadata["relevance_score"]
        
        # Check for relevance indicators in the result
        if context.result is not None:
            if hasattr(context.result, "relevance"):
                return context.result.relevance
            elif hasattr(context.result, "score"):
                return context.result.score
        
        # Default relevance based on action appropriateness
        action = context.action
        if action is not None:
            # Simple heuristic: check if action is in expected actions
            expected_actions = context.agent_state.get("expected_actions", [])
            if expected_actions and action in expected_actions:
                return 0.8
        
        return 0.5


class ContentRelevanceReward(ScalarRewardFunction):
    """Reward function based on content relevance."""
    
    name: str = "content_relevance_reward"
    description: str = "Rewards agents for generating relevant content"
    
    def _calculate_raw_reward(self, context: RewardContext) -> float:
        """Calculate reward based on content relevance."""
        # Check for content relevance metrics
        if "content_relevance" in context.metadata:
            return context.metadata["content_relevance"]
        
        # Check for keyword matching
        if "keywords" in context.metadata and "generated_content" in context.agent_state:
            keywords = context.metadata["keywords"]
            content = context.agent_state["generated_content"]
            
            if keywords and content:
                matches = sum(1 for keyword in keywords if keyword.lower() in content.lower())
                relevance = matches / len(keywords) if keywords else 0.0
                return relevance
        
        return 0.5


class ContextAwarenessReward(ScalarRewardFunction):
    """Reward function based on context awareness."""
    
    name: str = "context_awareness_reward"
    description: str = "Rewards agents for being aware of context"
    
    def _calculate_raw_reward(self, context: RewardContext) -> float:
        """Calculate reward based on context awareness."""
        # Check if agent used available context
        available_context = context.agent_state.get("available_context", {})
        used_context = context.agent_state.get("used_context", {})
        
        if not available_context:
            return 0.5
        
        # Calculate context utilization ratio
        utilization = len(used_context) / len(available_context)
        return utilization

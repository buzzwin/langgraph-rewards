# LangGraph Rewards

A framework for building reward functions in LangGraph agents.

## Installation

```bash
pip install -e .
```

## Basic Usage

```python
from langgraph_rewards import CompletionReward, RelevanceReward, RewardBuilder

# Create reward functions
completion_reward = CompletionReward()
relevance_reward = RelevanceReward()

# Build composite rewards
builder = RewardBuilder()
builder.add_reward_function("completion", completion_reward)
builder.add_reward_function("relevance", relevance_reward)
```

## LangGraph Integration Example

Here's how to integrate reward functions with LangGraph agents:

```python
from typing import Dict, Any, TypedDict
from langgraph.graph import StateGraph
from langgraph.constants import END
from langgraph_rewards import (
    RewardBuilder, CompletionReward, RelevanceReward,
    RewardContext, RewardEvaluator
)

class AgentState(TypedDict):
    """State for the agent."""
    messages: list
    current_task: str
    completed_tasks: list
    total_tasks: int
    reward_history: list

def create_agent_with_rewards():
    """Create a LangGraph agent with integrated reward functions."""

    # Create reward functions
    builder = RewardBuilder()
    builder.add_reward_function("completion", CompletionReward())
    builder.add_reward_function("relevance", RelevanceReward())

    # Create composite reward
    composite_reward = builder.create_composite_reward(
        function_names=["completion", "relevance"],
        weights=[0.7, 0.3]
    )

    # Create reward evaluator
    evaluator = RewardEvaluator(composite_reward)

    def agent_node(state: AgentState) -> AgentState:
        """Agent node that processes tasks and calculates rewards."""

        # Simulate task processing
        current_task = state["current_task"]

        # Process the task (simplified)
        if current_task == "analyze_data":
            # Simulate successful completion
            state["completed_tasks"].append(current_task)
            completion_status = "success"
            relevance_score = 0.9
        else:
            completion_status = "partial"
            relevance_score = 0.6

        # Create reward context
        context = RewardContext(
            agent_state={
                "completed": len(state["completed_tasks"]) == state["total_tasks"],
                "completed_steps": len(state["completed_tasks"]),
                "total_steps": state["total_tasks"],
                "goals": [f"task_{i}" for i in range(state["total_tasks"])],
                "achieved_goals": state["completed_tasks"]
            },
            action=current_task,
            result={"status": completion_status, "score": relevance_score},
            metadata={
                "completion_status": completion_status,
                "relevance_score": relevance_score,
                "current_task": current_task
            }
        )

        # Calculate reward
        reward = composite_reward(context)
        state["reward_history"].append(reward)

        # Evaluate performance
        evaluation = evaluator.evaluate_context(context)

        print(f"Task: {current_task}")
        print(f"Reward: {reward:.3f}")
        print(f"Completion: {len(state['completed_tasks'])}/{state['total_tasks']}")
        print("---")

        return state

    def should_continue(state: AgentState) -> str:
        """Determine if the agent should continue or end."""
        if len(state["completed_tasks"]) >= state["total_tasks"]:
            return END
        return "agent"

    # Create the graph
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_conditional_edges("agent", should_continue)
    workflow.set_entry_point("agent")

    return workflow.compile()

def run_agent_example():
    """Run the agent example."""

    # Create the agent
    agent = create_agent_with_rewards()

    # Initial state
    initial_state = {
        "messages": [],
        "current_task": "analyze_data",
        "completed_tasks": [],
        "total_tasks": 3,
        "reward_history": []
    }

    # Run the agent
    print("Starting agent with reward functions...")
    print("=" * 50)

    result = agent.invoke(initial_state)

    print("=" * 50)
    print("Final Results:")
    print(f"Completed Tasks: {result['completed_tasks']}")
    print(f"Total Reward: {sum(result['reward_history']):.3f}")
    print(f"Average Reward: {sum(result['reward_history']) / len(result['reward_history']):.3f}")
    print(f"Reward History: {[f'{r:.3f}' for r in result['reward_history']]}")

if __name__ == "__main__":
    run_agent_example()
```

## Features

- Modular reward function system
- Built-in completion and relevance rewards
- Composite reward functions
- Reward evaluation metrics
- Seamless LangGraph integration
- Real-time reward calculation and tracking

## Examples

- **Basic Usage**: See `examples/basic_usage.py` for simple reward function examples
- **LangGraph Integration**: See `examples/langgraph_integration.py` for full agent workflows
- **Tests**: Run `python -m pytest tests/` to verify functionality

## Documentation

Explore the full API in the `src/langgraph_rewards/` directory.

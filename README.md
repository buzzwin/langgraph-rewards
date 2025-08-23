# LangGraph Rewards

A framework for building reward functions in LangGraph agents.

## Installation

```bash
pip install -e .
```

## Usage

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

## Features

- Modular reward function system
- Built-in completion and relevance rewards
- Composite reward functions
- Reward evaluation metrics

# Reward Function Design: How They Work Without Knowing Agent Actions

## Overview

One of the most powerful aspects of the LangGraph Rewards framework is that **reward functions don't need to know what the agent does specifically** - they work by evaluating **outcomes and patterns** rather than understanding the action itself. This document explains how this works and why it's beneficial.

## The Core Principle: Outcome-Based Evaluation

Reward functions evaluate **what happened** rather than **what was done**. This means:

- ✅ **What matters**: Did the task succeed? Was the customer satisfied? How long did it take?
- ❌ **What doesn't matter**: What specific algorithm was used? What words were said? What tools were employed?

## How It Works in Practice

### 1. **The Agent Provides Its Own Evaluation**

The agent determines the quality of its work and provides this information in the reward context:

```python
def agent_node(state: AgentState) -> AgentState:
    # Process the task (the agent knows what it's doing)
    if current_task == "analyze_data":
        # The AGENT decides if it succeeded
        completion_status = "success"
        relevance_score = 0.9
    else:
        completion_status = "partial"
        relevance_score = 0.6

    # Create reward context with the agent's self-assessment
    context = RewardContext(
        agent_state={
            "completed": len(state["completed_tasks"]) == state["total_tasks"],
            "completed_steps": len(state["completed_tasks"]),
            "total_steps": state["total_tasks"],
        },
        action=current_task,
        result={"status": completion_status, "score": relevance_score},
        metadata={
            "completion_status": completion_status,  # Agent's self-rating
            "relevance_score": relevance_score,     # Agent's self-rating
        }
    )
```

### 2. **Reward Functions Evaluate Outcomes, Not Actions**

The `CompletionReward` function doesn't care about the specific action - it only cares about success indicators:

```python
class CompletionReward(ScalarRewardFunction):
    def _calculate_raw_reward(self, context: RewardContext) -> float:
        # Check if the task is marked as completed
        if "completed" in context.agent_state:
            if context.agent_state["completed"]:
                return 1.0  # Success!

        # Check for completion indicators in metadata
        if "completion_status" in context.metadata:
            status = context.metadata["completion_status"]
            if status == "success":
                return 1.0  # Success!
            elif status == "partial":
                return 0.5  # Partial success
            elif status == "failed":
                return 0.0  # Failure

        return 0.0
```

## Three Ways to Provide Evaluation Data

### **A. Agent Self-Assessment**

The agent evaluates its own work and provides scores:

```python
metadata = {
    "completion_status": "success",  # Agent says: "I succeeded"
    "relevance_score": 0.9,         # Agent says: "I was 90% relevant"
    "confidence": 0.85,             # Agent says: "I'm 85% confident"
    "quality_assessment": "high"    # Agent says: "This is high quality work"
}
```

### **B. External Validation**

External systems or humans provide evaluation:

```python
metadata = {
    "human_rating": 4.2,                # Human says: "4.2/5 stars"
    "automated_test_passed": True,      # Test suite says: "All tests pass"
    "customer_satisfaction": 0.8,       # Survey says: "80% satisfied"
    "peer_review_score": 0.9,          # Colleague says: "9/10"
    "business_impact_score": 0.75      # Metrics say: "75% improvement"
}
```

### **C. Observable Outcomes**

The system measures what actually happened:

```python
agent_state = {
    "completed": True,              # System observes: task is done
    "time_taken": 45,               # System measures: took 45 seconds
    "resources_used": 0.3,          # System tracks: used 30% resources
    "errors_encountered": 0,        # System counts: no errors
    "customer_wait_time": 120,      # System measures: customer waited 2 min
    "escalation_count": 0           # System tracks: no escalations needed
}
```

## Real-World Examples

### **Customer Service Agent**

```python
# The agent doesn't know it's handling a "refund request"
# It just knows it's processing some customer interaction

def customer_service_agent():
    # ... agent logic ...

    # After the interaction, the agent provides context
    context = RewardContext(
        agent_state={
            "customer_satisfied": True,        # Agent observed: customer smiled
            "issue_resolved": True,            # Agent observed: customer said "thank you"
            "time_spent": 180                 # System measured: 3 minutes
        },
        action="customer_interaction",
        result={"status": "completed"},
        metadata={
            "customer_rating": 5.0,           # Customer gave 5 stars
            "resolution_time": "under_5min",  # Business rule: under 5 min = good
            "escalation_needed": False        # Agent determined: no escalation
        }
    )

    # Reward functions evaluate the OUTCOMES, not the specific actions
    reward = completion_reward(context)  # High reward because issue resolved
    reward += efficiency_reward(context) # High reward because under 5 min
    reward += satisfaction_reward(context) # High reward because 5 stars
```

### **Data Analysis Agent**

```python
# The agent doesn't know it's analyzing "sales data"
# It just knows it's processing some dataset

def data_analysis_agent():
    # ... agent logic ...

    context = RewardContext(
        agent_state={
            "analysis_complete": True,         # Agent determined: analysis finished
            "data_quality_score": 0.85,       # Agent assessed: data is 85% clean
            "insights_generated": 5            # Agent counted: found 5 insights
        },
        action="data_analysis",
        result={"status": "completed"},
        metadata={
            "accuracy_score": 0.92,           # Validation test: 92% accurate
            "completeness": 0.98,             # Coverage: 98% of data analyzed
            "business_value": "high"          # Stakeholder: "This is valuable"
        }
    )

    # Same reward functions work for completely different tasks
    reward = completion_reward(context)  # High reward because analysis complete
    reward += quality_reward(context)    # High reward because 92% accurate
    reward += value_reward(context)      # High reward because high business value
```

## Key Benefits of This Approach

### 1. **Action-Agnostic**

- Reward functions work with any type of agent or task
- No need to modify reward functions when adding new agent types
- Same evaluation criteria can apply across different domains

### 2. **Self-Correcting**

- Agents learn to provide honest self-assessments
- Poor self-assessment leads to poor rewards, encouraging accuracy
- Creates a feedback loop for better self-evaluation

### 3. **Flexible**

- Can handle completely new types of tasks without code changes
- Easy to add new evaluation criteria
- Supports both simple and complex reward structures

### 4. **Auditable**

- Clear record of what was evaluated and why
- Transparent reward calculation process
- Easy to debug and optimize reward functions

### 5. **Scalable**

- Same reward functions work across different domains
- Easy to replicate successful reward patterns
- Supports multi-agent systems with different specializations

## Best Practices

### 1. **Design for Outcomes, Not Actions**

```python
# ❌ Don't do this - too specific to actions
def action_specific_reward(context):
    if context.action == "send_email":
        return 1.0
    elif context.action == "make_call":
        return 0.8
    # What if agent uses a new action type?

# ✅ Do this - focus on outcomes
def outcome_based_reward(context):
    if context.metadata.get("customer_satisfied"):
        return 1.0
    elif context.metadata.get("issue_resolved"):
        return 0.8
    # Works with any action that achieves these outcomes
```

### 2. **Use Multiple Data Sources**

```python
# Combine agent self-assessment with external validation
context = RewardContext(
    agent_state=agent_self_assessment,
    metadata={
        **agent_self_assessment,
        **external_validation,
        **system_measurements
    }
)
```

### 3. **Normalize Your Metrics**

```python
# Ensure all reward functions return values in the same range
class NormalizedReward(ScalarRewardFunction):
    min_value: float = 0.0
    max_value: float = 1.0

    def normalize_reward(self, value: float) -> float:
        return np.clip((value - self.min_value) / (self.max_value - self.min_value), 0.0, 1.0)
```

## Conclusion

The beauty of this approach is that you can take a reward function designed for one type of agent (like a customer service bot) and use it with a completely different agent (like a data analysis bot) - as long as both agents provide the same types of outcome data in their context.

This makes the framework incredibly powerful for building diverse AI systems that can all benefit from the same reward learning principles, while maintaining the flexibility to handle completely different types of tasks and domains.

## Related Documentation

- [Basic Usage Guide](basic_usage.md)
- [LangGraph Integration](langgraph_integration.md)
- [Custom Reward Functions](custom_rewards.md)
- [Reward Evaluation Metrics](evaluation_metrics.md)

# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Data models for the Supply Chain Environment (v4).

Defines typed Action, Observation, State, and Rubrics following the OpenEnv spec
and RFC 004 (rubric system). All are exported from supply_chain_env.__init__.

Rubric usage:
    from supply_chain_env.models import SupplyChainRubric, ExactGoalRubric

    # Default rubric (layered outcome + process)
    rubric = SupplyChainRubric()

    # Inject at reset to get rubric_reward alongside raw reward:
    obs = env.reset(task_id=12)
    result = env.step(action)
    print(result.reward)         # raw layered reward (0.0–1.30)
    print(result.rubric_reward)  # rubric score (0.0–1.0, normalised)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.types import Action, Observation
except ImportError:
    # Graceful fallback so models can be imported without openenv-core installed
    Action = BaseModel       # type: ignore[assignment,misc]
    Observation = BaseModel  # type: ignore[assignment,misc]


# ── Action ────────────────────────────────────────────────────────────────────

class SupplyChainAction(Action):
    """
    Action for the Supply Chain Environment.

    The agent picks exactly one tool per step and provides arguments.
    Unknown tools return an error observation, not a crash.
    """

    tool: str = Field(
        ...,
        description=(
            "Tool to call. One of: "
            "get_inventory, check_supplier_status, get_demand_forecast, "
            "place_order, reroute_shipment, cancel_shipment, get_pending_shipments, "
            "get_market_prices, get_quality_report, get_competing_bids"
        ),
        examples=["get_inventory", "place_order", "reroute_shipment"],
    )
    args: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Arguments for the tool as a flat key-value dict. "
            "get_inventory -> {}, "
            "check_supplier_status -> {'supplier_name': 'SupplierA'}, "
            "place_order -> {'supplier_name': 'SupplierA', 'product': 'bottled_water', 'quantity': 200}, "
            "reroute_shipment -> {'shipment_id': 'SHP-001', 'new_supplier': 'SupplierB'}, "
            "cancel_shipment -> {'shipment_id': 'SHP-001'}, "
            "get_demand_forecast -> {'product': 'bottled_water'}"
        ),
    )


# ── State ─────────────────────────────────────────────────────────────────────

class SupplyChainState(BaseModel):
    """
    Typed Pydantic model for the full episode state.

    Always present in observation.state. Can be constructed directly:
        state = SupplyChainState(**obs.state)

    All 13 standard fields are always present regardless of task type.
    competing_bids_countdown is additionally present on competing_buyer tasks.
    """

    task_id: int = Field(description="Task ID used at reset.")
    task_type: str = Field(description=(
        "One of: reorder, reroute, demand_spike, price_negotiation, "
        "multi_product_crisis, port_strike, quality_control, competing_buyer."
    ))
    difficulty: str = Field(description="One of: easy, medium, hard.")
    goal_description: str = Field(description="What the agent needs to accomplish.")
    inventory: Dict[str, int] = Field(description="Current stock levels keyed by product name.")
    steps: int = Field(description="Steps taken so far this episode.")
    max_steps: int = Field(default=25, description="Maximum steps allowed per episode.")
    orders_placed: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="All orders placed. Each has: supplier, product, quantity."
    )
    shipments_rerouted: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="All reroutes done. Each has: shipment_id, from_supplier, to_supplier."
    )
    shipments_cancelled: List[str] = Field(
        default_factory=list,
        description="Shipment IDs cancelled this episode."
    )
    pending_shipments: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="In-transit shipments not yet cancelled or received."
    )
    spent_budget: float = Field(default=0.0, description="Total cost of all orders placed.")
    remaining_budget: Optional[float] = Field(
        default=None,
        description="Budget remaining. Present only on tasks with a budget constraint."
    )
    competing_bids_countdown: Optional[Dict[str, int]] = Field(
        default=None,
        description="Steps until competitor locks supplier capacity. competing_buyer tasks only."
    )


# ── Observation ───────────────────────────────────────────────────────────────

class SupplyChainObservation(Observation):
    """
    Observation returned after each reset() and step().

    Contains a human-readable tool result, the full environment state dict,
    the raw layered reward, the RFC-004 rubric reward, and done flag.
    """

    text: str = Field(
        default="",
        description="Human-readable result of the last tool call.",
    )
    state: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Full environment state. Always contains all 13 standard keys. "
            "See SupplyChainState for the complete field listing."
        ),
    )
    done: bool = Field(
        default=False,
        description="True when reward >= 1.0 or max_steps (25) is reached.",
    )
    reward: float = Field(
        default=0.0,
        description=(
            "Raw layered reward. Range 0.0–1.30. "
            "0.10 (any action), 0.50 (order/reroute), 0.75 (sub-task), "
            "1.00 (all goals met), +0.15 step efficiency, +0.10 budget efficiency. "
            "Spam penalty applies for >5 consecutive identical tool calls."
        ),
    )
    rubric_reward: Optional[float] = Field(
        default=None,
        description=(
            "RFC-004 rubric reward. Range 0.0–1.0 (normalised). "
            "Populated when a SupplyChainRubric is active. "
            "On non-terminal steps: process reward (0.0 success, -0.05 per error). "
            "On terminal step: outcome reward (1.0 all goals met, partial credit otherwise). "
            "Use this for GRPO/RL training — it is more comparable across tasks than raw reward."
        ),
    )


# ── RFC 004 Rubric system ─────────────────────────────────────────────────────

class ExactGoalRubric:
    """
    Binary outcome rubric — 1.0 if all task goals are met, 0.0 otherwise.

    Equivalent to REPL's ExactMatchRubric. Use when you want clean binary
    signal for tasks with unambiguous success criteria (reorder, reroute).
    """

    def score(self, final_reward: float, expected: Any = None) -> float:
        return 1.0 if final_reward >= 1.0 else 0.0


class PartialCreditRubric:
    """
    Partial credit rubric — normalises raw reward to [0.0, 1.0].

    Equivalent to REPL's FuzzyMatchRubric. Use for hard multi-goal tasks
    where the agent deserves credit for completing sub-goals even if the
    episode ends before all goals are met.
    """

    def __init__(self, reward_max: float = 1.30):
        self.reward_max = reward_max

    def score(self, final_reward: float, expected: Any = None) -> float:
        return round(min(max(final_reward / self.reward_max, 0.0), 1.0), 4)


class ProcessRubric:
    """
    Per-step process rubric — penalises errors, rewards efficient actions.

    Equivalent to REPL's CodeExecutionRubric. Returns:
      -0.05  tool returned an error / unknown tool
       0.00  any valid tool call
      +0.05  action that directly advances the goal (order placed, reroute done)

    Use this to shape intermediate steps in RL training.
    """

    def score(self, obs_text: str, action_advanced_goal: bool = False) -> float:
        text_lower = obs_text.lower()
        if any(k in text_lower for k in ("error", "rejected", "unknown tool", "invalid")):
            return -0.05
        if action_advanced_goal or "success" in text_lower:
            return 0.05
        return 0.0


class CustomMetricRubric:
    """
    User-provided scoring function rubric.

    Equivalent to REPL's CustomMetricRubric. Wrap any callable:
        def my_metric(reward, expected):
            return 1.0 if reward >= 1.0 and expected == "SupplierB" else 0.0

        rubric = SupplyChainRubric(outcome=CustomMetricRubric(my_metric))
    """

    def __init__(self, metric_fn: Callable[[float, Any], float]):
        self.metric_fn = metric_fn

    def score(self, final_reward: float, expected: Any = None) -> float:
        return float(self.metric_fn(final_reward, expected))


class SupplyChainRubric:
    """
    Composite RFC-004 rubric for the Supply Chain Environment.

    Combines an outcome rubric (terminal steps) and a process rubric
    (non-terminal steps) into a single rubric_reward signal.

    This is the rubric returned in observation.rubric_reward and is what
    judges use to compare reward quality across submissions.

    Usage:
        rubric = SupplyChainRubric()                        # defaults
        rubric = SupplyChainRubric(outcome=ExactGoalRubric())  # binary only
        rubric = SupplyChainRubric(
            outcome=PartialCreditRubric(reward_max=1.30),
            process=ProcessRubric(),
        )

    In inference.py / training loops:
        rubric = SupplyChainRubric()
        for step in episode:
            rubric_reward = rubric.score(
                obs_text=result.text,
                raw_reward=result.reward,
                done=result.done,
                action_advanced_goal="SUCCESS" in result.text,
            )
            # use rubric_reward for GRPO loss

    Reward semantics:
        Terminal step (done=True):
            Uses outcome rubric — PartialCreditRubric by default.
            Range: 0.0–1.0.
        Non-terminal step (done=False):
            Uses process rubric — ProcessRubric by default.
            Range: -0.05–0.05.
        Failure (max steps exhausted without done):
            Returns failure_reward (default -0.10).
    """

    def __init__(
        self,
        outcome: Any = None,
        process: Any = None,
        failure_reward: float = -0.10,
        gamma: float = 1.0,
    ):
        self.outcome = outcome or PartialCreditRubric(reward_max=1.30)
        self.process = process or ProcessRubric()
        self.failure_reward = failure_reward
        self.gamma = gamma          # temporal discount for multi-step credit assignment

    def score(
        self,
        obs_text: str,
        raw_reward: float,
        done: bool,
        action_advanced_goal: bool = False,
        expected: Any = None,
        step: int = 0,
        max_steps: int = 25,
    ) -> float:
        """
        Compute rubric reward for one step.

        Args:
            obs_text:            Observation text from the environment.
            raw_reward:          Raw reward returned by the environment.
            done:                Whether the episode is finished.
            action_advanced_goal: True if the action directly moved toward the goal.
            expected:            Optional expected outcome for custom rubrics.
            step:                Current step number (used for gamma discounting).
            max_steps:           Max steps (used to detect timeout).

        Returns:
            float: rubric_reward in range [-0.10, 1.0].
        """
        if done:
            score = self.outcome.score(raw_reward, expected)
            if self.gamma < 1.0:
                score *= (self.gamma ** step)
            return round(score, 4)

        if step >= max_steps:
            return self.failure_reward

        return round(self.process.score(obs_text, action_advanced_goal), 4)
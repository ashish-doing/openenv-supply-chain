# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Data models for the Supply Chain Environment (v4).

Defines typed Action, Observation, and State using OpenEnv base classes.
All three are required by the OpenEnv spec and used by create_app().
"""

from typing import Any, Dict, List, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class SupplyChainAction(Action):
    """
    Action for the Supply Chain Environment.

    The agent picks exactly one tool per step and provides arguments.
    The environment validates the tool name and argument types before
    executing — unknown tools return an error observation, not a crash.
    """

    tool: str = Field(
        ...,
        description=(
            "Tool to call. One of: "
            "get_inventory, "
            "check_supplier_status, "
            "get_demand_forecast, "
            "place_order, "
            "reroute_shipment, "
            "cancel_shipment, "
            "get_pending_shipments, "
            "get_market_prices, "
            "get_quality_report, "
            "get_competing_bids"
        ),
        examples=["get_inventory", "place_order", "reroute_shipment"],
    )
    args: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Arguments for the tool as a flat key-value dict. "
            "Examples: "
            "get_inventory -> {}, "
            "check_supplier_status -> {'supplier_name': 'SupplierA'}, "
            "place_order -> {'supplier_name': 'SupplierA', 'product': 'bottled_water', 'quantity': 200}, "
            "reroute_shipment -> {'shipment_id': 'SHP-001', 'new_supplier': 'SupplierB'}, "
            "cancel_shipment -> {'shipment_id': 'SHP-001'}, "
            "get_demand_forecast -> {'product': 'bottled_water'}"
        ),
    )


class SupplyChainObservation(Observation):
    """
    Observation returned after each step.

    Contains a human-readable tool result, the full environment state dict,
    the current reward, and whether the episode is finished.

    The state dict always contains all 13 standard keys regardless of task type.
    See SupplyChainState for the full field listing.
    """

    text: str = Field(
        default="",
        description="Human-readable result of the last tool call.",
    )
    state: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Full environment state. Always contains: "
            "task_id, task_type, difficulty, goal_description, "
            "inventory, steps, max_steps, "
            "orders_placed, shipments_rerouted, shipments_cancelled, "
            "pending_shipments, spent_budget, remaining_budget. "
            "competing_bids_countdown is additionally present on competing_buyer tasks."
        ),
    )
    done: bool = Field(
        default=False,
        description="True when reward >= 1.0 or max_steps (25) is reached.",
    )
    reward: float = Field(
        default=0.0,
        description=(
            "Reward signal. Range 0.0–1.30. "
            "Layered: 0.10 (any action taken), "
            "0.50 (any order placed or shipment action), "
            "0.75 (key sub-task complete: reroute or cancel), "
            "1.00 (all task goals met), "
            "up to +0.15 step efficiency bonus (fewer steps = higher), "
            "up to +0.10 budget efficiency bonus (hard tasks only, budget remaining). "
            "Spam penalty applies for >5 consecutive identical tool calls."
        ),
    )


class SupplyChainState:
    """
    Typed reference for the state dict returned inside SupplyChainObservation.

    This is not a Pydantic model used by the server — it documents the
    exact shape of observation.state so training code can rely on it.

    All fields are always present. Optional fields are noted below.

    Fields
    ------
    task_id : int
        The task ID used in the reset() call.
    task_type : str
        One of: reorder, reroute, demand_spike, price_negotiation,
        multi_product_crisis, port_strike, quality_control, competing_buyer.
    difficulty : str
        One of: easy, medium, hard.
    goal_description : str
        First 120 chars of the task description — what the agent needs to do.
    inventory : dict[str, int]
        Current stock levels keyed by product name.
    steps : int
        Number of steps taken so far in this episode.
    max_steps : int
        Maximum steps allowed (always 25).
    orders_placed : list[dict]
        All orders placed this episode. Each has: supplier, product, quantity.
    shipments_rerouted : list[dict]
        All reroutes done. Each has: shipment_id, from_supplier, to_supplier.
    shipments_cancelled : list[str]
        Shipment IDs that have been cancelled.
    pending_shipments : list[dict]
        In-transit shipments not yet cancelled or received.
    spent_budget : float
        Total cost of all orders placed so far.
    remaining_budget : float
        Budget remaining (only on tasks with a budget constraint, else absent).
    competing_bids_countdown : dict[str, int]  (optional)
        Steps until competitor locks supplier capacity. Only on competing_buyer tasks.
    """

    task_id: int
    task_type: str
    difficulty: str
    goal_description: str
    inventory: Dict[str, int]
    steps: int
    max_steps: int
    orders_placed: List[Dict[str, Any]]
    shipments_rerouted: List[Dict[str, Any]]
    shipments_cancelled: List[str]
    pending_shipments: List[Dict[str, Any]]
    spent_budget: float
    remaining_budget: Optional[float]                  # present if task has budget
    competing_bids_countdown: Optional[Dict[str, int]] # present on competing_buyer only
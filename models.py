# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Data models for the Supply Chain Environment.
Defines typed Action and Observation using OpenEnv base classes.
"""

from typing import Any, Dict, List, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class SupplyChainAction(Action):
    """
    Action for the Supply Chain Environment.
    The agent picks a tool and provides arguments.
    """
    tool: str = Field(..., description=(
        "Tool to call. One of: get_inventory, check_supplier_status, "
        "get_demand_forecast, place_order, reroute_shipment, "
        "cancel_shipment, get_pending_shipments"
    ))
    args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for the tool, e.g. {'supplier_name': 'SupplierA'}"
    )


class SupplyChainObservation(Observation):
    """
    Observation returned after each step.
    Contains human-readable text and full environment state.
    """
    text: str = Field(default="", description="Human-readable result of the last action")
    state: Dict[str, Any] = Field(default_factory=dict, description="Full environment state")
    done: bool = Field(default=False, description="Whether the episode is finished")
    reward: float = Field(default=0.0, description="Reward signal between 0.0 and 1.0")
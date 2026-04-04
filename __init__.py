# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Supply Chain Disruption Environment for OpenEnv.

Quick start (async):
    from supply_chain_env import SupplyChainEnv, SupplyChainAction

    async with SupplyChainEnv(base_url="http://localhost:7860") as env:
        obs = await env.reset(task_id=0)
        obs = await env.step(SupplyChainAction(
            tool="place_order",
            args={"supplier_name": "SupplierA", "product": "bottled_water", "quantity": 200}
        ))
        print(obs.reward)

Quick start (sync):
    from supply_chain_env import SupplyChainEnv, SupplyChainAction

    with SupplyChainEnv(base_url="http://localhost:7860").sync() as env:
        obs = env.reset(task_id=0)
        obs = env.step(SupplyChainAction(tool="get_inventory", args={}))

Local (in-process, no server needed):
    from supply_chain_env import SupplyChainEnvironment, SupplyChainAction

    env = SupplyChainEnvironment()
    obs = env.reset(task_id=7)
    obs = env.step(SupplyChainAction(tool="get_market_prices", args={}))

Rubric usage (RFC 004):
    from supply_chain_env import SupplyChainRubric

    rubric = SupplyChainRubric()
    rubric_reward = rubric.score(
        obs_text=result.text,
        raw_reward=result.reward,
        done=result.done,
        action_advanced_goal="SUCCESS" in result.text,
    )
"""

# ── Remote client (HTTP + WebSocket) ──────────────────────────────────────────
try:
    from .client import SupplyChainEnv
except ImportError:
    SupplyChainEnv = None  # type: ignore[assignment,misc]

# ── Typed models + rubric system ──────────────────────────────────────────────
try:
    from .models import (
        SupplyChainAction,
        SupplyChainObservation,
        SupplyChainState,
        SupplyChainRubric,
        ExactGoalRubric,
        PartialCreditRubric,
        ProcessRubric,
        CustomMetricRubric,
    )
except ImportError:
    from models import (  # type: ignore[no-redef]
        SupplyChainAction,
        SupplyChainObservation,
        SupplyChainState,
        SupplyChainRubric,
        ExactGoalRubric,
        PartialCreditRubric,
        ProcessRubric,
        CustomMetricRubric,
    )

# ── Local in-process environment (no server required) ─────────────────────────
try:
    from .server.supply_chain_env_environment import SupplyChainEnvironment
except ImportError:
    try:
        from server.supply_chain_env_environment import SupplyChainEnvironment  # type: ignore[no-redef]
    except ImportError:
        SupplyChainEnvironment = None  # type: ignore[assignment,misc]

__version__ = "4.2.0"

__all__ = [
    "SupplyChainEnv",
    "SupplyChainAction",
    "SupplyChainObservation",
    "SupplyChainState",
    "SupplyChainRubric",
    "ExactGoalRubric",
    "PartialCreditRubric",
    "ProcessRubric",
    "CustomMetricRubric",
    "SupplyChainEnvironment",
    "__version__",
]
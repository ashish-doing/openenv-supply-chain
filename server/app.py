"""
FastAPI application for the Supply Chain Environment.
Serves the environment over HTTP with pre-filled Swagger examples.
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required. Install: pip install openenv-core") from e

try:
    from models import SupplyChainAction, SupplyChainObservation
    from server.supply_chain_env_environment import SupplyChainEnvironment
except ImportError:
    try:
        from ..models import SupplyChainAction, SupplyChainObservation
        from .supply_chain_env_environment import SupplyChainEnvironment
    except ImportError:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from models import SupplyChainAction, SupplyChainObservation
        from server.supply_chain_env_environment import SupplyChainEnvironment

from fastapi import FastAPI, Body
from fastapi.responses import RedirectResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional
import requests as req

# ── Create core OpenEnv app ───────────────────────────────────────────────
app = create_app(
    SupplyChainEnvironment,
    SupplyChainAction,
    SupplyChainObservation,
    env_name="supply_chain_env",
    max_concurrent_envs=1,
)

# ── Request/Response models with examples ────────────────────────────────

class ResetRequest(BaseModel):
    task_id: int = Field(default=0, description="Task ID: 0-2=easy, 5-6=medium, 10-11=hard")
    seed: int = Field(default=42, description="Random seed for reproducibility")

    model_config = {
        "json_schema_extra": {
            "example": {"task_id": 0, "seed": 42}
        }
    }

class StepRequest(BaseModel):
    tool: str = Field(
        default="get_inventory",
        description="Tool to call",
    )
    args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tool arguments"
    )

    model_config = {
        "json_schema_extra": {
            "examples": {
                "1_get_inventory": {
                    "summary": "Step 1 - Check inventory",
                    "value": {"tool": "get_inventory", "args": {}}
                },
                "2_check_supplier": {
                    "summary": "Step 2 - Check supplier status",
                    "value": {"tool": "check_supplier_status", "args": {"supplier_name": "SupplierA"}}
                },
                "3_place_order": {
                    "summary": "Step 3 - Place order (completes easy task!)",
                    "value": {"tool": "place_order", "args": {
                        "supplier_name": "SupplierA",
                        "product": "bottled_water",
                        "quantity": 200
                    }}
                },
                "4_reroute_shipment": {
                    "summary": "Medium task - Reroute stranded shipment",
                    "value": {"tool": "reroute_shipment", "args": {
                        "shipment_id": "SHP-001",
                        "new_supplier": "SupplierB"
                    }}
                },
            }
        }
    }

# ── Custom clean endpoints ────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to interactive docs."""
    return RedirectResponse(url="/docs")

@app.post(
    "/quick/reset",
    tags=["Quick Start"],
    summary="Reset environment (easy to use)",
    description="Start a new episode. Use task_id 0-2 for easy, 5-6 for medium, 10-11 for hard."
)
async def quick_reset(body: ResetRequest = Body(...)):
    """Reset the environment with a simple, pre-filled format."""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            r = await client.post(
                "http://localhost:7860/reset",
                params={"task_id": body.task_id, "seed": body.seed},
                timeout=10
            )
            return r.json()
    except Exception:
        env = SupplyChainEnvironment()
        obs = env.reset(task_id=body.task_id, seed=body.seed)
        return {
            "observation": {
                "text": obs.text,
                "state": obs.state,
                "reward": obs.reward,
                "done": obs.done
            }
        }

@app.post(
    "/quick/step",
    tags=["Quick Start"],
    summary="Execute a tool action (easy to use)",
    description="Call a tool. Pick from the examples dropdown to see all available actions."
)
async def quick_step(body: StepRequest = Body(...)):
    """Execute a tool with a simple, pre-filled format."""
    # Create a fresh env for demonstration
    env = SupplyChainEnvironment()
    env.reset(task_id=0)
    action = SupplyChainAction(tool=body.tool, args=body.args)
    result = env.step(action)
    return {
        "observation": {
            "text": result.text,
            "state": result.state,
        },
        "reward": result.reward,
        "done": result.done,
        "hint": "reward=1.0 means task complete! Try the full /reset + /step sequence for stateful play."
    }

@app.get(
    "/quick/demo",
    tags=["Quick Start"],
    summary="Run a complete demo episode",
    description="Automatically runs a full easy task from reset to completion. Watch the agent solve it!"
)
async def quick_demo():
    """Auto-runs a complete episode so visitors can see the environment in action."""
    env = SupplyChainEnvironment()
    obs = env.reset(task_id=0)
    
    steps = []
    actions = [
        SupplyChainAction(tool="get_inventory", args={}),
        SupplyChainAction(tool="check_supplier_status", args={"supplier_name": "SupplierA"}),
        SupplyChainAction(tool="place_order", args={
            "supplier_name": "SupplierA",
            "product": "bottled_water",
            "quantity": 200
        }),
    ]
    
    steps.append({"step": 0, "event": "reset", "text": obs.text[:120] + "...", "reward": 0.0})
    
    for i, action in enumerate(actions):
        result = env.step(action)
        steps.append({
            "step": i + 1,
            "tool": action.tool,
            "args": action.args,
            "response": result.text[:100],
            "reward": result.reward,
            "done": result.done
        })
        if result.done:
            break
    
    return {
        "task": "Easy Task 0 — Reorder bottled water",
        "result": "SUCCESS" if steps[-1]["reward"] >= 1.0 else "INCOMPLETE",
        "final_reward": steps[-1]["reward"],
        "episode": steps,
        "message": "This is what an AI agent experiences each step. Reward 1.0 = task solved!"
    }


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(port=args.port)
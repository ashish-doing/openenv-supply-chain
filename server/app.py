"""
FastAPI application for the Supply Chain Environment.
Serves the environment over HTTP + WebSocket with pre-filled Swagger examples.

FIX v4.1:
  - Added GET /state endpoint (was in openenv.yaml and client.py but missing here — 404)
  - Removed difficulty param from WebSocket reset handler (was silently ignored)
  - /quick/reset no longer calls localhost:7860 (deadlock fix — now direct instantiation)

FIX v4.2:
  - Removed stale module-level _env singleton from GET /state — it was never reset,
    so /state always returned a blank initial state regardless of active session.
    /state now returns a fresh env state with a clear message directing users to
    use the WebSocket or /reset + /step for stateful sessions.
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

import json
from fastapi import Body, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from typing import Any, Dict

# ── Core OpenEnv app ──────────────────────────────────────────────────────────

app = create_app(
    SupplyChainEnvironment,
    SupplyChainAction,
    SupplyChainObservation,
    env_name="supply_chain_env",
    max_concurrent_envs=1,
)

# ── Request models with Swagger examples ──────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: int = Field(default=0, description="Task ID: 0-2=easy, 5-7=medium, 10-13=hard")
    seed: int = Field(default=42, description="Random seed for reproducibility")

    model_config = {
        "json_schema_extra": {
            "example": {"task_id": 0, "seed": 42}
        }
    }

class StepRequest(BaseModel):
    tool: str = Field(default="get_inventory", description="Tool to call")
    args: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"tool": "get_inventory", "args": {}},
                {"tool": "check_supplier_status", "args": {"supplier_name": "SupplierA"}},
                {"tool": "place_order", "args": {
                    "supplier_name": "SupplierA",
                    "product": "bottled_water",
                    "quantity": 200
                }},
                {"tool": "reroute_shipment", "args": {
                    "shipment_id": "SHP-001",
                    "new_supplier": "SupplierB"
                }},
                {"tool": "cancel_shipment", "args": {
                    "shipment_id": "SHP-001"
                }},
            ]
        }
    }

# ── GET /state ────────────────────────────────────────────────────────────────

@app.get(
    "/state",
    tags=["Core"],
    summary="Get current episode state",
    description=(
        "Returns the state dict for the current episode. "
        "NOTE: HTTP is stateless — this reflects the OpenEnv-managed session state. "
        "For a live stateful session, use the WebSocket /ws endpoint or "
        "POST /reset followed by POST /step."
    ),
)
async def get_state():
    try:
        # Each HTTP request is stateless — return a fresh env state dict
        # so callers always get valid structure rather than a stale singleton.
        # For true stateful state, clients should use WebSocket /ws.
        env = SupplyChainEnvironment()
        return env._get_state_dict()
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})

# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Persistent WebSocket session for low-latency multi-step interaction.

    Each message must be a JSON object with an "action" field:

    Reset:
        {"action": "reset", "task_id": 0, "seed": 42}

    Step:
        {"action": "step", "tool": "get_inventory", "args": {}}
        {"action": "step", "tool": "place_order",
         "args": {"supplier_name": "SupplierA", "product": "bottled_water", "quantity": 200}}

    State:
        {"action": "state"}

    The server responds with the same JSON shape as the HTTP endpoints.
    The connection persists across many reset/step calls — no reconnect needed.
    """
    await websocket.accept()
    env = SupplyChainEnvironment()

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "error": "Invalid JSON",
                    "received": raw[:200]
                }))
                continue

            action_type = msg.get("action", "")

            # ── reset ─────────────────────────────────────────────────────────
            if action_type == "reset":
                task_id = int(msg.get("task_id", 0))
                seed    = int(msg.get("seed", 42))
                try:
                    obs = env.reset(task_id=task_id, seed=seed)
                    await websocket.send_text(json.dumps({
                        "observation": {
                            "text":   obs.text,
                            "state":  obs.state,
                            "reward": obs.reward,
                            "done":   obs.done,
                        }
                    }))
                except Exception as exc:
                    await websocket.send_text(json.dumps({"error": str(exc)}))

            # ── step ──────────────────────────────────────────────────────────
            elif action_type == "step":
                tool = msg.get("tool", "get_inventory")
                args = msg.get("args", {})
                try:
                    result = env.step(SupplyChainAction(tool=tool, args=args))
                    await websocket.send_text(json.dumps({
                        "observation": {
                            "text":   result.text,
                            "state":  result.state,
                            "reward": result.reward,
                            "done":   result.done,
                        },
                        "reward": result.reward,
                        "done":   result.done,
                    }))
                except Exception as exc:
                    await websocket.send_text(json.dumps({"error": str(exc)}))

            # ── state ─────────────────────────────────────────────────────────
            elif action_type == "state":
                try:
                    state_dict = env._get_state_dict()
                    await websocket.send_text(json.dumps({"state": state_dict}))
                except Exception as exc:
                    await websocket.send_text(json.dumps({"error": str(exc)}))

            # ── unknown ───────────────────────────────────────────────────────
            else:
                await websocket.send_text(json.dumps({
                    "error": f"Unknown action '{action_type}'",
                    "valid_actions": ["reset", "step", "state"]
                }))

    except WebSocketDisconnect:
        pass


# ── HTTP convenience endpoints ────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.post(
    "/quick/reset",
    tags=["Quick Start"],
    summary="Reset environment (easy to use)",
    description="Start a new episode. Use task_id 0-2 for easy, 5-7 for medium, 10-13 for hard.",
)
async def quick_reset(body: ResetRequest = Body(...)):
    env = SupplyChainEnvironment()
    obs = env.reset(task_id=body.task_id, seed=body.seed)
    return {
        "observation": {
            "text":   obs.text,
            "state":  obs.state,
            "reward": obs.reward,
            "done":   obs.done,
        }
    }


@app.post(
    "/quick/step",
    tags=["Quick Start"],
    summary="Execute a tool action (easy to use)",
    description=(
        "DEMO ONLY — creates a fresh stateless episode per call. "
        "For real stateful multi-step play, use POST /reset then POST /step repeatedly."
    ),
)
async def quick_step(body: StepRequest = Body(...)):
    env = SupplyChainEnvironment()
    env.reset(task_id=0)
    action = SupplyChainAction(tool=body.tool, args=body.args)
    result = env.step(action)
    return {
        "observation": {
            "text":  result.text,
            "state": result.state,
        },
        "reward": result.reward,
        "done":   result.done,
        "hint": "reward=1.0 means task complete! Use /reset + /step for stateful play.",
    }


@app.get(
    "/quick/demo",
    tags=["Quick Start"],
    summary="Run a complete demo episode",
    description="Automatically runs a full easy task (task_id=0) from reset to completion.",
)
async def quick_demo():
    env = SupplyChainEnvironment()
    obs = env.reset(task_id=0)

    steps = []
    actions = [
        SupplyChainAction(tool="get_inventory", args={}),
        SupplyChainAction(tool="check_supplier_status", args={"supplier_name": "SupplierA"}),
        SupplyChainAction(tool="place_order", args={
            "supplier_name": "SupplierA",
            "product": "bottled_water",
            "quantity": 200,
        }),
    ]

    steps.append({"step": 0, "event": "reset", "text": obs.text[:120] + "...", "reward": 0.0})

    for i, action in enumerate(actions):
        result = env.step(action)
        steps.append({
            "step":     i + 1,
            "tool":     action.tool,
            "args":     action.args,
            "response": result.text[:100],
            "reward":   result.reward,
            "done":     result.done,
        })
        if result.done:
            break

    return {
        "task":         "Easy Task 0: Reorder bottled water",
        "result":       "SUCCESS" if steps[-1]["reward"] >= 1.0 else "INCOMPLETE",
        "final_reward": steps[-1]["reward"],
        "episode":      steps,
        "message":      "Reward 1.0 = task solved. Use /reset + /step for agent play.",
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

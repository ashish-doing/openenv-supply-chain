"""
Supply Chain Environment — client.

Provides:
  SupplyChainEnv       async client (WebSocket-first, HTTP fallback)
  SyncSupplyChainEnv   synchronous wrapper — use via SupplyChainEnv(...).sync()

Usage (async):
    from supply_chain_env import SupplyChainEnv, SupplyChainAction

    async with SupplyChainEnv(base_url="http://localhost:7860") as env:
        obs = await env.reset(task_id=0)
        obs = await env.step(SupplyChainAction(
            tool="place_order",
            args={"supplier_name": "SupplierA", "product": "bottled_water", "quantity": 200}
        ))
        print(obs.reward)   # 1.15

Usage (sync):
    from supply_chain_env import SupplyChainEnv, SupplyChainAction

    with SupplyChainEnv(base_url="http://localhost:7860").sync() as env:
        obs = env.reset(task_id=0)
        obs = env.step(SupplyChainAction(tool="get_inventory", args={}))
        print(obs.reward)
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional

import requests

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

try:
    from .models import SupplyChainAction, SupplyChainObservation, SupplyChainState
except ImportError:
    from models import SupplyChainAction, SupplyChainObservation, SupplyChainState  # type: ignore[no-redef]


def _http_to_ws(url: str) -> str:
    """Convert http(s):// base URL to ws(s):// WebSocket URL."""
    return url.replace("https://", "wss://").replace("http://", "ws://")


def _parse_obs(data: dict) -> SupplyChainObservation:
    """Parse a raw response dict into a typed SupplyChainObservation."""
    obs = data.get("observation", data)
    if isinstance(obs, dict):
        return SupplyChainObservation(
            text=obs.get("text", ""),
            state=obs.get("state", {}),
            reward=float(obs.get("reward", data.get("reward", 0.0))),
            done=bool(obs.get("done", data.get("done", False))),
        )
    return SupplyChainObservation(
        text=str(obs),
        state={},
        reward=float(data.get("reward", 0.0)),
        done=bool(data.get("done", False)),
    )


# ── Async client ──────────────────────────────────────────────────────────────

class SupplyChainEnv:
    """
    Async OpenEnv client for the Supply Chain Environment.

    Connects over WebSocket for persistent low-latency sessions,
    with transparent HTTP fallback if websockets is not installed.

    Use as an async context manager:
        async with SupplyChainEnv(base_url="http://localhost:7860") as env:
            obs = await env.reset(task_id=0)

    Or call .sync() for a synchronous wrapper:
        with SupplyChainEnv(...).sync() as env:
            obs = env.reset(task_id=0)
    """

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self._ws_url = _http_to_ws(self.base_url) + "/ws"
        self._ws = None          # active WebSocket connection
        self._session_id: Optional[str] = None

    # ── Context manager ───────────────────────────────────────────────────────

    async def __aenter__(self) -> "SupplyChainEnv":
        await self.connect()
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    # ── Connection ────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Open WebSocket connection. Falls back silently if unavailable."""
        if not HAS_WEBSOCKETS:
            return
        try:
            self._ws = await websockets.connect(self._ws_url, open_timeout=5)
        except Exception:
            self._ws = None  # fall back to HTTP

    async def close(self) -> None:
        """Close the WebSocket connection."""
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    # ── Core API ──────────────────────────────────────────────────────────────

    async def reset(
        self,
        task_id: int = 0,
        seed: int = 42,
        difficulty: Optional[str] = None,
    ) -> SupplyChainObservation:
        """
        Start a new episode.

        Args:
            task_id:    Fixed task (0–13) or procedural (14+).
            seed:       Random seed for reproducibility.
            difficulty: Optional filter — "easy", "medium", or "hard".
                        Overrides task_id with a random task of that difficulty.
        """
        if self._ws is not None:
            return await self._ws_reset(task_id=task_id, seed=seed, difficulty=difficulty)
        return self._http_reset(task_id=task_id, seed=seed, difficulty=difficulty)

    async def step(self, action: SupplyChainAction) -> SupplyChainObservation:
        """
        Execute one tool action.

        Args:
            action: SupplyChainAction(tool="...", args={...})
        """
        if self._ws is not None:
            return await self._ws_step(action)
        return self._http_step(action)

    async def state(self) -> SupplyChainState:
        """Fetch the current episode state (all 13 fields)."""
        r = requests.get(f"{self.base_url}/state", timeout=10)
        r.raise_for_status()
        return SupplyChainState(**r.json())

    # ── WebSocket transport ───────────────────────────────────────────────────

    async def _ws_reset(self, **kwargs) -> SupplyChainObservation:
        payload = {"action": "reset", **kwargs}
        await self._ws.send(json.dumps(payload))
        raw = await self._ws.recv()
        return _parse_obs(json.loads(raw))

    async def _ws_step(self, action: SupplyChainAction) -> SupplyChainObservation:
        payload = {
            "action": "step",
            "tool": action.tool,
            "args": action.args,
        }
        await self._ws.send(json.dumps(payload))
        raw = await self._ws.recv()
        return _parse_obs(json.loads(raw))

    # ── HTTP transport (fallback) ─────────────────────────────────────────────

    def _http_reset(
        self,
        task_id: int = 0,
        seed: int = 42,
        difficulty: Optional[str] = None,
    ) -> SupplyChainObservation:
        params: Dict[str, Any] = {"task_id": task_id, "seed": seed}
        if difficulty:
            params["difficulty"] = difficulty
        r = requests.post(f"{self.base_url}/reset", params=params, timeout=15)
        r.raise_for_status()
        return _parse_obs(r.json())

    def _http_step(self, action: SupplyChainAction) -> SupplyChainObservation:
        r = requests.post(
            f"{self.base_url}/step",
            json={"tool": action.tool, "args": action.args},
            timeout=15,
        )
        r.raise_for_status()
        return _parse_obs(r.json())

    # ── Sync wrapper ──────────────────────────────────────────────────────────

    def sync(self) -> "SyncSupplyChainEnv":
        """Return a synchronous wrapper around this async client."""
        return SyncSupplyChainEnv(self)

    # ── Convenience ───────────────────────────────────────────────────────────

    def health(self) -> dict:
        """Check server health synchronously."""
        r = requests.get(f"{self.base_url}/health", timeout=10)
        r.raise_for_status()
        return r.json()


# ── Sync wrapper ──────────────────────────────────────────────────────────────

class SyncSupplyChainEnv:
    """
    Synchronous wrapper around SupplyChainEnv.

    Returned by SupplyChainEnv(...).sync(). Use as a context manager:

        with SupplyChainEnv(base_url="http://localhost:7860").sync() as env:
            obs = env.reset(task_id=0)
            obs = env.step(SupplyChainAction(tool="get_inventory", args={}))
    """

    def __init__(self, async_env: SupplyChainEnv):
        self._env = async_env
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def __enter__(self) -> "SyncSupplyChainEnv":
        self._loop = asyncio.new_event_loop()
        self._loop.run_until_complete(self._env.connect())
        return self

    def __exit__(self, *_) -> None:
        if self._loop:
            self._loop.run_until_complete(self._env.close())
            self._loop.close()

    def _run(self, coro):
        return self._loop.run_until_complete(coro)

    def reset(self, task_id: int = 0, seed: int = 42, difficulty: Optional[str] = None) -> SupplyChainObservation:
        return self._run(self._env.reset(task_id=task_id, seed=seed, difficulty=difficulty))

    def step(self, action: SupplyChainAction) -> SupplyChainObservation:
        return self._run(self._env.step(action))

    def state(self) -> SupplyChainState:
        return self._run(self._env.state())

    def health(self) -> dict:
        return self._env.health()
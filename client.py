"""Supply Chain Environment — HTTP client."""
from typing import Dict, Any
import requests


class SupplyChainEnv:
    """
    HTTP client for the Supply Chain Environment.

    Example:
        env = SupplyChainEnv("http://localhost:7860")
        obs = env.reset(task_id=0)
        result = env.step("place_order", {
            "supplier_name": "SupplierA",
            "product": "bottled_water",
            "quantity": 200
        })
        print(result["reward"])
    """

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict:
        r = requests.get(f"{self.base_url}/health", timeout=10)
        r.raise_for_status()
        return r.json()

    def reset(self, task_id: int = 0, seed: int = 42) -> dict:
        r = requests.post(
            f"{self.base_url}/reset",
            params={"task_id": task_id, "seed": seed},
            timeout=15,
        )
        r.raise_for_status()
        return r.json()

    def step(self, tool: str, args: Dict[str, Any] = None) -> dict:
        r = requests.post(
            f"{self.base_url}/step",
            json={"tool": tool, "args": args or {}},
            timeout=15,
        )
        r.raise_for_status()
        return r.json()

    def state(self) -> dict:
        r = requests.get(f"{self.base_url}/state", timeout=10)
        r.raise_for_status()
        return r.json()
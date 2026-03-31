"""
FastAPI application for the Supply Chain Env Environment.
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: pip install openenv-core"
    ) from e

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

# ── Create app FIRST ──────────────────────────────────────────────────────
app = create_app(
    SupplyChainEnvironment,
    SupplyChainAction,
    SupplyChainObservation,
    env_name="supply_chain_env",
    max_concurrent_envs=1,
)

# ── THEN add custom routes ────────────────────────────────────────────────
from fastapi.responses import RedirectResponse

@app.get("/")
async def root():
    """Redirect root URL to interactive API docs."""
    return RedirectResponse(url="/docs")


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(port=args.port)
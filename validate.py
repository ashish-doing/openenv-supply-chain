"""
validate.py — Pre-submission validator
Run: python validate.py
"""
import os, sys

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
results = []

def check(label, ok, detail=""):
    sym = "OK  " if ok else "FAIL"
    print(f"  [{sym}] {label}" + (f"  ->  {detail}" if detail else ""))
    results.append(ok)
    return ok

print("=" * 50)
print("  Supply Chain Env — Pre-submission Validator")
print("=" * 50)

# 1. Required files exist
print("\n[1] Required files")
check("openenv.yaml exists",  os.path.exists("openenv.yaml"))
check("Dockerfile exists",    os.path.exists("Dockerfile"))
check("inference.py exists",  os.path.exists("inference.py"))
check("README.md exists",     os.path.exists("README.md"))
check("models.py exists",     os.path.exists("models.py"))
check("server/app.py exists", os.path.exists(os.path.join("server","app.py")))
check("server/requirements.txt exists",
      os.path.exists(os.path.join("server","requirements.txt")))

# 2. Imports
print("\n[2] Imports")
try:
    from server.supply_chain_env_environment import SupplyChainEnvironment
    from models import SupplyChainAction
    check("Core imports work", True)
except Exception as e:
    check("Core imports work", False, str(e))
    print("\nCannot continue — fix imports first.")
    sys.exit(1)

# 3. reset()
print("\n[3] reset()")
env = SupplyChainEnvironment()
try:
    obs = env.reset(task_id=0)
    check("reset() returns observation",    hasattr(obs, "text"),  type(obs).__name__)
    check("obs.text is a string",           isinstance(obs.text, str))
    check("obs.state is a dict",            isinstance(obs.state, dict))
    check("obs.done is False after reset",  obs.done == False)
except Exception as e:
    check("reset() works", False, str(e))

# 4. step()
print("\n[4] step()")
try:
    env.reset(task_id=0)
    r = env.step(SupplyChainAction(tool="get_inventory", args={}))
    check("step() returns observation",   hasattr(r, "reward"))
    check("step() reward is float",       isinstance(r.reward, float), str(r.reward))
    check("step() reward in [0.0, 1.0]",  0.0 <= r.reward <= 1.0,     str(r.reward))
    check("step() done is bool",          isinstance(r.done, bool))
except Exception as e:
    check("step() works", False, str(e))

# 5. Reward logic
print("\n[5] Reward logic")
try:
    env.reset(task_id=0)
    r_partial = env.step(SupplyChainAction(tool="get_inventory", args={}))
    check("Partial reward > 0",        r_partial.reward > 0.0, str(r_partial.reward))
    check("Partial reward < 1",        r_partial.reward < 1.0, str(r_partial.reward))

    env.reset(task_id=0)
    r_full = env.step(SupplyChainAction(
        tool="place_order",
        args={"supplier_name": "SupplierA",
              "product": "bottled_water",
              "quantity": 200}
    ))
    check("Full reward (1.0) achievable", r_full.reward == 1.0, str(r_full.reward))
    check("Done=True when reward=1.0",    r_full.done == True)
except Exception as e:
    check("Reward logic", False, str(e))

# 6. All 3 difficulty levels
print("\n[6] Task difficulty levels")
try:
    env.reset(task_id=0);  easy = env.task["difficulty"]
    env.reset(task_id=5);  med  = env.task["difficulty"]
    env.reset(task_id=10); hard = env.task["difficulty"]
    check("Easy task (id=0) exists",   easy == "easy",   easy)
    check("Medium task (id=5) exists", med  == "medium", med)
    check("Hard task (id=10) exists",  hard == "hard",   hard)
except Exception as e:
    check("Difficulty levels", False, str(e))

# 7. Server ping (only if already running)
print("\n[7] Server (skip if not running)")
if HAS_REQUESTS:
    try:
        h = requests.get(f"{BASE_URL}/health", timeout=3)
        check("GET /health returns 200", h.status_code == 200, str(h.status_code))
        rr = requests.post(f"{BASE_URL}/reset", params={"task_id": 0}, timeout=5)
        check("POST /reset returns 200", rr.status_code == 200, str(rr.status_code))
        data = rr.json()
        obs_data = data.get("observation", data)
        check("reset response has 'text'", "text" in obs_data)
        check("reset response has 'state'", "state" in obs_data)
    except Exception:
        print("  [SKIP] Server not running — that is OK for now")
        print("         Start with: uvicorn server.app:app --host 0.0.0.0 --port 7860")
else:
    print("  [SKIP] requests not installed")

# Summary
passed = sum(results)
total  = len(results)
print(f"\n{'='*50}")
print(f"  Result: {passed}/{total} checks passed")
if passed == total:
    print("  STATUS: READY TO SUBMIT")
else:
    failed = total - passed
    print(f"  STATUS: {failed} issue(s) to fix")
print("=" * 50)
sys.exit(0 if passed == total else 1)
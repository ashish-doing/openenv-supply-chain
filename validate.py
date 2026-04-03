"""
validate.py — Pre-submission validator (v2)
Covers: original 7 tasks + 3 new task types + new tools.
Run: python validate.py
Expected: STATUS: READY TO SUBMIT
"""
import os, sys

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
results  = []

def check(label, ok, detail=""):
    sym = "OK  " if ok else "FAIL"
    print(f"  [{sym}] {label}" + (f" → {detail}" if detail else ""))
    results.append(ok)
    return ok

print("=" * 55)
print("  Supply Chain Env v2 — Pre-submission Validator")
print("=" * 55)

# ── 1. Required files ─────────────────────────────────────────────────────────
print("\n[1] Required files")
check("openenv.yaml exists",          os.path.exists("openenv.yaml"))
check("Dockerfile exists",            os.path.exists("Dockerfile"))
check("inference.py exists",          os.path.exists("inference.py"))
check("README.md exists",             os.path.exists("README.md"))
check("models.py exists",             os.path.exists("models.py"))
check("server/app.py exists",         os.path.exists(os.path.join("server", "app.py")))
check("server/requirements.txt exists",
      os.path.exists(os.path.join("server", "requirements.txt")))

# ── 2. Imports ────────────────────────────────────────────────────────────────
print("\n[2] Imports")
try:
    from server.supply_chain_env_environment import SupplyChainEnvironment
    from models import SupplyChainAction
    check("Core imports work", True)
except Exception as e:
    check("Core imports work", False, str(e))
    print("\nCannot continue — fix imports first.")
    sys.exit(1)

env = SupplyChainEnvironment()

# ── 3. reset() ────────────────────────────────────────────────────────────────
print("\n[3] reset()")
try:
    obs = env.reset(task_id=0)
    check("reset() returns observation",    hasattr(obs, "text"), type(obs).__name__)
    check("obs.text is a string",           isinstance(obs.text, str))
    check("obs.state is a dict",            isinstance(obs.state, dict))
    check("obs.done is False after reset",  obs.done == False)
    check("obs.state has 'steps' key",      "steps" in obs.state)
    check("obs.state has 'inventory' key",  "inventory" in obs.state)
except Exception as e:
    check("reset() works", False, str(e))

# ── 4. step() ─────────────────────────────────────────────────────────────────
print("\n[4] step()")
try:
    env.reset(task_id=0)
    r = env.step(SupplyChainAction(tool="get_inventory", args={}))
    check("step() returns observation",         hasattr(r, "reward"))
    check("step() reward is float",             isinstance(r.reward, float), str(r.reward))
    check("step() reward in [0.0, 1.15]",       0.0 <= r.reward <= 1.15, str(r.reward))
    check("step() done is bool",                isinstance(r.done, bool))
    check("step() state has max_steps",         "max_steps" in r.state)
    check("MAX_STEPS == 25",                    r.state.get("max_steps") == 25, str(r.state.get("max_steps")))
except Exception as e:
    check("step() works", False, str(e))

# ── 5. Reward logic ───────────────────────────────────────────────────────────
print("\n[5] Reward logic")
try:
    env.reset(task_id=0)
    r_partial = env.step(SupplyChainAction(tool="get_inventory", args={}))
    check("Partial reward > 0",   r_partial.reward > 0.0,  str(r_partial.reward))
    check("Partial reward < 1",   r_partial.reward < 1.0,  str(r_partial.reward))

    env.reset(task_id=0)
    r_full = env.step(SupplyChainAction(
        tool="place_order",
        args={"supplier_name": "SupplierA", "product": "bottled_water", "quantity": 200}
    ))
    check("Full reward (1.0) achievable",   r_full.reward >= 1.0, str(r_full.reward))
    check("Done=True when reward>=1.0",     r_full.done == True)
except Exception as e:
    check("Reward logic", False, str(e))

# ── 6. Hard task budget-efficiency bonus ──────────────────────────────────────
print("\n[6] Hard task budget-efficiency bonus")
try:
    env.reset(task_id=11)  # port strike task, budget=$5000
    # Place all goal orders efficiently
    env.step(SupplyChainAction(tool="cancel_shipment", args={"shipment_id": "SHP-011"}))
    env.step(SupplyChainAction(tool="place_order",
        args={"supplier_name": "SupplierC", "product": "engine_part", "quantity": 30}))
    r_eff = env.step(SupplyChainAction(tool="place_order",
        args={"supplier_name": "SupplierC", "product": "brake_pad", "quantity": 40}))
    check("Hard reward >= 1.0 (all goals met)", r_eff.reward >= 1.0, str(r_eff.reward))
    check("Hard reward can exceed 1.0 (efficiency bonus)", r_eff.reward > 1.0, str(r_eff.reward))
except Exception as e:
    check("Budget-efficiency bonus", False, str(e))

# ── 7. All 3 original difficulty levels ───────────────────────────────────────
print("\n[7] Original difficulty levels")
try:
    env.reset(task_id=0);  easy = env.task["difficulty"]
    env.reset(task_id=5);  med  = env.task["difficulty"]
    env.reset(task_id=10); hard = env.task["difficulty"]
    check("Easy task (id=0) exists",   easy == "easy",   easy)
    check("Medium task (id=5) exists", med  == "medium", med)
    check("Hard task (id=10) exists",  hard == "hard",   hard)
except Exception as e:
    check("Difficulty levels", False, str(e))

# ── 8. New task types ─────────────────────────────────────────────────────────
print("\n[8] New task types (v2)")
try:
    env.reset(task_id=7)
    check("Task 7: price_negotiation type",     env.task.get("type") == "price_negotiation",  env.task.get("type"))
    check("Task 7: has market_prices",          "market_prices" in env.task)

    env.reset(task_id=12)
    check("Task 12: quality_control type",      env.task.get("type") == "quality_control",    env.task.get("type"))
    check("Task 12: has quality_threshold",     "quality_threshold" in env.task)

    env.reset(task_id=13)
    check("Task 13: competing_buyer type",      env.task.get("type") == "competing_buyer",    env.task.get("type"))
    check("Task 13: has competing_bids",        "competing_bids" in env.task)
except Exception as e:
    check("New task types", False, str(e))

# ── 9. New tools ──────────────────────────────────────────────────────────────
print("\n[9] New tools (v2)")
try:
    env.reset(task_id=7)
    r = env.step(SupplyChainAction(tool="get_market_prices", args={}))
    check("get_market_prices works",    "Market price" in r.text or "supplier" in r.text.lower(), r.text[:60])

    env.reset(task_id=12)
    r = env.step(SupplyChainAction(tool="get_quality_report", args={}))
    check("get_quality_report works",   "Quality Report" in r.text or "defect" in r.text.lower(), r.text[:60])

    env.reset(task_id=13)
    r = env.step(SupplyChainAction(tool="get_competing_bids", args={}))
    check("get_competing_bids works",   "Competing buyer" in r.text or "competitor" in r.text.lower(), r.text[:60])
except Exception as e:
    check("New tools", False, str(e))

# ── 10. Quality control enforcement ───────────────────────────────────────────
print("\n[10] Quality gate enforcement")
try:
    env.reset(task_id=12)
    # Ordering from defective SupplierA should be REJECTED
    r = env.step(SupplyChainAction(
        tool="place_order",
        args={"supplier_name": "SupplierA", "product": "vaccine_vial", "quantity": 100}
    ))
    check("Defective supplier order rejected", "REJECTED" in r.text or "defect" in r.text.lower(), r.text[:80])
except Exception as e:
    check("Quality gate", False, str(e))

# ── 11. Competing buyer countdown ─────────────────────────────────────────────
print("\n[11] Competing buyer adversarial mechanic")
try:
    env.reset(task_id=13)
    initial_countdown = dict(env._competing_bids_remaining)
    env.step(SupplyChainAction(tool="get_inventory", args={}))
    after_countdown = dict(env._competing_bids_remaining)
    check(
        "Competing bid countdown ticks each step",
        any(after_countdown.get(p, 0) < initial_countdown.get(p, 0)
            for p in initial_countdown),
        f"{initial_countdown} → {after_countdown}"
    )
    check("State exposes countdown",
          "competing_bids_countdown" in env._get_state_dict())
except Exception as e:
    check("Competing buyer countdown", False, str(e))

# ── 12. Server ping ───────────────────────────────────────────────────────────
print("\n[12] Server (skip if not running)")
if HAS_REQUESTS:
    try:
        h = requests.get(f"{BASE_URL}/health", timeout=3)
        check("GET /health returns 200", h.status_code == 200, str(h.status_code))

        rr = requests.post(f"{BASE_URL}/reset", params={"task_id": 0}, timeout=5)
        check("POST /reset returns 200", rr.status_code == 200, str(rr.status_code))

        data = rr.json()
        obs_data = data.get("observation", data)
        check("reset response has 'text'",  "text"  in obs_data)
        check("reset response has 'state'", "state" in obs_data)

        # Test new task types via HTTP
        for tid, label in [(7, "price_negotiation"), (12, "quality_control"), (13, "competing_buyer")]:
            rr2 = requests.post(f"{BASE_URL}/reset", params={"task_id": tid}, timeout=5)
            check(f"POST /reset task_id={tid} ({label}) OK",
                  rr2.status_code == 200, str(rr2.status_code))
    except Exception:
        print("  [SKIP] Server not running — OK for now")
        print(f"  Start: uvicorn server.app:app --host 0.0.0.0 --port 7860")
else:
    print("  [SKIP] requests not installed")

# ── Summary ───────────────────────────────────────────────────────────────────
passed = sum(results)
total  = len(results)
print(f"\n{'='*55}")
print(f"  Result: {passed}/{total} checks passed")
if passed == total:
    print("  STATUS: READY TO SUBMIT ✓")
else:
    failed = total - passed
    print(f"  STATUS: {failed} issue(s) to fix")
print("=" * 55)
sys.exit(0 if passed == total else 1)
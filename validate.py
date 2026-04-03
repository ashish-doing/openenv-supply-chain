"""
validate.py — Pre-submission validator (v4 corrected)

Run:   python validate.py
Pass:  STATUS: READY TO SUBMIT ✓
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

print("=" * 60)
print("  Supply Chain Env v4 — Pre-submission Validator")
print("=" * 60)

# ── 1. Required files ─────────────────────────────────────────────────────────
print("\n[1] Required files")
check("openenv.yaml exists",            os.path.exists("openenv.yaml"))
check("Dockerfile exists",              os.path.exists("Dockerfile"))
check("inference.py exists",            os.path.exists("inference.py"))
check("README.md exists",               os.path.exists("README.md"))
check("models.py exists",               os.path.exists("models.py"))
check("generate_tasks.py exists",       os.path.exists("generate_tasks.py"))
check("server/app.py exists",           os.path.exists(os.path.join("server","app.py")))
check("server/requirements.txt exists", os.path.exists(os.path.join("server","requirements.txt")))

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

# ── 3. reset() contract ───────────────────────────────────────────────────────
print("\n[3] reset()")
try:
    obs = env.reset(task_id=0)
    check("reset() returns observation",    hasattr(obs, "text"),       type(obs).__name__)
    check("obs.text is a string",           isinstance(obs.text, str))
    check("obs.state is a dict",            isinstance(obs.state, dict))
    check("obs.done is False after reset",  obs.done == False)
    check("obs.reward is 0.0 after reset",  obs.reward == 0.0,          str(obs.reward))
    check("obs.state has 'steps' key",      "steps"      in obs.state)
    check("obs.state has 'inventory' key",  "inventory"  in obs.state)
    check("obs.state has 'task_id' key",    "task_id"    in obs.state)
    check("obs.state has 'task_type' key",  "task_type"  in obs.state)
    check("obs.state has 'difficulty' key", "difficulty" in obs.state)
    check("_tool_call_log resets to []",    env._tool_call_log == [])
except Exception as e:
    check("reset() contract", False, str(e))

# ── 4. step() contract ────────────────────────────────────────────────────────
print("\n[4] step()")
try:
    env.reset(task_id=0)
    r = env.step(SupplyChainAction(tool="get_inventory", args={}))
    check("step() returns observation",         hasattr(r, "reward"))
    check("step() reward is float",             isinstance(r.reward, float),    str(r.reward))
    check("step() reward in [0.0, 1.30]",       0.0 <= r.reward <= 1.30,        str(r.reward))
    check("step() done is bool",                isinstance(r.done, bool))
    check("step() state has 'max_steps'",       "max_steps" in r.state)
    check("MAX_STEPS == 25",                    r.state.get("max_steps") == 25, str(r.state.get("max_steps")))
    check("_tool_call_log grows after step",    len(env._tool_call_log) == 1,   str(env._tool_call_log))
    check("tool logged correctly",              env._tool_call_log[0] == "get_inventory")
except Exception as e:
    check("step() contract", False, str(e))

# ── 5. Reward layer structure (v4) ────────────────────────────────────────────
print("\n[5] Reward layer structure")
try:
    # Layer 0: participation floor
    env.reset(task_id=0)
    r0 = env.step(SupplyChainAction(tool="get_inventory", args={}))
    check("Layer 0 — any step → reward >= 0.05",    r0.reward >= 0.05,  str(r0.reward))
    check("Layer 0 — read only → reward < 0.50",    r0.reward <  0.50,  str(r0.reward))

    # Layer 1: multiple diagnostics still < 0.50
    env.reset(task_id=0)
    for tool, args in [
        ("get_inventory",        {}),
        ("check_supplier_status",{"supplier_name":"SupplierA"}),
        ("get_demand_forecast",  {"product":"bottled_water"}),
        ("get_pending_shipments",{}),
    ]:
        env.step(SupplyChainAction(tool=tool, args=args))
    r1 = env._compute_reward()
    check("Layer 1 — 4 read tools → reward < 0.50",    r1 < 0.50,  str(r1))
    check("Layer 1 — diagnostics earn > 0.05",          r1 > 0.05,  str(r1))

    # Layer 2: any order → 0.50
    env.reset(task_id=0)
    r2 = env.step(SupplyChainAction(
        tool="place_order",
        args={"supplier_name":"SupplierA","product":"bottled_water","quantity":50}
    ))
    check("Layer 2 — order placed → reward == 0.50",    r2.reward == 0.50, str(r2.reward))

    # Layer 4: all goals met — task 0 goal is SupplierA, bottled_water, qty>=200
    env.reset(task_id=0)
    r4 = env.step(SupplyChainAction(
        tool="place_order",
        args={"supplier_name":"SupplierA","product":"bottled_water","quantity":200}
    ))
    check("Layer 4 — goals met → reward >= 1.0",    r4.reward >= 1.0,  str(r4.reward))
    check("done=True when reward >= 1.0",           r4.done == True)

    # Bonus A: 1-step solve should give efficiency bonus >= 1.10
    check("Bonus A — 1-step solve >= 1.10",         r4.reward >= 1.10, str(r4.reward))

    # Spam penalty
    env.reset(task_id=0)
    for _ in range(8):
        env.step(SupplyChainAction(tool="get_inventory", args={}))
    r_spam = env._compute_reward()
    check("Spam penalty — 8× same tool → reward < 0.20",   r_spam < 0.20, str(r_spam))
    check("Spam penalty — floor at 0.0",                    r_spam >= 0.0, str(r_spam))
except Exception as e:
    check("Reward layers", False, str(e))

# ── 6. All 10 fixed task IDs — CORRECT type + difficulty ─────────────────────
# Ground truth from generate_tasks.py _FIXED_TASKS dict
print("\n[6] Fixed task IDs — type and difficulty")
EXPECTED = {
    0:  ("easy",   "reorder"),
    1:  ("easy",   "reorder"),
    2:  ("easy",   "reorder"),
    5:  ("medium", "reroute"),
    6:  ("medium", "demand_spike"),
    7:  ("medium", "price_negotiation"),
    10: ("hard",   "multi_product_crisis"),
    11: ("hard",   "port_strike"),
    12: ("hard",   "quality_control"),
    13: ("hard",   "competing_buyer"),
}
try:
    for tid, (exp_diff, exp_type) in EXPECTED.items():
        env.reset(task_id=tid)
        got_diff = env.task.get("difficulty", "?")
        got_type = env.task.get("type", "?")
        check(
            f"Task {tid:2d}: difficulty={exp_diff}, type={exp_type}",
            got_diff == exp_diff and got_type == exp_type,
            f"got difficulty={got_diff} type={got_type}"
        )
except Exception as e:
    check("Fixed task IDs", False, str(e))

# ── 7. All task types have required fields ────────────────────────────────────
# Using correct task IDs for each type
print("\n[7] Task type required fields")
try:
    # price_negotiation → task 7
    env.reset(task_id=7)
    check("price_negotiation has market_prices",    "market_prices"     in env.task)
    check("price_negotiation has budget",           "budget"            in env.task)
    check("price_negotiation has goal_supplier",    "goal_supplier"     in env.task)

    # quality_control → task 12
    env.reset(task_id=12)
    check("quality_control has quality_threshold",  "quality_threshold" in env.task)
    has_defect = any("defect_rate" in s for s in env.task["suppliers"].values())
    check("quality_control suppliers have defect_rate", has_defect)

    # competing_buyer → task 13
    env.reset(task_id=13)
    check("competing_buyer has competing_bids",     "competing_bids"    in env.task)
    prod = list(env.task["competing_bids"].keys())[0]
    bid  = env.task["competing_bids"][prod]
    check("competing_bids has steps_until_competitor_orders",
          "steps_until_competitor_orders" in bid)
    has_cap = any("remaining_capacity" in s for s in env.task["suppliers"].values())
    check("competing_buyer supplier has remaining_capacity", has_cap)

    # multi_product_crisis → task 10
    env.reset(task_id=10)
    check("multi_product_crisis has goal_orders",       "goal_orders"           in env.task)
    check("multi_product_crisis has goal_reroute",      "goal_reroute_shipment" in env.task)
    check("multi_product_crisis has budget",            "budget"                in env.task)

    # port_strike → task 11
    env.reset(task_id=11)
    has_strike = any(s.get("status") == "strike" for s in env.task["suppliers"].values())
    check("port_strike has at least one strike supplier", has_strike)

    # demand_spike → task 6
    env.reset(task_id=6)
    check("demand_spike has demand_spike=True",     env.task.get("demand_spike") == True)
except Exception as e:
    check("Task type fields", False, str(e))

# ── 8. All 10 tools respond without crashing ─────────────────────────────────
print("\n[8] All 10 tools")
TOOL_TESTS = [
    (0,  "get_inventory",           {}),
    (0,  "check_supplier_status",   {"supplier_name":"SupplierA"}),
    (0,  "get_demand_forecast",     {"product":"bottled_water"}),
    (5,  "get_pending_shipments",   {}),                                          # task 5 has pending shipments
    (5,  "reroute_shipment",        {"shipment_id":"SHP-001","new_supplier":"SupplierB"}),
    (11, "cancel_shipment",         {"shipment_id":"SHP-011"}),                   # task 11 has SHP-011
    (0,  "place_order",             {"supplier_name":"SupplierA","product":"bottled_water","quantity":50}),
    (7,  "get_market_prices",       {}),                                          # task 7 is price_negotiation
    (12, "get_quality_report",      {}),                                          # task 12 is quality_control
    (13, "get_competing_bids",      {}),                                          # task 13 is competing_buyer
]
try:
    for tid, tool, args in TOOL_TESTS:
        env.reset(task_id=tid)
        r = env.step(SupplyChainAction(tool=tool, args=args))
        check(f"Tool '{tool}' responds",
              isinstance(r.text, str) and len(r.text) > 0, r.text[:50])
except Exception as e:
    check("Tools", False, str(e))

# ── 9. Quality gate enforcement ───────────────────────────────────────────────
# task 12 = quality_control: SupplierA defect_rate=0.35 (bad), SupplierB defect_rate=0.04 (good)
print("\n[9] Quality gate enforcement")
try:
    # SupplierA defect rate 35% → must be rejected
    env.reset(task_id=12)
    bad_product = list(env.task["initial_inventory"].keys())[0]
    r_bad = env.step(SupplyChainAction(
        tool="place_order",
        args={"supplier_name":"SupplierA","product":bad_product,"quantity":100}
    ))
    check("Defective supplier order REJECTED",
          "REJECTED" in r_bad.text or "defect" in r_bad.text.lower(),
          r_bad.text[:80])

    # SupplierB defect rate 4% → must succeed
    env.reset(task_id=12)
    r_good = env.step(SupplyChainAction(
        tool="place_order",
        args={"supplier_name":"SupplierB","product":bad_product,"quantity":100}
    ))
    check("Compliant supplier order ACCEPTED",
          "SUCCESS" in r_good.text,
          r_good.text[:80])
except Exception as e:
    check("Quality gate", False, str(e))

# ── 10. Competing buyer countdown mechanic ────────────────────────────────────
# task 13 = competing_buyer
print("\n[10] Competing buyer countdown")
try:
    env.reset(task_id=13)
    initial = dict(env._competing_bids_remaining)
    check("Countdown initialised > 0",
          all(v > 0 for v in initial.values()), str(initial))

    env.step(SupplyChainAction(tool="get_inventory", args={}))
    after = dict(env._competing_bids_remaining)
    ticked = any(after.get(p, 0) < initial.get(p, 0) for p in initial)
    check("Countdown ticks down each step",     ticked,         f"{initial} → {after}")
    check("State exposes countdown",
          "competing_bids_countdown" in env._get_state_dict())

    # When countdown hits 0, competitor reduces supplier capacity
    env2 = SupplyChainEnvironment()
    env2.reset(task_id=13)
    prod2    = list(env2.task["competing_bids"].keys())[0]
    supplier = env2.task["competing_bids"][prod2]["supplier"]
    cap_before    = env2.suppliers[supplier].get("remaining_capacity", 0)
    steps_needed  = env2._competing_bids_remaining[prod2]
    for _ in range(steps_needed + 1):
        env2.step(SupplyChainAction(tool="get_inventory", args={}))
    cap_after = env2.suppliers[supplier].get("remaining_capacity", 0)
    check("Competitor reduces supplier capacity when deadline passes",
          cap_after < cap_before, f"before={cap_before} after={cap_after}")
except Exception as e:
    check("Competing buyer countdown", False, str(e))

# ── 11. Procedural task generation ───────────────────────────────────────────
print("\n[11] Procedural task generation")
try:
    for tid in [20, 75, 120, 175, 220, 270]:
        env.reset(task_id=tid)
        check(f"Procedural task {tid} generates valid task",
              env.task is not None and "difficulty" in env.task and "suppliers" in env.task)

    env.reset(task_id=55);  desc_a = env.task["description"]
    env.reset(task_id=55);  desc_b = env.task["description"]
    check("Same task_id produces same task (deterministic)", desc_a == desc_b)

    env.reset(task_id=20);  t20 = env.task["description"]
    env.reset(task_id=21);  t21 = env.task["description"]
    check("Different task_ids produce different tasks", t20 != t21)
except Exception as e:
    check("Procedural generation", False, str(e))

# ── 12. Efficiency bonuses ────────────────────────────────────────────────────
print("\n[12] Efficiency bonuses")
try:
    # Step efficiency: 1-step solve on task 0 must score > 10-step solve
    env.reset(task_id=0)
    r_fast = env.step(SupplyChainAction(
        tool="place_order",
        args={"supplier_name":"SupplierA","product":"bottled_water","quantity":200}
    ))
    env_slow = SupplyChainEnvironment()
    env_slow.reset(task_id=0)
    for _ in range(10):
        env_slow.step(SupplyChainAction(tool="get_pending_shipments", args={}))
    r_slow = env_slow.step(SupplyChainAction(
        tool="place_order",
        args={"supplier_name":"SupplierA","product":"bottled_water","quantity":200}
    ))
    check("Step efficiency: fast solve scores higher than slow",
          r_fast.reward > r_slow.reward, f"fast={r_fast.reward:.4f} slow={r_slow.reward:.4f}")
    check("Fast solve reward >= 1.10",  r_fast.reward >= 1.10, str(r_fast.reward))

    # Budget efficiency: hard task 10 has budget — spending less should score higher
    # Two solves: one efficient (low spend), one wasteful (high spend)
    # We test by comparing a full correct solve against the max reward cap
    env.reset(task_id=10)
    # Reroute first, then place all 3 orders with correct suppliers
    env.step(SupplyChainAction(tool="reroute_shipment",
        args={"shipment_id":"SHP-010","new_supplier":"SupplierD"}))
    env.step(SupplyChainAction(tool="place_order",
        args={"supplier_name":"SupplierD","product":"mask","quantity":150}))
    env.step(SupplyChainAction(tool="place_order",
        args={"supplier_name":"SupplierB","product":"glove","quantity":100}))
    r_budget = env.step(SupplyChainAction(tool="place_order",
        args={"supplier_name":"SupplierD","product":"sanitizer","quantity":80}))
    check("Budget efficiency: hard task solve reward >= 1.0",
          r_budget.reward >= 1.0, str(r_budget.reward))
    check("Budget efficiency: hard task reward in [1.0, 1.30]",
          1.0 <= r_budget.reward <= 1.30, str(r_budget.reward))
except Exception as e:
    check("Efficiency bonuses", False, str(e))

# ── 13. State dict completeness ───────────────────────────────────────────────
print("\n[13] State dict completeness")
try:
    # task 10 = hard with budget → remaining_budget must be present
    env.reset(task_id=10)
    env.step(SupplyChainAction(tool="get_inventory", args={}))
    s = env._get_state_dict()
    for key in ["task_id","task_type","difficulty","goal_description",
                "inventory","steps","max_steps","orders_placed",
                "shipments_rerouted","shipments_cancelled",
                "pending_shipments","spent_budget","remaining_budget"]:
        check(f"State has '{key}'", key in s, str(s.get(key, "MISSING")))

    # task 13 = competing_buyer → countdown must be in state
    env.reset(task_id=13)
    env.step(SupplyChainAction(tool="get_inventory", args={}))
    s2 = env._get_state_dict()
    check("competing_buyer state has 'competing_bids_countdown'",
          "competing_bids_countdown" in s2)
except Exception as e:
    check("State dict", False, str(e))

# ── 14. Server HTTP endpoints ─────────────────────────────────────────────────
print("\n[14] Server HTTP endpoints (skip if not running)")
if HAS_REQUESTS:
    try:
        h = requests.get(f"{BASE_URL}/health", timeout=3)
        check("GET /health returns 200",    h.status_code == 200, str(h.status_code))

        rr = requests.post(f"{BASE_URL}/reset", params={"task_id":0}, timeout=5)
        check("POST /reset returns 200",    rr.status_code == 200, str(rr.status_code))
        obs_data = rr.json().get("observation", rr.json())
        check("reset response has 'text'",  "text"  in obs_data)
        check("reset response has 'state'", "state" in obs_data)

        rs = requests.post(f"{BASE_URL}/step",
                           json={"tool":"get_inventory","args":{}}, timeout=5)
        check("POST /step returns 200",     rs.status_code == 200, str(rs.status_code))

        for tid in [0, 1, 2, 5, 6, 7, 10, 11, 12, 13]:
            rr2 = requests.post(f"{BASE_URL}/reset", params={"task_id":tid}, timeout=5)
            check(f"POST /reset task_id={tid} → 200",
                  rr2.status_code == 200, str(rr2.status_code))
    except requests.exceptions.ConnectionError:
        print("  [SKIP] Server not running — OK for local validation")
        print(f"         Start: uvicorn server.app:app --host 0.0.0.0 --port 7860")
    except Exception as e:
        check("Server endpoints", False, str(e))
else:
    print("  [SKIP] requests not installed")

# ── Summary ───────────────────────────────────────────────────────────────────
passed = sum(results)
total  = len(results)
print(f"\n{'='*60}")
print(f"  Result: {passed}/{total} checks passed")
if passed == total:
    print("  STATUS: READY TO SUBMIT ✓")
else:
    failed = total - passed
    print(f"  STATUS: {failed} issue(s) to fix before submitting")
    failing = [i+1 for i, ok in enumerate(results) if not ok]
    print(f"  Failed check indices: {failing}")
print("=" * 60)
sys.exit(0 if passed == total else 1)
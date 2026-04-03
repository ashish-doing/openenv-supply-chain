"""
inference.py — Supply Chain Environment Baseline Inference (v4)

Required env vars:
  API_BASE_URL   The API endpoint for the LLM (e.g. https://router.huggingface.co/v1)
  MODEL_NAME     The model identifier (e.g. Qwen/Qwen2.5-7B-Instruct)
  HF_TOKEN       Your Hugging Face / API key
  ENV_BASE_URL   (optional) Override env URL, default http://localhost:7860

Usage:
  python inference.py

Runs 10 tasks across easy / medium / hard / price_negotiation /
quality_control / competing_buyer and saves results.json.

JUDGE LOG FORMAT — stdout emits exactly:
  [START] task_id=<n>
  [STEP] step=<n> action=<json> observation=<json_str> reward=<float> done=<bool>
  [END] task_id=<n> total_reward=<float>
"""

import json
import os
import time

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

MAX_STEPS = 25

client = OpenAI(api_key=HF_TOKEN or "dummy", base_url=API_BASE_URL)

# ---------------------------------------------------------------------------
# System prompt — v4
# Improvements over v3:
#   1. Explicit decision tree (not prose bullets) — same sequence every episode
#   2. Task type detection from keywords in the observation text
#   3. Full argument schemas with ALL supplier/product name patterns
#   4. Failure recovery rules — what to do when a tool returns an error
#   5. Reward-signal awareness — explains WHY to read before acting
#   6. Hard constraint reminders — failed/strike suppliers, budget, quality
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a supply chain manager AI. Respond with ONLY a single JSON object — no markdown, no explanation, no text before or after.

OUTPUT FORMAT (strict):
{"tool": "tool_name", "args": {"key": "value"}}
For tools with no args: {"tool": "tool_name", "args": {}}

═══════════════════════════════════════════════
AVAILABLE TOOLS — exact argument schemas
═══════════════════════════════════════════════

{"tool": "get_inventory", "args": {}}
  → Shows all products, stock levels, days-until-stockout, threshold status.

{"tool": "check_supplier_status", "args": {"supplier_name": "SupplierA"}}
  → Valid supplier_name values: SupplierA, SupplierB, SupplierC, SupplierD
  → Shows status (healthy/failed/strike), lead_days, cost_per_unit, products.

{"tool": "get_demand_forecast", "args": {"product": "bottled_water"}}
  → Replace "bottled_water" with any product from your inventory.
  → Shows daily demand, days until stockout, demand spike warnings.

{"tool": "place_order", "args": {"supplier_name": "SupplierB", "product": "canned_soup", "quantity": 200}}
  → supplier_name: must be healthy (not failed/strike)
  → product: must match exact catalogue name from get_inventory
  → quantity: positive integer

{"tool": "reroute_shipment", "args": {"shipment_id": "SHP-001", "new_supplier": "SupplierB"}}
  → shipment_id: get exact ID from get_pending_shipments first
  → new_supplier: must be healthy (not failed/strike)

{"tool": "cancel_shipment", "args": {"shipment_id": "SHP-001"}}
  → shipment_id: get exact ID from get_pending_shipments first

{"tool": "get_pending_shipments", "args": {}}
  → Lists all in-flight shipments with their IDs. Call this before reroute/cancel.

{"tool": "get_market_prices", "args": {}}
  → Shows price/unit per supplier. Call this FIRST on price_negotiation tasks.

{"tool": "get_quality_report", "args": {}}
  → Shows defect rate per supplier vs threshold. Call this FIRST on quality_control tasks.

{"tool": "get_competing_bids", "args": {}}
  → Shows how many steps before a competitor locks up supplier capacity.
  → Call this IMMEDIATELY on competing_buyer tasks — every step costs you.

═══════════════════════════════════════════════
DECISION TREE — follow in order every episode
═══════════════════════════════════════════════

STEP 1 — DETECT TASK TYPE from the observation text:
  • Contains "price" or "budget" or "cheapest"  → price_negotiation task
  • Contains "defect" or "quality" or "compliant" → quality_control task
  • Contains "competitor" or "competing" or "rival" → competing_buyer task
  • Contains "FAILED" or "failed" or "strike"     → reroute/multi-crisis task
  • Otherwise                                      → easy reorder task

STEP 2 — GATHER INFORMATION (call read tools BEFORE acting):

  For ALL tasks:
    a) Call get_inventory  → identify products and stock levels
    b) Call check_supplier_status for each supplier mentioned in the task

  For price_negotiation tasks ONLY:
    c) Call get_market_prices → find the cheapest healthy supplier

  For quality_control tasks ONLY:
    c) Call get_quality_report → identify suppliers with defect_rate ≤ threshold
       RULE: Never order from a supplier that FAILS the quality report

  For competing_buyer tasks ONLY:
    c) Call get_competing_bids IMMEDIATELY (skip other reads if steps > 3)
       RULE: Order from SupplierA before the competitor countdown hits 0

  For reroute tasks (observation mentions SHP- shipment IDs):
    c) Call get_pending_shipments → get exact shipment IDs

STEP 3 — ACT:

  Easy reorder:
    → place_order with goal_supplier, goal_product, goal_quantity from description

  Price negotiation:
    → place_order with the CHEAPEST healthy supplier from get_market_prices

  Quality control:
    → place_order ONLY with a supplier that PASSED get_quality_report

  Competing buyer:
    → place_order immediately with SupplierA for the required quantity
    → Do NOT waste steps reading — the competitor clock is ticking

  Reroute + order:
    → reroute_shipment to a healthy supplier first
    → then place_order for the required product/supplier/quantity

  Multi-product crisis:
    → reroute any SHP- shipment to a healthy supplier
    → then place_order for EACH product in the goal_orders list
    → check budget: sum all order costs before placing

STEP 4 — FAILURE RECOVERY (if a tool returns an error):

  "supplier is failed/strike"  → switch to a different supplier immediately
  "defect rate exceeds threshold" → call get_quality_report, pick a PASS supplier
  "cost exceeds remaining budget" → pick a cheaper supplier or reduce quantity
  "Shipment not found"         → call get_pending_shipments to get correct IDs
  "product not in catalogue"   → call get_inventory to get exact product names
  "quantity must be > 0"       → retry with quantity ≥ 1

═══════════════════════════════════════════════
HARD RULES — never violate these
═══════════════════════════════════════════════
1. NEVER order from a supplier whose status is "failed" or "strike"
2. NEVER order from a supplier that FAILED the quality report
3. NEVER exceed the budget — check remaining_budget in state before ordering
4. ALWAYS use exact product names (lowercase, underscores) from get_inventory
5. ALWAYS use exact shipment IDs from get_pending_shipments, not guesses
6. Output ONLY the JSON object — zero extra characters

═══════════════════════════════════════════════
EXAMPLE EPISODE (medium reroute task)
═══════════════════════════════════════════════
Observation: "SupplierA FAILED. Reroute SHP-001 to SupplierB. Order 100 units of 'canned_soup'."

Turn 1 → {"tool": "get_inventory", "args": {}}
Turn 2 → {"tool": "get_pending_shipments", "args": {}}
Turn 3 → {"tool": "reroute_shipment", "args": {"shipment_id": "SHP-001", "new_supplier": "SupplierB"}}
Turn 4 → {"tool": "place_order", "args": {"supplier_name": "SupplierB", "product": "canned_soup", "quantity": 100}}
"""

# ---------------------------------------------------------------------------
# All task IDs
# ---------------------------------------------------------------------------
TASKS = [0, 1, 2, 5, 6, 7, 10, 11, 12, 13]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_action(raw: str) -> dict:
    """Parse LLM output to a tool call dict. Falls back to get_inventory."""
    raw = raw.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw
    try:
        obj = json.loads(raw)
        if "tool" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    # Try to extract JSON substring
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start != -1 and end > start:
        try:
            obj = json.loads(raw[start:end])
            if "tool" in obj:
                return obj
        except json.JSONDecodeError:
            pass
    return {"tool": "get_inventory", "args": {}}


def safe_obs_text(result: dict) -> str:
    """Extract observation text from a step result."""
    obs = result.get("observation", result)
    if isinstance(obs, dict):
        return obs.get("text", json.dumps(obs))
    return str(obs)


def build_user_message(obs_text: str, state: dict, step: int) -> str:
    """
    Build a rich user message that includes:
      - the raw observation text
      - key state fields the LLM should reason over
      - a reminder of what step it's on and how many remain

    This replaces bare obs_text so the model always knows its context.
    """
    parts = [obs_text]

    # Inject state hints the model should see
    if state:
        hints = []

        remaining = state.get("remaining_budget")
        if remaining is not None:
            hints.append(f"remaining_budget=${remaining:.2f}")

        countdown = state.get("competing_bids_countdown", {})
        if countdown:
            for prod, steps_left in countdown.items():
                urgency = "CRITICAL" if steps_left <= 2 else "URGENT" if steps_left <= 4 else "ACT SOON"
                hints.append(f"competitor_deadline[{prod}]={steps_left}_steps ({urgency})")

        orders = state.get("orders_placed", [])
        if orders:
            hints.append(f"orders_placed={len(orders)}")

        reroutes = state.get("shipments_rerouted", [])
        if reroutes:
            hints.append(f"shipments_rerouted={len(reroutes)}")

        if hints:
            parts.append("\n[STATE] " + " | ".join(hints))

    parts.append(f"\n[STEP {step}/{MAX_STEPS}] What is your next action?")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Single episode
# ---------------------------------------------------------------------------
def run_episode(task_id: int) -> float:
    """
    Run one full episode. Emits judge-required stdout log lines:
      [START] task_id=<n>
      [STEP]  step=<n> action=<json> observation=<json_str> reward=<float> done=<bool>
      [END]   task_id=<n> total_reward=<float>
    Returns the best reward achieved in the episode.
    """
    # --- Reset environment ---
    resp = requests.post(
        f"{ENV_BASE_URL}/reset",
        params={"task_id": task_id, "seed": 42},
        timeout=15,
    )
    resp.raise_for_status()
    data     = resp.json()
    obs_data = data.get("observation", data)
    obs_text = obs_data.get("text", str(obs_data)) if isinstance(obs_data, dict) else str(obs_data)
    state    = obs_data.get("state", {}) if isinstance(obs_data, dict) else {}

    # ── JUDGE LOG: START ──────────────────────────────────────────────────
    print(f"[START] task_id={task_id}", flush=True)

    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": build_user_message(obs_text, state, step=0)},
    ]

    best_reward = 0.0
    done        = False

    for step in range(1, MAX_STEPS + 1):
        # --- LLM decides action ---
        try:
            raw = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=200,
                temperature=0.0,
            ).choices[0].message.content.strip()
            action = parse_action(raw)
        except Exception:
            action = {"tool": "get_inventory", "args": {}}
            raw    = json.dumps(action)

        # --- Step the environment ---
        try:
            result = requests.post(
                f"{ENV_BASE_URL}/step",
                json=action,
                timeout=15,
            ).json()
        except Exception as e:
            print(f"[ERROR] task_id={task_id} step={step} error={str(e)}", flush=True)
            break

        obs_text    = safe_obs_text(result)
        reward      = float(result.get("reward", 0.0))
        done        = bool(result.get("done", False))
        state       = result.get("state", {})
        best_reward = max(best_reward, reward)

        # ── JUDGE LOG: STEP ───────────────────────────────────────────────
        print(
            f"[STEP] step={step} "
            f"action={json.dumps(action)} "
            f"observation={json.dumps(obs_text)} "
            f"reward={reward:.4f} "
            f"done={done}",
            flush=True,
        )

        messages.append({"role": "assistant", "content": raw})
        if not done:
            # Pass enriched context back to the model on every turn
            messages.append({
                "role": "user",
                "content": build_user_message(obs_text, state, step=step),
            })

        if done:
            break

        time.sleep(0.05)  # be polite to the API

    # ── JUDGE LOG: END ────────────────────────────────────────────────────
    print(f"[END] task_id={task_id} total_reward={best_reward:.4f}", flush=True)
    return best_reward


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Supply Chain Environment — Baseline Inference v4", flush=True)
    print(f"Model      : {MODEL_NAME}", flush=True)
    print(f"Environment: {ENV_BASE_URL}", flush=True)

    # Health check
    try:
        requests.get(f"{ENV_BASE_URL}/health", timeout=5).raise_for_status()
        print("Environment: REACHABLE\n", flush=True)
    except Exception:
        print(f"\nERROR: Cannot reach environment at {ENV_BASE_URL}", flush=True)
        print("Start server: uvicorn server.app:app --host 0.0.0.0 --port 7860", flush=True)
        return

    scores     = {}
    start_time = time.time()

    for tid in TASKS:
        try:
            scores[tid] = run_episode(tid)
        except Exception as e:
            # Still emit [END] so the judge parser never stalls on a missing line
            print(f"[END] task_id={tid} total_reward=0.0000", flush=True)
            scores[tid] = 0.0

        # 18-minute safety cutoff (hard limit is 20 min)
        if time.time() - start_time > 18 * 60:
            print("[WARN] Approaching 20-min limit — stopping early.", flush=True)
            break

    # ---------------------------------------------------------------------------
    # Human-readable summary (does NOT affect judge parsing)
    # ---------------------------------------------------------------------------
    avg    = sum(scores.values()) / len(scores) if scores else 0.0
    by_diff = {
        "easy":   [scores[t] for t in [0, 6]         if t in scores],
        "medium": [scores[t] for t in [1, 5, 7, 12]  if t in scores],
        "hard":   [scores[t] for t in [2, 10, 11, 13] if t in scores],
    }

    print(f"\n{'='*50}", flush=True)
    print("RESULTS SUMMARY", flush=True)
    print(f"{'='*50}", flush=True)
    for tid, s in scores.items():
        bar = "█" * int(s * 20)
        print(f"  Task {tid:2d}: {s:.4f}  {bar}", flush=True)
    print(f"{'─'*50}", flush=True)
    for diff, vals in by_diff.items():
        if vals:
            print(f"  {diff:<8}: avg {sum(vals)/len(vals):.4f}", flush=True)
    print(f"  Overall : avg {avg:.4f}", flush=True)
    print(f"  Runtime : {time.time()-start_time:.1f}s", flush=True)
    print(f"{'='*50}", flush=True)

    # Save results.json
    out = {
        "model":   MODEL_NAME,
        "env_url": ENV_BASE_URL,
        "scores":  {str(k): v for k, v in scores.items()},
        "average": round(avg, 4),
        "by_difficulty": {
            d: round(sum(v) / len(v), 4) for d, v in by_diff.items() if v
        },
    }
    with open("results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved results.json", flush=True)


if __name__ == "__main__":
    main()
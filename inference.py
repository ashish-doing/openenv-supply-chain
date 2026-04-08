"""
inference.py — Supply Chain Environment Baseline Inference (v4.5)

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
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>

FIXES v4.3 (all changes over v4.2):
  - FIX 7: /reset now sends task_id + seed as JSON body (not query params),
            matching the server's ResetRequest Pydantic model
  - FIX 8: seed variable defined as EPISODE_SEED constant — was used but
            never defined in v4.2, causing NameError on every /reset call
  - FIX 9: 402 "credits depleted" detected in call_llm_with_retry and raises
            SystemExit immediately; main() catches it and aborts cleanly
  - FIX 10: /step body sends action nested AND flat so both openenv-core
            wrapper and any custom /step handler accept it without changes

  Carried from v4.2:
  - FIX 1: log_start() always emitted before any possible exception path
  - FIX 2: state extracted from both top-level and nested observation
  - FIX 3: message history trimmed to last 10 exchanges
  - FIX 4: retry logic on rate-limit (429) errors with back-off
  - FIX 5: score/success uses final step reward (rewards[-1])
  - FIX 6: log_end() always uses the helper, never a bare print
"""

import json
import os
import time
from typing import List, Optional

import requests
from openai import OpenAI, RateLimitError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK    = "supply_chain_env"

# FIX 8: seed defined as a module-level constant — was previously used but
# never defined, causing NameError on every /reset call.
EPISODE_SEED = 42

MAX_STEPS          = 25
MSG_HISTORY_KEEP   = 10  # keep last N user+assistant pairs (+ system prompt)
RATE_LIMIT_RETRIES = 3   # max retries on 429
RATE_LIMIT_BACKOFF = 5   # seconds to wait between retries

client = OpenAI(api_key=HF_TOKEN or "dummy", base_url=API_BASE_URL)

# ---------------------------------------------------------------------------
# System prompt — v4.3 (unchanged from v4.2)
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
  → Use this when a shipment is from a failed/strike/defective supplier

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
  • Contains "price" or "budget" or "cheapest"      → price_negotiation task
  • Contains "defect" or "quality" or "compliant"   → quality_control task
  • Contains "competitor" or "competing" or "rival" → competing_buyer task
  • Contains "FAILED" or "failed" or "strike"       → reroute/multi-crisis task
  • Contains "DEMAND SPIKE" or "spiked"             → demand_spike task
  • Otherwise                                        → easy reorder task

STEP 2 — GATHER INFORMATION (call read tools BEFORE acting):

  For ALL tasks:
    a) Call get_inventory  → identify products and stock levels
    b) Call check_supplier_status for each supplier mentioned in the task

  For price_negotiation tasks ONLY:
    c) Call get_market_prices → find the cheapest healthy supplier

  For quality_control tasks ONLY:
    c) Call get_quality_report → identify suppliers with defect_rate ≤ threshold
    d) Call get_pending_shipments → get the defective shipment ID to cancel
       RULE: Never order from a supplier that FAILS the quality report

  For competing_buyer tasks ONLY:
    c) Call get_competing_bids IMMEDIATELY (skip other reads if steps > 3)
       RULE: Order from the scarce supplier before the competitor countdown hits 0

  For reroute/port_strike tasks (observation mentions SHP- shipment IDs):
    c) Call get_pending_shipments → get exact shipment IDs

STEP 3 — ACT:

  Easy reorder:
    → place_order with goal_supplier, goal_product, goal_quantity from description

  Demand spike:
    → place_order with the healthy supplier for the large quantity in description

  Price negotiation:
    → place_order with the CHEAPEST healthy supplier from get_market_prices

  Quality control — EXACT SEQUENCE (both steps required):
    1. cancel_shipment for the defective shipment (get ID from get_pending_shipments)
    2. place_order for EACH required product from a PASSED supplier ONLY
    → RULE: cancel first, then order. Missing either step means task incomplete.

  Competing buyer:
    → place_order immediately with the scarce supplier for the required quantity
    → Then place_order for the second product from its supplier
    → Do NOT waste steps reading — the competitor clock is ticking

  Reroute + order:
    → reroute_shipment to a healthy supplier first
    → then place_order for the required product/supplier/quantity

  Port strike:
    → cancel_shipment for the stuck shipment
    → place_order for EACH product from the only healthy supplier

  Multi-product crisis:
    → reroute any SHP- shipment to a healthy supplier
    → then place_order for EACH product in the goal_orders list
    → check budget: sum all order costs before placing

STEP 4 — FAILURE RECOVERY (if a tool returns an error):

  "supplier is failed/strike"     → switch to a different supplier immediately
  "defect rate exceeds threshold" → call get_quality_report, pick a PASS supplier
  "cost exceeds remaining budget" → pick a cheaper supplier or reduce quantity
  "Shipment not found"            → call get_pending_shipments to get correct IDs
  "product not in catalogue"      → call get_inventory to get exact product names
  "quantity must be > 0"          → retry with quantity ≥ 1

═══════════════════════════════════════════════
HARD RULES — never violate these
═══════════════════════════════════════════════
1. NEVER order from a supplier whose status is "failed" or "strike"
2. NEVER order from a supplier that FAILED the quality report
3. NEVER exceed the budget — check remaining_budget in state before ordering
4. ALWAYS use exact product names (lowercase, underscores) from get_inventory
5. ALWAYS use exact shipment IDs from get_pending_shipments, not guesses
6. For quality_control tasks: ALWAYS cancel the defective shipment first, THEN order
7. Output ONLY the JSON object — zero extra characters

═══════════════════════════════════════════════
EXAMPLE EPISODES
═══════════════════════════════════════════════

Medium reroute task:
  Observation: "SupplierA FAILED. Reroute SHP-001 to SupplierB. Order 100 units of 'canned_soup'."
  Turn 1 → {"tool": "get_inventory", "args": {}}
  Turn 2 → {"tool": "get_pending_shipments", "args": {}}
  Turn 3 → {"tool": "reroute_shipment", "args": {"shipment_id": "SHP-001", "new_supplier": "SupplierB"}}
  Turn 4 → {"tool": "place_order", "args": {"supplier_name": "SupplierB", "product": "canned_soup", "quantity": 100}}

Hard quality_control task:
  Observation: "QUALITY CRISIS. SupplierA defect rate 35%. Cancel SHP-012. Order only from compliant suppliers."
  Turn 1 → {"tool": "get_quality_report", "args": {}}
  Turn 2 → {"tool": "get_pending_shipments", "args": {}}
  Turn 3 → {"tool": "cancel_shipment", "args": {"shipment_id": "SHP-012"}}
  Turn 4 → {"tool": "place_order", "args": {"supplier_name": "SupplierB", "product": "vaccine_vial", "quantity": 300}}
  Turn 5 → {"tool": "place_order", "args": {"supplier_name": "SupplierB", "product": "syringe", "quantity": 200}}
"""

# ---------------------------------------------------------------------------
# All task IDs
# ---------------------------------------------------------------------------
TASKS = [0, 1, 2, 5, 6, 7, 10, 11, 12, 13]


# ---------------------------------------------------------------------------
# Structured log helpers — MUST match official sample format exactly
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_action(raw: str) -> dict:
    """Parse LLM output to a tool call dict. Falls back to get_inventory."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw
    try:
        obj = json.loads(raw)
        if "tool" in obj:
            return obj
    except json.JSONDecodeError:
        pass
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
    obs = result.get("observation", result)
    if isinstance(obs, dict):
        return obs.get("text", json.dumps(obs))
    return str(obs)


def extract_state(data: dict) -> dict:
    """
    Robustly extract state dict from a /reset or /step response.
    Checks: data["state"], data["observation"]["state"], data["observation"] itself.
    """
    # FIX 2: top-level state key (most common for /step responses)
    if "state" in data and isinstance(data["state"], dict):
        return data["state"]

    obs = data.get("observation", {})
    if isinstance(obs, dict):
        # nested observation.state (some /reset responses)
        if "state" in obs and isinstance(obs["state"], dict):
            return obs["state"]
        # observation itself may carry state fields directly
        state_keys = {"remaining_budget", "competing_bids_countdown",
                      "orders_placed", "shipments_rerouted", "shipments_cancelled", "steps"}
        if any(k in obs for k in state_keys):
            return obs

    return {}


def build_user_message(obs_text: str, state: dict, step: int) -> str:
    parts = [obs_text]
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
        cancelled = state.get("shipments_cancelled", [])
        if cancelled:
            hints.append(f"shipments_cancelled={len(cancelled)}")
        if hints:
            parts.append("\n[STATE] " + " | ".join(hints))
    parts.append(f"\n[STEP {step}/{MAX_STEPS}] What is your next action?")
    return "\n".join(parts)


def trim_messages(messages: list) -> list:
    """
    FIX 3: Keep the system prompt (messages[0]) + the last MSG_HISTORY_KEEP
    user/assistant pairs to avoid token limit failures on long episodes.
    """
    system   = messages[:1]
    tail     = messages[1:]
    max_tail = MSG_HISTORY_KEEP * 2
    if len(tail) > max_tail:
        tail = tail[-max_tail:]
    return system + tail


def call_llm_with_retry(messages: list) -> str:
    """
    FIX 4 + FIX 9: Call the LLM with retry on 429; hard-stop on 402.
    - HTTP 429 (rate limit): retries up to RATE_LIMIT_RETRIES times with back-off.
    - HTTP 402 (credits depleted): raises SystemExit(1) immediately — no retry.
    - Other exceptions: re-raised immediately, not retried.
    """
    last_exc = None
    for attempt in range(1 + RATE_LIMIT_RETRIES):
        try:
            return client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=200,
                temperature=0.0,
            ).choices[0].message.content.strip()
        except RateLimitError as e:
            err_str = str(e)
            # FIX 9: credits gone — abort immediately, do not retry.
            if "402" in err_str or "depleted" in err_str.lower() or "credits" in err_str.lower():
                print(
                    "[FATAL] API credits depleted (402). "
                    "Set a new API_BASE_URL / HF_TOKEN and restart.",
                    flush=True,
                )
                raise SystemExit(1)
            last_exc = e
            wait = RATE_LIMIT_BACKOFF * (attempt + 1)
            print(f"[WARN] Rate limit hit (attempt {attempt + 1}), retrying in {wait}s…", flush=True)
            time.sleep(wait)
        except Exception as e:
            raise e
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Single episode
# ---------------------------------------------------------------------------

def run_episode(task_id: int) -> float:
    """
    Run one full episode. Emits the three required judge log line types.
    Returns the final step reward (matches judge scoring logic).
    """
    task_name = f"task_{task_id}"
    rewards: List[float] = []
    steps_taken = 0
    score   = 0.0
    success = False

    # FIX 1: log_start() emitted unconditionally BEFORE any code that could
    # raise — judge always sees a matching [START] for every [END].
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # FIX 7: JSON body instead of query params — matches ResetRequest model.
        # FIX 8: EPISODE_SEED constant used (was undefined `seed` in v4.2).
        resp = requests.post(
            f"{ENV_BASE_URL}/reset",
            json={"task_id": task_id, "seed": EPISODE_SEED},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        obs_data = data.get("observation", data)
        obs_text = obs_data.get("text", str(obs_data)) if isinstance(obs_data, dict) else str(obs_data)

        state = extract_state(data)  # FIX 2

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_message(obs_text, state, step=0)},
        ]

        done = False

        for step in range(1, MAX_STEPS + 1):
            error_msg = None

            trimmed_messages = trim_messages(messages)  # FIX 3

            try:
                raw    = call_llm_with_retry(trimmed_messages)  # FIX 4 + FIX 9
                action = parse_action(raw)
            except SystemExit:
                raise  # FIX 9: let 402 propagate — do not swallow it here
            except Exception as e:
                error_msg = str(e)[:80]
                action = {"tool": "get_inventory", "args": {}}
                raw    = json.dumps(action)

            action_str = json.dumps(action, separators=(',', ':'))

            try:
                # FIX 10: send action nested AND flat — works with both
                # openenv-core create_app wrapper and custom /step handlers.
                result = requests.post(
                    f"{ENV_BASE_URL}/step",
                    json={
                        "action": action,          # nested — openenv-core
                        "tool":   action["tool"],  # flat   — custom handlers
                        "args":   action["args"],  # flat   — custom handlers
                    },
                    timeout=15,
                ).json()
            except Exception as e:
                log_step(step=step, action=action_str, reward=0.0, done=True, error=str(e)[:80])
                break

            obs_text = safe_obs_text(result)
            reward   = float(result.get("reward", 0.0))
            done     = bool(result.get("done", False))
            state    = extract_state(result)  # FIX 2

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

            messages.append({"role": "assistant", "content": raw})
            if not done:
                messages.append({
                    "role": "user",
                    "content": build_user_message(obs_text, state, step=step),
                })

            if done:
                break

            time.sleep(0.05)

        # FIX 5: final step reward — consistent with judge scoring (not max).
        score   = rewards[-1] if rewards else 0.0
        success = score >= 0.99

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)  # FIX 6

    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Supply Chain Environment — Baseline Inference v4.5", flush=True)
    print(f"Model      : {MODEL_NAME}", flush=True)
    print(f"Environment: {ENV_BASE_URL}", flush=True)

    try:
        requests.get(f"{ENV_BASE_URL}/health", timeout=5).raise_for_status()
        print("Environment: REACHABLE\n", flush=True)
    except Exception:
        print(f"\nERROR: Cannot reach environment at {ENV_BASE_URL}", flush=True)
        print("Start server: uvicorn supply_chain_env.server.app:app --host 0.0.0.0 --port 7860", flush=True)
        return

    scores     = {}
    start_time = time.time()

    for tid in TASKS:
        try:
            scores[tid] = run_episode(tid)
        except SystemExit:
            # FIX 9: credits depleted — abort entire run cleanly.
            print("[FATAL] Aborting all tasks — no API credits remaining.", flush=True)
            break
        except Exception:
            scores[tid] = 0.0

        if time.time() - start_time > 18 * 60:
            print("[WARN] Approaching 20-min limit — stopping early.", flush=True)
            break

    if not scores:
        return

    avg = sum(scores.values()) / len(scores)
    by_diff = {
        "easy":   [scores[t] for t in [0, 1, 2]        if t in scores],
        "medium": [scores[t] for t in [5, 6, 7]         if t in scores],
        "hard":   [scores[t] for t in [10, 11, 12, 13]  if t in scores],
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

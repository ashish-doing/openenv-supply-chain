"""
inference.py — Supply Chain Environment Baseline Inference (v2)

Required env vars:
  API_BASE_URL   The API endpoint for the LLM (e.g. https://router.huggingface.co/v1)
  MODEL_NAME     The model identifier (e.g. Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN       Your Hugging Face / API key
  ENV_BASE_URL   (optional) Override env URL, default http://localhost:7860

Usage:
  python inference.py

Runs 10 tasks across easy / medium / hard / price_negotiation /
quality_control / competing_buyer and saves results.json.
"""

import os, json, time, requests
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

client = OpenAI(api_key=HF_TOKEN or "dummy", base_url=API_BASE_URL)

SYSTEM_PROMPT = """You are a supply chain manager AI agent.
Each turn respond with ONLY a valid JSON object (no markdown, no explanation):
{"tool": "tool_name", "args": {"key": "value"}}

Available tools:
- get_inventory              (no args)
- check_supplier_status      {"supplier_name": "SupplierA"}
- get_demand_forecast        {"product": "bottled_water"}
- place_order                {"supplier_name": "SupplierA", "product": "bottled_water", "quantity": 200}
- reroute_shipment           {"shipment_id": "SHP-001", "new_supplier": "SupplierB"}
- cancel_shipment            {"shipment_id": "SHP-001"}
- get_pending_shipments      (no args)
- get_market_prices          (no args)   ← use for price_negotiation tasks
- get_quality_report         (no args)   ← use for quality_control tasks
- get_competing_bids         (no args)   ← use for competing_buyer tasks

Strategy hints:
- Always check inventory FIRST to understand the situation
- For medium/hard: check supplier statuses before placing orders
- For price_negotiation: call get_market_prices to find cheapest healthy supplier
- For quality_control: call get_quality_report to identify compliant suppliers
- For competing_buyer: call get_competing_bids IMMEDIATELY — time is critical
- Cancelled or failed suppliers cannot fulfil orders
- Respond with ONLY the JSON object. No text before or after."""

# All tasks: easy (0,1,2), medium (5,6,7), hard (10,11,12,13)
TASKS = [0, 1, 2, 5, 6, 7, 10, 11, 12, 13]

MAX_STEPS = 25   # matches environment MAX_STEPS


def run_episode(task_id: int) -> float:
    print(f"\n{'─'*50}")
    print(f"Task {task_id}")
    resp = requests.post(
        f"{ENV_BASE_URL}/reset",
        params={"task_id": task_id, "seed": 42},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    obs  = data.get("observation", data)

    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": obs.get("text", str(obs))},
    ]

    reward = 0.0
    for step in range(MAX_STEPS):
        # LLM decides action
        try:
            raw = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=200,
                temperature=0.0,
            ).choices[0].message.content.strip()
            action = json.loads(raw)
        except Exception as e:
            print(f"  LLM error step {step+1}: {e}")
            action = {"tool": "get_inventory", "args": {}}
            raw = json.dumps(action)

        # Send to environment
        try:
            result = requests.post(
                f"{ENV_BASE_URL}/step",
                json=action,
                timeout=15,
            ).json()
        except Exception as e:
            print(f"  Env error: {e}")
            break

        obs_new = result.get("observation", {})
        reward  = result.get("reward", 0.0)
        done    = result.get("done", False)

        print(f"  Step {step+1:2d}: {action.get('tool'):<26} reward={reward:.4f}  done={done}")

        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user",      "content": obs_new.get("text", "continue")})

        if done:
            break

        time.sleep(0.05)  # be polite to the API

    print(f"  → Final reward: {reward:.4f}")
    return reward


def main():
    print("Supply Chain Environment v2 — Baseline Inference")
    print(f"Model      : {MODEL_NAME}")
    print(f"Environment: {ENV_BASE_URL}")

    # Health check
    try:
        requests.get(f"{ENV_BASE_URL}/health", timeout=5).raise_for_status()
        print("Environment: REACHABLE ✓\n")
    except Exception:
        print(f"\nERROR: Cannot reach environment at {ENV_BASE_URL}")
        print("Start server: uvicorn server.app:app --host 0.0.0.0 --port 7860")
        return

    scores = {}
    for tid in TASKS:
        try:
            scores[tid] = run_episode(tid)
        except Exception as e:
            print(f"  Task {tid} error: {e}")
            scores[tid] = 0.0

    # Summary
    avg = sum(scores.values()) / len(scores)
    by_diff = {
        "easy":   [scores[t] for t in [0,1,2]   if t in scores],
        "medium": [scores[t] for t in [5,6,7]   if t in scores],
        "hard":   [scores[t] for t in [10,11,12,13] if t in scores],
    }

    print(f"\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")
    for tid, s in scores.items():
        bar = "█" * int(s * 20)
        print(f"  Task {tid:2d}: {s:.4f}  {bar}")
    print(f"{'─'*50}")
    for diff, vals in by_diff.items():
        if vals:
            avg_d = sum(vals) / len(vals)
            print(f"  {diff:<8}: avg {avg_d:.4f}")
    print(f"  Overall : avg {avg:.4f}")
    print(f"{'='*50}")

    # Save results
    out = {
        "model":   MODEL_NAME,
        "env_url": ENV_BASE_URL,
        "scores":  {str(k): v for k, v in scores.items()},
        "average": round(avg, 4),
        "by_difficulty": {
            d: round(sum(v)/len(v), 4) for d, v in by_diff.items() if v
        },
    }
    with open("results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved → results.json")


if __name__ == "__main__":
    main()
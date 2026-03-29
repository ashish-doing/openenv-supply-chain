"""
inference.py — Supply Chain Environment Baseline Inference

Required env vars:  API_BASE_URL  MODEL_NAME  HF_TOKEN
Usage: python inference.py
"""
import os, json, time, requests
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

client = OpenAI(api_key=HF_TOKEN or "dummy", base_url=API_BASE_URL)

SYSTEM_PROMPT = """You are a supply chain manager AI agent.
Each turn respond with ONLY a JSON object:
{"tool": "tool_name", "args": {"key": "value"}}

Available tools:
- get_inventory  (no args)
- check_supplier_status: {"supplier_name": "SupplierA"}
- get_demand_forecast: {"product": "bottled_water"}
- place_order: {"supplier_name": "SupplierA", "product": "bottled_water", "quantity": 200}
- reroute_shipment: {"shipment_id": "SHP-001", "new_supplier": "SupplierB"}
- cancel_shipment: {"shipment_id": "SHP-001"}
- get_pending_shipments  (no args)

Respond with valid JSON only. No explanation, no markdown."""

TASKS = [0, 1, 5, 10]

def run_episode(task_id: int) -> float:
    print(f"\n--- Task {task_id} ---")
    resp = requests.post(f"{ENV_BASE_URL}/reset", params={"task_id": task_id, "seed": 42})
    resp.raise_for_status()
    data = resp.json()
    obs  = data.get("observation", data)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": obs.get("text", str(obs))},
    ]
    reward = 0.0
    for step in range(12):
        try:
            raw = client.chat.completions.create(
                model=MODEL_NAME, messages=messages,
                max_tokens=200, temperature=0.0,
            ).choices[0].message.content.strip()
            action = json.loads(raw)
        except Exception as e:
            print(f"  LLM error step {step}: {e}")
            action = {"tool": "get_inventory", "args": {}}
            raw = json.dumps(action)
        try:
            result = requests.post(f"{ENV_BASE_URL}/step", json=action, timeout=15).json()
        except Exception as e:
            print(f"  Env error: {e}"); break
        obs_new = result.get("observation", {})
        reward  = result.get("reward", 0.0)
        done    = result.get("done", False)
        print(f"  Step {step+1}: {action.get('tool')} -> reward={reward:.2f} done={done}")
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": obs_new.get("text", "continue")})
        if done:
            break
        time.sleep(0.1)
    print(f"  Final reward: {reward:.4f}")
    return reward

def main():
    print("Supply Chain Environment — Baseline Inference")
    print(f"Model: {MODEL_NAME}  |  Env: {ENV_BASE_URL}")
    try:
        requests.get(f"{ENV_BASE_URL}/health", timeout=5).raise_for_status()
        print("Environment: REACHABLE\n")
    except Exception as e:
        print(f"ERROR: Cannot reach environment at {ENV_BASE_URL}")
        print("Start server: uvicorn server.app:app --host 0.0.0.0 --port 7860")
        return
    scores = {}
    for tid in TASKS:
        try:
            scores[tid] = run_episode(tid)
        except Exception as e:
            print(f"Task {tid} error: {e}")
            scores[tid] = 0.0
    avg = sum(scores.values()) / len(scores)
    print(f"\n{'='*40}")
    print("RESULTS")
    for tid, s in scores.items():
        print(f"  Task {tid:3d}: {s:.4f}  {'█' * int(s*20)}")
    print(f"  Average: {avg:.4f}")
    print("="*40)
    with open("results.json", "w") as f:
        json.dump({"scores": {str(k): v for k, v in scores.items()}, "average": avg, "model": MODEL_NAME}, f, indent=2)
    print("Saved to results.json")

if __name__ == "__main__":
    main()
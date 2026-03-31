---
title: Supply Chain Environment
emoji: 🏭
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
  - supply-chain
  - reinforcement-learning
---

# Supply Chain Disruption Environment

An OpenEnv-compatible reinforcement learning environment where an AI agent
manages real-world supply chain crises using diagnostic and action tools.

**Live Space:** [HuggingFace](https://huggingface.co/spaces/ashish-doing/supply-chain-env-hf)  
**API Docs:** [Interactive Swagger UI](https://ashish-doing-supply-chain-env-hf.hf.space/docs)  
**Live Demo:** [Watch agent solve a task](https://ashish-doing-supply-chain-env-hf.hf.space/quick/demo)
---

## What the Agent Does

The agent plays a warehouse manager. Each step it calls one tool, receives a
text observation and a reward score, and must reach the goal before running
out of steps (max 25).

---

## Tasks and Rewards

### Easy — Tasks 0, 1, 2

**Scenario:** Single product, healthy suppliers, stock running low.

**Goal:** Check inventory → verify supplier → place reorder.

| Score | Condition |
|---|---|
| 0.10 | Agent took at least one action |
| 0.50 | Agent placed any order |
| 1.00 | Correct supplier + correct product + correct quantity |

---

### Medium — Tasks 5, 6

**Scenario:** Primary supplier failed OR demand spiked 3×. Shipment is stranded.

**Goal:** Identify failure → reroute shipment → place emergency order.

| Score | Condition |
|---|---|
| 0.10 | Agent took at least one action |
| 0.50 | Agent placed any order or rerouted any shipment |
| 0.75 | Correct shipment rerouted to healthy supplier |
| 1.00 | Reroute done AND correct emergency order placed |

---

### Hard — Tasks 10, 11

**Scenario:** Multi-product crisis, budget constraint, supplier on strike.

**Goal:** Assess all suppliers → cancel/reroute shipment → order ALL products within budget.

| Score | Condition |
|---|---|
| 0.10 | Agent took at least one action |
| 0.50 | Agent placed any order or rerouted any shipment |
| 0.75 | Required shipment action (cancel/reroute) completed |
| 1.00 | All product orders placed with correct suppliers within budget |

---

## Available Tools

| Tool | Arguments | Description |
|---|---|---|
| `get_inventory` | none | Current stock levels |
| `check_supplier_status` | `supplier_name` | Supplier health, lead time, cost |
| `get_demand_forecast` | `product` | Daily demand and days until stockout |
| `place_order` | `supplier_name`, `product`, `quantity` | Place a purchase order |
| `reroute_shipment` | `shipment_id`, `new_supplier` | Redirect a stranded shipment |
| `cancel_shipment` | `shipment_id` | Cancel a stuck shipment |
| `get_pending_shipments` | none | List all in-transit shipments |

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset?task_id=0` | POST | Start a new episode |
| `/step` | POST | Execute one tool action |
| `/state` | GET | Current episode state |
| `/health` | GET | Liveness check |
| `/docs` | GET | Interactive Swagger UI |

---

## Action Format
```json
{
  "tool": "place_order",
  "args": {
    "supplier_name": "SupplierA",
    "product": "bottled_water",
    "quantity": 200
  }
}
```

## Observation Format
```json
{
  "text": "SUCCESS: Order placed! Supplier: SupplierA ...",
  "state": {
    "inventory": {"bottled_water": 50},
    "steps": 2,
    "orders_placed": [],
    "shipments_rerouted": [],
    "spent_budget": 0.0
  },
  "reward": 0.5,
  "done": false
}
```

---

## Quick Start
```bash
pip install openenv-core
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Running Inference
```bash
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

## Pre-submission Validation
```bash
python validate.py
# Expected: Result: 23/23 checks passed — STATUS: READY TO SUBMIT
```
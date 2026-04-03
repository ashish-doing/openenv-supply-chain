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

# Supply Chain Disruption Environment v2

An OpenEnv-compatible reinforcement learning environment where an AI agent manages real-world supply chain crises — enhanced with adversarial market dynamics, quality control, and price negotiation.

**Live Space:** [HuggingFace](https://huggingface.co/spaces/ashish-doing/supply-chain-env-hf)  
**API Docs:** [Interactive Swagger UI](https://ashish-doing-supply-chain-env-hf.hf.space/docs)  
**Live Demo:** [Watch agent solve a task](https://ashish-doing-supply-chain-env-hf.hf.space/quick/demo)

---

## What the Agent Does

The agent plays a warehouse manager. Each step it calls one tool, receives a text observation and a reward score, and must reach the goal before running out of steps (max 25).

---

## Task Types & Rewards

### Easy — Tasks 0, 1, 2 · `reorder`

**Scenario:** Single product, healthy suppliers, stock running low.  
**Goal:** Check inventory → verify supplier → place reorder.

| Score | Condition |
|-------|-----------|
| 0.10 | Agent took at least one action |
| 0.50 | Agent placed any order |
| 1.00 | Correct supplier + correct product + correct quantity |

---

### Medium — Tasks 5, 6 · `reroute` / `demand_spike`

**Scenario:** Primary supplier failed OR demand spiked 3×. Shipment is stranded.  
**Goal:** Identify failure → reroute shipment → place emergency order.

| Score | Condition |
|-------|-----------|
| 0.10 | Agent took at least one action |
| 0.50 | Agent placed any order or rerouted any shipment |
| 0.75 | Correct shipment rerouted to healthy supplier |
| 1.00 | Reroute done AND correct emergency order placed |

---

### Medium — Task 7 · `price_negotiation` 🆕

**Scenario:** Critical shortage. 3 suppliers with different prices. Budget cap enforced.  
**Goal:** Use `get_market_prices` → find cheapest healthy supplier → order within budget.

| Score | Condition |
|-------|-----------|
| 0.10 | Agent took at least one action |
| 0.50 | Agent placed any order |
| 1.00 | Correct supplier (cheapest healthy), correct product, quantity ≥ 400, within budget |

---

### Hard — Tasks 10, 11 · `multi_product_crisis` / `port_strike`

**Scenario:** Multi-product crisis, budget constraint, supplier on strike.  
**Goal:** Assess all suppliers → cancel/reroute shipment → order ALL products within budget.

| Score | Condition |
|-------|-----------|
| 0.10 | Agent took at least one action |
| 0.50 | Agent placed any order or rerouted any shipment |
| 0.75 | Required shipment action (cancel/reroute) completed |
| 1.00+ | All product orders placed with correct suppliers within budget |

> **Budget-efficiency bonus:** On hard tasks, reward scales up to 1.15 based on how much budget remains. Spending only 50% of budget yields ~1.075. This creates a rich continuous reward surface for RL training.

---

### Hard — Task 12 · `quality_control` 🆕

**Scenario:** Defective supplier batches detected in medical supply chain. SupplierA has 35% defect rate — unacceptable. Only suppliers with < 5% defect rate may be used.  
**Goal:** Use `get_quality_report` → cancel defective shipment → order ONLY from compliant suppliers.

| Score | Condition |
|-------|-----------|
| 0.10 | Agent took at least one action |
| 0.50 | Agent placed any order or cancelled shipment |
| 0.75 | Defective shipment cancelled |
| 1.00+ | All orders placed with quality-compliant suppliers + efficiency bonus |

---

### Hard — Task 13 · `competing_buyer` 🆕 (Adversarial)

**Scenario:** A rival company (RivalCorp) is also bidding for the same limited semiconductor supply from SupplierA. The competitor will lock up capacity in ~6 steps. Time pressure is real.  
**Goal:** Use `get_competing_bids` → act fast to secure semiconductors before competitor → also order capacitors.

| Score | Condition |
|-------|-----------|
| 0.10 | Agent took at least one action |
| 0.50 | Agent placed any order |
| 0.75 | Semiconductor secured before competitor locked capacity |
| 1.00+ | Both products ordered + efficiency bonus |

---

## Available Tools

| Tool | Arguments | Description |
|------|-----------|-------------|
| `get_inventory` | none | Current stock levels with days-until-stockout |
| `check_supplier_status` | `supplier_name` | Supplier health, lead time, cost, capacity |
| `get_demand_forecast` | `product` | Daily demand and days until stockout |
| `place_order` | `supplier_name`, `product`, `quantity` | Place a purchase order |
| `reroute_shipment` | `shipment_id`, `new_supplier` | Redirect a stranded shipment |
| `cancel_shipment` | `shipment_id` | Cancel a stuck/defective shipment |
| `get_pending_shipments` | none | List all in-transit shipments |
| `get_market_prices` | none | 🆕 Compare all supplier quotes and budget impact |
| `get_quality_report` | none | 🆕 Defect rates per supplier vs threshold |
| `get_competing_bids` | none | 🆕 Competing buyer urgency countdown |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset?task_id=0` | POST | Start a new episode |
| `/step` | POST | Execute one tool action |
| `/state` | GET | Current episode state |
| `/health` | GET | Liveness check |
| `/docs` | GET | Interactive Swagger UI |
| `/quick/demo` | GET | Auto-runs a complete demo episode |

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
    "max_steps": 25,
    "orders_placed": [],
    "shipments_rerouted": [],
    "shipments_cancelled": [],
    "pending_shipments": [],
    "spent_budget": 0.0,
    "remaining_budget": 2000.0,
    "task_type": "reorder",
    "difficulty": "easy",
    "competing_bids_countdown": {"semiconductor": 5}
  },
  "reward": 0.5,
  "done": false
}
```

---

## Enhancements over v1 (inspired by SF reference projects)

| Feature | Inspired by | What changed |
|---------|-------------|--------------|
| 3 new task types | Reasoning Gym (task diversity) | price_negotiation, quality_control, competing_buyer |
| Budget-efficiency bonus | CARLA (continuous reward surface) | Hard rewards scale 1.0–1.15 based on budget saved |
| Adversarial competing buyer | Calendar Env (multi-agent coordination) | Countdown mechanic — competitor locks capacity after N steps |
| Quality gate enforcement | TB2 Env (multi-hop reasoning) | Orders from defective suppliers are rejected |
| get_market_prices tool | Calendar Env (rich tool set) | Compare quotes, budget impact, quote validity |
| get_quality_report tool | Reasoning Gym (programmatic grading) | Defect rates, pass/fail verdict per supplier |
| get_competing_bids tool | CARLA (real-time state) | Urgency countdown visible to agent |
| Max steps raised to 25 | All reference envs | Enables long-horizon reasoning evaluation |
| State includes max_steps | All reference envs | Agent can plan step budget |
| Dense inventory feedback | REPL Env (rich observations) | Stockout days + threshold status per product |

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
# Expected: STATUS: READY TO SUBMIT ✓
```

## Task Summary

| Task ID | Difficulty | Type | Key Challenge |
|---------|------------|------|---------------|
| 0, 1, 2 | Easy | reorder | Basic inventory check + reorder |
| 5 | Medium | reroute | Supplier failure + shipment reroute |
| 6 | Medium | demand_spike | 3× demand spike detection |
| 7 | Medium | price_negotiation 🆕 | Find cheapest healthy supplier under budget |
| 10 | Hard | multi_product_crisis | 3 products + budget + reroute |
| 11 | Hard | port_strike | Strike scenario + cancel + domestic supplier |
| 12 | Hard | quality_control 🆕 | Defect rate gate + cancel defective shipment |
| 13 | Hard | competing_buyer 🆕 | Adversarial time pressure + capacity lock |
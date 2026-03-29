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

An OpenEnv-compatible AI agent environment for training and evaluating agents
on real-world supply chain crisis management.

## What Is This?

This environment places an AI agent in the role of a supply chain manager.
Suppliers fail, shipments get stranded, demand spikes occur — and the agent
must use tools to inspect the situation, reroute deliveries, and place
emergency orders before stockouts happen.

**Real-world domain:** Every major company (Amazon, Toyota, hospitals)
faces these exact crises. An agent that can manage supply chain disruptions
has immediate, tangible business value.

## Quick Start
```python
from supply_chain_env import SupplyChainAction, SupplyChainEnv

with SupplyChainEnv(base_url="http://localhost:7860") as env:
    result = env.reset()
    print(result.observation.text)

    result = env.step(SupplyChainAction(
        tool="get_inventory",
        args={}
    ))
    result = env.step(SupplyChainAction(
        tool="place_order",
        args={"supplier_name": "SupplierA", "product": "bottled_water", "quantity": 200}
    ))
    print(f"Reward: {result.reward}, Done: {result.done}")
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Reset environment to a new task |
| `/step` | POST | Execute a tool action |
| `/state` | GET | Get current episode state |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive API documentation |

## Action Space
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

## Observation Space
```json
{
  "text": "Human-readable result of the action",
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

## Available Tools

| Tool | Description |
|---|---|
| `get_inventory` | Returns current stock levels |
| `check_supplier_status` | Returns supplier health and lead time |
| `get_demand_forecast` | Returns daily demand and days until stockout |
| `place_order` | Places a purchase order |
| `reroute_shipment` | Redirects stranded shipment to new supplier |
| `cancel_shipment` | Cancels a pending shipment |
| `get_pending_shipments` | Lists all in-transit shipments |

## Difficulty Levels

| Level | Scenario | Challenge |
|---|---|---|
| Easy | Single product, healthy suppliers | Check inventory → place reorder |
| Medium | Supplier failure or demand spike | Reroute shipment + emergency order |
| Hard | Multi-product crisis, budget limit | Prioritize, reroute, order all products |

## Reward Structure

| Score | Meaning |
|---|---|
| 0.10 | Agent started acting |
| 0.50 | Agent placed order or rerouted shipment |
| 0.75 | Required reroutes completed |
| 1.00 | All goals fully achieved |

## Running Locally
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
"""
generate_tasks.py — Procedural task generator for Supply Chain Environment.

Every task is fully deterministic given (task_id, seed).
The environment calls generate_task(task_id) at reset time — no hardcoded list.

Task ID → difficulty mapping:
  0–49   → easy              (reorder)
  50–99  → medium            (reroute if even, demand_spike if odd)
  100–149→ medium            (price_negotiation)
  150–199→ hard              (multi_product_crisis)
  200–249→ hard              (quality_control)
  250–299→ hard              (competing_buyer)
  300+   → cycles all hard types

  Special IDs kept for backward compatibility with old hardcoded tasks:
  0,1,2 → easy
  5,6   → medium reroute/demand_spike
  7     → medium price_negotiation
  10,11 → hard multi_product/port_strike
  12    → hard quality_control
  13    → hard competing_buyer

FIX v4.1:
  - _medium_demand_spike() is now a proper generator (was incorrectly reusing
    _medium_reroute() and just renaming the type, which produced tasks with the
    wrong goal structure — no demand_spike flag, wrong goal_supplier/goal_product).

Usage (CLI):
  python generate_tasks.py --task_id 42
  python generate_tasks.py --count 10 --seed 99
  python generate_tasks.py --difficulty hard --count 5
"""

import argparse
import json
import random

# ── Pools ──────────────────────────────────────────────────────────────────────

PRODUCTS = [
    "bottled_water", "canned_soup", "coffee_beans", "circuit_board",
    "pain_reliever", "mask", "glove", "sanitizer", "engine_part", "brake_pad",
    "raw_steel", "vaccine_vial", "syringe", "semiconductor", "capacitor",
    "laptop_screen", "battery_cell", "solar_panel", "air_filter", "rubber_seal",
]

SUPPLIER_NAMES = [
    "SupplierA", "SupplierB", "SupplierC", "SupplierD",
    "SupplierX", "SupplierY", "SupplierZ",
    "AlphaLogistics", "BetaWholesale", "GammaSupply",
]

COMPETITOR_NAMES = ["RivalCorp", "NexusInc", "OmegaTrade", "ApexBuyers", "ZenithGroup"]

# ── Backward-compatible fixed tasks (IDs 0–13) ─────────────────────────────────

_FIXED_TASKS = {
    0: {
        "id": 0, "difficulty": "easy", "type": "reorder",
        "description": (
            "You manage a warehouse that sells bottled water. "
            "Current stock: 50 units. Daily demand: 30 units. "
            "Your primary supplier (SupplierA) is healthy. "
            "Reorder threshold is 60 units. "
            "Task: Check inventory, identify the shortage risk, "
            "and place a reorder with SupplierA for 200 units."
        ),
        "initial_inventory": {"bottled_water": 50},
        "daily_demand": {"bottled_water": 30},
        "reorder_threshold": {"bottled_water": 60},
        "suppliers": {
            "SupplierA": {"status": "healthy", "lead_days": 2, "cost_per_unit": 1.5},
            "SupplierB": {"status": "healthy", "lead_days": 4, "cost_per_unit": 1.2},
        },
        "goal_product": "bottled_water", "goal_supplier": "SupplierA", "goal_quantity": 200,
    },
    1: {
        "id": 1, "difficulty": "easy", "type": "reorder",
        "description": (
            "You manage a warehouse selling canned soup. "
            "Current stock: 20 units. Daily demand: 25 units. "
            "SupplierA is healthy with 1-day lead time. "
            "Task: Check inventory, verify SupplierA is healthy, "
            "and place an order for 150 units."
        ),
        "initial_inventory": {"canned_soup": 20},
        "daily_demand": {"canned_soup": 25},
        "reorder_threshold": {"canned_soup": 30},
        "suppliers": {
            "SupplierA": {"status": "healthy", "lead_days": 1, "cost_per_unit": 0.8},
            "SupplierB": {"status": "delayed", "lead_days": 7, "cost_per_unit": 0.6},
        },
        "goal_product": "canned_soup", "goal_supplier": "SupplierA", "goal_quantity": 150,
    },
    2: {
        "id": 2, "difficulty": "easy", "type": "reorder",
        "description": (
            "Warehouse stocks coffee beans. Stock: 100 units. "
            "Daily demand: 40 units. SupplierC is the cheapest and healthy. "
            "Task: Check inventory, check demand forecast, "
            "and place order of 300 units with SupplierC."
        ),
        "initial_inventory": {"coffee_beans": 100},
        "daily_demand": {"coffee_beans": 40},
        "reorder_threshold": {"coffee_beans": 80},
        "suppliers": {
            "SupplierC": {"status": "healthy", "lead_days": 3, "cost_per_unit": 2.0},
            "SupplierD": {"status": "healthy", "lead_days": 5, "cost_per_unit": 2.5},
        },
        "goal_product": "coffee_beans", "goal_supplier": "SupplierC", "goal_quantity": 300,
    },
    5: {
        "id": 5, "difficulty": "medium", "type": "reroute",
        "description": (
            "PRIMARY SUPPLIER FAILED. You manage electronics components. "
            "Stock: 30 units of 'circuit_board'. Daily demand: 20 units. "
            "SupplierA status: FAILED. SupplierB is healthy but slower. "
            "Task: Check supplier status, identify SupplierA is failed, "
            "reroute the pending shipment SHP-001 from SupplierA to SupplierB, "
            "then place an emergency order of 100 units with SupplierB."
        ),
        "initial_inventory": {"circuit_board": 30},
        "daily_demand": {"circuit_board": 20},
        "reorder_threshold": {"circuit_board": 40},
        "suppliers": {
            "SupplierA": {"status": "failed", "lead_days": 0, "cost_per_unit": 5.0},
            "SupplierB": {"status": "healthy", "lead_days": 5, "cost_per_unit": 6.5},
        },
        "pending_shipments": [
            {"shipment_id": "SHP-001", "supplier": "SupplierA", "quantity": 80, "product": "circuit_board"}
        ],
        "goal_product": "circuit_board", "goal_supplier": "SupplierB",
        "goal_quantity": 100, "goal_reroute_shipment": "SHP-001",
    },
    6: {
        "id": 6, "difficulty": "medium", "type": "demand_spike",
        "description": (
            "DEMAND SPIKE DETECTED. You manage a pharmacy warehouse "
            "stocking 'pain_reliever'. Normal demand: 50 units/day. "
            "Demand just spiked 3x to 150 units/day due to flu season. "
            "Stock: 200 units. SupplierA is healthy. "
            "Task: Check demand forecast (you will see the spike), "
            "check inventory, and place an urgent order of 500 units."
        ),
        "initial_inventory": {"pain_reliever": 200},
        "daily_demand": {"pain_reliever": 150},
        "demand_spike": True,
        "reorder_threshold": {"pain_reliever": 100},
        "suppliers": {
            "SupplierA": {"status": "healthy", "lead_days": 2, "cost_per_unit": 3.0},
            "SupplierB": {"status": "delayed", "lead_days": 10, "cost_per_unit": 2.5},
        },
        "goal_product": "pain_reliever", "goal_supplier": "SupplierA", "goal_quantity": 500,
    },
    7: {
        "id": 7, "difficulty": "medium", "type": "price_negotiation",
        "description": (
            "PRICE NEGOTIATION REQUIRED. You need 400 units of 'raw_steel'. "
            "Stock: 15 units. Daily demand: 80 units — critical shortage. "
            "3 suppliers quote different prices. Budget cap: $1200. "
            "Task: Use get_market_prices to compare all supplier quotes, "
            "identify the cheapest healthy supplier under budget, "
            "and place an order of 400 units with them."
        ),
        "initial_inventory": {"raw_steel": 15},
        "daily_demand": {"raw_steel": 80},
        "reorder_threshold": {"raw_steel": 100},
        "budget": 1200.0,
        "suppliers": {
            "SupplierX": {"status": "healthy", "lead_days": 3, "cost_per_unit": 2.8},
            "SupplierY": {"status": "healthy", "lead_days": 2, "cost_per_unit": 2.5},
            "SupplierZ": {"status": "delayed", "lead_days": 10, "cost_per_unit": 1.9},
        },
        "market_prices": {
            "SupplierX": {"raw_steel": 2.8, "quote_valid_hours": 24},
            "SupplierY": {"raw_steel": 2.5, "quote_valid_hours": 12},
            "SupplierZ": {"raw_steel": 1.9, "quote_valid_hours": 48},
        },
        "goal_product": "raw_steel", "goal_supplier": "SupplierY", "goal_quantity": 400,
    },
    10: {
        "id": 10, "difficulty": "hard", "type": "multi_product_crisis",
        "description": (
            "MULTI-PRODUCT CRISIS. Three products are critically low. "
            "Inventory: {'mask': 10, 'glove': 5, 'sanitizer': 15}. "
            "SupplierA (masks) just failed. SupplierB (gloves) is healthy. "
            "Shipment SHP-010 from SupplierA must be rerouted to SupplierD. "
            "Budget: $2000. Task: Check all inventory, reroute SHP-010, "
            "and place orders for all three products within budget."
        ),
        "initial_inventory": {"mask": 10, "glove": 5, "sanitizer": 15},
        "daily_demand": {"mask": 50, "glove": 40, "sanitizer": 30},
        "reorder_threshold": {"mask": 60, "glove": 50, "sanitizer": 40},
        "budget": 2000.0,
        "suppliers": {
            "SupplierA": {"status": "failed", "lead_days": 0, "cost_per_unit": 2.0, "products": ["mask"]},
            "SupplierB": {"status": "healthy", "lead_days": 3, "cost_per_unit": 1.5, "products": ["glove"]},
            "SupplierC": {"status": "delayed", "lead_days": 8, "cost_per_unit": 1.8, "products": ["sanitizer"]},
            "SupplierD": {"status": "healthy", "lead_days": 4, "cost_per_unit": 2.5, "products": ["mask", "sanitizer"]},
        },
        "pending_shipments": [
            {"shipment_id": "SHP-010", "supplier": "SupplierA", "quantity": 200, "product": "mask"}
        ],
        "goal_orders": [
            {"product": "mask",      "supplier": "SupplierD", "min_quantity": 150},
            {"product": "glove",     "supplier": "SupplierB", "min_quantity": 100},
            {"product": "sanitizer", "supplier": "SupplierD", "min_quantity": 80},
        ],
        "goal_reroute_shipment": "SHP-010",
    },
    11: {
        "id": 11, "difficulty": "hard", "type": "port_strike",
        "description": (
            "PORT STRIKE + SUPPLIER FAILURE. You manage auto parts. "
            "Stock: {'engine_part': 8, 'brake_pad': 12}. "
            "All overseas suppliers are on strike. "
            "Shipment SHP-011 is stuck at port. Budget: $5000. "
            "Task: Assess all supplier statuses, cancel the stuck shipment, "
            "find the only available supplier, and place emergency orders."
        ),
        "initial_inventory": {"engine_part": 8, "brake_pad": 12},
        "daily_demand": {"engine_part": 10, "brake_pad": 15},
        "reorder_threshold": {"engine_part": 20, "brake_pad": 25},
        "budget": 5000.0,
        "suppliers": {
            "SupplierA": {"status": "strike", "lead_days": 0, "cost_per_unit": 20.0, "products": ["engine_part"]},
            "SupplierB": {"status": "strike", "lead_days": 0, "cost_per_unit": 18.0, "products": ["brake_pad"]},
            "SupplierC": {"status": "healthy", "lead_days": 1, "cost_per_unit": 35.0, "products": ["engine_part", "brake_pad"]},
        },
        "pending_shipments": [
            {"shipment_id": "SHP-011", "supplier": "SupplierA", "quantity": 50, "product": "engine_part"}
        ],
        "goal_orders": [
            {"product": "engine_part", "supplier": "SupplierC", "min_quantity": 30},
            {"product": "brake_pad",   "supplier": "SupplierC", "min_quantity": 40},
        ],
        "goal_cancel_shipment": "SHP-011",
    },
    12: {
        "id": 12, "difficulty": "hard", "type": "quality_control",
        "description": (
            "QUALITY CRISIS. Defective batches detected. "
            "SupplierA defect rate: 35% — unacceptable. "
            "SupplierB: 4% — within threshold. SupplierC: 0% defects. "
            "Cancel defective shipment SHP-012. Budget: $8000. "
            "Task: Use get_quality_report, cancel SHP-012, "
            "order only from suppliers with defect rate < 5%."
        ),
        "initial_inventory": {"vaccine_vial": 50, "syringe": 30},
        "daily_demand": {"vaccine_vial": 100, "syringe": 80},
        "reorder_threshold": {"vaccine_vial": 120, "syringe": 100},
        "budget": 8000.0,
        "quality_threshold": 0.05,
        "suppliers": {
            "SupplierA": {"status": "healthy", "lead_days": 1, "cost_per_unit": 8.0,  "products": ["vaccine_vial", "syringe"], "defect_rate": 0.35},
            "SupplierB": {"status": "healthy", "lead_days": 2, "cost_per_unit": 10.0, "products": ["vaccine_vial", "syringe"], "defect_rate": 0.04},
            "SupplierC": {"status": "healthy", "lead_days": 3, "cost_per_unit": 12.0, "products": ["vaccine_vial", "syringe"], "defect_rate": 0.00},
        },
        "pending_shipments": [
            {"shipment_id": "SHP-012", "supplier": "SupplierA", "quantity": 200, "product": "vaccine_vial"}
        ],
        "goal_orders": [
            {"product": "vaccine_vial", "supplier": "SupplierB", "min_quantity": 300, "allowed_suppliers": ["SupplierB", "SupplierC"]},
            {"product": "syringe",      "supplier": "SupplierB", "min_quantity": 200, "allowed_suppliers": ["SupplierB", "SupplierC"]},
        ],
        "goal_cancel_shipment": "SHP-012",
    },
    13: {
        "id": 13, "difficulty": "hard", "type": "competing_buyer",
        "description": (
            "ADVERSARIAL MARKET. RivalCorp is bidding for the same capacity. "
            "SupplierA has only 500 units of semiconductor left. "
            "Use get_competing_bids to see urgency. Budget: $6000. "
            "Task: Secure semiconductor from SupplierA (min 300) "
            "before competitor, AND order capacitors from SupplierB (min 200)."
        ),
        "initial_inventory": {"semiconductor": 20, "capacitor": 10},
        "daily_demand": {"semiconductor": 60, "capacitor": 45},
        "reorder_threshold": {"semiconductor": 80, "capacitor": 60},
        "budget": 6000.0,
        "competing_bids": {
            "semiconductor": {
                "competitor": "RivalCorp", "competitor_quantity": 400,
                "steps_until_competitor_orders": 6, "supplier": "SupplierA",
            }
        },
        "suppliers": {
            "SupplierA": {"status": "healthy", "lead_days": 2, "cost_per_unit": 15.0, "products": ["semiconductor"], "remaining_capacity": 500},
            "SupplierB": {"status": "healthy", "lead_days": 3, "cost_per_unit": 0.5,  "products": ["capacitor"]},
            "SupplierC": {"status": "delayed", "lead_days": 12, "cost_per_unit": 12.0, "products": ["semiconductor"]},
        },
        "goal_orders": [
            {"product": "semiconductor", "supplier": "SupplierA", "min_quantity": 300},
            {"product": "capacitor",     "supplier": "SupplierB", "min_quantity": 200},
        ],
    },
}

# ── Procedural generators ──────────────────────────────────────────────────────

def _easy(task_id: int, rng: random.Random) -> dict:
    p      = rng.choice(PRODUCTS[:12])
    qty    = rng.randint(100, 400)
    stock  = rng.randint(15, 80)
    demand = rng.randint(20, 55)
    threshold = demand * 2
    sa_name = rng.choice(SUPPLIER_NAMES[:4])
    sb_name = rng.choice([s for s in SUPPLIER_NAMES[:6] if s != sa_name])
    cost_a  = round(rng.uniform(0.8, 3.5), 2)
    cost_b  = round(rng.uniform(0.6, cost_a), 2)
    return {
        "id": task_id,
        "difficulty": "easy",
        "type": "reorder",
        "description": (
            f"Warehouse stocks '{p}'. Stock: {stock} units. "
            f"Daily demand: {demand} units/day. "
            f"{sa_name} is healthy (lead: 2 days). "
            f"Task: Check inventory, verify {sa_name} is healthy, "
            f"and place an order of {qty} units with {sa_name}."
        ),
        "initial_inventory": {p: stock},
        "daily_demand": {p: demand},
        "reorder_threshold": {p: threshold},
        "suppliers": {
            sa_name: {"status": "healthy", "lead_days": 2, "cost_per_unit": cost_a},
            sb_name: {"status": rng.choice(["healthy", "delayed"]), "lead_days": 5, "cost_per_unit": cost_b},
        },
        "goal_product": p, "goal_supplier": sa_name, "goal_quantity": qty,
    }


def _medium_reroute(task_id: int, rng: random.Random) -> dict:
    p   = rng.choice(PRODUCTS[:14])
    qty = rng.randint(80, 250)
    shp = f"SHP-{task_id:03d}"
    sa  = rng.choice(SUPPLIER_NAMES[:3])
    sb  = rng.choice([s for s in SUPPLIER_NAMES[:5] if s != sa])
    stock  = rng.randint(10, 40)
    demand = rng.randint(15, 45)
    return {
        "id": task_id,
        "difficulty": "medium",
        "type": "reroute",
        "description": (
            f"SUPPLIER FAILED. You manage '{p}' stock: {stock} units. "
            f"Daily demand: {demand} units. "
            f"{sa} status: FAILED. {sb} is healthy. "
            f"Task: Reroute shipment {shp} from {sa} to {sb}, "
            f"then place emergency order of {qty} units with {sb}."
        ),
        "initial_inventory": {p: stock},
        "daily_demand": {p: demand},
        "reorder_threshold": {p: 50},
        "suppliers": {
            sa: {"status": "failed", "lead_days": 0, "cost_per_unit": round(rng.uniform(4.0, 8.0), 2)},
            sb: {"status": "healthy", "lead_days": rng.randint(2, 6), "cost_per_unit": round(rng.uniform(3.0, 7.0), 2)},
        },
        "pending_shipments": [
            {"shipment_id": shp, "supplier": sa, "quantity": rng.randint(60, 120), "product": p}
        ],
        "goal_product": p, "goal_supplier": sb, "goal_quantity": qty, "goal_reroute_shipment": shp,
    }


def _medium_demand_spike(task_id: int, rng: random.Random) -> dict:
    """
    FIX v4.1: Proper demand spike generator.
    Previously this was _medium_reroute() with type renamed — wrong goal structure,
    no demand_spike flag, and _all_goals_met() would check reroute goals that
    didn't exist, making the task impossible to complete correctly.
    """
    p          = rng.choice(PRODUCTS[:12])
    normal_demand = rng.randint(30, 70)
    # Spike is 3x normal demand
    spike_demand  = normal_demand * 3
    stock      = rng.randint(50, 150)
    # Order quantity must cover at least 5 days of spike demand
    qty        = rng.randint(spike_demand * 5, spike_demand * 8)
    sa_name    = rng.choice(SUPPLIER_NAMES[:3])
    sb_name    = rng.choice([s for s in SUPPLIER_NAMES[:5] if s != sa_name])
    cost_a     = round(rng.uniform(1.5, 5.0), 2)
    cost_b     = round(rng.uniform(1.2, cost_a), 2)
    return {
        "id": task_id,
        "difficulty": "medium",
        "type": "demand_spike",
        "description": (
            f"DEMAND SPIKE DETECTED. You manage '{p}'. "
            f"Normal demand: {normal_demand} units/day. "
            f"Demand just spiked 3x to {spike_demand} units/day. "
            f"Stock: {stock} units. {sa_name} is healthy. "
            f"Task: Check demand forecast (spike warning visible), "
            f"check inventory, and place an urgent order of {qty} units with {sa_name}."
        ),
        "initial_inventory": {p: stock},
        "daily_demand": {p: spike_demand},
        "demand_spike": True,
        "reorder_threshold": {p: normal_demand * 3},
        "suppliers": {
            sa_name: {"status": "healthy", "lead_days": rng.randint(1, 3), "cost_per_unit": cost_a},
            sb_name: {"status": "delayed", "lead_days": rng.randint(8, 14), "cost_per_unit": cost_b},
        },
        "goal_product": p,
        "goal_supplier": sa_name,
        "goal_quantity": qty,
    }


def _medium_price(task_id: int, rng: random.Random) -> dict:
    p       = rng.choice(PRODUCTS[10:])
    qty     = rng.randint(200, 500)
    budget  = round(qty * rng.uniform(2.2, 3.5), 2)
    prices  = sorted([round(rng.uniform(1.8, 4.0), 2) for _ in range(3)])
    names   = rng.sample(SUPPLIER_NAMES[4:], 3)
    sup = {
        names[0]: {"status": "healthy",  "lead_days": rng.randint(2, 4),  "cost_per_unit": prices[2]},
        names[1]: {"status": "healthy",  "lead_days": rng.randint(1, 3),  "cost_per_unit": prices[1]},
        names[2]: {"status": "delayed",  "lead_days": rng.randint(8, 15), "cost_per_unit": prices[0]},
    }
    mkt = {n: {p: v["cost_per_unit"], "quote_valid_hours": rng.randint(12, 48)} for n, v in sup.items()}
    return {
        "id": task_id,
        "difficulty": "medium",
        "type": "price_negotiation",
        "description": (
            f"PRICE NEGOTIATION. Need {qty} units of '{p}'. "
            f"Budget cap: ${budget:.2f}. "
            f"3 suppliers quote different prices. {names[2]} is delayed. "
            f"Task: Use get_market_prices, find cheapest healthy supplier, "
            f"and place order of {qty} units within budget."
        ),
        "initial_inventory": {p: rng.randint(5, 25)},
        "daily_demand": {p: rng.randint(50, 100)},
        "reorder_threshold": {p: 80},
        "budget": budget,
        "suppliers": sup,
        "market_prices": mkt,
        "goal_product": p, "goal_supplier": names[1], "goal_quantity": qty,
    }


def _hard_multicrisis(task_id: int, rng: random.Random) -> dict:
    prods  = rng.sample(PRODUCTS, 3)
    shp    = f"SHP-{task_id:03d}"
    budget = round(rng.uniform(1500, 5000), 2)
    sa, sb, sc, sd = rng.sample(SUPPLIER_NAMES, 4)
    return {
        "id": task_id,
        "difficulty": "hard",
        "type": "multi_product_crisis",
        "description": (
            f"MULTI-PRODUCT CRISIS. Products: {prods}. "
            f"{sa} FAILED. Reroute {shp} to {sd}. "
            f"Budget: ${budget:.2f}. "
            f"Task: Check all inventory, reroute {shp}, "
            f"order all 3 products within budget."
        ),
        "initial_inventory": {p: rng.randint(5, 20) for p in prods},
        "daily_demand": {p: rng.randint(20, 55) for p in prods},
        "reorder_threshold": {p: 60 for p in prods},
        "budget": budget,
        "suppliers": {
            sa: {"status": "failed",  "lead_days": 0, "cost_per_unit": round(rng.uniform(3, 6), 2), "products": [prods[0]]},
            sb: {"status": "healthy", "lead_days": rng.randint(2, 5), "cost_per_unit": round(rng.uniform(2, 5), 2), "products": [prods[1]]},
            sc: {"status": "delayed", "lead_days": rng.randint(6, 10), "cost_per_unit": round(rng.uniform(1, 4), 2), "products": [prods[2]]},
            sd: {"status": "healthy", "lead_days": rng.randint(3, 6), "cost_per_unit": round(rng.uniform(3, 7), 2), "products": prods},
        },
        "pending_shipments": [
            {"shipment_id": shp, "supplier": sa, "quantity": rng.randint(80, 150), "product": prods[0]}
        ],
        "goal_orders": [
            {"product": prods[0], "supplier": sd, "min_quantity": rng.randint(60, 120)},
            {"product": prods[1], "supplier": sb, "min_quantity": rng.randint(50, 100)},
            {"product": prods[2], "supplier": sd, "min_quantity": rng.randint(40, 90)},
        ],
        "goal_reroute_shipment": shp,
    }


def _hard_quality(task_id: int, rng: random.Random) -> dict:
    prods     = rng.sample(PRODUCTS[11:16], 2)
    shp       = f"SHP-{task_id:03d}"
    budget    = round(rng.uniform(5000, 12000), 2)
    threshold = rng.choice([0.03, 0.05, 0.08])
    bad_rate  = round(rng.uniform(0.15, 0.45), 3)
    good_rate = round(rng.uniform(0.0, threshold - 0.005), 3)
    names     = rng.sample(SUPPLIER_NAMES, 3)
    sa, sb, sc = names
    return {
        "id": task_id,
        "difficulty": "hard",
        "type": "quality_control",
        "description": (
            f"QUALITY CRISIS. Products: {prods}. "
            f"{sa} defect rate: {bad_rate*100:.1f}% — exceeds {threshold*100:.0f}% threshold. "
            f"{sb} defect rate: {good_rate*100:.1f}% — compliant. "
            f"Cancel defective shipment {shp}. Budget: ${budget:.2f}. "
            f"Task: Use get_quality_report, cancel {shp}, "
            f"order ONLY from suppliers with defect rate < {threshold*100:.0f}%."
        ),
        "initial_inventory": {p: rng.randint(20, 60) for p in prods},
        "daily_demand": {p: rng.randint(50, 120) for p in prods},
        "reorder_threshold": {p: 100 for p in prods},
        "budget": budget,
        "quality_threshold": threshold,
        "suppliers": {
            sa: {"status": "healthy", "lead_days": 1, "cost_per_unit": round(rng.uniform(5, 10), 2),  "products": prods, "defect_rate": bad_rate},
            sb: {"status": "healthy", "lead_days": 2, "cost_per_unit": round(rng.uniform(8, 15), 2),  "products": prods, "defect_rate": good_rate},
            sc: {"status": "healthy", "lead_days": 3, "cost_per_unit": round(rng.uniform(10, 18), 2), "products": prods, "defect_rate": 0.0},
        },
        "pending_shipments": [
            {"shipment_id": shp, "supplier": sa, "quantity": rng.randint(100, 250), "product": prods[0]}
        ],
        "goal_orders": [
            {"product": prods[0], "supplier": sb, "min_quantity": rng.randint(200, 400), "allowed_suppliers": [sb, sc]},
            {"product": prods[1], "supplier": sb, "min_quantity": rng.randint(150, 300), "allowed_suppliers": [sb, sc]},
        ],
        "goal_cancel_shipment": shp,
    }


def _hard_competing(task_id: int, rng: random.Random) -> dict:
    prods    = rng.sample(PRODUCTS[13:], 2)
    budget   = round(rng.uniform(4000, 9000), 2)
    steps    = rng.randint(4, 8)
    comp     = rng.choice(COMPETITOR_NAMES)
    cap      = rng.randint(400, 700)
    comp_qty = rng.randint(250, cap - 100)
    sa, sb   = rng.sample(SUPPLIER_NAMES, 2)
    price_a  = round(rng.uniform(10, 20), 2)
    price_b  = round(rng.uniform(0.3, 2.0), 2)
    min_qty_a = rng.randint(200, 350)
    min_qty_b = rng.randint(150, 300)
    return {
        "id": task_id,
        "difficulty": "hard",
        "type": "competing_buyer",
        "description": (
            f"ADVERSARIAL MARKET. {comp} will order {comp_qty} units of "
            f"'{prods[0]}' from {sa} in ~{steps} steps. "
            f"{sa} only has {cap} units left. Budget: ${budget:.2f}. "
            f"Task: Use get_competing_bids to assess urgency, "
            f"secure {prods[0]} from {sa} (min {min_qty_a}), "
            f"and order {prods[1]} from {sb} (min {min_qty_b})."
        ),
        "initial_inventory": {prods[0]: rng.randint(10, 30), prods[1]: rng.randint(5, 20)},
        "daily_demand": {prods[0]: rng.randint(40, 80), prods[1]: rng.randint(30, 60)},
        "reorder_threshold": {prods[0]: 70, prods[1]: 50},
        "budget": budget,
        "competing_bids": {
            prods[0]: {
                "competitor": comp, "competitor_quantity": comp_qty,
                "steps_until_competitor_orders": steps, "supplier": sa,
            }
        },
        "suppliers": {
            sa: {"status": "healthy", "lead_days": 2, "cost_per_unit": price_a, "products": [prods[0]], "remaining_capacity": cap},
            sb: {"status": "healthy", "lead_days": 3, "cost_per_unit": price_b, "products": [prods[1]]},
        },
        "goal_orders": [
            {"product": prods[0], "supplier": sa, "min_quantity": min_qty_a},
            {"product": prods[1], "supplier": sb, "min_quantity": min_qty_b},
        ],
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_task(task_id: int, seed: int = None) -> dict:
    """
    Generate a single task deterministically from task_id.

    Fixed IDs (0,1,2,5,6,7,10,11,12,13) return the original hardcoded tasks.
    Any other ID is generated procedurally using task_id as the seed.

    Difficulty routing for non-fixed IDs:
      0–49   → easy (reorder)
      50–99  → medium: even=reroute, odd=demand_spike
      100–149→ medium (price_negotiation)
      150–199→ hard (multi_product_crisis)
      200–249→ hard (quality_control)
      250–299→ hard (competing_buyer)
      300+   → cycles through all hard types
    """
    if task_id in _FIXED_TASKS:
        return dict(_FIXED_TASKS[task_id])

    rng = random.Random(seed if seed is not None else task_id)

    if task_id < 50:
        return _easy(task_id, rng)
    elif task_id < 100:
        # FIX: even=reroute, odd=proper demand_spike (not reroute renamed)
        if task_id % 2 == 0:
            return _medium_reroute(task_id, rng)
        else:
            return _medium_demand_spike(task_id, rng)
    elif task_id < 150:
        return _medium_price(task_id, rng)
    elif task_id < 200:
        return _hard_multicrisis(task_id, rng)
    elif task_id < 250:
        return _hard_quality(task_id, rng)
    elif task_id < 300:
        return _hard_competing(task_id, rng)
    else:
        funcs = [_hard_multicrisis, _hard_quality, _hard_competing]
        return funcs[task_id % 3](task_id, rng)


def generate_batch(
    count: int = 6,
    seed: int = 42,
    difficulty: str = "all",
) -> list[dict]:
    """Generate a batch of tasks. Used by CLI and tests."""
    rng    = random.Random(seed)
    levels = ["easy", "medium", "hard"] if difficulty == "all" else [difficulty]
    tasks  = []
    for i in range(count):
        level = levels[i % len(levels)]
        if level == "easy":
            start = 20
        elif level == "medium":
            start = 60
        else:
            start = 160
        tid = start + rng.randint(0, 29)
        tasks.append(generate_task(tid, seed=seed + i))
    return tasks


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Supply chain task generator")
    ap.add_argument("--task_id",    type=int,   default=None)
    ap.add_argument("--count",      type=int,   default=6)
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--difficulty", type=str,   default="all",
                    choices=["easy", "medium", "hard", "all"])
    ap.add_argument("--pretty",     action="store_true")
    args = ap.parse_args()

    indent = 2 if args.pretty else None

    if args.task_id is not None:
        task = generate_task(args.task_id, seed=args.seed)
        print(json.dumps(task, indent=indent))
    else:
        tasks = generate_batch(count=args.count, seed=args.seed, difficulty=args.difficulty)
        print(json.dumps(tasks, indent=indent))
        print(f"\nGenerated {len(tasks)} tasks (seed={args.seed}).", flush=True)


if __name__ == "__main__":
    main()
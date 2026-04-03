# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Supply Chain Disruption Environment — Enhanced v2.

Inspired by the 5 SF Hackathon reference projects:
- Calendar Env  → multi-user coordination, ACL permissions, ambiguity
- Reasoning Gym → diverse task types, programmatic exact-match grading
- TB2 Env       → delayed rewards, multi-hop reasoning chains
- CARLA Env     → rich state, continuous metrics, adversarial elements
- REPL Env      → persistent session state, iterative tool use

Enhancements over v1:
  1. 3 NEW task types: price_negotiation, quality_control, competing_buyer
  2. Continuous budget-efficiency bonus on Hard tasks (richer reward surface)
  3. Adversarial competing-buyer element on Hard tasks (time pressure)
  4. Dense per-tool observation feedback (not just tool result strings)
  5. max_steps raised to 25 across all difficulties
  6. get_market_prices tool (price negotiation tasks)
  7. get_quality_report tool (quality control tasks)
  8. get_competing_bids tool (adversarial tasks)
"""

import random
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SupplyChainAction, SupplyChainObservation
except ImportError:
    from models import SupplyChainAction, SupplyChainObservation

# ── Task Definitions ──────────────────────────────────────────────────────────

TASKS = [
    # ── EASY ─────────────────────────────────────────────────────────────────
    {
        "id": 0,
        "difficulty": "easy",
        "type": "reorder",
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
        "goal_product": "bottled_water",
        "goal_supplier": "SupplierA",
        "goal_quantity": 200,
    },
    {
        "id": 1,
        "difficulty": "easy",
        "type": "reorder",
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
        "goal_product": "canned_soup",
        "goal_supplier": "SupplierA",
        "goal_quantity": 150,
    },
    {
        "id": 2,
        "difficulty": "easy",
        "type": "reorder",
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
        "goal_product": "coffee_beans",
        "goal_supplier": "SupplierC",
        "goal_quantity": 300,
    },

    # ── MEDIUM ────────────────────────────────────────────────────────────────
    {
        "id": 5,
        "difficulty": "medium",
        "type": "reroute",
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
            {"shipment_id": "SHP-001", "supplier": "SupplierA",
             "quantity": 80, "product": "circuit_board"}
        ],
        "goal_product": "circuit_board",
        "goal_supplier": "SupplierB",
        "goal_quantity": 100,
        "goal_reroute_shipment": "SHP-001",
    },
    {
        "id": 6,
        "difficulty": "medium",
        "type": "demand_spike",
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
        "goal_product": "pain_reliever",
        "goal_supplier": "SupplierA",
        "goal_quantity": 500,
    },

    # ── NEW: MEDIUM — Price Negotiation ───────────────────────────────────────
    {
        "id": 7,
        "difficulty": "medium",
        "type": "price_negotiation",
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
        "goal_product": "raw_steel",
        "goal_supplier": "SupplierY",   # cheapest healthy, within budget (400*2.5=1000 < 1200)
        "goal_quantity": 400,
    },

    # ── HARD ──────────────────────────────────────────────────────────────────
    {
        "id": 10,
        "difficulty": "hard",
        "type": "multi_product_crisis",
        "description": (
            "MULTI-PRODUCT CRISIS. Three products are critically low. "
            "Inventory: {'mask': 10, 'glove': 5, 'sanitizer': 15}. "
            "Daily demand: {'mask': 50, 'glove': 40, 'sanitizer': 30}. "
            "SupplierA (masks) just failed. SupplierB (gloves) is healthy. "
            "SupplierC (sanitizer) is delayed. "
            "Shipment SHP-010 from SupplierA (200 masks) must be rerouted to SupplierD. "
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
            {"shipment_id": "SHP-010", "supplier": "SupplierA",
             "quantity": 200, "product": "mask"}
        ],
        "goal_orders": [
            {"product": "mask", "supplier": "SupplierD", "min_quantity": 150},
            {"product": "glove", "supplier": "SupplierB", "min_quantity": 100},
            {"product": "sanitizer", "supplier": "SupplierD", "min_quantity": 80},
        ],
        "goal_reroute_shipment": "SHP-010",
    },
    {
        "id": 11,
        "difficulty": "hard",
        "type": "port_strike",
        "description": (
            "PORT STRIKE + SUPPLIER FAILURE. You manage auto parts. "
            "Stock: {'engine_part': 8, 'brake_pad': 12}. "
            "All overseas suppliers are on strike (SupplierA, SupplierB). "
            "Only domestic SupplierC is available but expensive. "
            "Shipment SHP-011 is stuck at port. Budget: $5000. "
            "Daily demand: {'engine_part': 10, 'brake_pad': 15}. "
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
            {"shipment_id": "SHP-011", "supplier": "SupplierA",
             "quantity": 50, "product": "engine_part"}
        ],
        "goal_orders": [
            {"product": "engine_part", "supplier": "SupplierC", "min_quantity": 30},
            {"product": "brake_pad",   "supplier": "SupplierC", "min_quantity": 40},
        ],
        "goal_cancel_shipment": "SHP-011",
    },

    # ── NEW: HARD — Quality Control ───────────────────────────────────────────
    {
        "id": 12,
        "difficulty": "hard",
        "type": "quality_control",
        "description": (
            "QUALITY CRISIS. Defective batches detected across suppliers. "
            "Inventory: {'vaccine_vial': 50, 'syringe': 30}. "
            "Daily demand: {'vaccine_vial': 100, 'syringe': 80}. "
            "SupplierA has a 35% defect rate — unacceptable for medical supply. "
            "SupplierB has a 4% defect rate — within acceptable threshold (< 5%). "
            "SupplierC is healthy with 0% defects but costs more. "
            "Pending shipment SHP-012 from SupplierA must be cancelled. "
            "Budget: $8000. "
            "Task: Use get_quality_report to check defect rates, "
            "cancel the defective shipment, and place orders ONLY with "
            "suppliers whose defect rate is below 5%."
        ),
        "initial_inventory": {"vaccine_vial": 50, "syringe": 30},
        "daily_demand": {"vaccine_vial": 100, "syringe": 80},
        "reorder_threshold": {"vaccine_vial": 120, "syringe": 100},
        "budget": 8000.0,
        "quality_threshold": 0.05,   # 5% max defect rate
        "suppliers": {
            "SupplierA": {"status": "healthy", "lead_days": 1, "cost_per_unit": 8.0,  "products": ["vaccine_vial", "syringe"], "defect_rate": 0.35},
            "SupplierB": {"status": "healthy", "lead_days": 2, "cost_per_unit": 10.0, "products": ["vaccine_vial", "syringe"], "defect_rate": 0.04},
            "SupplierC": {"status": "healthy", "lead_days": 3, "cost_per_unit": 12.0, "products": ["vaccine_vial", "syringe"], "defect_rate": 0.00},
        },
        "pending_shipments": [
            {"shipment_id": "SHP-012", "supplier": "SupplierA",
             "quantity": 200, "product": "vaccine_vial"}
        ],
        "goal_orders": [
            {"product": "vaccine_vial", "supplier": "SupplierB", "min_quantity": 300,
             "allowed_suppliers": ["SupplierB", "SupplierC"]},
            {"product": "syringe",      "supplier": "SupplierB", "min_quantity": 200,
             "allowed_suppliers": ["SupplierB", "SupplierC"]},
        ],
        "goal_cancel_shipment": "SHP-012",
    },

    # ── NEW: HARD — Competing Buyer (Adversarial) ─────────────────────────────
    {
        "id": 13,
        "difficulty": "hard",
        "type": "competing_buyer",
        "description": (
            "ADVERSARIAL MARKET. A competing company is also bidding for "
            "the same limited supplier capacity. "
            "Inventory: {'semiconductor': 20, 'capacitor': 10}. "
            "Daily demand: {'semiconductor': 60, 'capacitor': 45}. "
            "SupplierA has only 500 units of semiconductor left — "
            "a competing buyer is placing their order SOON. "
            "Use get_competing_bids to see how much time you have. "
            "SupplierB is healthy for capacitors. Budget: $6000. "
            "Task: Check competing bids urgency, secure semiconductor "
            "from SupplierA (min 300 units) before competitor, "
            "AND place capacitor order with SupplierB (min 200 units)."
        ),
        "initial_inventory": {"semiconductor": 20, "capacitor": 10},
        "daily_demand": {"semiconductor": 60, "capacitor": 45},
        "reorder_threshold": {"semiconductor": 80, "capacitor": 60},
        "budget": 6000.0,
        "competing_bids": {
            "semiconductor": {
                "competitor": "RivalCorp",
                "competitor_quantity": 400,
                "steps_until_competitor_orders": 6,   # agent has 6 steps to act
                "supplier": "SupplierA",
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
]


# ── Environment Class ─────────────────────────────────────────────────────────

class SupplyChainEnvironment(Environment):
    """
    Enhanced stateful supply chain disruption environment.
    Implements the OpenEnv Environment interface.

    New in v2:
    - 4 new task types: price_negotiation, quality_control, competing_buyer
    - Continuous budget-efficiency bonus on hard tasks
    - Adversarial competing-buyer time pressure
    - 3 new tools: get_market_prices, get_quality_report, get_competing_bids
    - Max steps raised to 25
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    MAX_STEPS: int = 25

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.task = None
        self.inventory = {}
        self.suppliers = {}
        self.pending_shipments = []
        self.orders_placed = []
        self.shipments_rerouted = []
        self.shipments_cancelled = []
        self.spent_budget = 0.0
        self.done = False
        self._competing_bids_remaining = {}  # tracks steps until competitor acts

    def reset(self, task_id: int = 0, seed: int = None) -> SupplyChainObservation:
        """Reset environment to a fresh task."""
        if seed is not None:
            random.seed(seed)

        task = next((t for t in TASKS if t["id"] == task_id), TASKS[0])
        self.task = task
        self.inventory = dict(task["initial_inventory"])
        self.suppliers = {k: dict(v) for k, v in task["suppliers"].items()}
        self.pending_shipments = [dict(s) for s in task.get("pending_shipments", [])]
        self.orders_placed = []
        self.shipments_rerouted = []
        self.shipments_cancelled = []
        self.spent_budget = 0.0
        self.done = False
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # Init competing buyer countdown
        self._competing_bids_remaining = {}
        for product, bid in task.get("competing_bids", {}).items():
            self._competing_bids_remaining[product] = bid["steps_until_competitor_orders"]

        return SupplyChainObservation(
            text=task["description"],
            state=self._get_state_dict(),
            done=False,
            reward=0.0,
        )

    def step(self, action: SupplyChainAction) -> SupplyChainObservation:
        """Execute one agent action."""
        if self.done:
            return SupplyChainObservation(
                text="Episode is already finished.",
                state=self._get_state_dict(),
                done=True,
                reward=0.0,
            )

        self._state.step_count += 1

        # Tick down competing buyer countdowns
        self._tick_competing_bids()

        tool = action.tool
        args = action.args

        handlers = {
            "get_inventory":         self._tool_get_inventory,
            "check_supplier_status": self._tool_check_supplier_status,
            "get_demand_forecast":   self._tool_get_demand_forecast,
            "place_order":           self._tool_place_order,
            "reroute_shipment":      self._tool_reroute_shipment,
            "cancel_shipment":       self._tool_cancel_shipment,
            "get_pending_shipments": self._tool_get_pending_shipments,
            # New tools
            "get_market_prices":     self._tool_get_market_prices,
            "get_quality_report":    self._tool_get_quality_report,
            "get_competing_bids":    self._tool_get_competing_bids,
        }

        if tool in handlers:
            result_text = handlers[tool](args)
        else:
            result_text = (
                f"Unknown tool '{tool}'. "
                f"Available: {list(handlers.keys())}"
            )

        reward = self._compute_reward()
        self.done = (reward >= 1.0) or (self._state.step_count >= self.MAX_STEPS)

        return SupplyChainObservation(
            text=result_text,
            state=self._get_state_dict(),
            done=self.done,
            reward=reward,
        )

    @property
    def state(self) -> State:
        return self._state

    # ── Tools ─────────────────────────────────────────────────────────────────

    def _tool_get_inventory(self, args: dict) -> str:
        lines = [f"  {p}: {q} units" for p, q in self.inventory.items()]
        threshold_lines = []
        for p, q in self.inventory.items():
            thresh = self.task.get("reorder_threshold", {}).get(p, 0)
            demand = self.task.get("daily_demand", {}).get(p, 1)
            days_left = round(q / demand, 1) if demand > 0 else "∞"
            status = "⚠️ BELOW THRESHOLD" if q < thresh else "OK"
            threshold_lines.append(f"  {p}: {q} units | days left: {days_left} | {status}")
        return "Current inventory:\n" + "\n".join(threshold_lines)

    def _tool_check_supplier_status(self, args: dict) -> str:
        name = args.get("supplier_name", "")
        if name not in self.suppliers:
            return f"Supplier '{name}' not found. Known: {list(self.suppliers.keys())}"
        s = self.suppliers[name]
        products = s.get("products", list(self.inventory.keys()))
        defect_line = ""
        if "defect_rate" in s:
            pct = s["defect_rate"] * 100
            defect_line = f"\n  Defect rate: {pct:.1f}%"
        capacity_line = ""
        if "remaining_capacity" in s:
            capacity_line = f"\n  Remaining capacity: {s['remaining_capacity']} units"
        return (
            f"Supplier '{name}':\n"
            f"  Status: {s['status']}\n"
            f"  Lead time: {s['lead_days']} day(s)\n"
            f"  Cost/unit: ${s['cost_per_unit']:.2f}\n"
            f"  Products: {products}"
            f"{defect_line}{capacity_line}"
        )

    def _tool_get_demand_forecast(self, args: dict) -> str:
        product = args.get("product", "")
        demand = self.task.get("daily_demand", {})
        if product not in demand:
            return f"No demand data for '{product}'. Known: {list(demand.keys())}"
        spike = self.task.get("demand_spike", False)
        spike_note = " ⚠️ WARNING: ACTIVE DEMAND SPIKE (3x normal)" if spike else ""
        days = (
            self.inventory.get(product, 0) / demand[product]
            if demand[product] > 0 else float("inf")
        )
        return (
            f"Demand forecast for '{product}':\n"
            f"  Daily demand: {demand[product]} units/day{spike_note}\n"
            f"  Current stock: {self.inventory.get(product, 0)} units\n"
            f"  Days until stockout: {days:.1f} days"
        )

    def _tool_place_order(self, args: dict) -> str:
        supplier_name = args.get("supplier_name", "")
        product       = args.get("product", "")
        quantity      = int(args.get("quantity", 0))

        if quantity <= 0:
            return "Order failed: quantity must be greater than 0."
        if supplier_name not in self.suppliers:
            return f"Order failed: supplier '{supplier_name}' not found. Known: {list(self.suppliers.keys())}"
        supplier = self.suppliers[supplier_name]
        if supplier["status"] in ("failed", "strike"):
            return f"Order failed: '{supplier_name}' is {supplier['status']} and cannot fulfil orders."
        if product not in self.inventory:
            return f"Order failed: product '{product}' not in catalogue. Known: {list(self.inventory.keys())}"

        # Quality gate
        if "quality_threshold" in self.task:
            defect = supplier.get("defect_rate", 0.0)
            if defect > self.task["quality_threshold"]:
                return (
                    f"Order REJECTED: '{supplier_name}' defect rate {defect*100:.1f}% "
                    f"exceeds threshold of {self.task['quality_threshold']*100:.1f}%. "
                    f"Use get_quality_report to find compliant suppliers."
                )

        # Capacity gate (competing buyer tasks)
        if "remaining_capacity" in supplier:
            if quantity > supplier["remaining_capacity"]:
                return (
                    f"Order failed: '{supplier_name}' only has "
                    f"{supplier['remaining_capacity']} units left. "
                    f"Requested {quantity}."
                )
            supplier["remaining_capacity"] -= quantity

        cost = quantity * supplier["cost_per_unit"]
        if "budget" in self.task:
            remaining = self.task["budget"] - self.spent_budget
            if cost > remaining:
                return (
                    f"Order failed: cost ${cost:.2f} exceeds remaining "
                    f"budget ${remaining:.2f}."
                )
            self.spent_budget += cost

        self.orders_placed.append({
            "supplier": supplier_name,
            "product": product,
            "quantity": quantity,
        })
        budget_note = ""
        if "budget" in self.task:
            budget_note = f"\n  Remaining budget: ${self.task['budget'] - self.spent_budget:.2f}"
        return (
            f"SUCCESS: Order placed!\n"
            f"  Supplier: {supplier_name}\n"
            f"  Product: {product}\n"
            f"  Quantity: {quantity} units\n"
            f"  Cost: ${cost:.2f}\n"
            f"  ETA: {supplier['lead_days']} day(s)"
            f"{budget_note}"
        )

    def _tool_reroute_shipment(self, args: dict) -> str:
        shipment_id  = args.get("shipment_id", "")
        new_supplier = args.get("new_supplier", "")
        shipment = next(
            (s for s in self.pending_shipments if s["shipment_id"] == shipment_id), None
        )
        if not shipment:
            ids = [s["shipment_id"] for s in self.pending_shipments]
            return f"Shipment '{shipment_id}' not found. Pending: {ids}"
        if new_supplier not in self.suppliers:
            return f"Supplier '{new_supplier}' not found."
        new_sup = self.suppliers[new_supplier]
        if new_sup["status"] in ("failed", "strike"):
            return f"Cannot reroute to '{new_supplier}': supplier is {new_sup['status']}."
        old = shipment["supplier"]
        shipment["supplier"] = new_supplier
        self.shipments_rerouted.append({
            "shipment_id": shipment_id,
            "from_supplier": old,
            "to_supplier": new_supplier,
        })
        return (
            f"SUCCESS: Shipment {shipment_id} rerouted!\n"
            f"  From: {old} → To: {new_supplier}\n"
            f"  New ETA: {new_sup['lead_days']} day(s)"
        )

    def _tool_cancel_shipment(self, args: dict) -> str:
        shipment_id = args.get("shipment_id", "")
        shipment = next(
            (s for s in self.pending_shipments if s["shipment_id"] == shipment_id), None
        )
        if not shipment:
            ids = [s["shipment_id"] for s in self.pending_shipments]
            return f"Shipment '{shipment_id}' not found. Pending: {ids}"
        self.pending_shipments.remove(shipment)
        self.shipments_cancelled.append(shipment_id)
        return f"SUCCESS: Shipment {shipment_id} cancelled."

    def _tool_get_pending_shipments(self, args: dict) -> str:
        if not self.pending_shipments:
            return "No pending shipments."
        lines = [
            f"  [{s['shipment_id']}] {s['quantity']}x {s['product']} from {s['supplier']}"
            for s in self.pending_shipments
        ]
        return "Pending shipments:\n" + "\n".join(lines)

    # ── New tools ──────────────────────────────────────────────────────────────

    def _tool_get_market_prices(self, args: dict) -> str:
        """Compare supplier quotes for price negotiation tasks."""
        prices = self.task.get("market_prices", {})
        if not prices:
            return "No market price data available for this task."
        lines = []
        for supplier, data in prices.items():
            status = self.suppliers.get(supplier, {}).get("status", "unknown")
            for product, price in data.items():
                if product == "quote_valid_hours":
                    continue
                valid = data.get("quote_valid_hours", "?")
                total_400 = price * 400
                lines.append(
                    f"  {supplier} [{status}]: ${price:.2f}/unit | "
                    f"400 units = ${total_400:.2f} | "
                    f"quote valid {valid}h"
                )
        budget = self.task.get("budget", None)
        budget_line = f"\nYour budget cap: ${budget:.2f}" if budget else ""
        return "Market price comparison:\n" + "\n".join(lines) + budget_line

    def _tool_get_quality_report(self, args: dict) -> str:
        """Get defect rate report for quality control tasks."""
        threshold = self.task.get("quality_threshold", None)
        if threshold is None:
            return "No quality data available for this task."
        lines = []
        for name, s in self.suppliers.items():
            defect = s.get("defect_rate", None)
            if defect is None:
                lines.append(f"  {name}: No quality data available")
                continue
            pct = defect * 100
            verdict = "✅ PASS" if defect <= threshold else "❌ FAIL — DO NOT ORDER"
            lines.append(f"  {name}: {pct:.1f}% defect rate | {verdict}")
        return (
            f"Quality Report (threshold: {threshold*100:.1f}% max defects):\n"
            + "\n".join(lines)
        )

    def _tool_get_competing_bids(self, args: dict) -> str:
        """Show competing buyer urgency for adversarial tasks."""
        bids = self.task.get("competing_bids", {})
        if not bids:
            return "No competing bids in this market."
        lines = []
        for product, bid in bids.items():
            steps_left = self._competing_bids_remaining.get(product, 0)
            urgency = "🚨 CRITICAL" if steps_left <= 2 else "⚠️ URGENT" if steps_left <= 4 else "⏳ ACT SOON"
            lines.append(
                f"  {product}: {bid['competitor']} will order "
                f"{bid['competitor_quantity']} units from {bid['supplier']} "
                f"in ~{steps_left} steps. {urgency}"
            )
        return "Competing buyer intelligence:\n" + "\n".join(lines)

    # ── Reward ─────────────────────────────────────────────────────────────────

    def _compute_reward(self) -> float:
        score = 0.0

        # Step 1 — took any action
        if self._state.step_count >= 1:
            score = max(score, 0.10)

        # Step 2 — placed any order or rerouted
        if self.orders_placed or self.shipments_rerouted:
            score = max(score, 0.50)

        difficulty = self.task["difficulty"]

        # Step 3 — key sub-task for medium/hard
        if difficulty in ("medium", "hard"):
            reroute_goal = self.task.get("goal_reroute_shipment")
            cancel_goal  = self.task.get("goal_cancel_shipment")
            if reroute_goal:
                if reroute_goal in [r["shipment_id"] for r in self.shipments_rerouted]:
                    score = max(score, 0.75)
            elif cancel_goal:
                if cancel_goal in self.shipments_cancelled:
                    score = max(score, 0.75)
            elif self.orders_placed:
                score = max(score, 0.65)

        # Step 4 — all goals met → 1.0 base, then apply efficiency bonus
        if self._all_goals_met():
            base = 1.0

            # Continuous budget-efficiency bonus on hard tasks
            # Agent gets rewarded more for spending less of the budget
            if difficulty == "hard" and "budget" in self.task:
                total_budget = self.task["budget"]
                spent = self.spent_budget
                if total_budget > 0 and spent <= total_budget:
                    efficiency = (total_budget - spent) / total_budget
                    # Bonus: up to +0.15 for using only 50% of budget
                    # Smoothly interpolated so any saving counts
                    bonus = round(0.15 * efficiency, 4)
                    base = min(1.15, 1.0 + bonus)

            score = base

        # Speed bonus: completing in ≤ 5 steps on easy (clean signal for RL)
        if difficulty == "easy" and self._all_goals_met() and self._state.step_count <= 5:
            score = min(score, 1.0)  # cap at 1.0 for easy

        return round(score, 4)

    def _all_goals_met(self) -> bool:
        task = self.task
        difficulty = task["difficulty"]
        task_type  = task.get("type", "reorder")

        if difficulty == "easy":
            for order in self.orders_placed:
                if (order["supplier"] == task.get("goal_supplier")
                        and order["product"] == task.get("goal_product")
                        and order["quantity"] >= task.get("goal_quantity", 0)):
                    return True
            return False

        if difficulty == "medium":
            if task_type == "price_negotiation":
                # Must place order with cheapest healthy supplier within budget
                for order in self.orders_placed:
                    if (order["supplier"] == task.get("goal_supplier")
                            and order["product"] == task.get("goal_product")
                            and order["quantity"] >= task.get("goal_quantity", 0)):
                        return True
                return False

            reroute_goal = task.get("goal_reroute_shipment")
            reroute_ok = (
                reroute_goal in [r["shipment_id"] for r in self.shipments_rerouted]
                if reroute_goal else True
            )
            order_ok = any(
                o["supplier"] == task.get("goal_supplier")
                and o["product"] == task.get("goal_product")
                and o["quantity"] >= task.get("goal_quantity", 0)
                for o in self.orders_placed
            )
            return reroute_ok and order_ok

        if difficulty == "hard":
            # Check all goal orders placed
            for goal in task.get("goal_orders", []):
                allowed = goal.get("allowed_suppliers", [goal["supplier"]])
                if not any(
                    o["supplier"] in allowed
                    and o["product"] == goal["product"]
                    and o["quantity"] >= goal["min_quantity"]
                    for o in self.orders_placed
                ):
                    return False

            # Check reroute goal
            if task.get("goal_reroute_shipment"):
                if task["goal_reroute_shipment"] not in [
                    r["shipment_id"] for r in self.shipments_rerouted
                ]:
                    return False

            # Check cancel goal
            if task.get("goal_cancel_shipment"):
                if task["goal_cancel_shipment"] not in self.shipments_cancelled:
                    return False

            return True

        return False

    # ── Competing buyer ticker ─────────────────────────────────────────────────

    def _tick_competing_bids(self):
        """Decrement competing buyer countdown each step."""
        for product in list(self._competing_bids_remaining.keys()):
            if self._competing_bids_remaining[product] > 0:
                self._competing_bids_remaining[product] -= 1
            # When countdown hits 0, competitor locks up supplier capacity
            if self._competing_bids_remaining[product] <= 0:
                bid = self.task.get("competing_bids", {}).get(product, {})
                supplier_name = bid.get("supplier")
                if supplier_name and supplier_name in self.suppliers:
                    sup = self.suppliers[supplier_name]
                    cap = sup.get("remaining_capacity", 0)
                    competitor_qty = bid.get("competitor_quantity", 0)
                    sup["remaining_capacity"] = max(0, cap - competitor_qty)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_state_dict(self) -> dict:
        state = {
            "inventory": dict(self.inventory),
            "steps": self._state.step_count,
            "max_steps": self.MAX_STEPS,
            "orders_placed": list(self.orders_placed),
            "shipments_rerouted": list(self.shipments_rerouted),
            "shipments_cancelled": list(self.shipments_cancelled),
            "pending_shipments": list(self.pending_shipments),
            "spent_budget": round(self.spent_budget, 2),
            "task_type": self.task.get("type", "reorder") if self.task else None,
            "difficulty": self.task.get("difficulty") if self.task else None,
        }
        if "budget" in (self.task or {}):
            state["remaining_budget"] = round(
                self.task["budget"] - self.spent_budget, 2
            )
        if self._competing_bids_remaining:
            state["competing_bids_countdown"] = dict(self._competing_bids_remaining)
        return state
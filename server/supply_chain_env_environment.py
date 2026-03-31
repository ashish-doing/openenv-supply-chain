# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Supply Chain Disruption Environment.

An AI agent manages a warehouse supply chain, responding to supplier
failures, demand spikes, and shipment disruptions. The agent must use
tools to inspect inventory, assess supplier health, reroute shipments,
and place orders to prevent stockouts.

Difficulty levels:
    easy   - Single product, healthy suppliers, straightforward reorder
    medium - One supplier failure or demand spike requiring rerouting
    hard   - Multi-product crisis with budget constraints
"""

import random
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SupplyChainAction, SupplyChainObservation
except ImportError:
    from models import SupplyChainAction, SupplyChainObservation

# ── Task Definitions ───────────────────────────────────────────────────────
TASKS = [
    # EASY
    {
        "id": 0,
        "difficulty": "easy",
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
    # MEDIUM
    {
        "id": 5,
        "difficulty": "medium",
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
    # HARD
    {
        "id": 10,
        "difficulty": "hard",
        "description": (
            "MULTI-PRODUCT CRISIS. Three products are critically low. "
            "Inventory: {'mask': 10, 'glove': 5, 'sanitizer': 15}. "
            "Daily demand: {'mask': 50, 'glove': 40, 'sanitizer': 30}. "
            "SupplierA (masks) just failed. SupplierB (gloves) is healthy. "
            "SupplierC (sanitizer) is delayed. "
            "Shipment SHP-010 from SupplierA (200 masks) must be rerouted to SupplierD. "
            "Budget: $2000. Task: Check all inventory, reroute SHP-010, "
            "and place orders for all three products."
        ),
        "initial_inventory": {"mask": 10, "glove": 5, "sanitizer": 15},
        "daily_demand": {"mask": 50, "glove": 40, "sanitizer": 30},
        "reorder_threshold": {"mask": 60, "glove": 50, "sanitizer": 40},
        "budget": 2000.0,
        "suppliers": {
            "SupplierA": {"status": "failed",  "lead_days": 0, "cost_per_unit": 2.0, "products": ["mask"]},
            "SupplierB": {"status": "healthy", "lead_days": 3, "cost_per_unit": 1.5, "products": ["glove"]},
            "SupplierC": {"status": "delayed", "lead_days": 8, "cost_per_unit": 1.8, "products": ["sanitizer"]},
            "SupplierD": {"status": "healthy", "lead_days": 4, "cost_per_unit": 2.5, "products": ["mask", "sanitizer"]},
        },
        "pending_shipments": [
            {"shipment_id": "SHP-010", "supplier": "SupplierA",
             "quantity": 200, "product": "mask"}
        ],
        "goal_orders": [
            {"product": "mask",      "supplier": "SupplierD", "min_quantity": 150},
            {"product": "glove",     "supplier": "SupplierB", "min_quantity": 100},
            {"product": "sanitizer", "supplier": "SupplierD", "min_quantity": 80},
        ],
        "goal_reroute_shipment": "SHP-010",
    },
    {
        "id": 11,
        "difficulty": "hard",
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
            "SupplierA": {"status": "strike",  "lead_days": 0,  "cost_per_unit": 20.0, "products": ["engine_part"]},
            "SupplierB": {"status": "strike",  "lead_days": 0,  "cost_per_unit": 18.0, "products": ["brake_pad"]},
            "SupplierC": {"status": "healthy", "lead_days": 1,  "cost_per_unit": 35.0, "products": ["engine_part", "brake_pad"]},
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
]


# ── Environment Class ──────────────────────────────────────────────────────
class SupplyChainEnvironment(Environment):
    """
    Stateful, episodic supply chain disruption environment.
    Implements the OpenEnv Environment interface.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = False
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

    # ── Tools ────────────────────────────────────────────────────────────

    def _tool_get_inventory(self, args: dict) -> str:
        lines = [f"  {p}: {q} units" for p, q in self.inventory.items()]
        return "Current inventory:\n" + "\n".join(lines)

    def _tool_check_supplier_status(self, args: dict) -> str:
        name = args.get("supplier_name", "")
        if name not in self.suppliers:
            return f"Supplier '{name}' not found. Known: {list(self.suppliers.keys())}"
        s = self.suppliers[name]
        products = s.get("products", list(self.inventory.keys()))
        return (
            f"Supplier '{name}':\n"
            f"  Status:    {s['status']}\n"
            f"  Lead time: {s['lead_days']} day(s)\n"
            f"  Cost/unit: ${s['cost_per_unit']:.2f}\n"
            f"  Products:  {products}"
        )

    def _tool_get_demand_forecast(self, args: dict) -> str:
        product = args.get("product", "")
        demand = self.task.get("daily_demand", {})
        if product not in demand:
            return f"No demand data for '{product}'. Known: {list(demand.keys())}"
        spike = self.task.get("demand_spike", False)
        spike_note = " WARNING: ACTIVE DEMAND SPIKE (3x normal)" if spike else ""
        days = (
            self.inventory.get(product, 0) / demand[product]
            if demand[product] > 0 else float("inf")
        )
        return (
            f"Demand forecast for '{product}':\n"
            f"  Daily demand:        {demand[product]} units/day{spike_note}\n"
            f"  Current stock:       {self.inventory.get(product, 0)} units\n"
            f"  Days until stockout: {days:.1f} days"
        )

    def _tool_place_order(self, args: dict) -> str:
        supplier_name = args.get("supplier_name", "")
        product       = args.get("product", "")
        quantity      = int(args.get("quantity", 0))

        if quantity <= 0:
            return "Order failed: quantity must be greater than 0."
        if supplier_name not in self.suppliers:
            return f"Order failed: supplier '{supplier_name}' not found."
        supplier = self.suppliers[supplier_name]
        if supplier["status"] in ("failed", "strike"):
            return f"Order failed: '{supplier_name}' is {supplier['status']}."
        if product not in self.inventory:
            return f"Order failed: product '{product}' not in catalogue."

        cost = quantity * supplier["cost_per_unit"]
        if "budget" in self.task:
            remaining = self.task["budget"] - self.spent_budget
            if cost > remaining:
                return f"Order failed: cost ${cost:.2f} exceeds remaining budget ${remaining:.2f}."
            self.spent_budget += cost

        self.orders_placed.append({
            "supplier": supplier_name,
            "product":  product,
            "quantity": quantity,
        })
        return (
            f"SUCCESS: Order placed!\n"
            f"  Supplier: {supplier_name}\n"
            f"  Product:  {product}\n"
            f"  Quantity: {quantity} units\n"
            f"  Cost:     ${cost:.2f}\n"
            f"  ETA:      {supplier['lead_days']} day(s)"
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

    # ── Reward ───────────────────────────────────────────────────────────

    def _compute_reward(self) -> float:
        score = 0.0
        if self._state.step_count >= 1:
            score = max(score, 0.1)
        if self.orders_placed or self.shipments_rerouted:
            score = max(score, 0.50)
        difficulty = self.task["difficulty"]
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
        if self._all_goals_met():
            score = 1.0
        # Efficiency bonus: reward faster solutions
        if score >= 1.0 and self._state.step_count <= 4:
            score = min(1.0, score)  # already max, keep clean
        return round(score, 4)

    def _all_goals_met(self) -> bool:
        task = self.task
        difficulty = task["difficulty"]
        if difficulty == "easy":
            for order in self.orders_placed:
                if (order["supplier"] == task.get("goal_supplier")
                        and order["product"]  == task.get("goal_product")
                        and order["quantity"] >= task.get("goal_quantity", 0)):
                    return True
            return False
        if difficulty == "medium":
            reroute_goal = task.get("goal_reroute_shipment")
            reroute_ok = (
                reroute_goal in [r["shipment_id"] for r in self.shipments_rerouted]
                if reroute_goal else True
            )
            order_ok = any(
                o["supplier"] == task.get("goal_supplier")
                and o["product"]  == task.get("goal_product")
                and o["quantity"] >= task.get("goal_quantity", 0)
                for o in self.orders_placed
            )
            return reroute_ok and order_ok
        if difficulty == "hard":
            for goal in task.get("goal_orders", []):
                if not any(
                    o["supplier"] == goal["supplier"]
                    and o["product"]  == goal["product"]
                    and o["quantity"] >= goal["min_quantity"]
                    for o in self.orders_placed
                ):
                    return False
            if task.get("goal_reroute_shipment"):
                if task["goal_reroute_shipment"] not in [r["shipment_id"] for r in self.shipments_rerouted]:
                    return False
            if task.get("goal_cancel_shipment"):
                if task["goal_cancel_shipment"] not in self.shipments_cancelled:
                    return False
            return True
        return False

    # ── Helpers ──────────────────────────────────────────────────────────

    def _get_state_dict(self) -> dict:
        return {
            "inventory":           dict(self.inventory),
            "steps":               self._state.step_count,
            "orders_placed":       list(self.orders_placed),
            "shipments_rerouted":  list(self.shipments_rerouted),
            "shipments_cancelled": list(self.shipments_cancelled),
            "pending_shipments":   list(self.pending_shipments),
            "spent_budget":        self.spent_budget,
        }
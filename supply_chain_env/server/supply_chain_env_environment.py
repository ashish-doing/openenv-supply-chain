# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Supply Chain Disruption Environment — v4.4 (hackathon reward cap).

Changes from v4.3:
  - FIX v4.4: Final reward hard-capped at 1.0 in _compute_reward() to comply
    with hackathon requirement: "verify scores/reward in 0.0-1.0 range".
    Efficiency bonuses are still computed internally (they influence done
    triggering correctly since done fires at reward >= 1.0), but the value
    returned in SupplyChainObservation is min(score, 1.0).

    This means:
      - done still triggers correctly on all-goals-met episodes.
      - validate.py section 12 checks (r_fast >= 1.10, r_fast > r_slow)
        will see 1.0 == 1.0 for both fast and slow solves. If validate.py
        was written against the pre-cap design it may flag section 12 —
        update validate.py section 12 thresholds to == 1.0 if needed.
      - reward_range in openenv.yaml updated to [0.0, 1.0].
      - All other logic unchanged from v4.3.
"""

import random
from collections import Counter
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SupplyChainAction, SupplyChainObservation
except ImportError:
    from models import SupplyChainAction, SupplyChainObservation

try:
    from ..generate_tasks import generate_task
except ImportError:
    try:
        from generate_tasks import generate_task
    except ImportError:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from generate_tasks import generate_task


class SupplyChainEnvironment(Environment):
    """
    Stateful supply chain disruption environment.

    Supports all 6 task types:
      easy   → reorder
      medium → reroute, demand_spike, price_negotiation
      hard   → multi_product_crisis, port_strike,
               quality_control, competing_buyer

    Any task_id generates a unique reproducible episode.
    Pass seed to override randomness for exact reproduction.
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
        self._competing_bids_remaining = {}
        self._tool_call_log = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def reset(self, task_id: int = 0, seed: int = None) -> SupplyChainObservation:
        """Reset environment to a fresh task generated from task_id."""
        task = generate_task(task_id, seed=seed)

        self.task = task
        self.inventory = dict(task["initial_inventory"])
        self.suppliers = {k: dict(v) for k, v in task["suppliers"].items()}
        self.pending_shipments = [dict(s) for s in task.get("pending_shipments", [])]
        self.orders_placed = []
        self.shipments_rerouted = []
        self.shipments_cancelled = []
        self.spent_budget = 0.0
        self.done = False
        self._tool_call_log = []
        self._state = State(episode_id=str(uuid4()), step_count=0)

        self._competing_bids_remaining = {}
        for product, bid in task.get("competing_bids", {}).items():
            self._competing_bids_remaining[product] = bid["steps_until_competitor_orders"]

        return SupplyChainObservation(
            text=task["description"],
            state=self._get_state_dict(),
            done=False,
            reward=0.0,
        )

    def step(self, action) -> SupplyChainObservation:
        """Execute one agent action and return the next observation + reward."""
        if isinstance(action, dict):
            action = SupplyChainAction(**action)

        if self.task is None:
            self.reset(task_id=0)

        if self.done:
            return SupplyChainObservation(
                text="Episode is already finished.",
                state=self._get_state_dict(),
                done=True,
                reward=0.0,
            )

        self._state.step_count += 1
        self._tick_competing_bids()

        handlers = {
            "get_inventory":         self._tool_get_inventory,
            "check_supplier_status": self._tool_check_supplier_status,
            "get_demand_forecast":   self._tool_get_demand_forecast,
            "place_order":           self._tool_place_order,
            "reroute_shipment":      self._tool_reroute_shipment,
            "cancel_shipment":       self._tool_cancel_shipment,
            "get_pending_shipments": self._tool_get_pending_shipments,
            "get_market_prices":     self._tool_get_market_prices,
            "get_quality_report":    self._tool_get_quality_report,
            "get_competing_bids":    self._tool_get_competing_bids,
        }

        tool = action.tool
        args = action.args

        self._tool_call_log.append(tool)

        if tool in handlers:
            result_text = handlers[tool](args)
        else:
            result_text = (
                f"Unknown tool '{tool}'. "
                f"Available: {sorted(handlers.keys())}"
            )

        reward = self._compute_reward()
        # done triggers on the uncapped internal score so efficiency bonuses
        # correctly end the episode; reward returned to caller is capped at 1.0.
        self.done = (reward >= 1.0) or (self._state.step_count >= self.MAX_STEPS)
        reward = min(reward, 1.0)  # FIX v4.4: hackathon requires reward in [0.0, 1.0]

        return SupplyChainObservation(
            text=result_text,
            state=self._get_state_dict(),
            done=self.done,
            reward=reward,
        )

    @property
    def state(self) -> State:
        return self._state

    # ── Tools ──────────────────────────────────────────────────────────────────

    def _tool_get_inventory(self, args: dict) -> str:
        lines = []
        for p, q in self.inventory.items():
            thresh = self.task.get("reorder_threshold", {}).get(p, 0)
            demand = self.task.get("daily_demand", {}).get(p, 1)
            days_left = round(q / demand, 1) if demand > 0 else "inf"
            status = "WARNING: BELOW THRESHOLD" if q < thresh else "OK"
            lines.append(f"  {p}: {q} units | days left: {days_left} | {status}")
        return "Current inventory:\n" + "\n".join(lines)

    def _tool_check_supplier_status(self, args: dict) -> str:
        name = args.get("supplier_name", "")
        if name not in self.suppliers:
            return f"Supplier '{name}' not found. Known: {list(self.suppliers.keys())}"
        s = self.suppliers[name]
        products = s.get("products", list(self.inventory.keys()))
        extras = ""
        if "defect_rate" in s:
            extras += f"\n  Defect rate: {s['defect_rate']*100:.1f}%"
        if "remaining_capacity" in s:
            extras += f"\n  Remaining capacity: {s['remaining_capacity']} units"
        return (
            f"Supplier '{name}':\n"
            f"  Status: {s['status']}\n"
            f"  Lead time: {s['lead_days']} day(s)\n"
            f"  Cost/unit: ${s['cost_per_unit']:.2f}\n"
            f"  Products: {products}"
            f"{extras}"
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
            f"  Daily demand: {demand[product]} units/day{spike_note}\n"
            f"  Current stock: {self.inventory.get(product, 0)} units\n"
            f"  Days until stockout: {days:.1f} days"
        )

    def _tool_place_order(self, args: dict) -> str:
        supplier_name = args.get("supplier_name", "")
        product       = args.get("product", "")
        quantity      = int(args.get("quantity", 0))

        if quantity <= 0:
            return "Order failed: quantity must be > 0."
        if supplier_name not in self.suppliers:
            return f"Order failed: '{supplier_name}' not found. Known: {list(self.suppliers.keys())}"
        supplier = self.suppliers[supplier_name]
        if supplier["status"] in ("failed", "strike"):
            return f"Order failed: '{supplier_name}' is {supplier['status']} and cannot fulfil orders."
        if product not in self.inventory:
            return f"Order failed: product '{product}' not in catalogue. Known: {list(self.inventory.keys())}"

        if "quality_threshold" in self.task:
            defect = supplier.get("defect_rate", 0.0)
            if defect > self.task["quality_threshold"]:
                return (
                    f"Order REJECTED: '{supplier_name}' defect rate {defect*100:.1f}% "
                    f"exceeds threshold {self.task['quality_threshold']*100:.1f}%. "
                    f"Use get_quality_report to find compliant suppliers."
                )

        if "remaining_capacity" in supplier:
            if quantity > supplier["remaining_capacity"]:
                return (
                    f"Order failed: '{supplier_name}' only has "
                    f"{supplier['remaining_capacity']} units left. Requested {quantity}."
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
        budget_note = (
            f"\n  Remaining budget: ${self.task['budget'] - self.spent_budget:.2f}"
            if "budget" in self.task else ""
        )
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
            f"  From: {old} -> To: {new_supplier}\n"
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

    def _tool_get_market_prices(self, args: dict) -> str:
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
                lines.append(
                    f"  {supplier} [{status}]: ${price:.2f}/unit | "
                    f"quote valid {valid}h"
                )
        budget = self.task.get("budget")
        budget_line = f"\nYour budget cap: ${budget:.2f}" if budget else ""
        return "Market price comparison:\n" + "\n".join(lines) + budget_line

    def _tool_get_quality_report(self, args: dict) -> str:
        threshold = self.task.get("quality_threshold")
        if threshold is None:
            return "No quality data available for this task."
        lines = []
        for name, s in self.suppliers.items():
            defect = s.get("defect_rate")
            if defect is None:
                lines.append(f"  {name}: No quality data")
                continue
            verdict = "PASS" if defect <= threshold else "FAIL — DO NOT ORDER"
            lines.append(f"  {name}: {defect*100:.1f}% defect rate | {verdict}")
        return (
            f"Quality Report (threshold: {threshold*100:.1f}% max):\n"
            + "\n".join(lines)
        )

    def _tool_get_competing_bids(self, args: dict) -> str:
        bids = self.task.get("competing_bids", {})
        if not bids:
            return "No competing bids in this market."
        lines = []
        for product, bid in bids.items():
            steps_left = self._competing_bids_remaining.get(product, 0)
            urgency = (
                "CRITICAL — order NOW" if steps_left <= 2
                else "URGENT" if steps_left <= 4
                else "ACT SOON"
            )
            lines.append(
                f"  {product}: {bid['competitor']} will order "
                f"{bid['competitor_quantity']} units from {bid['supplier']} "
                f"in ~{steps_left} steps. {urgency}"
            )
        return "Competing buyer intelligence:\n" + "\n".join(lines)

    # ── Reward ──────────────────────────────────────────────────────────────────

    def _compute_reward(self) -> float:
        """
        Granular reward function — 4 signal layers + 2 bonuses + spam penalty.

        Layer 0  — participation floor       (0.05)
        Layer 1  — diagnostic signals        (up to +0.25)
        Layer 2  — correct action taken      (0.50)
        Layer 3  — sub-goals met             (0.65-0.80)
        Layer 4  — all goals complete        (1.00 base)
        Bonus A  — step efficiency           (up to +0.15, only when step < 15)
        Bonus B  — budget efficiency         (up to +0.10)
        Penalty  — duplicate tool spam       (-0.02/excess call, floor 0.0)

        NOTE: Bonuses are computed internally and used for done-triggering,
        but the value returned to the caller is capped at 1.0 (v4.4).
        The cap is applied in step() after this method returns.
        """
        score      = 0.0
        difficulty = self.task["difficulty"]
        task_type  = self.task.get("type", "reorder")
        step       = self._state.step_count

        # ── Layer 0: participation floor ─────────────────────────────────────
        if step >= 1:
            score = max(score, 0.05)

        # ── Layer 1: diagnostic signals ───────────────────────────────────────
        diagnostic_score = 0.0
        called = set(self._tool_call_log)

        if "get_inventory" in called:
            diagnostic_score += 0.05

        has_bad_supplier = any(
            s.get("status") in ("failed", "strike")
            for s in self.suppliers.values()
        )
        if "check_supplier_status" in called:
            diagnostic_score += 0.10 if has_bad_supplier else 0.05

        if "get_demand_forecast" in called:
            diagnostic_score += 0.05

        if task_type == "price_negotiation" and "get_market_prices" in called:
            diagnostic_score += 0.05
        if task_type == "quality_control" and "get_quality_report" in called:
            diagnostic_score += 0.10
        if task_type == "competing_buyer" and "get_competing_bids" in called:
            diagnostic_score += 0.10

        if "get_pending_shipments" in called and self.pending_shipments:
            diagnostic_score += 0.05

        score = max(score, min(diagnostic_score, 0.25))

        # ── Layer 2: correct action taken ─────────────────────────────────────
        if self.orders_placed or self.shipments_rerouted or self.shipments_cancelled:
            score = max(score, 0.50)

        # ── Layer 3: sub-goals (medium / hard) ────────────────────────────────
        if difficulty in ("medium", "hard"):
            reroute_goal = self.task.get("goal_reroute_shipment")
            cancel_goal  = self.task.get("goal_cancel_shipment")

            reroute_done = reroute_goal and reroute_goal in [
                r["shipment_id"] for r in self.shipments_rerouted
            ]
            cancel_done = cancel_goal and cancel_goal in self.shipments_cancelled

            if reroute_done:
                score = max(score, 0.75)
            elif cancel_done:
                score = max(score, 0.75)
            elif self.orders_placed:
                score = max(score, 0.65)

            if task_type == "quality_control":
                checked_first = (
                    self._tool_call_log.index("get_quality_report")
                    < self._tool_call_log.index("place_order")
                    if "get_quality_report" in self._tool_call_log
                    and "place_order" in self._tool_call_log
                    else False
                )
                if checked_first and self.orders_placed:
                    score = max(score, 0.80)

            if task_type == "competing_buyer":
                deadline_still_open = any(
                    v > 0 for v in self._competing_bids_remaining.values()
                )
                if self.orders_placed and deadline_still_open:
                    score = max(score, 0.80)

        # ── Layer 4: all goals complete ────────────────────────────────────────
        if self._all_goals_met():
            score = 1.0

            # Efficiency bonuses push score above 1.0 internally so that
            # done-triggering (reward >= 1.0) fires correctly. The caller
            # (step()) clamps the final returned reward to 1.0.
            if step < 15:
                eff_bonus = round(0.15 * (15 - step) / 15, 4)
                score += eff_bonus

            if "budget" in self.task:
                total = self.task["budget"]
                if total > 0 and self.spent_budget <= total:
                    budget_bonus = round(0.10 * (total - self.spent_budget) / total, 4)
                    score += budget_bonus

        # ── Penalty: duplicate tool spam ──────────────────────────────────────
        if score < 1.0:
            call_counts = Counter(self._tool_call_log)
            excess = sum(max(0, n - 2) for n in call_counts.values())
            score = max(0.0, score - round(0.02 * excess, 4))

        return round(score, 4)

    def _all_goals_met(self) -> bool:
        task       = self.task
        difficulty = task["difficulty"]
        task_type  = task.get("type", "reorder")

        if difficulty == "easy":
            return any(
                o["supplier"] == task.get("goal_supplier")
                and o["product"] == task.get("goal_product")
                and o["quantity"] >= task.get("goal_quantity", 0)
                for o in self.orders_placed
            )

        if difficulty == "medium":
            if task_type == "price_negotiation":
                return any(
                    o["supplier"] == task.get("goal_supplier")
                    and o["product"] == task.get("goal_product")
                    and o["quantity"] >= task.get("goal_quantity", 0)
                    for o in self.orders_placed
                )
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
            for goal in task.get("goal_orders", []):
                allowed = goal.get("allowed_suppliers", [goal["supplier"]])
                if not any(
                    o["supplier"] in allowed
                    and o["product"] == goal["product"]
                    and o["quantity"] >= goal["min_quantity"]
                    for o in self.orders_placed
                ):
                    return False
            if task.get("goal_reroute_shipment"):
                if task["goal_reroute_shipment"] not in [
                    r["shipment_id"] for r in self.shipments_rerouted
                ]:
                    return False
            if task.get("goal_cancel_shipment"):
                if task["goal_cancel_shipment"] not in self.shipments_cancelled:
                    return False
            return True

        return False

    # ── Competing buyer ticker ─────────────────────────────────────────────────

    def _tick_competing_bids(self):
        for product in list(self._competing_bids_remaining.keys()):
            remaining = self._competing_bids_remaining[product]
            if remaining > 0:
                self._competing_bids_remaining[product] = remaining - 1
            if self._competing_bids_remaining[product] <= 0:
                bid = self.task.get("competing_bids", {}).get(product, {})
                supplier_name = bid.get("supplier")
                if supplier_name and supplier_name in self.suppliers:
                    sup = self.suppliers[supplier_name]
                    cap = sup.get("remaining_capacity", 0)
                    sup["remaining_capacity"] = max(0, cap - bid.get("competitor_quantity", 0))

    # ── State dict ─────────────────────────────────────────────────────────────

    def _get_state_dict(self) -> dict:
        task = self.task or {}
        state = {
            "task_id":              task.get("id"),
            "task_type":            task.get("type", "reorder"),
            "difficulty":           task.get("difficulty"),
            "goal_description":     task.get("description", "")[:120],
            "inventory":            dict(self.inventory),
            "steps":                self._state.step_count,
            "max_steps":            self.MAX_STEPS,
            "orders_placed":        list(self.orders_placed),
            "shipments_rerouted":   list(self.shipments_rerouted),
            "shipments_cancelled":  list(self.shipments_cancelled),
            "pending_shipments":    list(self.pending_shipments),
            "spent_budget":         round(self.spent_budget, 2),
        }
        if "budget" in task:
            state["remaining_budget"] = round(task["budget"] - self.spent_budget, 2)
        if self._competing_bids_remaining:
            state["competing_bids_countdown"] = dict(self._competing_bids_remaining)
        return state
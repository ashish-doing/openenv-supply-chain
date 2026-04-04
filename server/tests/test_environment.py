"""
tests/test_environment.py — pytest suite for SupplyChainEnvironment.

Run:
    pytest tests/ -v

All tests run in-process (no server required).
"""
import pytest
from server.supply_chain_env_environment import SupplyChainEnvironment
from models import SupplyChainAction


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def env():
    return SupplyChainEnvironment()


# ── reset() ───────────────────────────────────────────────────────────────────

def test_reset_returns_observation(env):
    obs = env.reset(task_id=0)
    assert obs is not None
    assert isinstance(obs.text, str)
    assert len(obs.text) > 0

def test_reset_done_is_false(env):
    obs = env.reset(task_id=0)
    assert obs.done == False

def test_reset_reward_is_zero(env):
    obs = env.reset(task_id=0)
    assert obs.reward == 0.0

def test_reset_state_is_dict(env):
    obs = env.reset(task_id=0)
    assert isinstance(obs.state, dict)

def test_reset_state_has_required_keys(env):
    obs = env.reset(task_id=0)
    required = [
        "task_id", "task_type", "difficulty", "goal_description",
        "inventory", "steps", "max_steps", "orders_placed",
        "shipments_rerouted", "shipments_cancelled", "pending_shipments",
        "spent_budget",
    ]
    for key in required:
        assert key in obs.state, f"Missing key: {key}"

def test_reset_is_reproducible(env):
    obs1 = env.reset(task_id=0, seed=42)
    obs2 = env.reset(task_id=0, seed=42)
    assert obs1.text == obs2.text

def test_reset_different_seeds_differ(env):
    obs1 = env.reset(task_id=20, seed=1)
    obs2 = env.reset(task_id=20, seed=99)
    assert obs1.text != obs2.text


# ── step() ────────────────────────────────────────────────────────────────────

def test_step_returns_observation(env):
    env.reset(task_id=0)
    obs = env.step(SupplyChainAction(tool="get_inventory", args={}))
    assert obs is not None
    assert isinstance(obs.text, str)

def test_step_reward_is_float(env):
    env.reset(task_id=0)
    obs = env.step(SupplyChainAction(tool="get_inventory", args={}))
    assert isinstance(obs.reward, float)

def test_step_reward_in_range(env):
    env.reset(task_id=0)
    obs = env.step(SupplyChainAction(tool="get_inventory", args={}))
    assert 0.0 <= obs.reward <= 1.0

def test_step_done_is_bool(env):
    env.reset(task_id=0)
    obs = env.step(SupplyChainAction(tool="get_inventory", args={}))
    assert isinstance(obs.done, bool)

def test_step_increments_step_count(env):
    env.reset(task_id=0)
    env.step(SupplyChainAction(tool="get_inventory", args={}))
    assert env._state.step_count == 1
    env.step(SupplyChainAction(tool="get_inventory", args={}))
    assert env._state.step_count == 2

def test_step_after_done_returns_done(env):
    env.reset(task_id=0)
    env.step(SupplyChainAction(tool="place_order", args={
        "supplier_name": "SupplierA", "product": "bottled_water", "quantity": 200
    }))
    obs = env.step(SupplyChainAction(tool="get_inventory", args={}))
    assert obs.done == True

def test_unknown_tool_returns_error_text(env):
    env.reset(task_id=0)
    obs = env.step(SupplyChainAction(tool="nonexistent_tool", args={}))
    assert "Unknown tool" in obs.text or "unknown" in obs.text.lower()

def test_step_accepts_dict_action(env):
    env.reset(task_id=0)
    obs = env.step({"tool": "get_inventory", "args": {}})
    assert obs is not None


# ── Reward logic ──────────────────────────────────────────────────────────────

def test_easy_task_full_reward(env):
    env.reset(task_id=0)
    obs = env.step(SupplyChainAction(tool="place_order", args={
        "supplier_name": "SupplierA", "product": "bottled_water", "quantity": 200
    }))
    assert obs.reward == 1.0
    assert obs.done == True

def test_participation_floor(env):
    env.reset(task_id=0)
    obs = env.step(SupplyChainAction(tool="get_inventory", args={}))
    assert obs.reward >= 0.05

def test_partial_reward_below_one(env):
    env.reset(task_id=0)
    obs = env.step(SupplyChainAction(tool="get_inventory", args={}))
    assert obs.reward < 1.0

def test_wrong_supplier_no_full_reward(env):
    env.reset(task_id=0)
    obs = env.step(SupplyChainAction(tool="place_order", args={
        "supplier_name": "SupplierB", "product": "bottled_water", "quantity": 200
    }))
    assert obs.reward < 1.0

def test_reward_never_exceeds_one(env):
    for tid in [0, 1, 2, 5, 10]:
        env.reset(task_id=tid)
        for _ in range(25):
            obs = env.step(SupplyChainAction(tool="get_inventory", args={}))
            assert obs.reward <= 1.0
            if obs.done:
                break


# ── Difficulty levels ─────────────────────────────────────────────────────────

def test_easy_task_difficulty(env):
    env.reset(task_id=0)
    assert env.task["difficulty"] == "easy"

def test_medium_task_difficulty(env):
    env.reset(task_id=5)
    assert env.task["difficulty"] == "medium"

def test_hard_task_difficulty(env):
    env.reset(task_id=10)
    assert env.task["difficulty"] == "hard"

def test_all_fixed_tasks_load(env):
    for tid in [0, 1, 2, 5, 6, 7, 10, 11, 12, 13]:
        obs = env.reset(task_id=tid)
        assert obs.text != "", f"Task {tid} returned empty text"

def test_procedural_task_loads(env):
    for tid in [20, 55, 110, 160, 210, 260]:
        obs = env.reset(task_id=tid)
        assert obs.text != "", f"Procedural task {tid} returned empty text"


# ── Tools ─────────────────────────────────────────────────────────────────────

def test_tool_get_inventory(env):
    env.reset(task_id=0)
    obs = env.step(SupplyChainAction(tool="get_inventory", args={}))
    assert "inventory" in obs.text.lower() or "units" in obs.text.lower()

def test_tool_check_supplier_status(env):
    env.reset(task_id=0)
    obs = env.step(SupplyChainAction(
        tool="check_supplier_status", args={"supplier_name": "SupplierA"}
    ))
    assert "SupplierA" in obs.text

def test_tool_get_demand_forecast(env):
    env.reset(task_id=0)
    obs = env.step(SupplyChainAction(
        tool="get_demand_forecast", args={"product": "bottled_water"}
    ))
    assert "demand" in obs.text.lower()

def test_tool_get_pending_shipments_medium(env):
    env.reset(task_id=5)
    obs = env.step(SupplyChainAction(tool="get_pending_shipments", args={}))
    assert "SHP" in obs.text

def test_tool_reroute_shipment(env):
    env.reset(task_id=5)
    obs = env.step(SupplyChainAction(
        tool="reroute_shipment",
        args={"shipment_id": "SHP-005", "new_supplier": "SupplierB"}
    ))
    assert "rerouted" in obs.text.lower() or "not found" in obs.text.lower()

def test_tool_get_market_prices_medium(env):
    env.reset(task_id=7)
    obs = env.step(SupplyChainAction(tool="get_market_prices", args={}))
    assert "price" in obs.text.lower() or "budget" in obs.text.lower()

def test_tool_get_quality_report_hard(env):
    env.reset(task_id=12)
    obs = env.step(SupplyChainAction(tool="get_quality_report", args={}))
    assert "defect" in obs.text.lower() or "quality" in obs.text.lower()

def test_tool_get_competing_bids_hard(env):
    env.reset(task_id=13)
    obs = env.step(SupplyChainAction(tool="get_competing_bids", args={}))
    assert "competitor" in obs.text.lower() or "competing" in obs.text.lower()


# ── State dict completeness ───────────────────────────────────────────────────

def test_state_dict_budget_tasks(env):
    env.reset(task_id=10)
    state = env._get_state_dict()
    assert "remaining_budget" in state

def test_state_dict_competing_buyer(env):
    env.reset(task_id=13)
    state = env._get_state_dict()
    assert "competing_bids_countdown" in state

def test_state_step_count_matches(env):
    env.reset(task_id=0)
    env.step(SupplyChainAction(tool="get_inventory", args={}))
    env.step(SupplyChainAction(tool="get_inventory", args={}))
    state = env._get_state_dict()
    assert state["steps"] == 2
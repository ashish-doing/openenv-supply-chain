"""generate_tasks.py — Procedural task generator
Usage: python generate_tasks.py --count 5 --seed 42"""
import argparse, json, random

PRODUCTS = ["bottled_water","canned_soup","coffee_beans","circuit_board",
            "pain_reliever","mask","glove","sanitizer","engine_part","brake_pad"]

def easy(tid, rng):
    p = rng.choice(PRODUCTS); qty = rng.randint(100, 350)
    return {"id": tid, "difficulty": "easy",
            "description": f"Warehouse stocks '{p}'. Place order of {qty} units with SupplierA.",
            "initial_inventory": {p: rng.randint(20,80)}, "daily_demand": {p: rng.randint(20,50)},
            "reorder_threshold": {p: 60},
            "suppliers": {"SupplierA": {"status":"healthy","lead_days":2,"cost_per_unit":1.5},
                          "SupplierB": {"status":"healthy","lead_days":4,"cost_per_unit":1.2}},
            "goal_product": p, "goal_supplier": "SupplierA", "goal_quantity": qty}

def medium(tid, rng):
    p = rng.choice(PRODUCTS); shp = f"SHP-{tid:03d}"; qty = rng.randint(80, 200)
    return {"id": tid, "difficulty": "medium",
            "description": f"SupplierA FAILED. Reroute {shp} to SupplierB. Order {qty} units of '{p}'.",
            "initial_inventory": {p: rng.randint(10,35)}, "daily_demand": {p: rng.randint(20,40)},
            "reorder_threshold": {p: 50},
            "suppliers": {"SupplierA": {"status":"failed","lead_days":0,"cost_per_unit":5.0},
                          "SupplierB": {"status":"healthy","lead_days":3,"cost_per_unit":6.5}},
            "pending_shipments": [{"shipment_id":shp,"supplier":"SupplierA","quantity":80,"product":p}],
            "goal_product": p, "goal_supplier": "SupplierB",
            "goal_quantity": qty, "goal_reroute_shipment": shp}

def hard(tid, rng):
    prods = rng.sample(PRODUCTS, 3); shp = f"SHP-{tid:03d}"
    return {"id": tid, "difficulty": "hard",
            "description": f"MULTI-CRISIS. Products {prods}. SupplierA failed. Reroute {shp}. Order all.",
            "initial_inventory": {p: rng.randint(5,20) for p in prods},
            "daily_demand": {p: rng.randint(20,50) for p in prods},
            "reorder_threshold": {p: 60 for p in prods}, "budget": round(rng.uniform(1500,5000),2),
            "suppliers": {"SupplierA":{"status":"failed","lead_days":0,"cost_per_unit":5.0,"products":[prods[0]]},
                          "SupplierB":{"status":"healthy","lead_days":3,"cost_per_unit":4.0,"products":[prods[1]]},
                          "SupplierD":{"status":"healthy","lead_days":4,"cost_per_unit":6.0,"products":prods}},
            "pending_shipments": [{"shipment_id":shp,"supplier":"SupplierA","quantity":100,"product":prods[0]}],
            "goal_orders": [{"product":prods[0],"supplier":"SupplierD","min_quantity":80},
                            {"product":prods[1],"supplier":"SupplierB","min_quantity":60},
                            {"product":prods[2],"supplier":"SupplierD","min_quantity":50}],
            "goal_reroute_shipment": shp}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=6)
    ap.add_argument("--seed",  type=int, default=42)
    ap.add_argument("--difficulty", default="all", choices=["easy","medium","hard","all"])
    args = ap.parse_args()
    rng = random.Random(args.seed)
    gens = {"easy": easy, "medium": medium, "hard": hard}
    levels = list(gens) if args.difficulty == "all" else [args.difficulty]
    tasks = [gens[levels[i % len(levels)]](i+100, rng) for i in range(args.count)]
    print(json.dumps(tasks, indent=2))
    print(f"\nGenerated {len(tasks)} tasks.")

if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
r"""
01_dataset_validator.py
CVRP dataset + solution validator for the WOA/SSA FPGA study.

Kullanım:
1) Bu dosyayı şu ana klasöre koy:
   C:\Users\Cem Deniz KUMRAL\Desktop\Aktif Çalışmalar\WOA & SSA on FPGA

2) Klasör yapısı:
   datasets/
      A-n32-k5.vrp
      ...
   solutions/
      A-n32-k5.sol
      ...

3) Çalıştır:
   python 01_dataset_validator.py

Çıktılar:
   01_validation_results/
      dataset_validation_report.csv
      dataset_validation_report.xlsx
      dataset_validation_report.md
"""

import math
import re
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import pandas as pd


EXPECTED_DATASETS = [
    {"name": "A-n32-k5",  "file": "A-n32-k5.vrp",  "solution": "A-n32-k5.sol",  "nodes": 32, "customers": 31, "vehicles": 5,  "capacity": 100, "known_best": 784},
    {"name": "A-n37-k6",  "file": "A-n37-k6.vrp",  "solution": "A-n37-k6.sol",  "nodes": 37, "customers": 36, "vehicles": 6,  "capacity": 100, "known_best": 949},
    {"name": "A-n45-k6",  "file": "A-n45-k6.vrp",  "solution": "A-n45-k6.sol",  "nodes": 45, "customers": 44, "vehicles": 6,  "capacity": 100, "known_best": 944},
    {"name": "A-n54-k7",  "file": "A-n54-k7.vrp",  "solution": "A-n54-k7.sol",  "nodes": 54, "customers": 53, "vehicles": 7,  "capacity": 100, "known_best": 1167},
    {"name": "A-n60-k9",  "file": "A-n60-k9.vrp",  "solution": "A-n60-k9.sol",  "nodes": 60, "customers": 59, "vehicles": 9,  "capacity": 100, "known_best": 1354},
    {"name": "A-n80-k10", "file": "A-n80-k10.vrp", "solution": "A-n80-k10.sol", "nodes": 80, "customers": 79, "vehicles": 10, "capacity": 100, "known_best": 1763},
    {"name": "B-n50-k7",  "file": "B-n50-k7.vrp",  "solution": "B-n50-k7.sol",  "nodes": 50, "customers": 49, "vehicles": 7,  "capacity": 100, "known_best": 741},
    {"name": "P-n76-k4",  "file": "P-n76-k4.vrp",  "solution": "P-n76-k4.sol",  "nodes": 76, "customers": 75, "vehicles": 4,  "capacity": 350, "known_best": 593},
]


@dataclass
class VRPInstance:
    name: str
    capacity: int
    depot_original_id: int
    coords: Dict[int, Tuple[float, float]]
    demands: Dict[int, int]
    distance_matrix: Dict[Tuple[int, int], int]

    @property
    def node_count(self) -> int:
        return len(self.coords)

    @property
    def customer_original_ids(self) -> List[int]:
        return [i for i in sorted(self.coords.keys()) if i != self.depot_original_id]


def to_json_safe(obj):
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    return obj


def parse_vrp_file(path: Path) -> VRPInstance:
    if not path.exists():
        raise FileNotFoundError(f"VRP dosyası bulunamadı: {path}")

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    name = path.stem
    capacity = None
    coords: Dict[int, Tuple[float, float]] = {}
    demands: Dict[int, int] = {}
    depot_original_id = 1

    coord_section = False
    demand_section = False
    depot_section = False

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        upper = line.upper()

        if upper.startswith("NAME"):
            parts = re.split(r":|\s+", line)
            if len(parts) >= 2:
                name = parts[-1].strip()

        elif upper.startswith("CAPACITY"):
            capacity = int(re.sub(r"[^0-9]", " ", line).split()[-1])

        elif upper.startswith("NODE_COORD_SECTION"):
            coord_section = True
            demand_section = False
            depot_section = False
            continue

        elif upper.startswith("DEMAND_SECTION"):
            coord_section = False
            demand_section = True
            depot_section = False
            continue

        elif upper.startswith("DEPOT_SECTION"):
            coord_section = False
            demand_section = False
            depot_section = True
            continue

        elif upper.startswith("EOF"):
            break

        if coord_section:
            parts = line.split()
            if len(parts) >= 3:
                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                coords[node_id] = (x, y)

        elif demand_section:
            parts = line.split()
            if len(parts) >= 2:
                node_id = int(parts[0])
                demand = int(float(parts[1]))
                demands[node_id] = demand

        elif depot_section:
            try:
                val = int(line.split()[0])
                if val != -1:
                    depot_original_id = val
            except Exception:
                pass

    if capacity is None:
        raise ValueError(f"CAPACITY bilgisi okunamadı: {path}")
    if not coords:
        raise ValueError(f"NODE_COORD_SECTION okunamadı: {path}")
    if not demands:
        raise ValueError(f"DEMAND_SECTION okunamadı: {path}")

    missing_demands = sorted(set(coords.keys()) - set(demands.keys()))
    if missing_demands:
        raise ValueError(f"Eksik demand kaydı var: {path.name} -> {missing_demands[:10]}")

    distance_matrix = {}
    for i, (xi, yi) in coords.items():
        for j, (xj, yj) in coords.items():
            if i == j:
                d = 0
            else:
                d = int(round(math.hypot(xi - xj, yi - yj)))
            distance_matrix[(i, j)] = d

    return VRPInstance(
        name=name,
        capacity=capacity,
        depot_original_id=depot_original_id,
        coords=coords,
        demands=demands,
        distance_matrix=distance_matrix,
    )


def parse_solution_file(path: Path) -> Tuple[List[List[int]], Optional[int]]:
    if not path.exists():
        return [], None

    routes: List[List[int]] = []
    solution_cost: Optional[int] = None

    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue

        lower = line.lower()

        if lower.startswith("route"):
            if ":" in line:
                after_colon = line.split(":", 1)[1]
            else:
                after_colon = line
            nums = [int(x) for x in re.findall(r"-?\d+", after_colon)]
            nums = [x for x in nums if x >= 0]
            if nums:
                routes.append(nums)

        elif lower.startswith("cost"):
            nums = re.findall(r"-?\d+", line)
            if nums:
                solution_cost = int(nums[-1])

    return routes, solution_cost


def route_cost(instance: VRPInstance, route: List[int]) -> int:
    if not route:
        return 0
    depot = instance.depot_original_id
    cost = instance.distance_matrix[(depot, route[0])]
    for a, b in zip(route[:-1], route[1:]):
        cost += instance.distance_matrix[(a, b)]
    cost += instance.distance_matrix[(route[-1], depot)]
    return int(cost)


def evaluate_routes(instance: VRPInstance, routes: List[List[int]]) -> Dict[str, object]:
    flat = [int(x) for r in routes for x in r]
    customer_set = set(instance.customer_original_ids)

    visited_set = set(flat)
    missing = sorted(customer_set - visited_set)
    extra = sorted(visited_set - customer_set)
    duplicates = sorted({x for x in flat if flat.count(x) > 1})

    route_loads = [sum(instance.demands.get(int(c), 0) for c in r) for r in routes]
    route_costs = [route_cost(instance, r) for r in routes]
    capacity_violations = [idx + 1 for idx, load in enumerate(route_loads) if load > instance.capacity]

    return {
        "route_count": len(routes),
        "visited_customer_count": len(flat),
        "unique_customer_count": len(visited_set),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "duplicate_count": len(duplicates),
        "capacity_violation_count": len(capacity_violations),
        "max_route_load": max(route_loads) if route_loads else 0,
        "total_route_cost": int(sum(route_costs)),
        "route_loads": route_loads,
        "route_costs": route_costs,
        "missing": missing[:20],
        "extra": extra[:20],
        "duplicates": duplicates[:20],
        "capacity_violations": capacity_violations[:20],
    }


def normalize_solution_routes(instance: VRPInstance, raw_routes: List[List[int]], expected_cost: Optional[int]) -> Tuple[List[List[int]], str, Dict[str, object]]:
    if not raw_routes:
        return [], "no_solution", {}

    candidates = []

    try:
        original_routes = [[int(x) for x in r] for r in raw_routes]
        ev = evaluate_routes(instance, original_routes)
        score = (
            ev["missing_count"] * 100000
            + ev["extra_count"] * 100000
            + ev["duplicate_count"] * 100000
            + ev["capacity_violation_count"] * 1000
        )
        if expected_cost is not None:
            score += abs(ev["total_route_cost"] - expected_cost)
        candidates.append(("original_id", original_routes, ev, score))
    except Exception as e:
        candidates.append(("original_id_failed", [], {"error": str(e)}, 10**12))

    try:
        shifted_routes = [[int(x) + 1 for x in r] for r in raw_routes]
        ev = evaluate_routes(instance, shifted_routes)
        score = (
            ev["missing_count"] * 100000
            + ev["extra_count"] * 100000
            + ev["duplicate_count"] * 100000
            + ev["capacity_violation_count"] * 1000
        )
        if expected_cost is not None:
            score += abs(ev["total_route_cost"] - expected_cost)
        candidates.append(("customer_index_plus_one", shifted_routes, ev, score))
    except Exception as e:
        candidates.append(("customer_index_plus_one_failed", [], {"error": str(e)}, 10**12))

    best = min(candidates, key=lambda x: x[3])
    mode, routes, ev, _score = best
    return routes, mode, ev


def main():
    root = Path(__file__).resolve().parent
    datasets_dir = root / "datasets"
    solutions_dir = root / "solutions"
    out_dir = root / "01_validation_results"
    out_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 78)
    print("CVRP DATASET + SOLUTION VALIDATOR")
    print("=" * 78)
    print(f"Root       : {root}")
    print(f"Datasets   : {datasets_dir}")
    print(f"Solutions  : {solutions_dir}")
    print(f"Out        : {out_dir}")
    print("=" * 78)

    rows = []
    details = {}

    for cfg in EXPECTED_DATASETS:
        name = cfg["name"]
        vrp_path = datasets_dir / cfg["file"]
        sol_path = solutions_dir / cfg["solution"]

        row = {
            "dataset": name,
            "vrp_exists": vrp_path.exists(),
            "sol_exists": sol_path.exists(),
            "expected_nodes": cfg["nodes"],
            "expected_customers": cfg["customers"],
            "expected_vehicles": cfg["vehicles"],
            "expected_capacity": cfg["capacity"],
            "known_best_config": cfg["known_best"],
            "parsed_nodes": None,
            "parsed_customers": None,
            "parsed_capacity": None,
            "parsed_depot": None,
            "sol_cost": None,
            "computed_solution_cost": None,
            "cost_matches_known_best": False,
            "cost_matches_sol_file": False,
            "solution_route_count": None,
            "visited_customer_count": None,
            "unique_customer_count": None,
            "missing_count": None,
            "extra_count": None,
            "duplicate_count": None,
            "capacity_violation_count": None,
            "max_route_load": None,
            "solution_node_interpretation": None,
            "status": "FAIL",
            "notes": "",
        }

        try:
            instance = parse_vrp_file(vrp_path)
            row["parsed_nodes"] = instance.node_count
            row["parsed_customers"] = len(instance.customer_original_ids)
            row["parsed_capacity"] = instance.capacity
            row["parsed_depot"] = instance.depot_original_id

            raw_routes, sol_cost = parse_solution_file(sol_path)
            row["sol_cost"] = sol_cost

            routes, mode, ev = normalize_solution_routes(instance, raw_routes, sol_cost or cfg["known_best"])
            row["solution_node_interpretation"] = mode

            if ev:
                row["computed_solution_cost"] = ev.get("total_route_cost")
                row["solution_route_count"] = ev.get("route_count")
                row["visited_customer_count"] = ev.get("visited_customer_count")
                row["unique_customer_count"] = ev.get("unique_customer_count")
                row["missing_count"] = ev.get("missing_count")
                row["extra_count"] = ev.get("extra_count")
                row["duplicate_count"] = ev.get("duplicate_count")
                row["capacity_violation_count"] = ev.get("capacity_violation_count")
                row["max_route_load"] = ev.get("max_route_load")

                row["cost_matches_known_best"] = (ev.get("total_route_cost") == cfg["known_best"])
                row["cost_matches_sol_file"] = (sol_cost is not None and ev.get("total_route_cost") == sol_cost)

                details[name] = {
                    "config": cfg,
                    "solution_mode": mode,
                    "routes": routes,
                    "route_loads": ev.get("route_loads"),
                    "route_costs": ev.get("route_costs"),
                    "missing": ev.get("missing"),
                    "extra": ev.get("extra"),
                    "duplicates": ev.get("duplicates"),
                    "capacity_violations": ev.get("capacity_violations"),
                }

            checks = [
                row["vrp_exists"],
                row["sol_exists"],
                row["parsed_nodes"] == cfg["nodes"],
                row["parsed_customers"] == cfg["customers"],
                row["parsed_capacity"] == cfg["capacity"],
                row["solution_route_count"] == cfg["vehicles"],
                row["missing_count"] == 0,
                row["extra_count"] == 0,
                row["duplicate_count"] == 0,
                row["capacity_violation_count"] == 0,
                row["cost_matches_known_best"] or row["cost_matches_sol_file"],
            ]

            if all(checks):
                row["status"] = "PASS"
            else:
                problem_notes = []
                if row["parsed_nodes"] != cfg["nodes"]:
                    problem_notes.append("node_count_mismatch")
                if row["parsed_capacity"] != cfg["capacity"]:
                    problem_notes.append("capacity_mismatch")
                if row["solution_route_count"] != cfg["vehicles"]:
                    problem_notes.append("vehicle_count_mismatch")
                if row["missing_count"] not in (0, None):
                    problem_notes.append("missing_customers")
                if row["extra_count"] not in (0, None):
                    problem_notes.append("extra_nodes")
                if row["duplicate_count"] not in (0, None):
                    problem_notes.append("duplicate_customers")
                if row["capacity_violation_count"] not in (0, None):
                    problem_notes.append("capacity_violation")
                if not (row["cost_matches_known_best"] or row["cost_matches_sol_file"]):
                    problem_notes.append("cost_mismatch")
                row["notes"] = ", ".join(problem_notes)

        except Exception as e:
            row["status"] = "FAIL"
            row["notes"] = str(e)

        rows.append(row)

        status_icon = "OK" if row["status"] == "PASS" else "!!"
        print(
            f"[{status_icon}] {name:10s} | "
            f"nodes={row['parsed_nodes']} | cap={row['parsed_capacity']} | "
            f"routes={row['solution_route_count']} | "
            f"cost={row['computed_solution_cost']} | "
            f"known={cfg['known_best']} | status={row['status']} | {row['notes']}"
        )

    df = pd.DataFrame(rows)

    csv_path = out_dir / "dataset_validation_report.csv"
    xlsx_path = out_dir / "dataset_validation_report.xlsx"
    md_path = out_dir / "dataset_validation_report.md"
    json_path = out_dir / "solution_route_details.json"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Validation")
    except Exception as e:
        print(f"[UYARI] Excel yazılamadı: {e}")

    try:
        md_path.write_text(df.to_markdown(index=False), encoding="utf-8")
    except Exception:
        md_path.write_text(df.to_string(index=False), encoding="utf-8")

    json_path.write_text(json.dumps(to_json_safe(details), ensure_ascii=False, indent=2), encoding="utf-8")

    pass_count = int((df["status"] == "PASS").sum())
    total_count = len(df)

    print("\n" + "=" * 78)
    print(f"VALIDATION SUMMARY: {pass_count}/{total_count} PASS")
    print("=" * 78)
    print(f"CSV   : {csv_path}")
    print(f"Excel : {xlsx_path}")
    print(f"MD    : {md_path}")
    print(f"JSON  : {json_path}")

    if pass_count != total_count:
        print("\n[DIKKAT] Bazı veri setlerinde kontrol hatası var. Rapor dosyasındaki notes sütununa bak.")
    else:
        print("\nTüm veri setleri ve solution dosyaları başarıyla doğrulandı. PC deney altyapısına geçebiliriz.")


if __name__ == "__main__":
    main()

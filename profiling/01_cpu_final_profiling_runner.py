# -*- coding: utf-8 -*-
r"""
10_cpu_final_profiling_runner.py
WOA / SSA / Memetic WOA / Memetic SSA final CPU profiling runner for the selected P3D configuration.

Amaç:
- 8 doğrulanmış CVRP veri setinde 4 algoritmayı küçük ayarla çalıştırmak.
- Her algoritmanın geçerli rota üretip üretmediğini kontrol etmek.
- Müşteri tekrarı, müşteri eksikliği, kapasite ihlali, maliyet hesabı ve gap değerlerini raporlamak.
- Bu aşama nihai makale deneyi değildir; CPU darboğazını ölçmek için profiling testidir.

Klasör yapısı:
  WOA & SSA on FPGA/
    datasets/
    solutions/
    02_algorithm_integrity_checker.py

Çıktılar:
  FINAL_RESULTS_CPU/09_profiling/raw/
    profiling_run_results.csv
    profiling_summary.csv
    profiling_results.xlsx
    best_routes_profiling.json

Not:
- valid_core_solution: müşteri/kopya/kapasite kontrolü
- feasible_solution: valid_core_solution + route_count <= expected vehicle count
"""

import os
import math
import time
import json
import random
import re
import gc
import platform
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ==========================================================
# 0. QUICK INTEGRITY CONFIG
# ==========================================================

TEST_RUNS = 3
POP_SIZE = 40
MAX_ITER = 200
BASE_SEED = 2026

# Swarm/search-space dynamics logging
SWARM_DYNAMICS_DATASETS = {"A-n32-k5", "A-n54-k7"}
SWARM_DYNAMICS_ALGORITHMS = {"WOA", "SSA"}
SWARM_DYNAMICS_RUN = 1


# Feasibility handling
FEASIBLE_SEED_RATIO = 0.50
VEHICLE_EXCESS_PENALTY = 1_000_000_000

# Final P3D quality-balanced memory-safe local search controls.
ROUTE_CACHE_MAX_ITEMS = 8000
TWO_OPT_MAX_PASSES = 25
TWO_OPT_MAX_ACCEPTED_SWAPS = 500
TWO_OPT_MAX_CANDIDATE_CHECKS = 120000
BOUND_RELOCATE_MAX_PASSES = 2
BOUND_RELOCATE_MAX_ACCEPTED_MOVES = 8
BOUND_RELOCATE_MAX_CANDIDATE_CHECKS = 8000

# Bu scriptte amaç hızlı bütünlük kontrolü olduğu için
# memetik sürümlerde 2opt kullanıyoruz. Nihai deneyde 2opt_relocate ayrıca açılabilir.
MEMETIC_LS_MODE = "2opt_qualitybalanced_relocate"
STANDARD_LS_MODE = "none"

EXPECTED_DATASETS = [
    {"name": "A-n32-k5",  "file": "A-n32-k5.vrp",  "nodes": 32, "customers": 31, "vehicles": 5,  "capacity": 100, "known_best": 784},
    {"name": "A-n37-k6",  "file": "A-n37-k6.vrp",  "nodes": 37, "customers": 36, "vehicles": 6,  "capacity": 100, "known_best": 949},
    {"name": "A-n45-k6",  "file": "A-n45-k6.vrp",  "nodes": 45, "customers": 44, "vehicles": 6,  "capacity": 100, "known_best": 944},
    {"name": "A-n54-k7",  "file": "A-n54-k7.vrp",  "nodes": 54, "customers": 53, "vehicles": 7,  "capacity": 100, "known_best": 1167},
    {"name": "A-n60-k9",  "file": "A-n60-k9.vrp",  "nodes": 60, "customers": 59, "vehicles": 9,  "capacity": 100, "known_best": 1354},
    {"name": "A-n80-k10", "file": "A-n80-k10.vrp", "nodes": 80, "customers": 79, "vehicles": 10, "capacity": 100, "known_best": 1763},
    {"name": "B-n50-k7",  "file": "B-n50-k7.vrp",  "nodes": 50, "customers": 49, "vehicles": 7,  "capacity": 100, "known_best": 741},
    {"name": "P-n76-k4",  "file": "P-n76-k4.vrp",  "nodes": 76, "customers": 75, "vehicles": 4,  "capacity": 350, "known_best": 593},
]

ALGORITHM_CONFIGS = [
    {"name": "WOA", "base": "WOA", "ls_mode": STANDARD_LS_MODE},
    {"name": "SSA", "base": "SSA", "ls_mode": STANDARD_LS_MODE},
    {"name": "M-WOA", "base": "WOA", "ls_mode": MEMETIC_LS_MODE},
    {"name": "M-SSA", "base": "SSA", "ls_mode": MEMETIC_LS_MODE},
]


# ==========================================================
# 1. CVRP PROBLEM
# ==========================================================

@dataclass
class CVRPInstance:
    name: str
    capacity: int
    max_vehicles: int
    known_best: float
    depot_id: int
    coords: Dict[int, Tuple[float, float]]
    demands: Dict[int, int]
    dist: Dict[Tuple[int, int], int]

    @property
    def customers(self) -> List[int]:
        return [i for i in sorted(self.coords.keys()) if i != self.depot_id]

    @property
    def num_customers(self) -> int:
        return len(self.customers)


def parse_vrp_file(path: Path, cfg: dict) -> CVRPInstance:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    name = cfg["name"]
    capacity = cfg["capacity"]
    depot_id = 1
    coords = {}
    demands = {}

    coord_section = False
    demand_section = False
    depot_section = False

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        upper = line.upper()

        if upper.startswith("CAPACITY"):
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
                coords[int(parts[0])] = (float(parts[1]), float(parts[2]))

        elif demand_section:
            parts = line.split()
            if len(parts) >= 2:
                demands[int(parts[0])] = int(float(parts[1]))

        elif depot_section:
            try:
                val = int(line.split()[0])
                if val != -1:
                    depot_id = val
            except Exception:
                pass

    for node in coords:
        if node not in demands:
            raise ValueError(f"{path.name}: node {node} için demand yok.")

    dist = {}
    for i, (xi, yi) in coords.items():
        for j, (xj, yj) in coords.items():
            if i == j:
                d = 0
            else:
                d = int(round(math.hypot(xi - xj, yi - yj)))
            dist[(i, j)] = d

    return CVRPInstance(
        name=name,
        capacity=capacity,
        max_vehicles=cfg["vehicles"],
        known_best=float(cfg["known_best"]),
        depot_id=depot_id,
        coords=coords,
        demands=demands,
        dist=dist,
    )


class CVRPProblem:
    def __init__(self, instance: CVRPInstance):
        self.instance = instance
        self.customer_ids = instance.customers
        self.dim = len(self.customer_ids)
        self._route_cache = {}
        self.reset_profile()

    def reset_profile(self):
        self.profile = {
            "initialization_time": 0.0,
            "decode_time": 0.0,
            "split_time": 0.0,
            "vehicle_repair_time": 0.0,
            "two_opt_time": 0.0,
            "relocate_time": 0.0,
            "local_search_total_time": 0.0,
            "cost_evaluation_time": 0.0,
            "lamarckian_writeback_time": 0.0,
            "position_update_time": 0.0,
            "evaluation_total_time": 0.0,
            "route_distance_calls": 0,
            "evaluations": 0,
        }

    def add_profile_time(self, key: str, start_time: float):
        self.profile[key] = self.profile.get(key, 0.0) + (time.perf_counter() - start_time)

    def decode_random_key(self, agent: np.ndarray) -> List[int]:
        order = np.argsort(agent)
        return [self.customer_ids[int(i)] for i in order]

    def split_routes(self, permutation: List[int]) -> List[List[int]]:
        routes = []
        current_route = []
        current_load = 0

        for customer in permutation:
            demand = self.instance.demands[customer]
            if current_load + demand <= self.instance.capacity:
                current_route.append(customer)
                current_load += demand
            else:
                if current_route:
                    routes.append(current_route)
                current_route = [customer]
                current_load = demand

        if current_route:
            routes.append(current_route)

        return routes

    def route_distance_raw(self, route: List[int]) -> float:
        if not route:
            return 0.0
        depot = self.instance.depot_id
        cost = self.instance.dist[(depot, route[0])]
        for a, b in zip(route[:-1], route[1:]):
            cost += self.instance.dist[(a, b)]
        cost += self.instance.dist[(route[-1], depot)]
        return float(cost)

    def route_distance(self, route: List[int]) -> float:
        self.profile["route_distance_calls"] = self.profile.get("route_distance_calls", 0) + 1
        if not route:
            return 0.0
        key = tuple(route)
        val = self._route_cache.get(key)
        if val is not None:
            return val
        cost = self.route_distance_raw(route)
        if len(self._route_cache) >= ROUTE_CACHE_MAX_ITEMS:
            self._route_cache.clear()
        self._route_cache[key] = float(cost)
        return float(cost)

    def total_distance(self, routes: List[List[int]]) -> float:
        return float(sum(self.route_distance(r) for r in routes))

    def route_load(self, route: List[int]) -> int:
        return int(sum(self.instance.demands[c] for c in route))

    def two_opt_delta(self, route: List[int], i: int, j: int) -> float:
        depot = self.instance.depot_id
        dist = self.instance.dist
        a = depot if i == 0 else route[i - 1]
        b = route[i]
        c = route[j - 1]
        d = depot if j == len(route) else route[j]
        old_cost = dist[(a, b)] + dist[(c, d)]
        new_cost = dist[(a, c)] + dist[(b, d)]
        # Count this as a low-level route evaluation/cost operation for profiling rationale.
        self.profile["route_distance_calls"] = self.profile.get("route_distance_calls", 0) + 1
        return float(new_cost - old_cost)

    def apply_2opt(self, route: List[int]) -> List[int]:
        if len(route) < 4:
            return route[:]

        best = route[:]
        accepted_swaps = 0
        candidate_checks = 0

        for _pass in range(TWO_OPT_MAX_PASSES):
            improved = False
            for i in range(1, len(best) - 1):
                if accepted_swaps >= TWO_OPT_MAX_ACCEPTED_SWAPS:
                    return best
                for j in range(i + 2, len(best) + 1):
                    candidate_checks += 1
                    if candidate_checks >= TWO_OPT_MAX_CANDIDATE_CHECKS:
                        return best
                    delta = self.two_opt_delta(best, i, j)
                    if delta < -1e-9:
                        best[i:j] = reversed(best[i:j])
                        accepted_swaps += 1
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

        return best

    def inter_route_relocate(self, routes: List[List[int]]) -> List[List[int]]:
        routes = [r[:] for r in routes if r]
        if len(routes) <= 1:
            return routes

        cap = self.instance.capacity
        demands = self.instance.demands
        depot = self.instance.depot_id
        moves_done = 0
        candidate_checks = 0

        def removal_saving(route: List[int], pos: int) -> float:
            node = route[pos]
            before_node = depot if pos == 0 else route[pos - 1]
            after_node = depot if pos == len(route) - 1 else route[pos + 1]
            self.profile["route_distance_calls"] = self.profile.get("route_distance_calls", 0) + 1
            return (
                self.instance.dist[(before_node, node)]
                + self.instance.dist[(node, after_node)]
                - self.instance.dist[(before_node, after_node)]
            )

        for _pass in range(BOUND_RELOCATE_MAX_PASSES):
            moved_in_pass = False
            while moves_done < BOUND_RELOCATE_MAX_ACCEPTED_MOVES:
                routes = [r for r in routes if r]
                loads = [self.route_load(r) for r in routes]
                found_move = False

                for i in range(len(routes)):
                    if found_move:
                        break
                    if not routes[i]:
                        continue

                    for c_idx in range(len(routes[i])):
                        if found_move:
                            break

                        node = routes[i][c_idx]
                        demand = demands[node]
                        remove_gain = removal_saving(routes[i], c_idx)

                        for j in range(len(routes)):
                            if found_move:
                                break
                            if i == j:
                                continue
                            if loads[j] + demand > cap:
                                continue

                            for ins_idx in range(len(routes[j]) + 1):
                                candidate_checks += 1
                                if candidate_checks > BOUND_RELOCATE_MAX_CANDIDATE_CHECKS:
                                    return [r for r in routes if r]
                                insert_cost = self.insertion_delta(routes[j], node, ins_idx)
                                delta = insert_cost - remove_gain
                                if delta < -1e-9:
                                    removed = routes[i].pop(c_idx)
                                    routes[j].insert(ins_idx, removed)
                                    moves_done += 1
                                    moved_in_pass = True
                                    found_move = True
                                    break

                if not found_move:
                    break

            if not moved_in_pass:
                break

        return [r for r in routes if r]


    def insertion_delta(self, route: List[int], node: int, pos: int) -> float:
        """Cost increase caused by inserting node at position pos in route."""
        depot = self.instance.depot_id
        before_node = depot if pos == 0 else route[pos - 1]
        after_node = depot if pos == len(route) else route[pos]
        return (
            self.instance.dist[(before_node, node)]
            + self.instance.dist[(node, after_node)]
            - self.instance.dist[(before_node, after_node)]
        )

    def repair_vehicle_count(self, routes: List[List[int]]) -> List[List[int]]:
        """
        Tries to reduce route_count to max_vehicles by redistributing customers
        from small routes into other routes using capacity-feasible minimum insertion.

        This is a feasibility repair, not the memetic improvement operator.
        Therefore it is applied to all four algorithmic configurations.
        """
        routes = [r[:] for r in routes if r]
        k = self.instance.max_vehicles
        cap = self.instance.capacity

        if len(routes) <= k:
            return routes

        improved = True
        while len(routes) > k and improved:
            improved = False
            # Try to eliminate the smallest/lightest routes first.
            route_order = sorted(
                range(len(routes)),
                key=lambda idx: (len(routes[idx]), self.route_load(routes[idx]))
            )

            for source_idx in route_order:
                if source_idx >= len(routes):
                    continue

                candidate_routes = [r[:] for r in routes]
                source_nodes = sorted(
                    candidate_routes[source_idx],
                    key=lambda c: self.instance.demands[c],
                    reverse=True
                )
                success = True

                for node in source_nodes:
                    best_target = None
                    best_pos = None
                    best_delta = float("inf")

                    for target_idx, target_route in enumerate(candidate_routes):
                        if target_idx == source_idx:
                            continue
                        if self.route_load(target_route) + self.instance.demands[node] > cap:
                            continue

                        for pos in range(len(target_route) + 1):
                            delta = self.insertion_delta(target_route, node, pos)
                            if delta < best_delta:
                                best_delta = delta
                                best_target = target_idx
                                best_pos = pos

                    if best_target is None:
                        success = False
                        break

                    candidate_routes[best_target].insert(best_pos, node)

                if success:
                    # Remove the eliminated source route.
                    del candidate_routes[source_idx]
                    routes = [r for r in candidate_routes if r]
                    improved = True
                    break

        return routes

    def encode_permutation_to_agent(self, permutation: List[int], rng: np.random.Generator) -> np.ndarray:
        """Encodes a customer order back into random-key representation."""
        base_values = np.linspace(-9.5, 9.5, self.dim)
        noise = rng.normal(0.0, 1e-4, self.dim)
        values = base_values + noise
        agent = np.zeros(self.dim, dtype=float)
        pos_lookup = {customer: idx for idx, customer in enumerate(self.customer_ids)}
        for order_idx, customer in enumerate(permutation):
            agent[pos_lookup[customer]] = values[order_idx]
        return agent

    def create_feasible_seed_permutation(self, rng: np.random.Generator) -> List[int]:
        """
        Creates a capacity-feasible initial giant tour using a randomized
        first-fit-decreasing route construction. The resulting flattened order
        tends to produce <= k routes under the greedy split decoder.
        """
        k = self.instance.max_vehicles
        cap = self.instance.capacity
        routes = [[] for _ in range(k)]
        loads = [0 for _ in range(k)]

        customers = self.customer_ids[:]
        # Demand-prioritized, slightly randomized order.
        customers.sort(key=lambda c: (-self.instance.demands[c], rng.random()))

        for customer in customers:
            demand = self.instance.demands[customer]
            feasible_targets = [idx for idx in range(k) if loads[idx] + demand <= cap]
            if not feasible_targets:
                # Fallback should almost never happen for these benchmarks.
                rng.shuffle(customers)
                return customers

            # Prefer lower load, with slight randomness for diversity.
            target = min(feasible_targets, key=lambda idx: (loads[idx], rng.random()))
            routes[target].append(customer)
            loads[target] += demand

        for r in routes:
            rng.shuffle(r)
        rng.shuffle(routes)

        permutation = [c for r in routes for c in r]
        return permutation

    def initialize_population(self, pop_size: int, rng: np.random.Generator) -> np.ndarray:
        _t_init = time.perf_counter()
        positions = rng.uniform(-10, 10, size=(pop_size, self.dim))
        n_seeded = max(1, int(round(pop_size * FEASIBLE_SEED_RATIO)))
        for i in range(n_seeded):
            perm = self.create_feasible_seed_permutation(rng)
            positions[i] = self.encode_permutation_to_agent(perm, rng)
        self.add_profile_time("initialization_time", _t_init)
        return positions

    def local_search(self, routes: List[List[int]], ls_mode: str) -> List[List[int]]:
        routes = [r[:] for r in routes if r]

        # Fleet-size feasibility repair is applied to all configurations.
        _t = time.perf_counter()
        routes = self.repair_vehicle_count(routes)
        self.add_profile_time("vehicle_repair_time", _t)

        if ls_mode == "none":
            return routes

        _t = time.perf_counter()
        routes = [self.apply_2opt(r) for r in routes]
        self.add_profile_time("two_opt_time", _t)

        _t = time.perf_counter()
        routes = self.repair_vehicle_count(routes)
        self.add_profile_time("vehicle_repair_time", _t)

        if ls_mode in ("2opt_relocate", "2opt_qualitybalanced_relocate"):
            _t = time.perf_counter()
            routes = self.inter_route_relocate(routes)
            self.add_profile_time("relocate_time", _t)

            _t = time.perf_counter()
            routes = self.repair_vehicle_count(routes)
            self.add_profile_time("vehicle_repair_time", _t)

            _t = time.perf_counter()
            routes = [self.apply_2opt(r) for r in routes]
            self.add_profile_time("two_opt_time", _t)

        return routes

    def evaluate_and_improve(self, agent: np.ndarray, ls_mode: str):
        _t_eval = time.perf_counter()
        self.profile["evaluations"] = self.profile.get("evaluations", 0) + 1

        _t = time.perf_counter()
        permutation = self.decode_random_key(agent)
        self.add_profile_time("decode_time", _t)

        _t = time.perf_counter()
        routes = self.split_routes(permutation)
        self.add_profile_time("split_time", _t)

        _t = time.perf_counter()
        routes = self.local_search(routes, ls_mode)
        self.add_profile_time("local_search_total_time", _t)

        _t = time.perf_counter()
        cost = self.total_distance(routes)
        self.add_profile_time("cost_evaluation_time", _t)

        # Araç sayısı aşımı varsa çok yüksek ceza veriyoruz.
        # Böylece seçim mekanizması feasible çözümü daima önceliklendirir.
        if len(routes) > self.instance.max_vehicles:
            cost += VEHICLE_EXCESS_PENALTY * (len(routes) - self.instance.max_vehicles)

        _t = time.perf_counter()
        # Lamarckian write-back
        new_perm = []
        for r in routes:
            new_perm.extend(r)

        sorted_values = np.sort(agent)
        improved_agent = np.zeros_like(agent)

        pos_lookup = {customer: idx for idx, customer in enumerate(self.customer_ids)}
        for idx, customer in enumerate(new_perm):
            improved_agent[pos_lookup[customer]] = sorted_values[idx]
        self.add_profile_time("lamarckian_writeback_time", _t)

        self.add_profile_time("evaluation_total_time", _t_eval)
        return float(cost), routes, improved_agent

    def validate_solution(self, routes: List[List[int]]) -> Dict[str, object]:
        flat = [c for r in routes for c in r]
        customer_set = set(self.customer_ids)
        visited = set(flat)

        missing = sorted(customer_set - visited)
        extra = sorted(visited - customer_set)
        duplicates = sorted({x for x in flat if flat.count(x) > 1})
        loads = [self.route_load(r) for r in routes]
        cap_viol = [i + 1 for i, load in enumerate(loads) if load > self.instance.capacity]

        valid_core = (
            len(missing) == 0
            and len(extra) == 0
            and len(duplicates) == 0
            and len(cap_viol) == 0
        )
        vehicle_count_valid = len(routes) <= self.instance.max_vehicles
        feasible = valid_core and vehicle_count_valid

        return {
            "valid_core_solution": valid_core,
            "vehicle_count_valid": vehicle_count_valid,
            "feasible_solution": feasible,
            "missing_count": len(missing),
            "extra_count": len(extra),
            "duplicate_count": len(duplicates),
            "capacity_violation_count": len(cap_viol),
            "vehicle_violation_count": max(0, len(routes) - self.instance.max_vehicles),
            "route_count": len(routes),
            "expected_vehicles": self.instance.max_vehicles,
            "max_route_load": max(loads) if loads else 0,
            "loads": loads,
            "cost_without_vehicle_penalty": self.total_distance(routes),
        }


# ==========================================================
# 2. OPTIMIZERS
# ==========================================================

class BaseOptimizer:
    def __init__(self, problem: CVRPProblem, pop_size: int, max_iter: int, seed: int, ls_mode: str):
        self.problem = problem
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.seed = seed
        self.ls_mode = ls_mode
        self.rng = np.random.default_rng(seed)
        self.dim = problem.dim
        self.curve = []
        self.best_routes = []
        self.time_to_best = 0.0
        self.iter_to_best = 0
        self.eval_to_best = 0
        self.swarm_snapshots = {}
        self.best_trajectory = []


class WOA(BaseOptimizer):
    def optimize(self):
        positions = self.problem.initialize_population(self.pop_size, self.rng)
        best_pos = positions[0].copy()
        best_score = float("inf")
        eval_counter = 0
        start = time.perf_counter()

        snapshot_points = {0, max(0, self.max_iter // 2), self.max_iter - 1}
        if 0 in snapshot_points:
            self.swarm_snapshots["start"] = positions.copy().tolist()

        for t in range(self.max_iter):
            a = 2 - t * (2 / self.max_iter)
            a2 = -1 + t * ((-1) / self.max_iter)

            for i in range(self.pop_size):
                score, routes, new_agent = self.problem.evaluate_and_improve(positions[i], self.ls_mode)
                positions[i] = new_agent
                eval_counter += 1

                if score < best_score:
                    best_score = score
                    best_pos = positions[i].copy()
                    self.best_routes = [r[:] for r in routes]
                    self.time_to_best = time.perf_counter() - start
                    self.iter_to_best = t + 1
                    self.eval_to_best = eval_counter
                    self.best_trajectory.append(best_pos.copy().tolist())

            _t_position_update = time.perf_counter()
            for i in range(self.pop_size):
                r1, r2, p = self.rng.random(), self.rng.random(), self.rng.random()
                A = 2 * a * r1 - a
                C = 2 * r2
                b = 1
                l = (a2 - 1) * self.rng.random() + 1

                if p < 0.5:
                    if abs(A) < 1:
                        positions[i] = best_pos - A * np.abs(C * best_pos - positions[i])
                    else:
                        rand_pos = positions[int(self.rng.integers(0, self.pop_size))]
                        positions[i] = rand_pos - A * np.abs(C * rand_pos - positions[i])
                else:
                    dist_to_leader = np.abs(best_pos - positions[i])
                    positions[i] = dist_to_leader * math.exp(b * l) * math.cos(2 * math.pi * l) + best_pos

                positions[i] = np.clip(positions[i], -10, 10)
            self.problem.add_profile_time("position_update_time", _t_position_update)

            self.curve.append(float(best_score))
            if t in snapshot_points:
                tag = "middle" if t == max(0, self.max_iter // 2) else ("final" if t == self.max_iter - 1 else f"iter_{t+1}")
                self.swarm_snapshots[tag] = positions.copy().tolist()

        return float(best_score), self.best_routes, self.curve, time.perf_counter() - start


class SSA(BaseOptimizer):
    def optimize(self):
        positions = self.problem.initialize_population(self.pop_size, self.rng)
        best_pos = positions[0].copy()
        best_score = float("inf")
        eval_counter = 0
        start = time.perf_counter()

        snapshot_points = {0, max(0, self.max_iter // 2), self.max_iter - 1}
        if 0 in snapshot_points:
            self.swarm_snapshots["start"] = positions.copy().tolist()

        for t in range(self.max_iter):
            c1 = 2 * math.exp(-((4 * t / self.max_iter) ** 2))

            for i in range(self.pop_size):
                score, routes, new_agent = self.problem.evaluate_and_improve(positions[i], self.ls_mode)
                positions[i] = new_agent
                eval_counter += 1

                if score < best_score:
                    best_score = score
                    best_pos = positions[i].copy()
                    self.best_routes = [r[:] for r in routes]
                    self.time_to_best = time.perf_counter() - start
                    self.iter_to_best = t + 1
                    self.eval_to_best = eval_counter
                    self.best_trajectory.append(best_pos.copy().tolist())

            _t_position_update = time.perf_counter()
            new_positions = positions.copy()

            for i in range(self.pop_size):
                if i == 0:
                    for j in range(self.dim):
                        c2, c3 = self.rng.random(), self.rng.random()
                        span = (20 * c2 - 10)
                        if c3 >= 0.5:
                            new_positions[i, j] = best_pos[j] + c1 * span
                        else:
                            new_positions[i, j] = best_pos[j] - c1 * span
                else:
                    new_positions[i] = (new_positions[i] + new_positions[i - 1]) / 2.0

                new_positions[i] = np.clip(new_positions[i], -10, 10)

            positions = new_positions
            self.problem.add_profile_time("position_update_time", _t_position_update)
            self.curve.append(float(best_score))
            if t in snapshot_points:
                tag = "middle" if t == max(0, self.max_iter // 2) else ("final" if t == self.max_iter - 1 else f"iter_{t+1}")
                self.swarm_snapshots[tag] = positions.copy().tolist()

        return float(best_score), self.best_routes, self.curve, time.perf_counter() - start


# ==========================================================
# 3. MAIN
# ==========================================================

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


def run_algorithm(problem: CVRPProblem, algo_cfg: dict, seed: int):
    problem.reset_profile()
    if algo_cfg["base"] == "WOA":
        opt = WOA(problem, pop_size=POP_SIZE, max_iter=MAX_ITER, seed=seed, ls_mode=algo_cfg["ls_mode"])
    elif algo_cfg["base"] == "SSA":
        opt = SSA(problem, pop_size=POP_SIZE, max_iter=MAX_ITER, seed=seed, ls_mode=algo_cfg["ls_mode"])
    else:
        raise ValueError(f"Bilinmeyen base algorithm: {algo_cfg['base']}")

    best_score, routes, curve, runtime = opt.optimize()
    val = problem.validate_solution(routes)

    return {
        "best_score_with_penalty": float(best_score),
        "routes": routes,
        "curve": curve,
        "runtime": float(runtime),
        "time_to_best": float(opt.time_to_best),
        "iter_to_best": int(opt.iter_to_best),
        "eval_to_best": int(opt.eval_to_best),
        "validation": val,
        "profile": dict(problem.profile),
        "swarm_snapshots": getattr(opt, "swarm_snapshots", {}),
        "best_trajectory": getattr(opt, "best_trajectory", []),
    }


def main():
    root = Path(__file__).resolve().parent
    datasets_dir = root / "datasets"
    out_dir = root / "FINAL_RESULTS_CPU" / "09_profiling" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "profiling_system_info.json").write_text(json.dumps(to_json_safe({
        "python_version": __import__('sys').version.replace('\n',' '),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "note": "Final profiling uses the selected P3D CPU configuration."
    }), ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("FINAL CPU PROFILING - P3D CONFIGURATION - BOTTLENECK ANALYSIS")
    print("=" * 90)
    print(f"Root      : {root}")
    print(f"Datasets  : {datasets_dir}")
    print(f"Out       : {out_dir}")
    print(f"Runs      : {TEST_RUNS}")
    print(f"Pop       : {POP_SIZE}")
    print(f"Iter      : {MAX_ITER}")
    print(f"Memetic LS: {MEMETIC_LS_MODE}")
    print(f"Swarm dynamics datasets: {sorted(SWARM_DYNAMICS_DATASETS)}")
    print("=" * 90)

    all_rows = []
    curve_rows = []
    best_routes_store = {}
    swarm_dynamics_store = {}

    overall_start = time.perf_counter()

    for ds_idx, cfg in enumerate(EXPECTED_DATASETS):
        vrp_path = datasets_dir / cfg["file"]
        instance = parse_vrp_file(vrp_path, cfg)
        problem = CVRPProblem(instance)

        print(f"\n[DATASET] {cfg['name']} | customers={problem.dim} | capacity={instance.capacity} | known_best={instance.known_best}")

        for algo_idx, algo_cfg in enumerate(ALGORITHM_CONFIGS):
            algo_name = algo_cfg["name"]
            algo_best_cost = float("inf")
            algo_best_routes = None

            for run_idx in range(1, TEST_RUNS + 1):
                seed = BASE_SEED + ds_idx * 100000 + algo_idx * 1000 + run_idx

                print(f"  -> {algo_name:5s} run {run_idx}/{TEST_RUNS} ...", end=" ", flush=True)
                res = run_algorithm(problem, algo_cfg, seed)

                val = res["validation"]
                raw_cost = float(val["cost_without_vehicle_penalty"])
                gap = 100.0 * (raw_cost - instance.known_best) / instance.known_best
                penalty_gap = 100.0 * (res["best_score_with_penalty"] - instance.known_best) / instance.known_best

                row = {
                    "dataset": instance.name,
                    "algorithm": algo_name,
                    "base_algorithm": algo_cfg["base"],
                    "ls_mode": algo_cfg["ls_mode"],
                    "run": run_idx,
                    "seed": seed,
                    "pop_size": POP_SIZE,
                    "max_iter": MAX_ITER,
                    "known_best": instance.known_best,
                    "best_score_with_penalty": res["best_score_with_penalty"],
                    "cost_without_vehicle_penalty": raw_cost,
                    "gap_percent_without_vehicle_penalty": gap,
                    "gap_percent_with_penalty": penalty_gap,
                    "route_count": val["route_count"],
                    "expected_vehicles": instance.max_vehicles,
                    "max_route_load": val["max_route_load"],
                    "capacity": instance.capacity,
                    "valid_core_solution": val["valid_core_solution"],
                    "vehicle_count_valid": val["vehicle_count_valid"],
                    "feasible_solution": val["feasible_solution"],
                    "vehicle_violation_count": val["vehicle_violation_count"],
                    "missing_count": val["missing_count"],
                    "extra_count": val["extra_count"],
                    "duplicate_count": val["duplicate_count"],
                    "capacity_violation_count": val["capacity_violation_count"],
                    "runtime_sec": res["runtime"],
                    "time_to_best_sec": res["time_to_best"],
                    "iter_to_best": res["iter_to_best"],
                    "eval_to_best": res["eval_to_best"],
                }
                profile = res.get("profile", {})
                for profile_key, profile_value in profile.items():
                    row[f"profile_{profile_key}"] = profile_value

                stage_cols_for_other = [
                    "initialization_time", "decode_time", "split_time", "vehicle_repair_time",
                    "two_opt_time", "relocate_time", "cost_evaluation_time",
                    "lamarckian_writeback_time", "position_update_time"
                ]
                stage_sum = sum(float(profile.get(k, 0.0)) for k in stage_cols_for_other)
                row["profile_other_time"] = max(0.0, float(res["runtime"]) - stage_sum)

                all_rows.append(row)

                for iter_idx, curve_value in enumerate(res["curve"], start=1):
                    curve_rows.append({
                        "dataset": instance.name,
                        "algorithm": algo_name,
                        "run": run_idx,
                        "seed": seed,
                        "iteration": iter_idx,
                        "best_score_with_penalty": float(curve_value),
                    })

                if (instance.name in SWARM_DYNAMICS_DATASETS and algo_name in SWARM_DYNAMICS_ALGORITHMS and run_idx == SWARM_DYNAMICS_RUN):
                    swarm_dynamics_store[f"{instance.name}_{algo_name}_run{run_idx}"] = {
                        "dataset": instance.name,
                        "algorithm": algo_name,
                        "run": run_idx,
                        "seed": seed,
                        "snapshots": res.get("swarm_snapshots", {}),
                        "best_trajectory": res.get("best_trajectory", []),
                        "curve": res.get("curve", []),
                        "known_best": instance.known_best,
                    }

                print(
                    f"cost={raw_cost:.0f} | gap={gap:.2f}% | routes={val['route_count']}/{instance.max_vehicles} | "
                    f"core_valid={val['valid_core_solution']} | feasible={val['feasible_solution']} | time={res['runtime']:.2f}s"
                )

                gc.collect()

                if val["feasible_solution"] and raw_cost < algo_best_cost:
                    algo_best_cost = raw_cost
                    algo_best_routes = [r[:] for r in res["routes"]]

            if algo_best_routes is not None:
                best_routes_store[f"{instance.name}_{algo_name}"] = {
                    "dataset": instance.name,
                    "algorithm": algo_name,
                    "best_cost_without_vehicle_penalty": algo_best_cost,
                    "known_best": instance.known_best,
                    "gap_percent": 100.0 * (algo_best_cost - instance.known_best) / instance.known_best,
                    "route_count": len(algo_best_routes),
                    "routes": algo_best_routes,
                    "route_loads": [problem.route_load(r) for r in algo_best_routes],
                    "route_distances": [problem.route_distance(r) for r in algo_best_routes],
                }

    df = pd.DataFrame(all_rows)
    curve_df = pd.DataFrame(curve_rows)

    profile_stage_map = {
        "Initialization": "profile_initialization_time",
        "Decode": "profile_decode_time",
        "Split": "profile_split_time",
        "Vehicle repair": "profile_vehicle_repair_time",
        "2-opt local search": "profile_two_opt_time",
        "Relocate local search": "profile_relocate_time",
        "Final cost evaluation": "profile_cost_evaluation_time",
        "Lamarckian write-back": "profile_lamarckian_writeback_time",
        "Position update": "profile_position_update_time",
        "Other": "profile_other_time",
    }
    stage_rows = []
    for _, rr in df.iterrows():
        total_runtime = float(rr.get("runtime_sec", 0.0))
        for stage_name, col_name in profile_stage_map.items():
            seconds = float(rr.get(col_name, 0.0))
            stage_rows.append({
                "dataset": rr["dataset"],
                "algorithm": rr["algorithm"],
                "run": rr["run"],
                "seed": rr["seed"],
                "stage": stage_name,
                "seconds": seconds,
                "percent_of_runtime": 100.0 * seconds / total_runtime if total_runtime > 0 else 0.0,
                "runtime_sec": total_runtime,
            })
    stage_df = pd.DataFrame(stage_rows)

    bottleneck_df = (
        stage_df.groupby(["algorithm", "stage"])
        .agg(
            mean_seconds=("seconds", "mean"),
            mean_percent_of_runtime=("percent_of_runtime", "mean"),
            std_percent_of_runtime=("percent_of_runtime", "std"),
        )
        .reset_index()
        .sort_values(["algorithm", "mean_percent_of_runtime"], ascending=[True, False])
    )

    summary = (
        df.groupby(["dataset", "algorithm", "base_algorithm", "ls_mode"])
        .agg(
            runs=("run", "count"),
            all_core_valid=("valid_core_solution", "all"),
            all_feasible=("feasible_solution", "all"),
            feasible_runs=("feasible_solution", "sum"),
            best_cost=("cost_without_vehicle_penalty", "min"),
            best_feasible_cost=("cost_without_vehicle_penalty", lambda s: s[df.loc[s.index, "feasible_solution"]].min() if df.loc[s.index, "feasible_solution"].any() else float("nan")),
            mean_cost=("cost_without_vehicle_penalty", "mean"),
            std_cost=("cost_without_vehicle_penalty", "std"),
            mean_gap_percent=("gap_percent_without_vehicle_penalty", "mean"),
            best_gap_percent=("gap_percent_without_vehicle_penalty", "min"),
            mean_runtime_sec=("runtime_sec", "mean"),
            mean_profile_eval_total_sec=("profile_evaluation_total_time", "mean"),
            mean_profile_local_search_total_sec=("profile_local_search_total_time", "mean"),
            mean_profile_two_opt_sec=("profile_two_opt_time", "mean"),
            mean_profile_vehicle_repair_sec=("profile_vehicle_repair_time", "mean"),
            mean_profile_cost_eval_sec=("profile_cost_evaluation_time", "mean"),
            mean_profile_position_update_sec=("profile_position_update_time", "mean"),
            mean_profile_route_distance_calls=("profile_route_distance_calls", "mean"),
            mean_route_count=("route_count", "mean"),
            max_capacity_violation_count=("capacity_violation_count", "max"),
            max_vehicle_violation_count=("vehicle_violation_count", "max"),
            max_missing_count=("missing_count", "max"),
            max_duplicate_count=("duplicate_count", "max"),
        )
        .reset_index()
        .sort_values(["dataset", "best_gap_percent", "mean_gap_percent"])
    )

    # Output files
    run_csv = out_dir / "profiling_run_results.csv"
    summary_csv = out_dir / "profiling_summary.csv"
    xlsx_path = out_dir / "profiling_results.xlsx"
    json_path = out_dir / "best_routes_profiling.json"
    curve_csv = out_dir / "profiling_convergence_curves.csv"
    swarm_json = out_dir / "profiling_swarm_dynamics.json"
    stage_csv = out_dir / "profiling_stage_long.csv"
    bottleneck_csv = out_dir / "profiling_bottleneck_percentages.csv"

    df.to_csv(run_csv, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    curve_df.to_csv(curve_csv, index=False, encoding="utf-8-sig")
    stage_df.to_csv(stage_csv, index=False, encoding="utf-8-sig")
    bottleneck_df.to_csv(bottleneck_csv, index=False, encoding="utf-8-sig")

    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Run_Level", index=False)
            summary.to_excel(writer, sheet_name="Summary", index=False)
            stage_df.to_excel(writer, sheet_name="Stage_Long", index=False)
            bottleneck_df.to_excel(writer, sheet_name="Bottleneck", index=False)
            curve_df.to_excel(writer, sheet_name="Convergence_Curves", index=False)
    except Exception as e:
        print(f"[UYARI] Excel yazılamadı: {e}")

    json_path.write_text(json.dumps(to_json_safe(best_routes_store), ensure_ascii=False, indent=2), encoding="utf-8")
    swarm_json.write_text(json.dumps(to_json_safe(swarm_dynamics_store), ensure_ascii=False, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - overall_start

    print("\n" + "=" * 90)
    print("CPU PROFILING COMPLETED")
    print("=" * 90)
    print(f"Total time : {elapsed:.2f} s")
    print(f"Run CSV    : {run_csv}")
    print(f"Summary    : {summary_csv}")
    print(f"Stages     : {stage_csv}")
    print(f"Bottleneck : {bottleneck_csv}")
    print(f"Curves     : {curve_csv}")
    print(f"Swarm dyn. : {swarm_json}")
    print(f"Excel      : {xlsx_path}")
    print(f"Routes JSON: {json_path}")

    print("\nSUMMARY PREVIEW:")
    preview_cols = [
        "dataset", "algorithm", "runs", "all_core_valid", "all_feasible", "feasible_runs",
        "best_cost", "best_feasible_cost", "mean_gap_percent", "best_gap_percent", "mean_runtime_sec",
        "max_capacity_violation_count", "max_vehicle_violation_count", "max_missing_count", "max_duplicate_count"
    ]
    print(summary[preview_cols].to_string(index=False))

    invalid_core_rows = df[df["valid_core_solution"] == False]
    infeasible_rows = df[df["feasible_solution"] == False]

    if len(invalid_core_rows) > 0:
        print("\n[DIKKAT] Müşteri/kopya/kapasite açısından geçersiz çözüm üreten koşular var. profiling_run_results.csv dosyasını kontrol edelim.")
    elif len(infeasible_rows) > 0:
        print("\n[NOT] Tüm çözümler müşteri ve kapasite açısından doğru; ancak bazı koşularda araç sayısı beklenen k değerini aştı.")
        print("Bu orta deneyde not alınmalı. Nihai deneyde parametreler ve repair ayarları yeniden kontrol edilecek.")
    else:
        print("\nTüm algoritmalar profiling koşularında tam feasible CVRP çözümü üretti. Bottleneck çıktıları FPGA gerekçesi için kullanılabilir.")


if __name__ == "__main__":
    main()

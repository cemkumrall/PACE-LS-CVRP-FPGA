# -*- coding: utf-8 -*-
r"""
07_cpu_final_experiment_runner.py
WOA / SSA / Memetic WOA / Memetic SSA final 30-run CPU experiment runner with checkpoint/resume for CVRP.

Amaç:
- Seçilen final P3D parametre seti ile nihai 30-run CPU deneylerini yürütmek.
- Feasibility-preserving decoder ve vehicle-count repair yapısını bozmadan çalışmak.
- Yakınsama, çözüm kalitesi, feasibility ve execution-time sonuçlarını final klasör yapısına kaydetmek.
- Checkpoint/resume mantığı ile uzun pilot koşuların baştan başlamasını önlemek.

Klasör yapısı:
  WOA & SSA on FPGA/
    datasets/
    solutions/
    07_cpu_final_experiment_runner.py

Çıktılar:
  FINAL_RESULTS_CPU/
    pilot_run_results.csv
    pilot_summary.csv
    pilot_results.xlsx
    pilot_convergence_curves.csv
    best_routes_pilot.json

Not:
- valid_core_solution: müşteri/kopya/kapasite kontrolü
- feasible_solution: valid_core_solution + route_count <= expected vehicle count
"""

import os
import sys
import math
import time
import json
import random
import re
import platform
import gc
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ==========================================================
# 0. QUICK INTEGRITY CONFIG
# ==========================================================

TEST_RUNS = 10
POP_SIZE = 25
MAX_ITER = 100
BASE_SEED = 2026

# Swarm/search-space dynamics logging
SWARM_DYNAMICS_DATASETS = {"A-n32-k5", "A-n54-k7"}
SWARM_DYNAMICS_ALGORITHMS = {"WOA", "SSA"}
SWARM_DYNAMICS_RUN = 1


# Feasibility handling
FEASIBLE_SEED_RATIO = 0.50
VEHICLE_EXCESS_PENALTY = 1_000_000_000

# Bounded relocate controls for P3B.
# Amaç: P3'teki kalite avantajını korumak ama kontrolsüz local-search süre patlamasını engellemek.
BOUND_RELOCATE_MAX_PASSES = 2
BOUND_RELOCATE_MAX_ACCEPTED_MOVES = 8
BOUND_RELOCATE_MAX_CANDIDATE_CHECKS = 8000

# Quality-balanced memory-safe 2-opt controls.
# V3 çok hızlandı ama yakınsama düştü; bu V4 sürümünde 2-opt daha güçlü çalışır.
# Kritik nokta: 2-opt aday rotaları yine cache'e yazılmaz; RAM güvenliği korunur.
ROUTE_CACHE_MAX_ITEMS = 8000
TWO_OPT_MAX_PASSES = 25
TWO_OPT_MAX_ACCEPTED_SWAPS = 500
TWO_OPT_MAX_CANDIDATE_CHECKS = 120000

# Bu scriptte amaç hızlı bütünlük kontrolü olduğu için
# memetik sürümlerde 2opt kullanıyoruz. Nihai deneyde 2opt_relocate ayrıca açılabilir.
MEMETIC_LS_MODE = "2opt"      # "2opt" veya "2opt_relocate"
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
        """Cache kullanmadan rota maliyeti hesaplar."""
        if not route:
            return 0.0
        depot = self.instance.depot_id
        cost = self.instance.dist[(depot, route[0])]
        for a, b in zip(route[:-1], route[1:]):
            cost += self.instance.dist[(a, b)]
        cost += self.instance.dist[(route[-1], depot)]
        return float(cost)

    def route_distance(self, route: List[int]) -> float:
        """
        Bounded route-distance cache.

        V2'de MemoryError oluşmasının nedeni, 2-opt içindeki milyonlarca aday rotanın
        tuple(route) olarak cache'e yazılmasıydı. Bu sürümde cache sınırlıdır ve 2-opt
        adayları zaten route_distance_raw / delta formülü ile hesaplanır.
        """
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
        """
        route[i:j] ters çevrilirse maliyet değişimini O(1) hesaplar.
        Burada j, Python slice bitiş indeksidir; ters çevrilen segment i..j-1 aralığıdır.
        """
        depot = self.instance.depot_id
        dist = self.instance.dist

        a = depot if i == 0 else route[i - 1]
        b = route[i]
        c = route[j - 1]
        d = depot if j == len(route) else route[j]

        old_cost = dist[(a, b)] + dist[(c, d)]
        new_cost = dist[(a, c)] + dist[(b, d)]
        return float(new_cost - old_cost)

    def apply_2opt(self, route: List[int]) -> List[int]:
        """
        Memory-safe bounded 2-opt.

        Önceki sürümde her candidate route kopyalanıp route_distance(candidate)
        çağrıldığı için route cache büyüyebiliyordu. Bu sürüm:
        - candidate rota için tuple/cache üretmez,
        - delta hesabı kullanır,
        - pass / swap / candidate-check sınırı koyar,
        - kabul edilen iyileştirmede yalnızca o anda segmenti ters çevirir.
        """
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
                    # First-improvement: yeni rota üzerinde baştan devam et.
                    break

            if not improved:
                break

        return best

    def inter_route_relocate(self, routes: List[List[int]]) -> List[List[int]]:
        """
        Bounded inter-route relocate.

        Önceki pilotta kullanılan sınırsız relocate bazı M-SSA koşularında çok uzun
        iyileştirme döngülerine girebiliyordu. Bu sürümde:
        - accepted move sayısı sınırlıdır,
        - pass sayısı sınırlıdır,
        - candidate check sayısı sınırlıdır,
        - ilk iyileştirme kabul edilir.

        Böylece P3'ün kalite avantajı korunurken final deney için kontrol edilebilir
        runtime davranışı elde edilir.
        """
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
                                    # Apply first improving feasible move.
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
        positions = rng.uniform(-10, 10, size=(pop_size, self.dim))
        n_seeded = max(1, int(round(pop_size * FEASIBLE_SEED_RATIO)))
        for i in range(n_seeded):
            perm = self.create_feasible_seed_permutation(rng)
            positions[i] = self.encode_permutation_to_agent(perm, rng)
        return positions

    def local_search(self, routes: List[List[int]], ls_mode: str) -> List[List[int]]:
        routes = [r[:] for r in routes if r]

        # Fleet-size feasibility repair is applied to all configurations.
        routes = self.repair_vehicle_count(routes)

        if ls_mode == "none":
            return routes

        routes = [self.apply_2opt(r) for r in routes]
        routes = self.repair_vehicle_count(routes)

        if ls_mode in ("2opt_relocate", "2opt_bounded_relocate", "2opt_memorysafe_bounded_relocate", "2opt_qualitybalanced_relocate"):
            routes = self.inter_route_relocate(routes)
            routes = self.repair_vehicle_count(routes)
            routes = [self.apply_2opt(r) for r in routes]

        return routes

    def evaluate_and_improve(self, agent: np.ndarray, ls_mode: str):
        permutation = self.decode_random_key(agent)
        routes = self.split_routes(permutation)
        routes = self.local_search(routes, ls_mode)

        cost = self.total_distance(routes)

        # Araç sayısı aşımı varsa çok yüksek ceza veriyoruz.
        # Böylece seçim mekanizması feasible çözümü daima önceliklendirir.
        if len(routes) > self.instance.max_vehicles:
            cost += VEHICLE_EXCESS_PENALTY * (len(routes) - self.instance.max_vehicles)

        # Lamarckian write-back
        new_perm = []
        for r in routes:
            new_perm.extend(r)

        sorted_values = np.sort(agent)
        improved_agent = np.zeros_like(agent)

        pos_lookup = {customer: idx for idx, customer in enumerate(self.customer_ids)}
        for idx, customer in enumerate(new_perm):
            improved_agent[pos_lookup[customer]] = sorted_values[idx]

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
            self.curve.append(float(best_score))
            if t in snapshot_points:
                tag = "middle" if t == max(0, self.max_iter // 2) else ("final" if t == self.max_iter - 1 else f"iter_{t+1}")
                self.swarm_snapshots[tag] = positions.copy().tolist()

        return float(best_score), self.best_routes, self.curve, time.perf_counter() - start




# ==========================================================
# 3. FINAL CPU EXPERIMENT MAIN
# ==========================================================

FINAL_RESULTS_ROOT_NAME = "FINAL_RESULTS_CPU"

FINAL_RUNS = 30
FINAL_POP_SIZE = 40
FINAL_MAX_ITER = 200
FINAL_MEMETIC_LS_MODE = "2opt_qualitybalanced_relocate"
FINAL_CONFIG_ID = "FINAL_P3D_pop40_iter200_2opt_qualitybalanced_relocate"
FINAL_DESCRIPTION = "Final CPU configuration selected from pilot: quality-balanced memory-safe 2opt+bounded relocate"

FINAL_DATASET_NAMES = {cfg["name"] for cfg in EXPECTED_DATASETS}

# Finalde tüm algoritmalar aynı veri setlerinde çalışır.
FINAL_ALGORITHMS = [
    {"name": "WOA", "base": "WOA", "ls_mode": "none"},
    {"name": "SSA", "base": "SSA", "ls_mode": "none"},
    {"name": "M-WOA", "base": "WOA", "ls_mode": FINAL_MEMETIC_LS_MODE},
    {"name": "M-SSA", "base": "SSA", "ls_mode": FINAL_MEMETIC_LS_MODE},
]


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


def collect_system_info() -> Dict[str, object]:
    return {
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "note": "CPU runtime values are hardware- and environment-dependent. Fair FPGA comparison must use matched kernel-level CPU timings generated under 08_kernel_timing.",
    }


def make_final_dirs(root: Path) -> Dict[str, Path]:
    final_root = root / FINAL_RESULTS_ROOT_NAME
    dirs = {
        "root": final_root,
        "environment": final_root / "00_environment",
        "raw": final_root / "01_raw",
        "summaries": final_root / "02_summaries",
        "convergence": final_root / "03_convergence",
        "best_routes": final_root / "04_best_routes",
        "figures": final_root / "05_figures",
        "tables": final_root / "06_tables",
        "logs": final_root / "07_logs",
        "kernel_timing": final_root / "08_kernel_timing",
        "profiling": final_root / "09_profiling",
        "article_exports": final_root / "10_article_exports",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def write_experiment_config(dirs: Dict[str, Path]):
    cfg = {
        "config_id": FINAL_CONFIG_ID,
        "description": FINAL_DESCRIPTION,
        "runs": FINAL_RUNS,
        "pop_size": FINAL_POP_SIZE,
        "max_iter": FINAL_MAX_ITER,
        "memetic_ls_mode": FINAL_MEMETIC_LS_MODE,
        "algorithms": FINAL_ALGORITHMS,
        "datasets": EXPECTED_DATASETS,
        "base_seed": BASE_SEED,
        "two_opt": {
            "max_passes": TWO_OPT_MAX_PASSES,
            "max_accepted_swaps": TWO_OPT_MAX_ACCEPTED_SWAPS,
            "max_candidate_checks": TWO_OPT_MAX_CANDIDATE_CHECKS,
        },
        "bounded_relocate": {
            "max_passes": BOUND_RELOCATE_MAX_PASSES,
            "max_accepted_moves": BOUND_RELOCATE_MAX_ACCEPTED_MOVES,
            "max_candidate_checks": BOUND_RELOCATE_MAX_CANDIDATE_CHECKS,
        },
        "route_cache_max_items": ROUTE_CACHE_MAX_ITEMS,
        "fair_fpga_comparison_note": "Kernel-level CPU timings must be compared only with the same operation block implemented on FPGA.",
    }
    (dirs["environment"] / "experiment_config.json").write_text(
        json.dumps(to_json_safe(cfg), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def make_run_key(dataset: str, algorithm: str, run: int) -> str:
    return f"{FINAL_CONFIG_ID}||{dataset}||{algorithm}||{run}"


def load_completed_keys(run_csv: Path) -> set:
    if not run_csv.exists():
        return set()
    try:
        df = pd.read_csv(run_csv)
        if df.empty:
            return set()
        return {
            make_run_key(str(r["dataset"]), str(r["algorithm"]), int(r["run"]))
            for _, r in df.iterrows()
        }
    except Exception:
        return set()


def append_row_csv(path: Path, row: dict):
    df = pd.DataFrame([row])
    header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=header, encoding="utf-8-sig")


def append_rows_csv(path: Path, rows: list):
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=header, encoding="utf-8-sig")


def run_algorithm(problem: CVRPProblem, algo_cfg: dict, seed: int):
    if algo_cfg["base"] == "WOA":
        opt = WOA(problem, pop_size=FINAL_POP_SIZE, max_iter=FINAL_MAX_ITER, seed=seed, ls_mode=algo_cfg["ls_mode"])
    elif algo_cfg["base"] == "SSA":
        opt = SSA(problem, pop_size=FINAL_POP_SIZE, max_iter=FINAL_MAX_ITER, seed=seed, ls_mode=algo_cfg["ls_mode"])
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
    }


def compute_final_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["dataset", "algorithm", "base_algorithm", "ls_mode"])
        .agg(
            runs=("run", "count"),
            all_core_valid=("valid_core_solution", "all"),
            all_feasible=("feasible_solution", "all"),
            feasible_runs=("feasible_solution", "sum"),
            best_cost=("cost_without_vehicle_penalty", "min"),
            mean_cost=("cost_without_vehicle_penalty", "mean"),
            median_cost=("cost_without_vehicle_penalty", "median"),
            std_cost=("cost_without_vehicle_penalty", "std"),
            min_gap_percent=("gap_percent_without_vehicle_penalty", "min"),
            mean_gap_percent=("gap_percent_without_vehicle_penalty", "mean"),
            median_gap_percent=("gap_percent_without_vehicle_penalty", "median"),
            std_gap_percent=("gap_percent_without_vehicle_penalty", "std"),
            max_gap_percent=("gap_percent_without_vehicle_penalty", "max"),
            mean_runtime_sec=("runtime_sec", "mean"),
            median_runtime_sec=("runtime_sec", "median"),
            std_runtime_sec=("runtime_sec", "std"),
            min_runtime_sec=("runtime_sec", "min"),
            max_runtime_sec=("runtime_sec", "max"),
            mean_time_to_best_sec=("time_to_best_sec", "mean"),
            median_time_to_best_sec=("time_to_best_sec", "median"),
            mean_iter_to_best=("iter_to_best", "mean"),
            median_iter_to_best=("iter_to_best", "median"),
            mean_eval_to_best=("eval_to_best", "mean"),
            median_eval_to_best=("eval_to_best", "median"),
            mean_route_count=("route_count", "mean"),
            max_capacity_violation_count=("capacity_violation_count", "max"),
            max_vehicle_violation_count=("vehicle_violation_count", "max"),
            max_missing_count=("missing_count", "max"),
            max_duplicate_count=("duplicate_count", "max"),
        )
        .reset_index()
    )

    # Improvement against base algorithms on same dataset
    base_map = {}
    for _, row in summary.iterrows():
        base_map[(row["dataset"], row["algorithm"])] = row

    imp_rows = []
    for _, row in summary.iterrows():
        alg = row["algorithm"]
        base_alg = None
        if alg == "M-WOA":
            base_alg = "WOA"
        elif alg == "M-SSA":
            base_alg = "SSA"

        if base_alg and (row["dataset"], base_alg) in base_map:
            base_row = base_map[(row["dataset"], base_alg)]
            imp_mean = 100.0 * (base_row["mean_gap_percent"] - row["mean_gap_percent"]) / max(abs(base_row["mean_gap_percent"]), 1e-12)
            imp_best = 100.0 * (base_row["min_gap_percent"] - row["min_gap_percent"]) / max(abs(base_row["min_gap_percent"]), 1e-12)
        else:
            imp_mean = np.nan
            imp_best = np.nan

        imp_rows.append({
            "dataset": row["dataset"],
            "algorithm": alg,
            "gap_reduction_vs_base_mean_percent": imp_mean,
            "gap_reduction_vs_base_best_percent": imp_best,
        })

    imp_df = pd.DataFrame(imp_rows)
    summary = summary.merge(imp_df, on=["dataset", "algorithm"], how="left")

    order_map = {name: i for i, name in enumerate(["WOA", "SSA", "M-WOA", "M-SSA"])}
    dataset_order = {cfg["name"]: i for i, cfg in enumerate(EXPECTED_DATASETS)}
    summary["dataset_order"] = summary["dataset"].map(dataset_order)
    summary["algorithm_order"] = summary["algorithm"].map(order_map)
    summary = summary.sort_values(["dataset_order", "algorithm_order"]).drop(columns=["dataset_order", "algorithm_order"])

    return summary


def compute_algorithm_overall(summary: pd.DataFrame) -> pd.DataFrame:
    overall = (
        summary.groupby("algorithm")
        .agg(
            avg_mean_gap_percent=("mean_gap_percent", "mean"),
            avg_best_gap_percent=("min_gap_percent", "mean"),
            avg_median_gap_percent=("median_gap_percent", "mean"),
            avg_runtime_sec=("mean_runtime_sec", "mean"),
            avg_time_to_best_sec=("mean_time_to_best_sec", "mean"),
            all_feasible=("all_feasible", "all"),
            best_rank_count=("min_gap_percent", lambda x: 0),
        )
        .reset_index()
    )

    # Best-rank count by dataset
    rank_counts = {a: 0 for a in overall["algorithm"].tolist()}
    for ds, g in summary.groupby("dataset"):
        best_val = g["min_gap_percent"].min()
        winners = g[g["min_gap_percent"] == best_val]["algorithm"].tolist()
        for w in winners:
            rank_counts[w] = rank_counts.get(w, 0) + 1
    overall["best_rank_count"] = overall["algorithm"].map(rank_counts)

    order_map = {name: i for i, name in enumerate(["WOA", "SSA", "M-WOA", "M-SSA"])}
    overall["algorithm_order"] = overall["algorithm"].map(order_map)
    overall = overall.sort_values("algorithm_order").drop(columns=["algorithm_order"])
    return overall


def compute_dataset_best(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds, g in summary.groupby("dataset"):
        best = g.loc[g["min_gap_percent"].idxmin()]
        best_mem = g[g["algorithm"].isin(["M-WOA", "M-SSA"])].loc[g[g["algorithm"].isin(["M-WOA", "M-SSA"])]["min_gap_percent"].idxmin()]
        rows.append({
            "dataset": ds,
            "best_algorithm_overall": best["algorithm"],
            "best_gap_overall": best["min_gap_percent"],
            "best_cost_overall": best["best_cost"],
            "best_memetic_algorithm": best_mem["algorithm"],
            "best_gap_memetic": best_mem["min_gap_percent"],
            "best_cost_memetic": best_mem["best_cost"],
        })
    return pd.DataFrame(rows)


def update_final_summaries(run_csv: Path, dirs: Dict[str, Path]):
    if not run_csv.exists():
        return None, None, None

    df = pd.read_csv(run_csv)
    if df.empty:
        return None, None, None

    summary = compute_final_summary(df)
    overall = compute_algorithm_overall(summary)
    dataset_best = compute_dataset_best(summary)

    summary.to_csv(dirs["summaries"] / "final_summary.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(dirs["summaries"] / "final_algorithm_overall.csv", index=False, encoding="utf-8-sig")
    dataset_best.to_csv(dirs["summaries"] / "final_dataset_best.csv", index=False, encoding="utf-8-sig")

    article_main = summary[[
        "dataset", "algorithm", "runs", "all_feasible",
        "best_cost", "min_gap_percent", "mean_gap_percent", "std_gap_percent",
        "mean_runtime_sec", "std_runtime_sec", "mean_time_to_best_sec", "mean_iter_to_best"
    ]].copy()
    article_main.to_csv(dirs["tables"] / "article_main_cpu_results_table.csv", index=False, encoding="utf-8-sig")

    # Excel only at summary points; no repeated write during runs.
    xlsx_path = dirs["tables"] / "final_cpu_results.xlsx"
    try:
        curve_path = dirs["convergence"] / "final_convergence_curves.csv"
        curve_df = pd.read_csv(curve_path) if curve_path.exists() else pd.DataFrame()
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Run_Level", index=False)
            summary.to_excel(writer, sheet_name="Summary", index=False)
            overall.to_excel(writer, sheet_name="Algorithm_Overall", index=False)
            dataset_best.to_excel(writer, sheet_name="Dataset_Best", index=False)
            article_main.to_excel(writer, sheet_name="Article_Main_Table", index=False)
            if len(curve_df) > 0 and len(curve_df) <= 1_000_000:
                curve_df.to_excel(writer, sheet_name="Convergence", index=False)
    except Exception as e:
        (dirs["logs"] / "excel_warning.txt").write_text(str(e), encoding="utf-8")

    return summary, overall, dataset_best


def main():
    root = Path(__file__).resolve().parent
    datasets_dir = root / "datasets"
    dirs = make_final_dirs(root)

    system_info_path = dirs["environment"] / "system_info.json"
    system_info_path.write_text(json.dumps(to_json_safe(collect_system_info()), ensure_ascii=False, indent=2), encoding="utf-8")
    write_experiment_config(dirs)

    run_csv = dirs["raw"] / "final_run_results.csv"
    curve_csv = dirs["convergence"] / "final_convergence_curves.csv"
    json_path = dirs["best_routes"] / "best_routes_final.json"
    log_path = dirs["logs"] / "final_execution_log.txt"

    print("\n" + "=" * 110)
    print("FINAL CPU EXPERIMENT RUNNER - 30 RUNS - CHECKPOINT / RESUME ENABLED")
    print("=" * 110)
    print(f"Root           : {root}")
    print(f"Datasets       : {datasets_dir}")
    print(f"Final results  : {dirs['root']}")
    print(f"Runs           : {FINAL_RUNS}")
    print(f"Config         : {FINAL_CONFIG_ID}")
    print(f"pop / iter     : {FINAL_POP_SIZE} / {FINAL_MAX_ITER}")
    print(f"Memetic LS     : {FINAL_MEMETIC_LS_MODE}")
    print(f"2-opt bounds   : passes={TWO_OPT_MAX_PASSES}, swaps={TWO_OPT_MAX_ACCEPTED_SWAPS}, checks={TWO_OPT_MAX_CANDIDATE_CHECKS}")
    print(f"Relocate bounds: passes={BOUND_RELOCATE_MAX_PASSES}, moves={BOUND_RELOCATE_MAX_ACCEPTED_MOVES}, checks={BOUND_RELOCATE_MAX_CANDIDATE_CHECKS}")
    print("=" * 110)

    completed = load_completed_keys(run_csv)
    if completed:
        print(f"[RESUME] Tamamlanmış koşu sayısı: {len(completed)}")

    best_routes_store = {}
    if json_path.exists():
        try:
            best_routes_store = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            best_routes_store = {}

    overall_start = time.perf_counter()
    log_lines = []

    for ds_idx, ds_cfg in enumerate(EXPECTED_DATASETS):
        if ds_cfg["name"] not in FINAL_DATASET_NAMES:
            continue

        vrp_path = datasets_dir / ds_cfg["file"]
        instance = parse_vrp_file(vrp_path, ds_cfg)
        problem = CVRPProblem(instance)

        print(f"\n[DATASET] {instance.name} | customers={problem.dim} | capacity={instance.capacity} | known_best={instance.known_best}")

        for algo_idx, algo_cfg in enumerate(FINAL_ALGORITHMS):
            algo_name = algo_cfg["name"]

            # Existing best for resume
            algo_best_cost = float("inf")
            if run_csv.exists():
                try:
                    prev_df = pd.read_csv(run_csv)
                    prev_subset = prev_df[
                        (prev_df["dataset"].astype(str) == instance.name)
                        & (prev_df["algorithm"].astype(str) == algo_name)
                        & (prev_df["feasible_solution"] == True)
                    ]
                    if len(prev_subset) > 0:
                        algo_best_cost = float(prev_subset["cost_without_vehicle_penalty"].min())
                except Exception:
                    pass

            for run_idx in range(1, FINAL_RUNS + 1):
                run_key = make_run_key(instance.name, algo_name, run_idx)
                if run_key in completed:
                    print(f"  -> {algo_name:5s} run {run_idx:02d}/{FINAL_RUNS} SKIP")
                    continue

                seed = BASE_SEED + ds_idx * 100_000 + algo_idx * 1_000 + run_idx

                print(f"  -> {algo_name:5s} run {run_idx:02d}/{FINAL_RUNS} ...", end=" ", flush=True)

                try:
                    res = run_algorithm(problem, algo_cfg, seed)
                    val = res["validation"]
                    raw_cost = float(val["cost_without_vehicle_penalty"])
                    gap = 100.0 * (raw_cost - instance.known_best) / instance.known_best
                    penalty_gap = 100.0 * (res["best_score_with_penalty"] - instance.known_best) / instance.known_best

                    row = {
                        "config_id": FINAL_CONFIG_ID,
                        "description": FINAL_DESCRIPTION,
                        "dataset": instance.name,
                        "algorithm": algo_name,
                        "base_algorithm": algo_cfg["base"],
                        "ls_mode": algo_cfg["ls_mode"],
                        "run": run_idx,
                        "seed": seed,
                        "pop_size": FINAL_POP_SIZE,
                        "max_iter": FINAL_MAX_ITER,
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

                    append_row_csv(run_csv, row)

                    curve_rows = []
                    for iter_idx, curve_value in enumerate(res["curve"], start=1):
                        curve_gap = 100.0 * (float(curve_value) - instance.known_best) / instance.known_best
                        curve_rows.append({
                            "config_id": FINAL_CONFIG_ID,
                            "dataset": instance.name,
                            "algorithm": algo_name,
                            "run": run_idx,
                            "seed": seed,
                            "iteration": iter_idx,
                            "known_best": instance.known_best,
                            "best_score_with_penalty": float(curve_value),
                            "best_gap_percent_with_penalty": float(curve_gap),
                        })
                    append_rows_csv(curve_csv, curve_rows)

                    print(
                        f"cost={raw_cost:.0f} | gap={gap:.2f}% | routes={val['route_count']}/{instance.max_vehicles} | "
                        f"feasible={val['feasible_solution']} | time={res['runtime']:.2f}s | t_best={res['time_to_best']:.2f}s"
                    )

                    log_lines.append(
                        f"{FINAL_CONFIG_ID},{instance.name},{algo_name},run={run_idx},seed={seed},"
                        f"cost={raw_cost:.0f},gap={gap:.6f},feasible={val['feasible_solution']},"
                        f"time={res['runtime']:.6f},time_to_best={res['time_to_best']:.6f}"
                    )

                    if val["feasible_solution"] and raw_cost < algo_best_cost:
                        algo_best_cost = raw_cost
                        best_routes_store[f"{FINAL_CONFIG_ID}_{instance.name}_{algo_name}"] = {
                            "config_id": FINAL_CONFIG_ID,
                            "dataset": instance.name,
                            "algorithm": algo_name,
                            "best_cost_without_vehicle_penalty": algo_best_cost,
                            "known_best": instance.known_best,
                            "gap_percent": 100.0 * (algo_best_cost - instance.known_best) / instance.known_best,
                            "route_count": len(res["routes"]),
                            "routes": [r[:] for r in res["routes"]],
                            "route_loads": [problem.route_load(r) for r in res["routes"]],
                            "route_distances": [problem.route_distance(r) for r in res["routes"]],
                        }
                        json_path.write_text(json.dumps(to_json_safe(best_routes_store), ensure_ascii=False, indent=2), encoding="utf-8")

                    completed.add(run_key)

                    if log_lines:
                        with log_path.open("a", encoding="utf-8") as f:
                            f.write("\n".join(log_lines) + "\n")
                        log_lines = []

                    gc.collect()

                except MemoryError as e:
                    err = f"MEMORY_ERROR,{instance.name},{algo_name},run={run_idx},seed={seed},{repr(e)}"
                    print("\n[MEMORY ERROR] Run kaydedilmedi. Resume için aynı run tekrar denenebilir.")
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(err + "\n")
                    gc.collect()
                    raise

                except KeyboardInterrupt:
                    print("\n[INTERRUPTED] Çalışma durduruldu. Resume destekli; aynı script tekrar başlatılabilir.")
                    update_final_summaries(run_csv, dirs)
                    raise

                except Exception as e:
                    err = f"ERROR,{instance.name},{algo_name},run={run_idx},seed={seed},{repr(e)}"
                    print(f"\n[ERROR] {err}")
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(err + "\n")
                    gc.collect()
                    raise

        # Her dataset sonunda ara özet oluştur.
        update_final_summaries(run_csv, dirs)

    summary, overall, dataset_best = update_final_summaries(run_csv, dirs)
    elapsed = time.perf_counter() - overall_start

    print("\n" + "=" * 110)
    print("FINAL CPU EXPERIMENT COMPLETED")
    print("=" * 110)
    print(f"Elapsed this session : {elapsed:.2f} s")
    print(f"Final results root   : {dirs['root']}")
    print(f"Run CSV              : {run_csv}")
    print(f"Curves CSV           : {curve_csv}")
    print(f"Best routes JSON     : {json_path}")
    print(f"Summary CSV          : {dirs['summaries'] / 'final_summary.csv'}")
    print(f"Excel                : {dirs['tables'] / 'final_cpu_results.xlsx'}")

    if overall is not None:
        print("\n[ALGORITHM OVERALL]")
        print(overall.to_string(index=False))

    if summary is not None:
        infeasible = pd.read_csv(run_csv)
        bad = infeasible[infeasible["feasible_solution"] == False]
        if len(bad) > 0:
            print("\n[DIKKAT] Feasible olmayan final koşuları var. final_run_results.csv kontrol edilmeli.")
        else:
            print("\nTüm tamamlanan final koşularında feasible CVRP çözümü üretildi.")


if __name__ == "__main__":
    main()

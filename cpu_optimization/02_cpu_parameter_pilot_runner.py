# -*- coding: utf-8 -*-
r"""
06_cpu_parameter_pilot_runner_V4_qualitybalanced.py
WOA / SSA / Memetic WOA / Memetic SSA memory-safe quality-balanced local-search parameter pilot runner with checkpoint/resume for CVRP.

Amaç:
- Nihai CPU deneyine geçmeden önce parametre adaylarını kontrollü biçimde test etmek.
- Feasibility-preserving decoder ve vehicle-count repair yapısını bozmadan çalışmak.
- Popülasyon/iterasyon/local-search seçeneklerinin kalite-zaman dengesini ölçmek.
- Checkpoint/resume mantığı ile uzun pilot koşuların baştan başlamasını önlemek.

Klasör yapısı:
  WOA & SSA on FPGA/
    datasets/
    solutions/
    06_cpu_parameter_pilot_runner_V4_qualitybalanced.py

Çıktılar:
  06_cpu_parameter_pilot_results_V2_bounded/
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
# 3. PARAMETER PILOT MAIN
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




def collect_system_info() -> Dict[str, object]:
    """CPU deney ortamını kaydeder; execution time yorumları için gereklidir."""
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
        "note": "CPU runtime values are hardware- and environment-dependent; fair FPGA comparison must use matched kernel-level CPU timings.",
    }

# ==========================================================
# 3.1 PILOT CONFIGURATION
# ==========================================================

# Bu V2 pilot yalnızca kontrolsüz uzayan P3 yerine bounded relocate testini çalıştırır.
# P1/P2 sonuçlarını tekrar üretmez; bunlarla karşılaştırma için combined analyzer kullanılır.
PILOT_RUNS = 3

PARAMETER_CONFIGS = [
    {
        "config_id": "P3D_pop40_iter200_2opt_qualitybalanced_relocate",
        "description": "Quality-balanced memory-safe 2opt+relocate pilot; stronger convergence with controlled RAM",
        "pop_size": 40,
        "max_iter": 200,
        "memetic_ls_mode": "2opt_qualitybalanced_relocate",
        "enabled": True,
    },
]

# Tüm veri setleri açık. Çok uzun sürerse önce temsilî sete düşürülebilir:
# PILOT_DATASET_NAMES = {"A-n54-k7", "A-n80-k10", "P-n76-k4"}
PILOT_DATASET_NAMES = {cfg["name"] for cfg in EXPECTED_DATASETS}

# Final seçim için metrik ağırlıkları.
# Düşük gap daha önemli, runtime ikinci öncelik.
SELECTION_WEIGHT_GAP = 0.75
SELECTION_WEIGHT_RUNTIME = 0.25


def build_algorithm_configs(memetic_ls_mode: str):
    return [
        {"name": "WOA", "base": "WOA", "ls_mode": "none"},
        {"name": "SSA", "base": "SSA", "ls_mode": "none"},
        {"name": "M-WOA", "base": "WOA", "ls_mode": memetic_ls_mode},
        {"name": "M-SSA", "base": "SSA", "ls_mode": memetic_ls_mode},
    ]


def run_algorithm(problem: CVRPProblem, algo_cfg: dict, seed: int, pop_size: int, max_iter: int):
    if algo_cfg["base"] == "WOA":
        opt = WOA(problem, pop_size=pop_size, max_iter=max_iter, seed=seed, ls_mode=algo_cfg["ls_mode"])
    elif algo_cfg["base"] == "SSA":
        opt = SSA(problem, pop_size=pop_size, max_iter=max_iter, seed=seed, ls_mode=algo_cfg["ls_mode"])
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


def make_run_key(config_id: str, dataset: str, algorithm: str, run: int):
    return f"{config_id}||{dataset}||{algorithm}||{run}"


def load_completed_keys(run_csv: Path):
    if not run_csv.exists():
        return set()
    try:
        df = pd.read_csv(run_csv)
        if df.empty:
            return set()
        return {
            make_run_key(str(r["config_id"]), str(r["dataset"]), str(r["algorithm"]), int(r["run"]))
            for _, r in df.iterrows()
        }
    except Exception:
        return set()


def append_row_csv(path: Path, row: dict):
    df = pd.DataFrame([row])
    header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=header, encoding="utf-8-sig")


def append_curve_csv(path: Path, rows: list):
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=header, encoding="utf-8-sig")


def compute_summary(df: pd.DataFrame):
    summary = (
        df.groupby(["config_id", "description", "pop_size", "max_iter", "memetic_ls_mode", "dataset", "algorithm", "base_algorithm", "ls_mode"])
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
            std_gap_percent=("gap_percent_without_vehicle_penalty", "std"),
            mean_runtime_sec=("runtime_sec", "mean"),
            std_runtime_sec=("runtime_sec", "std"),
            mean_time_to_best_sec=("time_to_best_sec", "mean"),
            mean_iter_to_best=("iter_to_best", "mean"),
            mean_eval_to_best=("eval_to_best", "mean"),
            mean_route_count=("route_count", "mean"),
            max_capacity_violation_count=("capacity_violation_count", "max"),
            max_vehicle_violation_count=("vehicle_violation_count", "max"),
            max_missing_count=("missing_count", "max"),
            max_duplicate_count=("duplicate_count", "max"),
        )
        .reset_index()
        .sort_values(["config_id", "dataset", "best_gap_percent", "mean_gap_percent"])
    )

    return summary


def compute_config_ranking(summary: pd.DataFrame):
    # Her config için dataset-algorithm özetlerini birleştirir.
    # Final parametre seçimi için özellikle memetik algoritmalara daha çok bakacağız.
    rows = []
    for config_id, g in summary.groupby("config_id"):
        desc = g["description"].iloc[0]
        pop_size = int(g["pop_size"].iloc[0])
        max_iter = int(g["max_iter"].iloc[0])
        ls_mode = g["memetic_ls_mode"].iloc[0]

        all_alg = g.copy()
        mem = g[g["algorithm"].isin(["M-WOA", "M-SSA"])].copy()
        mwoa = g[g["algorithm"] == "M-WOA"].copy()
        mssa = g[g["algorithm"] == "M-SSA"].copy()

        rows.append({
            "config_id": config_id,
            "description": desc,
            "pop_size": pop_size,
            "max_iter": max_iter,
            "memetic_ls_mode": ls_mode,
            "all_feasible": bool(all_alg["all_feasible"].all()),
            "avg_mean_gap_all": float(all_alg["mean_gap_percent"].mean()),
            "avg_best_gap_all": float(all_alg["best_gap_percent"].mean()),
            "avg_runtime_all": float(all_alg["mean_runtime_sec"].mean()),
            "avg_mean_gap_memetic": float(mem["mean_gap_percent"].mean()) if len(mem) else float("nan"),
            "avg_best_gap_memetic": float(mem["best_gap_percent"].mean()) if len(mem) else float("nan"),
            "avg_runtime_memetic": float(mem["mean_runtime_sec"].mean()) if len(mem) else float("nan"),
            "avg_mean_gap_mwoa": float(mwoa["mean_gap_percent"].mean()) if len(mwoa) else float("nan"),
            "avg_best_gap_mwoa": float(mwoa["best_gap_percent"].mean()) if len(mwoa) else float("nan"),
            "avg_runtime_mwoa": float(mwoa["mean_runtime_sec"].mean()) if len(mwoa) else float("nan"),
            "avg_mean_gap_mssa": float(mssa["mean_gap_percent"].mean()) if len(mssa) else float("nan"),
            "avg_best_gap_mssa": float(mssa["best_gap_percent"].mean()) if len(mssa) else float("nan"),
            "avg_runtime_mssa": float(mssa["mean_runtime_sec"].mean()) if len(mssa) else float("nan"),
        })

    rank = pd.DataFrame(rows)

    # Normalize edilmiş seçim skoru: düşük daha iyi.
    # Gap ana ölçüt, runtime ikincil ölçüt.
    gap = rank["avg_mean_gap_memetic"]
    rt = np.log1p(rank["avg_runtime_memetic"])

    gap_norm = (gap - gap.min()) / (gap.max() - gap.min() + 1e-12)
    rt_norm = (rt - rt.min()) / (rt.max() - rt.min() + 1e-12)

    rank["selection_score_lower_is_better"] = (
        SELECTION_WEIGHT_GAP * gap_norm + SELECTION_WEIGHT_RUNTIME * rt_norm
    )

    rank = rank.sort_values("selection_score_lower_is_better").reset_index(drop=True)
    rank["selection_rank"] = np.arange(1, len(rank) + 1)

    return rank


def main():
    root = Path(__file__).resolve().parent
    datasets_dir = root / "datasets"
    out_dir = root / "06_cpu_parameter_pilot_results_V4_qualitybalanced"
    out_dir.mkdir(exist_ok=True)

    system_info_path = out_dir / "system_info.json"
    system_info_path.write_text(json.dumps(to_json_safe(collect_system_info()), ensure_ascii=False, indent=2), encoding="utf-8")

    run_csv = out_dir / "pilot_run_results.csv"
    curve_csv = out_dir / "pilot_convergence_curves.csv"
    summary_csv = out_dir / "pilot_summary.csv"
    ranking_csv = out_dir / "pilot_config_ranking.csv"
    xlsx_path = out_dir / "pilot_results.xlsx"
    json_path = out_dir / "best_routes_pilot.json"
    log_path = out_dir / "pilot_execution_log.txt"

    print("\n" + "=" * 100)
    print("FINAL CPU PARAMETER PILOT RUNNER - CHECKPOINT / RESUME ENABLED")
    print("=" * 100)
    print(f"Root      : {root}")
    print(f"Datasets  : {datasets_dir}")
    print(f"Out       : {out_dir}")
    print(f"Runs      : {PILOT_RUNS}")
    print(f"Datasets  : {sorted(PILOT_DATASET_NAMES)}")
    print("Configs:")
    for cfg in PARAMETER_CONFIGS:
        if cfg.get("enabled", True):
            print(f"  - {cfg['config_id']} | pop={cfg['pop_size']} | iter={cfg['max_iter']} | ls={cfg['memetic_ls_mode']}")
    print(f"Quality-balanced memory-safe 2-opt: passes={TWO_OPT_MAX_PASSES}, swaps={TWO_OPT_MAX_ACCEPTED_SWAPS}, checks={TWO_OPT_MAX_CANDIDATE_CHECKS}, cache_max={ROUTE_CACHE_MAX_ITEMS}")
    print(f"Bounded relocate: passes={BOUND_RELOCATE_MAX_PASSES}, moves={BOUND_RELOCATE_MAX_ACCEPTED_MOVES}, checks={BOUND_RELOCATE_MAX_CANDIDATE_CHECKS}")
    print("=" * 100)

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

    for exp_idx, exp_cfg in enumerate(PARAMETER_CONFIGS):
        if not exp_cfg.get("enabled", True):
            continue

        config_id = exp_cfg["config_id"]
        description = exp_cfg["description"]
        pop_size = int(exp_cfg["pop_size"])
        max_iter = int(exp_cfg["max_iter"])
        memetic_ls_mode = exp_cfg["memetic_ls_mode"]
        algorithm_configs = build_algorithm_configs(memetic_ls_mode)

        print("\n" + "-" * 100)
        print(f"[CONFIG] {config_id} | {description} | pop={pop_size} | iter={max_iter} | memetic_ls={memetic_ls_mode}")
        print("-" * 100)

        for ds_idx, ds_cfg in enumerate(EXPECTED_DATASETS):
            if ds_cfg["name"] not in PILOT_DATASET_NAMES:
                continue

            vrp_path = datasets_dir / ds_cfg["file"]
            instance = parse_vrp_file(vrp_path, ds_cfg)
            problem = CVRPProblem(instance)

            print(f"\n[DATASET] {instance.name} | customers={problem.dim} | capacity={instance.capacity} | known_best={instance.known_best}")

            for algo_idx, algo_cfg in enumerate(algorithm_configs):
                algo_name = algo_cfg["name"]
                algo_best_cost = float("inf")
                algo_best_routes = None

                # Mevcut CSV'de bu config/dataset/algorithm için önceki en iyi değeri al.
                if run_csv.exists():
                    try:
                        prev_df = pd.read_csv(run_csv)
                        prev_subset = prev_df[
                            (prev_df["config_id"].astype(str) == config_id)
                            & (prev_df["dataset"].astype(str) == instance.name)
                            & (prev_df["algorithm"].astype(str) == algo_name)
                            & (prev_df["feasible_solution"] == True)
                        ]
                        if len(prev_subset) > 0:
                            algo_best_cost = float(prev_subset["cost_without_vehicle_penalty"].min())
                    except Exception:
                        pass

                for run_idx in range(1, PILOT_RUNS + 1):
                    run_key = make_run_key(config_id, instance.name, algo_name, run_idx)
                    if run_key in completed:
                        print(f"  -> {algo_name:5s} run {run_idx}/{PILOT_RUNS} SKIP")
                        continue

                    # Config, dataset, algorithm ve run bazlı deterministik seed.
                    seed = BASE_SEED + exp_idx * 10_000_000 + ds_idx * 100_000 + algo_idx * 1_000 + run_idx

                    print(f"  -> {algo_name:5s} run {run_idx}/{PILOT_RUNS} ...", end=" ", flush=True)
                    res = run_algorithm(problem, algo_cfg, seed, pop_size=pop_size, max_iter=max_iter)

                    val = res["validation"]
                    raw_cost = float(val["cost_without_vehicle_penalty"])
                    gap = 100.0 * (raw_cost - instance.known_best) / instance.known_best
                    penalty_gap = 100.0 * (res["best_score_with_penalty"] - instance.known_best) / instance.known_best

                    row = {
                        "config_id": config_id,
                        "description": description,
                        "dataset": instance.name,
                        "algorithm": algo_name,
                        "base_algorithm": algo_cfg["base"],
                        "ls_mode": algo_cfg["ls_mode"],
                        "run": run_idx,
                        "seed": seed,
                        "pop_size": pop_size,
                        "max_iter": max_iter,
                        "memetic_ls_mode": memetic_ls_mode,
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
                        curve_rows.append({
                            "config_id": config_id,
                            "dataset": instance.name,
                            "algorithm": algo_name,
                            "run": run_idx,
                            "seed": seed,
                            "pop_size": pop_size,
                            "max_iter": max_iter,
                            "memetic_ls_mode": memetic_ls_mode,
                            "iteration": iter_idx,
                            "best_score_with_penalty": float(curve_value),
                        })
                    append_curve_csv(curve_csv, curve_rows)

                    completed.add(run_key)

                    print(
                        f"cost={raw_cost:.0f} | gap={gap:.2f}% | routes={val['route_count']}/{instance.max_vehicles} | "
                        f"feasible={val['feasible_solution']} | time={res['runtime']:.2f}s"
                    )

                    log_lines.append(
                        f"{config_id},{instance.name},{algo_name},run={run_idx},seed={seed},"
                        f"cost={raw_cost:.0f},gap={gap:.4f},feasible={val['feasible_solution']},time={res['runtime']:.4f}"
                    )

                    if val["feasible_solution"] and raw_cost < algo_best_cost:
                        algo_best_cost = raw_cost
                        algo_best_routes = [r[:] for r in res["routes"]]

                        best_routes_store[f"{config_id}_{instance.name}_{algo_name}"] = {
                            "config_id": config_id,
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

                    # Her run sonrası JSON ve log güncellensin.
                    json_path.write_text(json.dumps(to_json_safe(best_routes_store), ensure_ascii=False, indent=2), encoding="utf-8")
                    if log_lines:
                        with log_path.open("a", encoding="utf-8") as f:
                            f.write("\n".join(log_lines) + "\n")
                        log_lines = []

    if not run_csv.exists():
        print("[UYARI] Hiç koşu yapılmadı.")
        return

    df = pd.read_csv(run_csv)
    summary = compute_summary(df)
    ranking = compute_config_ranking(summary)

    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    ranking.to_csv(ranking_csv, index=False, encoding="utf-8-sig")

    curve_df = pd.read_csv(curve_csv) if curve_csv.exists() else pd.DataFrame()

    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Run_Level", index=False)
            summary.to_excel(writer, sheet_name="Summary", index=False)
            ranking.to_excel(writer, sheet_name="Config_Ranking", index=False)
            if len(curve_df) > 0:
                curve_df.to_excel(writer, sheet_name="Convergence_Curves", index=False)
    except Exception as e:
        print(f"[UYARI] Excel yazılamadı: {e}")

    elapsed = time.perf_counter() - overall_start

    print("\n" + "=" * 100)
    print("PARAMETER PILOT COMPLETED")
    print("=" * 100)
    print(f"Total elapsed this session : {elapsed:.2f} s")
    print(f"Run CSV                    : {run_csv}")
    print(f"Summary CSV                : {summary_csv}")
    print(f"Ranking CSV                : {ranking_csv}")
    print(f"Curves CSV                 : {curve_csv}")
    print(f"Excel                      : {xlsx_path}")
    print(f"Routes JSON                : {json_path}")

    print("\n[CONFIG RANKING PREVIEW]")
    cols = [
        "selection_rank", "config_id", "avg_mean_gap_memetic", "avg_best_gap_memetic",
        "avg_runtime_memetic", "all_feasible", "selection_score_lower_is_better"
    ]
    print(ranking[cols].to_string(index=False))

    invalid_core_rows = df[df["valid_core_solution"] == False]
    infeasible_rows = df[df["feasible_solution"] == False]

    if len(invalid_core_rows) > 0:
        print("\n[DIKKAT] Core-valid olmayan koşular var. pilot_run_results.csv dosyasını kontrol edelim.")
    elif len(infeasible_rows) > 0:
        print("\n[DIKKAT] Feasible olmayan koşular var. Nihai deneyden önce repair/penalty kontrol edilmeli.")
    else:
        print("\nTüm pilot koşularında tam feasible CVRP çözümü üretildi.")


if __name__ == "__main__":
    main()

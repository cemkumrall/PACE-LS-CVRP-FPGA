# -*- coding: utf-8 -*-
r"""
01_generate_p76_dual_workload.py

P76 için dual local-search kernel workload üretir:
- 2-opt best-delta scan workload
- inter-route relocate best-delta scan workload

Bu script proje ana klasöründe çalıştırılmalıdır:
  C:\Users\Cem Deniz KUMRAL\Desktop\Aktif Çalışmalar\WOA & SSA on FPGA

Beklenen girdiler:
  datasets/P-n76-k4.vrp
  FINAL_RESULTS_CPU/04_best_routes/best_routes_final.json

Çıktılar:
  C:\FPGA_ZedBoard\17_DUAL_LS_2OPT_RELOCATE_P76\dual_ls_p76_workload.h
  C:\FPGA_ZedBoard\17_DUAL_LS_2OPT_RELOCATE_P76\02_cpu_dual_ls_benchmark.cpp
  C:\FPGA_ZedBoard\17_DUAL_LS_2OPT_RELOCATE_P76\compile_and_run_cpu_baseline.bat
  C:\FPGA_ZedBoard\17_DUAL_LS_2OPT_RELOCATE_P76\workload_summary.txt

Not:
- Bu çalışma tam Python local-search akışını birebir taşımayı hedeflemez.
- Amaç CPU ve FPGA tarafında aynı kernel-level candidate-evaluation workload'unu karşılaştırmaktır.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any

DATASET_DEFAULT = "P-n76-k4"
MEMETIC_ALGS = {"M-WOA", "M-SSA"}


def parse_vrp(vrp_path: Path):
    coords: Dict[int, Tuple[float, float]] = {}
    demands: Dict[int, int] = {}
    depot = 1
    capacity = None
    section = None

    for raw in vrp_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = raw.strip()
        if not s:
            continue
        u = s.upper()
        if u.startswith("CAPACITY"):
            capacity = int(re.sub(r"[^0-9]", " ", s).split()[-1])
            continue
        if u.startswith("NODE_COORD_SECTION"):
            section = "coords"
            continue
        if u.startswith("DEMAND_SECTION"):
            section = "demands"
            continue
        if u.startswith("DEPOT_SECTION"):
            section = "depot"
            continue
        if u.startswith("EOF"):
            break

        if section == "coords":
            parts = s.split()
            if len(parts) >= 3:
                coords[int(parts[0])] = (float(parts[1]), float(parts[2]))
        elif section == "demands":
            parts = s.split()
            if len(parts) >= 2:
                demands[int(parts[0])] = int(float(parts[1]))
        elif section == "depot":
            parts = s.split()
            if parts:
                try:
                    v = int(parts[0])
                    if v != -1:
                        depot = v
                except ValueError:
                    pass

    if capacity is None:
        raise RuntimeError(f"CAPACITY okunamadı: {vrp_path}")
    if not coords or not demands:
        raise RuntimeError(f"VRP coords/demands okunamadı: {vrp_path}")

    return {"coords": coords, "demands": demands, "depot": depot, "capacity": capacity}


def find_best_routes_json(root: Path) -> Path:
    candidates = [
        root / "FINAL_RESULTS_CPU" / "04_best_routes" / "best_routes_final.json",
        root / "FINAL_RESULTS_CPU" / "09_profiling" / "raw" / "best_routes_profiling.json",
        root / "best_routes_final.json",
        root / "best_routes_profiling.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    hits = list(root.rglob("best_routes_final.json")) + list(root.rglob("best_routes_profiling.json"))
    if hits:
        return hits[0]
    raise FileNotFoundError("best_routes_final.json veya best_routes_profiling.json bulunamadı.")


def normalize_routes(raw_routes: Any) -> List[List[int]]:
    if raw_routes is None:
        return []
    routes = []
    for r in raw_routes:
        if not isinstance(r, (list, tuple)):
            continue
        rr = [int(x) for x in r if int(x) > 0]
        if rr:
            routes.append(rr)
    return routes


def choose_routes(best_routes: Dict[str, Any], dataset: str, algorithm: str = "auto"):
    rows = []
    for key, v in best_routes.items():
        if not isinstance(v, dict):
            continue
        if str(v.get("dataset", "")) != dataset:
            continue
        alg = str(v.get("algorithm", ""))
        if algorithm != "auto" and alg != algorithm:
            continue
        if algorithm == "auto" and alg not in MEMETIC_ALGS:
            continue
        routes = normalize_routes(v.get("routes"))
        if not routes:
            continue
        gap = float(v.get("gap_percent", v.get("best_gap_percent", v.get("gap", 1e18))))
        cost = float(v.get("cost_without_vehicle_penalty", v.get("best_cost_without_vehicle_penalty", v.get("best_cost", 1e18))))
        rows.append((gap, cost, alg, key, routes, v))

    if not rows and algorithm == "auto":
        # Fallback: dataset içindeki herhangi bir kayıt.
        for key, v in best_routes.items():
            if isinstance(v, dict) and str(v.get("dataset", "")) == dataset:
                routes = normalize_routes(v.get("routes"))
                if routes:
                    alg = str(v.get("algorithm", "UNKNOWN"))
                    gap = float(v.get("gap_percent", v.get("best_gap_percent", v.get("gap", 1e18))))
                    cost = float(v.get("cost_without_vehicle_penalty", v.get("best_cost_without_vehicle_penalty", v.get("best_cost", 1e18))))
                    rows.append((gap, cost, alg, key, routes, v))

    if not rows:
        raise RuntimeError(f"{dataset} için uygun route kaydı bulunamadı. algorithm={algorithm}")

    rows.sort(key=lambda x: (x[0], x[1], x[2]))
    return rows[0]


def route_load(route: List[int], demands: Dict[int, int]) -> int:
    return sum(int(demands[c]) for c in route)


def build_compact(vrp, routes: List[List[int]]):
    depot_orig = int(vrp["depot"])
    all_nodes = sorted({depot_orig} | {int(c) for r in routes for c in r})
    compact_of = {orig: idx for idx, orig in enumerate(all_nodes)}
    orig_of = {idx: orig for orig, idx in compact_of.items()}
    depot = compact_of[depot_orig]

    n = len(all_nodes)
    dist = [[0 for _ in range(n)] for _ in range(n)]
    coords = vrp["coords"]
    for ia, a in enumerate(all_nodes):
        ax, ay = coords[a]
        for ib, b in enumerate(all_nodes):
            bx, by = coords[b]
            dist[ia][ib] = int(round(math.hypot(ax - bx, ay - by)))

    demands = [0 for _ in range(n)]
    for orig in all_nodes:
        demands[compact_of[orig]] = int(vrp["demands"].get(orig, 0))

    compact_routes = [[compact_of[int(c)] for c in r] for r in routes]
    return {
        "depot": depot,
        "orig_depot": depot_orig,
        "compact_of": compact_of,
        "orig_of": orig_of,
        "dist": dist,
        "demands": demands,
        "routes": compact_routes,
    }


def dist_lookup(dist, a, b):
    return int(dist[int(a)][int(b)])


def twoopt_scan_all_routes(compact, relocate_cap: int = 8000):
    depot = compact["depot"]
    dist = compact["dist"]
    best_delta = 0
    best_r = -1
    best_i = -1
    best_j = -1
    checks = 0
    checksum_delta = 0
    checksum_idx = 0

    for r_idx, route_customers in enumerate(compact["routes"]):
        full = [depot] + route_customers + [depot]
        L = len(full)
        if L < 5:
            continue
        for i in range(1, L - 2):
            for j in range(i + 1, L - 1):
                a, b, c, d = full[i - 1], full[i], full[j], full[j + 1]
                old_cost = dist_lookup(dist, a, b) + dist_lookup(dist, c, d)
                new_cost = dist_lookup(dist, a, c) + dist_lookup(dist, b, d)
                delta = new_cost - old_cost
                checks += 1
                checksum_delta += int(delta)
                checksum_idx += (r_idx + 1) * 1000003 + i * 1009 + j
                if delta < best_delta:
                    best_delta = int(delta)
                    best_r = r_idx
                    best_i = i
                    best_j = j
    return {
        "checks": checks,
        "best_delta": best_delta,
        "best_route": best_r,
        "best_i": best_i,
        "best_j": best_j,
        "checksum_delta": checksum_delta,
        "checksum_idx": checksum_idx,
    }


def relocate_scan(compact, capacity: int, relocate_cap: int = 8000):
    depot = compact["depot"]
    dist = compact["dist"]
    demands = compact["demands"]
    routes = compact["routes"]
    loads = [sum(demands[c] for c in r) for r in routes]

    best_delta = 0
    best_src = -1
    best_pos = -1
    best_dst = -1
    best_ins = -1
    checks = 0
    feasible_checks = 0
    checksum_delta = 0
    checksum_idx = 0

    for src_idx, src_route in enumerate(routes):
        if not src_route:
            continue
        for pos, node in enumerate(src_route):
            demand = demands[node]
            prev_s = depot if pos == 0 else src_route[pos - 1]
            next_s = depot if pos == len(src_route) - 1 else src_route[pos + 1]
            remove_gain = (
                dist_lookup(dist, prev_s, node)
                + dist_lookup(dist, node, next_s)
                - dist_lookup(dist, prev_s, next_s)
            )

            for dst_idx, dst_route in enumerate(routes):
                if src_idx == dst_idx:
                    continue
                if loads[dst_idx] + demand > capacity:
                    continue

                for ins in range(0, len(dst_route) + 1):
                    if checks >= relocate_cap:
                        return {
                            "checks": checks,
                            "feasible_checks": feasible_checks,
                            "best_delta": best_delta,
                            "best_src": best_src,
                            "best_pos": best_pos,
                            "best_dst": best_dst,
                            "best_ins": best_ins,
                            "checksum_delta": checksum_delta,
                            "checksum_idx": checksum_idx,
                        }
                    prev_t = depot if ins == 0 else dst_route[ins - 1]
                    next_t = depot if ins == len(dst_route) else dst_route[ins]
                    insert_cost = (
                        dist_lookup(dist, prev_t, node)
                        + dist_lookup(dist, node, next_t)
                        - dist_lookup(dist, prev_t, next_t)
                    )
                    delta = insert_cost - remove_gain
                    checks += 1
                    feasible_checks += 1
                    checksum_delta += int(delta)
                    checksum_idx += (src_idx + 1) * 10000019 + (pos + 1) * 100003 + (dst_idx + 1) * 1009 + ins
                    if delta < best_delta:
                        best_delta = int(delta)
                        best_src = src_idx
                        best_pos = pos
                        best_dst = dst_idx
                        best_ins = ins

    return {
        "checks": checks,
        "feasible_checks": feasible_checks,
        "best_delta": best_delta,
        "best_src": best_src,
        "best_pos": best_pos,
        "best_dst": best_dst,
        "best_ins": best_ins,
        "checksum_delta": checksum_delta,
        "checksum_idx": checksum_idx,
    }


def c_array_1d(name: str, data: List[int], ctype: str = "int") -> str:
    lines = [f"static const {ctype} {name}[{len(data)}] = {{"]
    for i in range(0, len(data), 16):
        lines.append("    " + ", ".join(str(int(x)) for x in data[i:i+16]) + ("," if i + 16 < len(data) else ""))
    lines.append("};")
    return "\n".join(lines)


def c_array_2d(name: str, data: List[List[int]], ctype: str = "int") -> str:
    rows = len(data)
    cols = len(data[0]) if rows else 0
    lines = [f"static const {ctype} {name}[{rows}][{cols}] = {{"]
    for r, row in enumerate(data):
        suffix = "," if r + 1 < rows else ""
        lines.append("    {" + ", ".join(str(int(x)) for x in row) + "}" + suffix)
    lines.append("};")
    return "\n".join(lines)


def write_header(out_dir: Path, dataset: str, algorithm: str, compact, batch: int, capacity: int, relocate_cap: int, two_stats, rel_stats):
    routes = compact["routes"]
    max_routes = len(routes)
    max_route_len = max(len(r) for r in routes) if routes else 0
    route_lens = [len(r) for r in routes]
    route_loads = [sum(compact["demands"][c] for c in r) for r in routes]
    padded_routes = []
    for r in routes:
        padded_routes.append(r + [0] * (max_route_len - len(r)))

    n = len(compact["dist"])
    h = []
    h.append("#ifndef DUAL_LS_P76_WORKLOAD_H")
    h.append("#define DUAL_LS_P76_WORKLOAD_H")
    h.append("")
    h.append("// Auto-generated by 01_generate_p76_dual_workload.py")
    h.append(f"// Dataset: {dataset}")
    h.append(f"// Algorithm/source route set: {algorithm}")
    h.append("")
    h.append(f"#define DUAL_DATASET_NAME \"{dataset}\"")
    h.append(f"#define DUAL_SOURCE_ALGORITHM \"{algorithm}\"")
    h.append(f"#define DUAL_BATCH_SIZE {batch}")
    h.append(f"#define DUAL_NODE_COUNT {n}")
    h.append(f"#define DUAL_MAX_ROUTES {max_routes}")
    h.append(f"#define DUAL_MAX_ROUTE_LEN {max_route_len}")
    h.append(f"#define DUAL_DEPOT {compact['depot']}")
    h.append(f"#define DUAL_CAPACITY {capacity}")
    h.append(f"#define DUAL_RELOCATE_CAP {relocate_cap}")
    h.append(f"#define EXPECTED_TWOOPT_CHECKS_PER_ITEM {two_stats['checks']}")
    h.append(f"#define EXPECTED_RELOCATE_CHECKS_PER_ITEM {rel_stats['checks']}")
    h.append(f"#define EXPECTED_TWOOPT_CHECKS_TOTAL ({two_stats['checks']}LL * DUAL_BATCH_SIZE)")
    h.append(f"#define EXPECTED_RELOCATE_CHECKS_TOTAL ({rel_stats['checks']}LL * DUAL_BATCH_SIZE)")
    h.append(f"#define EXPECTED_TWOOPT_BEST_DELTA {two_stats['best_delta']}")
    h.append(f"#define EXPECTED_TWOOPT_BEST_ROUTE {two_stats['best_route']}")
    h.append(f"#define EXPECTED_TWOOPT_BEST_I {two_stats['best_i']}")
    h.append(f"#define EXPECTED_TWOOPT_BEST_J {two_stats['best_j']}")
    h.append(f"#define EXPECTED_RELOCATE_BEST_DELTA {rel_stats['best_delta']}")
    h.append(f"#define EXPECTED_RELOCATE_BEST_SRC {rel_stats['best_src']}")
    h.append(f"#define EXPECTED_RELOCATE_BEST_POS {rel_stats['best_pos']}")
    h.append(f"#define EXPECTED_RELOCATE_BEST_DST {rel_stats['best_dst']}")
    h.append(f"#define EXPECTED_RELOCATE_BEST_INS {rel_stats['best_ins']}")
    h.append(f"#define EXPECTED_TWOOPT_DELTA_CHECKSUM_PER_ITEM {two_stats['checksum_delta']}LL")
    h.append(f"#define EXPECTED_TWOOPT_INDEX_CHECKSUM_PER_ITEM {two_stats['checksum_idx']}LL")
    h.append(f"#define EXPECTED_RELOCATE_DELTA_CHECKSUM_PER_ITEM {rel_stats['checksum_delta']}LL")
    h.append(f"#define EXPECTED_RELOCATE_INDEX_CHECKSUM_PER_ITEM {rel_stats['checksum_idx']}LL")
    h.append("")
    h.append(c_array_1d("DUAL_ROUTE_LEN", route_lens, "int"))
    h.append("")
    h.append(c_array_1d("DUAL_ROUTE_LOAD", route_loads, "int"))
    h.append("")
    h.append(c_array_1d("DUAL_DEMAND", compact["demands"], "int"))
    h.append("")
    h.append(c_array_2d("DUAL_ROUTES", padded_routes, "int"))
    h.append("")
    h.append(c_array_2d("DUAL_DIST", compact["dist"], "int"))
    h.append("")
    h.append("#endif // DUAL_LS_P76_WORKLOAD_H")
    (out_dir / "dual_ls_p76_workload.h").write_text("\n".join(h), encoding="utf-8")


def write_cpp(out_dir: Path):
    cpp = r'''#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "dual_ls_p76_workload.h"

struct TwoOptResult {
    long long checks = 0;
    int best_delta = 0;
    int best_route = -1;
    int best_i = -1;
    int best_j = -1;
    long long checksum_delta = 0;
    long long checksum_index = 0;
};

struct RelocateResult {
    long long checks = 0;
    int best_delta = 0;
    int best_src = -1;
    int best_pos = -1;
    int best_dst = -1;
    int best_ins = -1;
    long long checksum_delta = 0;
    long long checksum_index = 0;
};

static inline int dmat(int a, int b) {
    return DUAL_DIST[a][b];
}

TwoOptResult twoopt_best_delta_scan_once() {
    TwoOptResult out;
    for (int r = 0; r < DUAL_MAX_ROUTES; ++r) {
        const int n_cust = DUAL_ROUTE_LEN[r];
        const int L = n_cust + 2;
        if (L < 5) continue;

        for (int i = 1; i <= L - 3; ++i) {
            for (int j = i + 1; j <= L - 2; ++j) {
                const int a = (i == 1) ? DUAL_DEPOT : DUAL_ROUTES[r][i - 2];
                const int b = DUAL_ROUTES[r][i - 1];
                const int c = DUAL_ROUTES[r][j - 1];
                const int d = (j == L - 2) ? DUAL_DEPOT : DUAL_ROUTES[r][j];

                const int old_cost = dmat(a, b) + dmat(c, d);
                const int new_cost = dmat(a, c) + dmat(b, d);
                const int delta = new_cost - old_cost;

                out.checks++;
                out.checksum_delta += static_cast<long long>(delta);
                out.checksum_index += static_cast<long long>(r + 1) * 1000003LL
                                    + static_cast<long long>(i) * 1009LL
                                    + static_cast<long long>(j);

                if (delta < out.best_delta) {
                    out.best_delta = delta;
                    out.best_route = r;
                    out.best_i = i;
                    out.best_j = j;
                }
            }
        }
    }
    return out;
}

RelocateResult relocate_best_delta_scan_once() {
    RelocateResult out;
    for (int src = 0; src < DUAL_MAX_ROUTES; ++src) {
        const int src_len = DUAL_ROUTE_LEN[src];
        if (src_len <= 0) continue;

        for (int pos = 0; pos < src_len; ++pos) {
            const int node = DUAL_ROUTES[src][pos];
            const int demand = DUAL_DEMAND[node];
            const int prev_s = (pos == 0) ? DUAL_DEPOT : DUAL_ROUTES[src][pos - 1];
            const int next_s = (pos == src_len - 1) ? DUAL_DEPOT : DUAL_ROUTES[src][pos + 1];
            const int remove_gain = dmat(prev_s, node) + dmat(node, next_s) - dmat(prev_s, next_s);

            for (int dst = 0; dst < DUAL_MAX_ROUTES; ++dst) {
                if (src == dst) continue;
                if (DUAL_ROUTE_LOAD[dst] + demand > DUAL_CAPACITY) continue;

                const int dst_len = DUAL_ROUTE_LEN[dst];
                for (int ins = 0; ins <= dst_len; ++ins) {
                    if (out.checks >= DUAL_RELOCATE_CAP) return out;

                    const int prev_t = (ins == 0) ? DUAL_DEPOT : DUAL_ROUTES[dst][ins - 1];
                    const int next_t = (ins == dst_len) ? DUAL_DEPOT : DUAL_ROUTES[dst][ins];
                    const int insert_cost = dmat(prev_t, node) + dmat(node, next_t) - dmat(prev_t, next_t);
                    const int delta = insert_cost - remove_gain;

                    out.checks++;
                    out.checksum_delta += static_cast<long long>(delta);
                    out.checksum_index += static_cast<long long>(src + 1) * 10000019LL
                                        + static_cast<long long>(pos + 1) * 100003LL
                                        + static_cast<long long>(dst + 1) * 1009LL
                                        + static_cast<long long>(ins);

                    if (delta < out.best_delta) {
                        out.best_delta = delta;
                        out.best_src = src;
                        out.best_pos = pos;
                        out.best_dst = dst;
                        out.best_ins = ins;
                    }
                }
            }
        }
    }
    return out;
}

struct TimingStats {
    double min_us = 0.0;
    double median_us = 0.0;
    double mean_us = 0.0;
    double max_us = 0.0;
};

TimingStats summarize(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    TimingStats s;
    s.min_us = v.front();
    s.max_us = v.back();
    s.median_us = v[v.size() / 2];
    s.mean_us = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
    return s;
}

int main() {
    constexpr int WARMUP = 10;
    constexpr int RUNS = 101;

    std::cout << "============================================================\n";
    std::cout << "CPU Dual Local-Search Baseline\n";
    std::cout << "Dataset     : " << DUAL_DATASET_NAME << "\n";
    std::cout << "Source alg. : " << DUAL_SOURCE_ALGORITHM << "\n";
    std::cout << "Batch       : " << DUAL_BATCH_SIZE << "\n";
    std::cout << "Nodes       : " << DUAL_NODE_COUNT << "\n";
    std::cout << "Routes      : " << DUAL_MAX_ROUTES << "\n";
    std::cout << "Max route   : " << DUAL_MAX_ROUTE_LEN << "\n";
    std::cout << "Reloc cap   : " << DUAL_RELOCATE_CAP << "\n";
    std::cout << "============================================================\n";

    const TwoOptResult t0 = twoopt_best_delta_scan_once();
    const RelocateResult r0 = relocate_best_delta_scan_once();

    bool pass = true;
    pass = pass && (t0.checks == EXPECTED_TWOOPT_CHECKS_PER_ITEM);
    pass = pass && (t0.best_delta == EXPECTED_TWOOPT_BEST_DELTA);
    pass = pass && (t0.best_route == EXPECTED_TWOOPT_BEST_ROUTE);
    pass = pass && (t0.best_i == EXPECTED_TWOOPT_BEST_I);
    pass = pass && (t0.best_j == EXPECTED_TWOOPT_BEST_J);
    pass = pass && (t0.checksum_delta == EXPECTED_TWOOPT_DELTA_CHECKSUM_PER_ITEM);
    pass = pass && (t0.checksum_index == EXPECTED_TWOOPT_INDEX_CHECKSUM_PER_ITEM);

    pass = pass && (r0.checks == EXPECTED_RELOCATE_CHECKS_PER_ITEM);
    pass = pass && (r0.best_delta == EXPECTED_RELOCATE_BEST_DELTA);
    pass = pass && (r0.best_src == EXPECTED_RELOCATE_BEST_SRC);
    pass = pass && (r0.best_pos == EXPECTED_RELOCATE_BEST_POS);
    pass = pass && (r0.best_dst == EXPECTED_RELOCATE_BEST_DST);
    pass = pass && (r0.best_ins == EXPECTED_RELOCATE_BEST_INS);
    pass = pass && (r0.checksum_delta == EXPECTED_RELOCATE_DELTA_CHECKSUM_PER_ITEM);
    pass = pass && (r0.checksum_index == EXPECTED_RELOCATE_INDEX_CHECKSUM_PER_ITEM);

    std::cout << "[GOLDEN] twoopt checks=" << t0.checks
              << " best_delta=" << t0.best_delta
              << " route=" << t0.best_route
              << " i=" << t0.best_i
              << " j=" << t0.best_j
              << " checksum_delta=" << t0.checksum_delta
              << " checksum_index=" << t0.checksum_index << "\n";

    std::cout << "[GOLDEN] relocate checks=" << r0.checks
              << " best_delta=" << r0.best_delta
              << " src=" << r0.best_src
              << " pos=" << r0.best_pos
              << " dst=" << r0.best_dst
              << " ins=" << r0.best_ins
              << " checksum_delta=" << r0.checksum_delta
              << " checksum_index=" << r0.checksum_index << "\n";

    if (!pass) {
        std::cerr << "[FAIL] Golden mismatch. Header and C++ logic differ.\n";
        return 2;
    }
    std::cout << "[PASS] Golden consistency check\n";

    volatile long long sink = 0;
    for (int w = 0; w < WARMUP; ++w) {
        for (int b = 0; b < DUAL_BATCH_SIZE; ++b) {
            auto t = twoopt_best_delta_scan_once();
            auto r = relocate_best_delta_scan_once();
            sink += t.checks + r.checks + t.best_delta + r.best_delta;
        }
    }

    std::vector<double> two_us, rel_us, total_us;
    two_us.reserve(RUNS);
    rel_us.reserve(RUNS);
    total_us.reserve(RUNS);

    for (int run = 0; run < RUNS; ++run) {
        auto s2 = std::chrono::high_resolution_clock::now();
        long long two_checks = 0;
        long long two_checksum = 0;
        for (int b = 0; b < DUAL_BATCH_SIZE; ++b) {
            auto t = twoopt_best_delta_scan_once();
            two_checks += t.checks;
            two_checksum += t.checksum_delta + t.checksum_index + t.best_delta;
        }
        auto e2 = std::chrono::high_resolution_clock::now();

        auto sr = std::chrono::high_resolution_clock::now();
        long long rel_checks = 0;
        long long rel_checksum = 0;
        for (int b = 0; b < DUAL_BATCH_SIZE; ++b) {
            auto r = relocate_best_delta_scan_once();
            rel_checks += r.checks;
            rel_checksum += r.checksum_delta + r.checksum_index + r.best_delta;
        }
        auto er = std::chrono::high_resolution_clock::now();

        sink += two_checks + rel_checks + two_checksum + rel_checksum;

        double t_us = std::chrono::duration<double, std::micro>(e2 - s2).count();
        double r_us = std::chrono::duration<double, std::micro>(er - sr).count();
        two_us.push_back(t_us);
        rel_us.push_back(r_us);
        total_us.push_back(t_us + r_us);
    }

    const auto ts = summarize(two_us);
    const auto rs = summarize(rel_us);
    const auto cs = summarize(total_us);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "============================================================\n";
    std::cout << "CPU_TIMING_US,kernel,min,median,mean,max,total_candidates,ns_per_candidate_median\n";
    const long long two_total = static_cast<long long>(EXPECTED_TWOOPT_CHECKS_PER_ITEM) * DUAL_BATCH_SIZE;
    const long long rel_total = static_cast<long long>(EXPECTED_RELOCATE_CHECKS_PER_ITEM) * DUAL_BATCH_SIZE;
    std::cout << "CPU_TIMING_US,twoopt," << ts.min_us << "," << ts.median_us << "," << ts.mean_us << "," << ts.max_us
              << "," << two_total << "," << (ts.median_us * 1000.0 / std::max(1LL, two_total)) << "\n";
    std::cout << "CPU_TIMING_US,relocate," << rs.min_us << "," << rs.median_us << "," << rs.mean_us << "," << rs.max_us
              << "," << rel_total << "," << (rs.median_us * 1000.0 / std::max(1LL, rel_total)) << "\n";
    std::cout << "CPU_TIMING_US,total_sequential," << cs.min_us << "," << cs.median_us << "," << cs.mean_us << "," << cs.max_us
              << "," << (two_total + rel_total) << "," << (cs.median_us * 1000.0 / std::max(1LL, two_total + rel_total)) << "\n";
    std::cout << "============================================================\n";
    std::cout << "sink=" << sink << "\n";
    return 0;
}
'''
    (out_dir / "02_cpu_dual_ls_benchmark.cpp").write_text(cpp, encoding="utf-8")


def write_bat(out_dir: Path):
    bat = r'''@echo off
setlocal
cd /d "%~dp0"

where cl >nul 2>nul
if errorlevel 1 (
    echo [INFO] cl bulunamadi. Visual Studio Developer Command Prompt icinde calistirin.
    echo Alternatif: g++ varsa asagidaki komutu deneyebilirsiniz:
    echo g++ -O3 -std=c++17 02_cpu_dual_ls_benchmark.cpp -o cpu_dual_ls_benchmark.exe
    pause
    exit /b 1
)

cl /O2 /std:c++17 /EHsc 02_cpu_dual_ls_benchmark.cpp /Fe:cpu_dual_ls_benchmark.exe
if errorlevel 1 (
    echo [FAIL] Derleme basarisiz.
    pause
    exit /b 1
)

cpu_dual_ls_benchmark.exe > cpu_dual_ls_p76_baseline_output.txt

type cpu_dual_ls_p76_baseline_output.txt

echo.
echo [OK] Sonuc dosyasi: %CD%\cpu_dual_ls_p76_baseline_output.txt
pause
'''
    (out_dir / "compile_and_run_cpu_baseline.bat").write_text(bat, encoding="utf-8")


def write_summary(out_dir: Path, root: Path, best_json_path: Path, dataset: str, alg: str, selected_key: str, routes, compact, batch, capacity, relocate_cap, two_stats, rel_stats):
    lines = []
    lines.append("P76 Dual Local-Search Workload Summary")
    lines.append("=" * 72)
    lines.append(f"Project root       : {root}")
    lines.append(f"Best routes JSON   : {best_json_path}")
    lines.append(f"Dataset            : {dataset}")
    lines.append(f"Selected algorithm : {alg}")
    lines.append(f"Selected key       : {selected_key}")
    lines.append(f"Batch size         : {batch}")
    lines.append(f"Capacity           : {capacity}")
    lines.append(f"Relocate cap/item  : {relocate_cap}")
    lines.append(f"Compact node count : {len(compact['dist'])}")
    lines.append(f"Depot compact id   : {compact['depot']}")
    lines.append(f"Route count        : {len(routes)}")
    lines.append("Route lengths      : " + ", ".join(str(len(r)) for r in routes))
    lines.append("Route loads        : " + ", ".join(str(sum(compact['demands'][c] for c in cr)) for cr in compact['routes']))
    lines.append("")
    lines.append("2-opt best-delta workload")
    lines.append(f"  checks/item       : {two_stats['checks']}")
    lines.append(f"  checks/total      : {two_stats['checks'] * batch}")
    lines.append(f"  best_delta        : {two_stats['best_delta']}")
    lines.append(f"  best route,i,j    : {two_stats['best_route']}, {two_stats['best_i']}, {two_stats['best_j']}")
    lines.append(f"  delta checksum    : {two_stats['checksum_delta']}")
    lines.append(f"  index checksum    : {two_stats['checksum_idx']}")
    lines.append("")
    lines.append("Relocate best-delta workload")
    lines.append(f"  checks/item       : {rel_stats['checks']}")
    lines.append(f"  checks/total      : {rel_stats['checks'] * batch}")
    lines.append(f"  best_delta        : {rel_stats['best_delta']}")
    lines.append(f"  best src,pos,dst,ins: {rel_stats['best_src']}, {rel_stats['best_pos']}, {rel_stats['best_dst']}, {rel_stats['best_ins']}")
    lines.append(f"  delta checksum    : {rel_stats['checksum_delta']}")
    lines.append(f"  index checksum    : {rel_stats['checksum_idx']}")
    lines.append("")
    lines.append("Next:")
    lines.append("  1) Visual Studio Developer Command Prompt acin.")
    lines.append("  2) compile_and_run_cpu_baseline.bat calistirin.")
    lines.append("  3) cpu_dual_ls_p76_baseline_output.txt ciktisini paylasin.")
    (out_dir / "workload_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="WOA & SSA on FPGA proje kok klasoru")
    ap.add_argument("--dataset", type=str, default=DATASET_DEFAULT)
    ap.add_argument("--algorithm", type=str, default="auto", help="auto, M-WOA veya M-SSA")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--relocate_cap", type=int, default=8000)
    ap.add_argument("--out", type=str, default=r"C:\FPGA_ZedBoard\17_DUAL_LS_2OPT_RELOCATE_P76")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    vrp_path = root / "datasets" / f"{args.dataset}.vrp"
    if not vrp_path.exists():
        hits = list((root / "datasets").glob(f"{args.dataset}*.vrp")) if (root / "datasets").exists() else []
        if hits:
            vrp_path = hits[0]
        else:
            raise FileNotFoundError(f"VRP bulunamadı: {vrp_path}")

    best_json_path = find_best_routes_json(root)
    best_routes = json.loads(best_json_path.read_text(encoding="utf-8"))
    gap, cost, alg, selected_key, routes, raw = choose_routes(best_routes, args.dataset, args.algorithm)

    vrp = parse_vrp(vrp_path)
    compact = build_compact(vrp, routes)

    two_stats = twoopt_scan_all_routes(compact, args.relocate_cap)
    rel_stats = relocate_scan(compact, int(vrp["capacity"]), args.relocate_cap)

    write_header(out_dir, args.dataset, alg, compact, args.batch, int(vrp["capacity"]), args.relocate_cap, two_stats, rel_stats)
    write_cpp(out_dir)
    write_bat(out_dir)
    write_summary(out_dir, root, best_json_path, args.dataset, alg, selected_key, routes, compact, args.batch, int(vrp["capacity"]), args.relocate_cap, two_stats, rel_stats)

    print("\n[OK] P76 dual workload hazirlandi")
    print(f"Out: {out_dir}")
    print(f"Selected: dataset={args.dataset}, algorithm={alg}, gap={gap}, cost={cost}")
    print(f"Routes: {len(routes)}, compact nodes: {len(compact['dist'])}")
    print(f"2opt checks/item={two_stats['checks']}, total={two_stats['checks'] * args.batch}, best_delta={two_stats['best_delta']}")
    print(f"relocate checks/item={rel_stats['checks']}, total={rel_stats['checks'] * args.batch}, best_delta={rel_stats['best_delta']}")
    print("\nSonraki adim:")
    print(f"  cd /d {out_dir}")
    print("  compile_and_run_cpu_baseline.bat")


if __name__ == "__main__":
    main()

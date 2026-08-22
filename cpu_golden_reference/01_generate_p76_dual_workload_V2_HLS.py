# -*- coding: utf-8 -*-
r"""
01_generate_p76_dual_workload_V2_HLS.py

P76 dual local-search workload headerini HLS icin genisletir.
V1 ile ayni kernel-level workload'u kullanir; ek olarak candidate list arrays uretir:
- DUAL_TWOOPT_R/I/J
- DUAL_RELOC_SRC/POS/DST/INS

Bu candidate listeleri batch boyunca tekrar kullanilir. Batch candidate array olarak saklanmaz.
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
        raise RuntimeError(f"CAPACITY okunamadi: {vrp_path}")
    if not coords or not demands:
        raise RuntimeError(f"VRP coords/demands okunamadi: {vrp_path}")
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
    raise FileNotFoundError("best_routes_final.json veya best_routes_profiling.json bulunamadi.")


def normalize_routes(raw_routes: Any) -> List[List[int]]:
    routes = []
    if raw_routes is None:
        return routes
    for r in raw_routes:
        if isinstance(r, (list, tuple)):
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
    if not rows:
        raise RuntimeError(f"{dataset} icin uygun route kaydi bulunamadi.")
    rows.sort(key=lambda x: (x[0], x[1], x[2]))
    return rows[0]


def build_compact(vrp, routes: List[List[int]]):
    depot_orig = int(vrp["depot"])
    all_nodes = sorted({depot_orig} | {int(c) for r in routes for c in r})
    compact_of = {orig: idx for idx, orig in enumerate(all_nodes)}
    depot = compact_of[depot_orig]
    n = len(all_nodes)

    coords = vrp["coords"]
    dist = [[0 for _ in range(n)] for _ in range(n)]
    for ia, a in enumerate(all_nodes):
        ax, ay = coords[a]
        for ib, b in enumerate(all_nodes):
            bx, by = coords[b]
            dist[ia][ib] = int(round(math.hypot(ax - bx, ay - by)))

    demands = [0 for _ in range(n)]
    for orig in all_nodes:
        demands[compact_of[orig]] = int(vrp["demands"].get(orig, 0))

    compact_routes = [[compact_of[int(c)] for c in r] for r in routes]
    return {"depot": depot, "dist": dist, "demands": demands, "routes": compact_routes}


def d(dist, a, b):
    return int(dist[int(a)][int(b)])


def twoopt_candidates_and_stats(compact):
    depot = compact["depot"]
    dist = compact["dist"]
    routes = compact["routes"]
    cand_r, cand_i, cand_j = [], [], []
    best_delta, best_r, best_i, best_j = 0, -1, -1, -1
    checksum_delta, checksum_idx = 0, 0

    for r_idx, route in enumerate(routes):
        n = len(route)
        L = n + 2
        if L < 5:
            continue
        # Matches V1: i in [1, L-3], j in [i+1, L-2]
        for i in range(1, L - 2):
            for j in range(i + 1, L - 1):
                a = depot if i == 1 else route[i - 2]
                b = route[i - 1]
                c = route[j - 1]
                nxt = depot if j == n else route[j]
                delta = (d(dist, a, c) + d(dist, b, nxt)) - (d(dist, a, b) + d(dist, c, nxt))
                cand_r.append(r_idx)
                cand_i.append(i)
                cand_j.append(j)
                checksum_delta += int(delta)
                checksum_idx += (r_idx + 1) * 1000003 + i * 1009 + j
                if delta < best_delta:
                    best_delta, best_r, best_i, best_j = int(delta), r_idx, i, j

    return {
        "r": cand_r, "i": cand_i, "j": cand_j,
        "checks": len(cand_r), "best_delta": best_delta,
        "best_route": best_r, "best_i": best_i, "best_j": best_j,
        "checksum_delta": checksum_delta, "checksum_idx": checksum_idx,
    }


def relocate_candidates_and_stats(compact, capacity: int, relocate_cap: int):
    depot = compact["depot"]
    dist = compact["dist"]
    demands = compact["demands"]
    routes = compact["routes"]
    loads = [sum(demands[c] for c in r) for r in routes]

    srcs, poss, dsts, inss = [], [], [], []
    best_delta, best_src, best_pos, best_dst, best_ins = 0, -1, -1, -1, -1
    checksum_delta, checksum_idx = 0, 0

    for src_idx, src_route in enumerate(routes):
        if not src_route:
            continue
        for pos, node in enumerate(src_route):
            demand = demands[node]
            prev_s = depot if pos == 0 else src_route[pos - 1]
            next_s = depot if pos == len(src_route) - 1 else src_route[pos + 1]
            remove_gain = d(dist, prev_s, node) + d(dist, node, next_s) - d(dist, prev_s, next_s)

            for dst_idx, dst_route in enumerate(routes):
                if src_idx == dst_idx:
                    continue
                if loads[dst_idx] + demand > capacity:
                    continue
                for ins in range(0, len(dst_route) + 1):
                    if len(srcs) >= relocate_cap:
                        return {
                            "src": srcs, "pos": poss, "dst": dsts, "ins": inss,
                            "checks": len(srcs), "best_delta": best_delta,
                            "best_src": best_src, "best_pos": best_pos,
                            "best_dst": best_dst, "best_ins": best_ins,
                            "checksum_delta": checksum_delta, "checksum_idx": checksum_idx,
                        }
                    prev_t = depot if ins == 0 else dst_route[ins - 1]
                    next_t = depot if ins == len(dst_route) else dst_route[ins]
                    insert_cost = d(dist, prev_t, node) + d(dist, node, next_t) - d(dist, prev_t, next_t)
                    delta = insert_cost - remove_gain
                    srcs.append(src_idx); poss.append(pos); dsts.append(dst_idx); inss.append(ins)
                    checksum_delta += int(delta)
                    checksum_idx += (src_idx + 1) * 10000019 + (pos + 1) * 100003 + (dst_idx + 1) * 1009 + ins
                    if delta < best_delta:
                        best_delta, best_src, best_pos, best_dst, best_ins = int(delta), src_idx, pos, dst_idx, ins

    return {
        "src": srcs, "pos": poss, "dst": dsts, "ins": inss,
        "checks": len(srcs), "best_delta": best_delta,
        "best_src": best_src, "best_pos": best_pos,
        "best_dst": best_dst, "best_ins": best_ins,
        "checksum_delta": checksum_delta, "checksum_idx": checksum_idx,
    }


def c_array_1d(name: str, data: List[int], ctype: str = "int") -> str:
    lines = [f"static const {ctype} {name}[{len(data)}] = {{"]
    for i in range(0, len(data), 20):
        chunk = ", ".join(str(int(x)) for x in data[i:i+20])
        suffix = "," if i + 20 < len(data) else ""
        lines.append("    " + chunk + suffix)
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


def write_header(out_dir: Path, dataset: str, algorithm: str, compact, batch: int, capacity: int, relocate_cap: int, two, rel):
    routes = compact["routes"]
    max_routes = len(routes)
    max_route_len = max(len(r) for r in routes) if routes else 0
    route_lens = [len(r) for r in routes]
    route_loads = [sum(compact["demands"][c] for c in r) for r in routes]
    padded_routes = [r + [0] * (max_route_len - len(r)) for r in routes]
    n = len(compact["dist"])

    h = []
    h.append("#ifndef DUAL_LS_P76_WORKLOAD_H")
    h.append("#define DUAL_LS_P76_WORKLOAD_H")
    h.append("")
    h.append("// Auto-generated by 01_generate_p76_dual_workload_V2_HLS.py")
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
    h.append(f"#define EXPECTED_TWOOPT_CHECKS_PER_ITEM {two['checks']}")
    h.append(f"#define EXPECTED_RELOCATE_CHECKS_PER_ITEM {rel['checks']}")
    h.append(f"#define EXPECTED_TWOOPT_CHECKS_TOTAL ({two['checks']}LL * DUAL_BATCH_SIZE)")
    h.append(f"#define EXPECTED_RELOCATE_CHECKS_TOTAL ({rel['checks']}LL * DUAL_BATCH_SIZE)")
    h.append(f"#define EXPECTED_TWOOPT_BEST_DELTA {two['best_delta']}")
    h.append(f"#define EXPECTED_TWOOPT_BEST_ROUTE {two['best_route']}")
    h.append(f"#define EXPECTED_TWOOPT_BEST_I {two['best_i']}")
    h.append(f"#define EXPECTED_TWOOPT_BEST_J {two['best_j']}")
    h.append(f"#define EXPECTED_RELOCATE_BEST_DELTA {rel['best_delta']}")
    h.append(f"#define EXPECTED_RELOCATE_BEST_SRC {rel['best_src']}")
    h.append(f"#define EXPECTED_RELOCATE_BEST_POS {rel['best_pos']}")
    h.append(f"#define EXPECTED_RELOCATE_BEST_DST {rel['best_dst']}")
    h.append(f"#define EXPECTED_RELOCATE_BEST_INS {rel['best_ins']}")
    h.append(f"#define EXPECTED_TWOOPT_DELTA_CHECKSUM_PER_ITEM {two['checksum_delta']}LL")
    h.append(f"#define EXPECTED_TWOOPT_INDEX_CHECKSUM_PER_ITEM {two['checksum_idx']}LL")
    h.append(f"#define EXPECTED_RELOCATE_DELTA_CHECKSUM_PER_ITEM {rel['checksum_delta']}LL")
    h.append(f"#define EXPECTED_RELOCATE_INDEX_CHECKSUM_PER_ITEM {rel['checksum_idx']}LL")
    h.append("")
    h.append(c_array_1d("DUAL_ROUTE_LEN", route_lens, "int")); h.append("")
    h.append(c_array_1d("DUAL_ROUTE_LOAD", route_loads, "int")); h.append("")
    h.append(c_array_1d("DUAL_DEMAND", compact["demands"], "int")); h.append("")
    h.append(c_array_2d("DUAL_ROUTES", padded_routes, "int")); h.append("")
    h.append(c_array_2d("DUAL_DIST", compact["dist"], "int")); h.append("")
    h.append("// Per-item candidate lists. Batch is processed by looping over these lists.")
    h.append(c_array_1d("DUAL_TWOOPT_R", two["r"], "short")); h.append("")
    h.append(c_array_1d("DUAL_TWOOPT_I", two["i"], "short")); h.append("")
    h.append(c_array_1d("DUAL_TWOOPT_J", two["j"], "short")); h.append("")
    h.append(c_array_1d("DUAL_RELOC_SRC", rel["src"], "short")); h.append("")
    h.append(c_array_1d("DUAL_RELOC_POS", rel["pos"], "short")); h.append("")
    h.append(c_array_1d("DUAL_RELOC_DST", rel["dst"], "short")); h.append("")
    h.append(c_array_1d("DUAL_RELOC_INS", rel["ins"], "short")); h.append("")
    h.append("#endif // DUAL_LS_P76_WORKLOAD_H")
    (out_dir / "dual_ls_p76_workload.h").write_text("\n".join(h), encoding="utf-8")


def write_summary(out_dir: Path, root: Path, best_json: Path, dataset: str, alg: str, selected_key: str, routes, compact, batch, capacity, relocate_cap, two, rel):
    lines = []
    lines.append("P76 Dual Local-Search HLS Workload Summary V2")
    lines.append("=" * 72)
    lines.append(f"Project root       : {root}")
    lines.append(f"Best routes JSON   : {best_json}")
    lines.append(f"Dataset            : {dataset}")
    lines.append(f"Selected algorithm : {alg}")
    lines.append(f"Selected key       : {selected_key}")
    lines.append(f"Batch size         : {batch}")
    lines.append(f"Capacity           : {capacity}")
    lines.append(f"Relocate cap/item  : {relocate_cap}")
    lines.append(f"Compact node count : {len(compact['dist'])}")
    lines.append(f"Route count        : {len(routes)}")
    lines.append("Route lengths      : " + ", ".join(str(len(r)) for r in routes))
    lines.append("Route loads        : " + ", ".join(str(sum(compact['demands'][c] for c in cr)) for cr in compact['routes']))
    lines.append("")
    lines.append(f"2-opt candidates/item     : {two['checks']}")
    lines.append(f"2-opt candidates/total    : {two['checks'] * batch}")
    lines.append(f"2-opt best_delta          : {two['best_delta']}")
    lines.append(f"2-opt checksum_delta/item : {two['checksum_delta']}")
    lines.append(f"2-opt checksum_index/item : {two['checksum_idx']}")
    lines.append("")
    lines.append(f"Relocate candidates/item     : {rel['checks']}")
    lines.append(f"Relocate candidates/total    : {rel['checks'] * batch}")
    lines.append(f"Relocate best_delta          : {rel['best_delta']}")
    lines.append(f"Relocate checksum_delta/item : {rel['checksum_delta']}")
    lines.append(f"Relocate checksum_index/item : {rel['checksum_idx']}")
    lines.append("")
    lines.append("This V2 header stores only per-item candidate lists, not full batch candidate arrays.")
    (out_dir / "workload_summary_V2_HLS.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=r"C:\Users\Cem Deniz KUMRAL\Desktop\Aktif Çalışmalar\WOA & SSA on FPGA")
    ap.add_argument("--dataset", type=str, default=DATASET_DEFAULT)
    ap.add_argument("--algorithm", type=str, default="auto")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--relocate_cap", type=int, default=8000)
    ap.add_argument("--out", type=str, default=".")
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
            raise FileNotFoundError(f"VRP bulunamadi: {vrp_path}")

    best_json = find_best_routes_json(root)
    best_routes = json.loads(best_json.read_text(encoding="utf-8"))
    gap, cost, alg, selected_key, routes, _raw = choose_routes(best_routes, args.dataset, args.algorithm)
    vrp = parse_vrp(vrp_path)
    compact = build_compact(vrp, routes)
    two = twoopt_candidates_and_stats(compact)
    rel = relocate_candidates_and_stats(compact, int(vrp["capacity"]), args.relocate_cap)

    write_header(out_dir, args.dataset, alg, compact, args.batch, int(vrp["capacity"]), args.relocate_cap, two, rel)
    write_summary(out_dir, root, best_json, args.dataset, alg, selected_key, routes, compact, args.batch, int(vrp["capacity"]), args.relocate_cap, two, rel)

    print("\n[OK] P76 dual HLS workload V2 hazirlandi")
    print(f"Out: {out_dir}")
    print(f"Selected: dataset={args.dataset}, algorithm={alg}, gap={gap}, cost={cost}")
    print(f"Routes: {len(routes)}, compact nodes: {len(compact['dist'])}")
    print(f"2opt checks/item={two['checks']}, total={two['checks'] * args.batch}, best_delta={two['best_delta']}")
    print(f"relocate checks/item={rel['checks']}, total={rel['checks'] * args.batch}, best_delta={rel['best_delta']}")
    print(f"Header: {out_dir / 'dual_ls_p76_workload.h'}")
    print("\nSonraki adim:")
    print(f"  cd /d {out_dir}")
    print("  run_hls_dual_o4r4.bat")


if __name__ == "__main__":
    main()

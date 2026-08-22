# -*- coding: utf-8 -*-
r"""
08_cpu_final_results_builder.py

FINAL_RESULTS_CPU içindeki final CPU sonuçlarından makaleye hazır tablo ve grafikleri üretir.
Ayrıca önceki validation / profiling / pilot aşamalarından oluşan kanıt dosyalarını
FINAL_RESULTS_CPU/10_article_exports/preliminary_evidence altına kopyalar.

Çalıştırma:
    python 08_cpu_final_results_builder.py

Girdi:
    FINAL_RESULTS_CPU/01_raw/final_run_results.csv
    FINAL_RESULTS_CPU/03_convergence/final_convergence_curves.csv
    FINAL_RESULTS_CPU/04_best_routes/best_routes_final.json
    datasets/*.vrp

Çıktı:
    FINAL_RESULTS_CPU/05_figures/*.png, *.pdf
    FINAL_RESULTS_CPU/06_tables/*.csv, *.xlsx
    FINAL_RESULTS_CPU/10_article_exports/*
"""

from pathlib import Path
import json
import math
import shutil
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FIG_DPI = 300
ALGORITHM_ORDER = ["WOA", "SSA", "M-WOA", "M-SSA"]
MEMETIC_ALGORITHMS = ["M-WOA", "M-SSA"]
DATASET_ORDER = ["A-n32-k5", "A-n37-k6", "A-n45-k6", "A-n54-k7", "A-n60-k9", "A-n80-k10", "B-n50-k7", "P-n76-k4"]


def set_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": FIG_DPI,
        "font.family": "DejaVu Serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.6,
        "ytick.labelsize": 8.6,
        "legend.fontsize": 8.4,
        "axes.grid": True,
        "grid.alpha": 0.20,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    fig.savefig(png, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[OK] {png}")
    print(f"[OK] {pdf}")


def make_dirs(root: Path):
    final_root = root / "FINAL_RESULTS_CPU"
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


def read_inputs(dirs):
    run_path = dirs["raw"] / "final_run_results.csv"
    curve_path = dirs["convergence"] / "final_convergence_curves.csv"
    routes_path = dirs["best_routes"] / "best_routes_final.json"

    if not run_path.exists():
        raise FileNotFoundError(f"Final run dosyası yok: {run_path}")

    run_df = pd.read_csv(run_path)
    curves = pd.read_csv(curve_path) if curve_path.exists() else pd.DataFrame()
    best_routes = json.loads(routes_path.read_text(encoding="utf-8")) if routes_path.exists() else {}

    return run_df, curves, best_routes


def ordered(df):
    df = df.copy()
    if "algorithm" in df.columns:
        df["algorithm"] = pd.Categorical(df["algorithm"], categories=ALGORITHM_ORDER, ordered=True)
    if "dataset" in df.columns:
        df["dataset"] = pd.Categorical(df["dataset"], categories=DATASET_ORDER, ordered=True)
    return df


def compute_tables(run_df: pd.DataFrame):
    run_df = ordered(run_df)

    summary = (
        run_df.groupby(["dataset", "algorithm", "base_algorithm", "ls_mode"], observed=False)
        .agg(
            runs=("run", "count"),
            all_feasible=("feasible_solution", "all"),
            best_cost=("cost_without_vehicle_penalty", "min"),
            mean_cost=("cost_without_vehicle_penalty", "mean"),
            median_cost=("cost_without_vehicle_penalty", "median"),
            std_cost=("cost_without_vehicle_penalty", "std"),
            best_gap_percent=("gap_percent_without_vehicle_penalty", "min"),
            mean_gap_percent=("gap_percent_without_vehicle_penalty", "mean"),
            median_gap_percent=("gap_percent_without_vehicle_penalty", "median"),
            std_gap_percent=("gap_percent_without_vehicle_penalty", "std"),
            worst_gap_percent=("gap_percent_without_vehicle_penalty", "max"),
            mean_runtime_sec=("runtime_sec", "mean"),
            std_runtime_sec=("runtime_sec", "std"),
            median_runtime_sec=("runtime_sec", "median"),
            mean_time_to_best_sec=("time_to_best_sec", "mean"),
            std_time_to_best_sec=("time_to_best_sec", "std"),
            mean_iter_to_best=("iter_to_best", "mean"),
            median_iter_to_best=("iter_to_best", "median"),
            mean_eval_to_best=("eval_to_best", "mean"),
            max_capacity_violation_count=("capacity_violation_count", "max"),
            max_vehicle_violation_count=("vehicle_violation_count", "max"),
            max_missing_count=("missing_count", "max"),
            max_duplicate_count=("duplicate_count", "max"),
        )
        .reset_index()
    )
    summary = ordered(summary).sort_values(["dataset", "algorithm"])

    overall = (
        summary.groupby("algorithm", observed=False)
        .agg(
            avg_best_gap_percent=("best_gap_percent", "mean"),
            avg_mean_gap_percent=("mean_gap_percent", "mean"),
            avg_runtime_sec=("mean_runtime_sec", "mean"),
            avg_time_to_best_sec=("mean_time_to_best_sec", "mean"),
            all_feasible=("all_feasible", "all"),
        )
        .reset_index()
    )
    overall = ordered(overall).sort_values("algorithm")

    dataset_best_rows = []
    for ds, g in summary.groupby("dataset", observed=False):
        if len(g) == 0:
            continue
        best_all = g.loc[g["best_gap_percent"].idxmin()]
        mem = g[g["algorithm"].astype(str).isin(MEMETIC_ALGORITHMS)]
        best_mem = mem.loc[mem["best_gap_percent"].idxmin()]
        dataset_best_rows.append({
            "dataset": str(ds),
            "best_algorithm_overall": str(best_all["algorithm"]),
            "best_cost_overall": best_all["best_cost"],
            "best_gap_overall": best_all["best_gap_percent"],
            "best_memetic_algorithm": str(best_mem["algorithm"]),
            "best_cost_memetic": best_mem["best_cost"],
            "best_gap_memetic": best_mem["best_gap_percent"],
        })
    dataset_best = pd.DataFrame(dataset_best_rows)

    article_table = summary[[
        "dataset", "algorithm", "runs", "all_feasible",
        "best_cost", "best_gap_percent", "mean_gap_percent", "std_gap_percent",
        "mean_runtime_sec", "std_runtime_sec", "mean_time_to_best_sec", "mean_iter_to_best"
    ]].copy()

    return summary, overall, dataset_best, article_table


def export_tables(dirs, run_df, curves, summary, overall, dataset_best, article_table):
    summary.to_csv(dirs["summaries"] / "final_summary.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(dirs["summaries"] / "final_algorithm_overall.csv", index=False, encoding="utf-8-sig")
    dataset_best.to_csv(dirs["summaries"] / "final_dataset_best.csv", index=False, encoding="utf-8-sig")
    article_table.to_csv(dirs["tables"] / "article_main_cpu_results_table.csv", index=False, encoding="utf-8-sig")

    xlsx = dirs["tables"] / "final_article_tables.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        article_table.to_excel(writer, sheet_name="Main_CPU_Table", index=False)
        overall.to_excel(writer, sheet_name="Algorithm_Overall", index=False)
        dataset_best.to_excel(writer, sheet_name="Dataset_Best", index=False)
        summary.to_excel(writer, sheet_name="Detailed_Summary", index=False)
        run_df.to_excel(writer, sheet_name="Run_Level", index=False)
        if len(curves) > 0 and len(curves) <= 1_000_000:
            curves.to_excel(writer, sheet_name="Convergence", index=False)
    print(f"[OK] {xlsx}")


def fig01_algorithm_gap(overall, out_dir):
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    overall = ordered(overall)
    x = np.arange(len(overall))
    w = 0.36

    ax.bar(x - w/2, overall["avg_best_gap_percent"], width=w, label="Average best gap")
    ax.bar(x + w/2, overall["avg_mean_gap_percent"], width=w, label="Average mean gap")

    ax.set_xticks(x)
    ax.set_xticklabels(overall["algorithm"].astype(str).tolist())
    ax.set_ylabel("Gap to known best (%)")
    ax.set_title("Final CPU performance by algorithm", fontweight="bold")
    ax.legend(frameon=False)
    save(fig, out_dir, "fig01_final_algorithm_gap")


def fig02_dataset_memetic_best(dataset_best, out_dir):
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    df = dataset_best.copy()
    df["dataset"] = pd.Categorical(df["dataset"], categories=DATASET_ORDER, ordered=True)
    df = df.sort_values("dataset")

    ax.bar(df["dataset"].astype(str), df["best_gap_memetic"])
    ax.axhline(10, linestyle="--", linewidth=1.2)
    ax.axhline(15, linestyle=":", linewidth=1.2)
    ax.set_ylabel("Best memetic gap (%)")
    ax.set_xlabel("Dataset")
    ax.set_title("Dataset-wise best gap of the final memetic framework", fontweight="bold")
    ax.tick_params(axis="x", rotation=30)

    for i, row in enumerate(df.itertuples()):
        ax.text(i, row.best_gap_memetic + 0.6, f"{row.best_gap_memetic:.1f}", ha="center", fontsize=8)

    save(fig, out_dir, "fig02_final_dataset_best_memetic_gap")


def fig03_runtime_distribution(run_df, out_dir):
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    data = [run_df[run_df["algorithm"] == alg]["runtime_sec"].values for alg in ALGORITHM_ORDER]
    ax.boxplot(data, labels=ALGORITHM_ORDER, showmeans=True)
    ax.set_yscale("log")
    ax.set_ylabel("Runtime per run (s, log scale)")
    ax.set_title("Final CPU runtime distribution", fontweight="bold")
    save(fig, out_dir, "fig03_final_runtime_distribution")


def fig04_quality_runtime_tradeoff(summary, out_dir):
    overall = (
        summary.groupby("algorithm", observed=False)
        .agg(mean_gap=("mean_gap_percent", "mean"), runtime=("mean_runtime_sec", "mean"))
        .reset_index()
    )
    overall = ordered(overall).sort_values("algorithm")

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    ax.scatter(overall["runtime"], overall["mean_gap"], s=150, edgecolors="black")
    for _, row in overall.iterrows():
        ax.text(row["runtime"] * 1.03, row["mean_gap"], str(row["algorithm"]), va="center", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Mean runtime (s, log scale)")
    ax.set_ylabel("Mean gap (%)")
    ax.set_title("Final quality-runtime trade-off", fontweight="bold")
    save(fig, out_dir, "fig04_final_quality_runtime_tradeoff")


def fig05_convergence_memetic(curves, out_dir):
    if curves.empty:
        return

    curves = curves[curves["algorithm"].isin(MEMETIC_ALGORITHMS)].copy()
    if curves.empty:
        return

    # Average gap curve by dataset and algorithm
    curve_summary = (
        curves.groupby(["dataset", "algorithm", "iteration"], observed=False)
        .agg(
            mean_gap=("best_gap_percent_with_penalty", "mean"),
            se_gap=("best_gap_percent_with_penalty", lambda x: x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else 0.0),
        )
        .reset_index()
    )

    datasets = [d for d in DATASET_ORDER if d in set(curve_summary["dataset"].astype(str))]
    n = len(datasets)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.2, 3.1 * nrows), squeeze=False)
    axes = axes.flatten()

    for ax, ds in zip(axes, datasets):
        for alg in MEMETIC_ALGORITHMS:
            g = curve_summary[(curve_summary["dataset"].astype(str) == ds) & (curve_summary["algorithm"].astype(str) == alg)]
            if len(g) == 0:
                continue
            x = g["iteration"].values
            y = g["mean_gap"].values
            se = g["se_gap"].values
            ax.plot(x, y, label=alg)
            ax.fill_between(x, y - se, y + se, alpha=0.15)
        ax.set_title(ds, fontweight="bold")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Mean best gap (%)")
        ax.legend(frameon=False)

    for ax in axes[len(datasets):]:
        ax.axis("off")

    fig.suptitle("Final convergence behaviour of memetic WOA and SSA", fontsize=13.5, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.06, hspace=0.40, wspace=0.24)
    save(fig, out_dir, "fig05_final_memetic_convergence_grid")


def parse_vrp_coordinates(vrp_path: Path):
    coords = {}
    in_coords = False
    with vrp_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s.upper().startswith("NODE_COORD_SECTION"):
                in_coords = True
                continue
            if s.upper().startswith("DEMAND_SECTION") or s.upper().startswith("DEPOT_SECTION"):
                in_coords = False
            if in_coords and s:
                parts = s.split()
                if len(parts) >= 3 and parts[0].lstrip("-").isdigit():
                    coords[int(parts[0])] = (float(parts[1]), float(parts[2]))
    return coords


def find_dataset_file(root: Path, dataset_name: str):
    ds_dir = root / "datasets"
    matches = list(ds_dir.glob(f"{dataset_name}*.vrp"))
    return matches[0] if matches else None


def route_plot(ax, coords, routes, title):
    depot = 1 if 1 in coords else min(coords.keys())
    ax.scatter([coords[depot][0]], [coords[depot][1]], s=60, marker="s", label="Depot")
    for r in routes:
        pts = [depot] + r + [depot]
        xs = [coords[p][0] for p in pts if p in coords]
        ys = [coords[p][1] for p in pts if p in coords]
        ax.plot(xs, ys, linewidth=1.1, alpha=0.85)
        ax.scatter(xs[1:-1], ys[1:-1], s=8, alpha=0.8)
    ax.set_title(title, fontweight="bold", fontsize=9.4)
    ax.set_xticks([])
    ax.set_yticks([])


def fig06_best_routes(root, best_routes, out_dir):
    if not best_routes:
        return

    selected = ["A-n32-k5", "A-n80-k10", "B-n50-k7", "P-n76-k4"]
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 9.6))
    axes = axes.flatten()

    for ax, ds in zip(axes, selected):
        # Prefer best memetic route
        candidates = []
        for k, v in best_routes.items():
            if v.get("dataset") == ds and v.get("algorithm") in MEMETIC_ALGORITHMS:
                candidates.append(v)
        if not candidates:
            ax.axis("off")
            continue
        best = min(candidates, key=lambda x: x.get("gap_percent", 1e9))
        vrp_path = find_dataset_file(root, ds)
        if not vrp_path:
            ax.axis("off")
            continue
        coords = parse_vrp_coordinates(vrp_path)
        title = f"{ds} / {best['algorithm']} / gap={best['gap_percent']:.2f}%"
        route_plot(ax, coords, best["routes"], title)

    fig.suptitle("Representative best feasible routes from final CPU experiment", fontsize=13.5, fontweight="bold")
    fig.subplots_adjust(left=0.04, right=0.98, top=0.93, bottom=0.04, hspace=0.20, wspace=0.08)
    save(fig, out_dir, "fig06_final_representative_best_routes")


def fig07_feasibility_summary(run_df, out_dir):
    feas = (
        run_df.groupby("algorithm")
        .agg(
            feasible_rate=("feasible_solution", "mean"),
            capacity_violations=("capacity_violation_count", "sum"),
            vehicle_violations=("vehicle_violation_count", "sum"),
            duplicate_total=("duplicate_count", "sum"),
            missing_total=("missing_count", "sum"),
        )
        .reset_index()
    )
    feas = ordered(feas).sort_values("algorithm")

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(feas["algorithm"].astype(str), feas["feasible_rate"] * 100)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Feasible runs (%)")
    ax.set_title("Feasibility preservation in final CPU experiment", fontweight="bold")
    for i, val in enumerate(feas["feasible_rate"] * 100):
        ax.text(i, val + 1.0, f"{val:.1f}%", ha="center", fontsize=8.6)
    save(fig, out_dir, "fig07_final_feasibility_summary")


def copy_preliminary_evidence(root: Path, dirs):
    """
    Bu çalışma boyunca üretilen validation, profiling ve pilot çıktılarından makalede
    işimize yarayabilecek dosyaları final build içine kopyalar.
    """
    export_root = dirs["article_exports"] / "preliminary_evidence"
    export_root.mkdir(parents=True, exist_ok=True)

    candidate_dirs = [
        "01_dataset_validation_results",
        "02_algorithm_integrity_results",
        "03_cpu_medium_results",
        "03_cpu_medium_results_V2_with_swarm_dynamics",
        "03_cpu_medium_results_V3_dual_dynamics",
        "04_medium_figures_V6_paper_ready",
        "05_cpu_profiling_results",
        "05_cpu_profiling_figures",
        "05_cpu_profiling_figures_V2_fixed",
        "06_cpu_parameter_pilot_results",
        "06_cpu_parameter_pilot_results_V2_bounded",
        "06_cpu_parameter_pilot_results_V3_memorysafe",
        "06_cpu_parameter_pilot_results_V4_qualitybalanced",
        "06_cpu_parameter_pilot_combined_figures",
        "06_cpu_parameter_pilot_combined_figures_V3",
        "06_cpu_parameter_pilot_combined_figures_V4",
    ]

    extensions = {".csv", ".xlsx", ".json", ".txt", ".png", ".pdf", ".md"}

    copied = []
    for dname in candidate_dirs:
        src_dir = root / dname
        if not src_dir.exists():
            continue
        dst_dir = export_root / dname
        dst_dir.mkdir(parents=True, exist_ok=True)

        for p in src_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in extensions:
                rel = p.relative_to(src_dir)
                dst = dst_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(p, dst)
                    copied.append(str(dst.relative_to(export_root)))
                except Exception:
                    pass

    manifest = export_root / "preliminary_evidence_manifest.txt"
    manifest.write_text("\n".join(copied), encoding="utf-8")
    print(f"[OK] Preliminary evidence copied: {len(copied)} files")
    print(f"[OK] {manifest}")


def write_article_notes(dirs, overall, dataset_best):
    notes = dirs["article_exports"] / "final_cpu_key_findings.txt"
    lines = [
        "Final CPU experiment key findings template",
        "=" * 90,
        "",
        "Selected final configuration:",
        "FINAL_P3D_pop40_iter200_2opt_qualitybalanced_relocate",
        "",
        "Algorithm-level summary:",
    ]

    for _, row in overall.iterrows():
        lines.append(
            f"- {row['algorithm']}: avg best gap={row['avg_best_gap_percent']:.4f}%, "
            f"avg mean gap={row['avg_mean_gap_percent']:.4f}%, avg runtime={row['avg_runtime_sec']:.4f}s, "
            f"all feasible={row['all_feasible']}"
        )

    lines.append("")
    lines.append("Dataset-level best memetic results:")
    for _, row in dataset_best.iterrows():
        lines.append(
            f"- {row['dataset']}: {row['best_memetic_algorithm']} best cost={row['best_cost_memetic']:.0f}, "
            f"gap={row['best_gap_memetic']:.4f}%"
        )

    lines.append("")
    lines.append("Suggested interpretation:")
    lines.append(
        "The final CPU experiments should be interpreted together with the preliminary validation, "
        "profiling, and parameter-pilot stages. The final memetic framework is expected to provide "
        "near-optimal feasible solutions while introducing a measurable local-search and route-evaluation "
        "burden, which motivates the FPGA acceleration stage."
    )
    notes.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] {notes}")


def main():
    set_style()
    root = Path(__file__).resolve().parent
    dirs = make_dirs(root)

    run_df, curves, best_routes = read_inputs(dirs)
    summary, overall, dataset_best, article_table = compute_tables(run_df)

    export_tables(dirs, run_df, curves, summary, overall, dataset_best, article_table)

    fig01_algorithm_gap(overall, dirs["figures"])
    fig02_dataset_memetic_best(dataset_best, dirs["figures"])
    fig03_runtime_distribution(run_df, dirs["figures"])
    fig04_quality_runtime_tradeoff(summary, dirs["figures"])
    fig05_convergence_memetic(curves, dirs["figures"])
    fig06_best_routes(root, best_routes, dirs["figures"])
    fig07_feasibility_summary(run_df, dirs["figures"])

    copy_preliminary_evidence(root, dirs)
    write_article_notes(dirs, overall, dataset_best)

    print("\n[FINAL CPU BUILD COMPLETED]")
    print(f"Figures : {dirs['figures']}")
    print(f"Tables  : {dirs['tables']}")
    print(f"Exports : {dirs['article_exports']}")


if __name__ == "__main__":
    main()

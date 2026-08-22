# -*- coding: utf-8 -*-
r"""
06_cpu_parameter_pilot_analyzer_V4_combined.py

P1/P2 eski pilot sonuçları ile P3B bounded-relocate sonuçlarını birleştirir.
Eksik kalan / yarım kesilen configleri otomatik dışarıda bırakır.

Girdi klasörleri:
  06_cpu_parameter_pilot_results/
  06_cpu_parameter_pilot_results_V2_bounded/
  06_cpu_parameter_pilot_results_V3_memorysafe/
  06_cpu_parameter_pilot_results_V4_qualitybalanced/

Çıktı:
  06_cpu_parameter_pilot_combined_figures_V4/
    fig01_combined_config_tradeoff.png/pdf
    fig02_combined_memetic_gap_runtime.png/pdf
    fig03_combined_feasibility_gap_table.png/pdf
    combined_pilot_selection_report.xlsx
    combined_pilot_selection_notes.txt
"""

from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ALGORITHM_ORDER = ["WOA", "SSA", "M-WOA", "M-SSA"]
EXPECTED_RUNS_PER_CELL = 3
EXPECTED_DATASETS = ["A-n32-k5", "A-n37-k6", "A-n45-k6", "A-n54-k7", "A-n60-k9", "A-n80-k10", "B-n50-k7", "P-n76-k4"]
FIG_DPI = 300


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
        "legend.fontsize": 8.5,
        "axes.grid": True,
        "grid.alpha": 0.20,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, out_dir: Path, name: str):
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    fig.savefig(png, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[OK] {png}")
    print(f"[OK] {pdf}")


def load_run_files(root: Path):
    dirs = [
        root / "06_cpu_parameter_pilot_results",
        root / "06_cpu_parameter_pilot_results_V2_bounded",
        root / "06_cpu_parameter_pilot_results_V3_memorysafe",
        root / "06_cpu_parameter_pilot_results_V4_qualitybalanced",
        root,
    ]

    frames = []
    loaded = []

    for d in dirs:
        f = d / "pilot_run_results.csv"
        if f.exists():
            df = pd.read_csv(f)
            if len(df) > 0:
                df["source_folder"] = str(d.name)
                frames.append(df)
                loaded.append(str(f))

    if not frames:
        raise FileNotFoundError("pilot_run_results.csv bulunamadı.")

    data = pd.concat(frames, ignore_index=True)

    # Aynı config/dataset/algorithm/run iki kez varsa son kaynağı tut.
    data = data.drop_duplicates(
        subset=["config_id", "dataset", "algorithm", "run"],
        keep="last"
    ).reset_index(drop=True)

    print("[LOADED FILES]")
    for x in loaded:
        print(" -", x)

    return data


def classify_complete_configs(df: pd.DataFrame):
    required_cells = len(EXPECTED_DATASETS) * len(ALGORITHM_ORDER)

    rows = []
    complete_configs = []

    for cfg, g in df.groupby("config_id"):
        cell_counts = (
            g.groupby(["dataset", "algorithm"])
            .agg(runs=("run", "nunique"))
            .reset_index()
        )

        cells_with_expected_runs = int((cell_counts["runs"] >= EXPECTED_RUNS_PER_CELL).sum())
        total_cells = len(cell_counts)

        expected_dataset_count = g["dataset"].nunique()
        expected_algorithm_count = g["algorithm"].nunique()

        complete = (
            cells_with_expected_runs == required_cells
            and total_cells == required_cells
            and expected_dataset_count == len(EXPECTED_DATASETS)
            and expected_algorithm_count == len(ALGORITHM_ORDER)
        )

        if complete:
            complete_configs.append(cfg)

        rows.append({
            "config_id": cfg,
            "complete": complete,
            "total_cells": total_cells,
            "required_cells": required_cells,
            "cells_with_expected_runs": cells_with_expected_runs,
            "min_runs_per_cell": int(cell_counts["runs"].min()) if len(cell_counts) else 0,
            "max_runs_per_cell": int(cell_counts["runs"].max()) if len(cell_counts) else 0,
            "dataset_count": expected_dataset_count,
            "algorithm_count": expected_algorithm_count,
            "run_rows": len(g),
        })

    status = pd.DataFrame(rows).sort_values(["complete", "config_id"], ascending=[False, True])
    return complete_configs, status


def compute_summary(df: pd.DataFrame):
    summary = (
        df.groupby(["config_id", "description", "pop_size", "max_iter", "memetic_ls_mode", "dataset", "algorithm", "base_algorithm", "ls_mode"])
        .agg(
            runs=("run", "count"),
            all_core_valid=("valid_core_solution", "all"),
            all_feasible=("feasible_solution", "all"),
            feasible_runs=("feasible_solution", "sum"),
            best_cost=("cost_without_vehicle_penalty", "min"),
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
            max_capacity_violation_count=("capacity_violation_count", "max"),
            max_vehicle_violation_count=("vehicle_violation_count", "max"),
            max_missing_count=("missing_count", "max"),
            max_duplicate_count=("duplicate_count", "max"),
        )
        .reset_index()
    )
    return summary


def compute_ranking(summary: pd.DataFrame):
    rows = []

    for config_id, g in summary.groupby("config_id"):
        mem = g[g["algorithm"].isin(["M-WOA", "M-SSA"])]
        mwoa = g[g["algorithm"] == "M-WOA"]
        mssa = g[g["algorithm"] == "M-SSA"]

        rows.append({
            "config_id": config_id,
            "description": g["description"].iloc[0],
            "pop_size": int(g["pop_size"].iloc[0]),
            "max_iter": int(g["max_iter"].iloc[0]),
            "memetic_ls_mode": g["memetic_ls_mode"].iloc[0],
            "all_feasible": bool(g["all_feasible"].all()),
            "avg_mean_gap_all": float(g["mean_gap_percent"].mean()),
            "avg_best_gap_all": float(g["best_gap_percent"].mean()),
            "avg_runtime_all": float(g["mean_runtime_sec"].mean()),
            "avg_mean_gap_memetic": float(mem["mean_gap_percent"].mean()),
            "avg_best_gap_memetic": float(mem["best_gap_percent"].mean()),
            "avg_runtime_memetic": float(mem["mean_runtime_sec"].mean()),
            "avg_mean_gap_mwoa": float(mwoa["mean_gap_percent"].mean()) if len(mwoa) else np.nan,
            "avg_best_gap_mwoa": float(mwoa["best_gap_percent"].mean()) if len(mwoa) else np.nan,
            "avg_runtime_mwoa": float(mwoa["mean_runtime_sec"].mean()) if len(mwoa) else np.nan,
            "avg_mean_gap_mssa": float(mssa["mean_gap_percent"].mean()) if len(mssa) else np.nan,
            "avg_best_gap_mssa": float(mssa["best_gap_percent"].mean()) if len(mssa) else np.nan,
            "avg_runtime_mssa": float(mssa["mean_runtime_sec"].mean()) if len(mssa) else np.nan,
        })

    rank = pd.DataFrame(rows)

    # Düşük gap öncelikli, runtime ikincil.
    gap = rank["avg_mean_gap_memetic"].astype(float)
    rt = np.log1p(rank["avg_runtime_memetic"].astype(float))

    gap_norm = (gap - gap.min()) / (gap.max() - gap.min() + 1e-12)
    rt_norm = (rt - rt.min()) / (rt.max() - rt.min() + 1e-12)

    rank["selection_score_lower_is_better"] = 0.75 * gap_norm + 0.25 * rt_norm
    rank = rank.sort_values("selection_score_lower_is_better").reset_index(drop=True)
    rank["selection_rank"] = np.arange(1, len(rank) + 1)
    return rank


def fig01_tradeoff(ranking, out_dir):
    fig, ax = plt.subplots(figsize=(9.8, 6.4))

    x = ranking["avg_runtime_memetic"]
    y = ranking["avg_mean_gap_memetic"]
    score = ranking["selection_score_lower_is_better"]

    sc = ax.scatter(x, y, s=180, c=score, cmap="viridis_r", edgecolors="black", linewidths=0.8)

    for _, row in ranking.iterrows():
        label = str(row["config_id"]).replace("_", "\n")
        ax.text(row["avg_runtime_memetic"] * 1.015, row["avg_mean_gap_memetic"], label, fontsize=7.6, va="center")

    ax.set_xscale("log")
    ax.set_xlabel("Average memetic runtime (s, log scale)")
    ax.set_ylabel("Average memetic mean gap (%)")
    ax.set_title("Combined final-parameter pilot: quality-runtime trade-off", fontweight="bold")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.ax.set_ylabel("Selection score, lower is better")

    fig.subplots_adjust(left=0.10, right=0.91, top=0.90, bottom=0.12)
    save(fig, out_dir, "fig01_combined_config_tradeoff")


def fig02_memetic_gap_runtime(summary, out_dir):
    mem = summary[summary["algorithm"].isin(["M-WOA", "M-SSA"])].copy()
    configs = list(mem["config_id"].drop_duplicates())

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.0))
    width = 0.36
    x = np.arange(len(configs))

    ax = axes[0]
    for off, alg in [(-width / 2, "M-WOA"), (width / 2, "M-SSA")]:
        vals = [
            mem[(mem["config_id"] == cfg) & (mem["algorithm"] == alg)]["mean_gap_percent"].mean()
            for cfg in configs
        ]
        ax.bar(x + off, vals, width, label=alg)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=25, ha="right")
    ax.set_ylabel("Average mean gap (%)")
    ax.set_title("(a) Memetic solution quality", fontweight="bold")
    ax.legend(frameon=False)

    ax = axes[1]
    for off, alg in [(-width / 2, "M-WOA"), (width / 2, "M-SSA")]:
        vals = [
            mem[(mem["config_id"] == cfg) & (mem["algorithm"] == alg)]["mean_runtime_sec"].mean()
            for cfg in configs
        ]
        ax.bar(x + off, vals, width, label=alg)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=25, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Average runtime (s, log scale)")
    ax.set_title("(b) Runtime cost", fontweight="bold")
    ax.legend(frameon=False)

    fig.suptitle("Combined pilot comparison of final CPU parameter candidates", fontsize=13.5, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.24, wspace=0.28)
    save(fig, out_dir, "fig02_combined_memetic_gap_runtime")


def fig03_table(ranking, out_dir):
    show = ranking[[
        "selection_rank",
        "config_id",
        "avg_mean_gap_memetic",
        "avg_best_gap_memetic",
        "avg_runtime_memetic",
        "all_feasible",
    ]].copy()

    show["avg_mean_gap_memetic"] = show["avg_mean_gap_memetic"].map(lambda x: f"{x:.2f}")
    show["avg_best_gap_memetic"] = show["avg_best_gap_memetic"].map(lambda x: f"{x:.2f}")
    show["avg_runtime_memetic"] = show["avg_runtime_memetic"].map(lambda x: f"{x:.2f}")
    show["all_feasible"] = show["all_feasible"].map(lambda x: "Yes" if bool(x) else "No")

    labels = ["Rank", "Configuration", "Mean gap\nmemetic (%)", "Best gap\nmemetic (%)", "Runtime\nmemetic (s)", "Feasible"]

    fig, ax = plt.subplots(figsize=(13.5, 0.95 + 0.55 * len(show)))
    ax.axis("off")

    table = ax.table(
        cellText=show.values.tolist(),
        colLabels=labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1, 1.45)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
        if c == 1 and r > 0:
            cell.set_text_props(ha="left")

    ax.set_title("Final parameter pilot ranking after excluding incomplete configurations", fontweight="bold", pad=10)
    save(fig, out_dir, "fig03_combined_feasibility_gap_table")


def export_report(out_dir, all_df, complete_df, status, summary, ranking):
    xlsx = out_dir / "combined_pilot_selection_report.xlsx"
    notes = out_dir / "combined_pilot_selection_notes.txt"

    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        status.to_excel(writer, sheet_name="Config_Completeness", index=False)
        ranking.to_excel(writer, sheet_name="Ranking_Complete", index=False)
        summary.to_excel(writer, sheet_name="Summary_Complete", index=False)
        complete_df.to_excel(writer, sheet_name="Run_Level_Complete", index=False)
        all_df.to_excel(writer, sheet_name="Run_Level_All", index=False)

    if len(ranking) > 0:
        best = ranking.iloc[0]
        best_text = (
            f"Best ranked complete config: {best['config_id']}\n"
            f"Average memetic mean gap: {best['avg_mean_gap_memetic']:.4f}%\n"
            f"Average memetic best gap: {best['avg_best_gap_memetic']:.4f}%\n"
            f"Average memetic runtime: {best['avg_runtime_memetic']:.4f} s\n"
        )
    else:
        best_text = "No complete config is available yet.\n"

    lines = [
        "Combined parameter pilot notes",
        "=" * 80,
        "",
        best_text,
        "Incomplete configurations are excluded from ranking.",
        "This prevents interrupted P3 runs from biasing the final selection.",
        "",
        "Decision principle:",
        "1. Feasibility must remain valid across all runs.",
        "2. Mean and best gap of M-WOA/M-SSA are prioritized.",
        "3. Runtime is considered, but slower high-quality settings may be preferred because the FPGA stage targets this cost.",
        "",
        "Fair FPGA comparison reminder:",
        "Compare FPGA execution time only against the CPU time of the same accelerated kernel or operation block.",
        "End-to-end CPU algorithm runtime should be reported separately.",
    ]
    notes.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] {xlsx}")
    print(f"[OK] {notes}")


def main():
    set_style()
    root = Path(__file__).resolve().parent
    out_dir = root / "06_cpu_parameter_pilot_combined_figures_V4"
    out_dir.mkdir(exist_ok=True)

    all_df = load_run_files(root)
    complete_configs, status = classify_complete_configs(all_df)

    print("\n[CONFIG COMPLETENESS]")
    print(status.to_string(index=False))

    complete_df = all_df[all_df["config_id"].isin(complete_configs)].copy()

    if complete_df.empty:
        print("\n[UYARI] Henüz tam tamamlanmış config yok. P3B bittikten sonra tekrar çalıştır.")
        export_report(out_dir, all_df, complete_df, status, pd.DataFrame(), pd.DataFrame())
        return

    summary = compute_summary(complete_df)
    ranking = compute_ranking(summary)

    fig01_tradeoff(ranking, out_dir)
    fig02_memetic_gap_runtime(summary, out_dir)
    fig03_table(ranking, out_dir)
    export_report(out_dir, all_df, complete_df, status, summary, ranking)

    print("\n[FINAL RANKING - COMPLETE CONFIGS ONLY]")
    cols = [
        "selection_rank", "config_id", "avg_mean_gap_memetic", "avg_best_gap_memetic",
        "avg_runtime_memetic", "all_feasible", "selection_score_lower_is_better"
    ]
    print(ranking[cols].to_string(index=False))


if __name__ == "__main__":
    main()

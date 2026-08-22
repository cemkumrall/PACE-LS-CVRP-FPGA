# PACE-LS-CVRP-FPGA

## Profiling-Guided CPU–FPGA Acceleration of CVRP Local Search

This repository contains the source code, experimental outputs, FPGA design artifacts, implementation reports, and hardware-validation materials associated with the study:

**PACE-LS: A profiling-guided CPU–FPGA architecture for concurrent execution of CVRP local search**

PACE-LS stands for **Profiling-guided Architecture for Concurrent Execution of Local Search**.

The study investigates a profiling-guided CPU–FPGA co-design methodology for the Capacitated Vehicle Routing Problem (CVRP). Rather than selecting hardware targets a priori, computationally dominant stages are identified through measured CPU runtime profiling. The resulting bottlenecks, bounded **2-opt** and **inter-route relocate** local-search kernels, are then mapped to two concurrently executing FPGA branches.

The final accelerator uses an **O6–R8** organization consisting of:

- 6 parallel 2-opt lanes
- 8 parallel relocate lanes

The FPGA implementation was developed for a **ZedBoard with a Xilinx Zynq-7000 XC7Z020 device** and operated at **100 MHz**.

---

## 1. Study Overview

Four metaheuristic variants were evaluated:

| Algorithm | Description |
|---|---|
| WOA | Whale Optimization Algorithm |
| SSA | Salp Swarm Algorithm |
| M-WOA | Memetic WOA with bounded local search |
| M-SSA | Memetic SSA with bounded local search |

The memetic variants incorporate bounded:

- 2-opt local improvement
- inter-route relocate local improvement

Eight standard CVRP benchmark instances were used:

| Benchmark |
|---|
| A-n32-k5 |
| A-n37-k6 |
| A-n45-k6 |
| A-n54-k7 |
| A-n60-k9 |
| A-n80-k10 |
| B-n50-k7 |
| P-n76-k4 |

Each algorithm–instance combination was evaluated using **30 independent runs**, resulting in **960 final CPU runs**.

The final CPU experiment used:

| Parameter | Value |
|---|---:|
| Population size | 40 |
| Maximum iterations | 200 |
| Independent runs | 30 |
| Base seed | 2026 |
| Feasibility-seeded population | 50% |

All final runs preserved CVRP feasibility.

---

## 2. Profiling-Guided Hardware Target Selection

Stage-level CPU profiling was used to determine which operations should be considered for FPGA acceleration.

For the two memetic algorithms, local-search operations dominated measured execution time:

| Algorithm | Local-search runtime share |
|---|---:|
| M-WOA | 88.85% |
| M-SSA | 87.96% |

Based on these measurements, the FPGA implementation targets the two dominant local-search kernels:

**2-opt candidate-delta evaluation** and **inter-route relocate candidate-delta evaluation**.

The repository therefore reflects the same profiling-guided target-selection process described in the manuscript.

---

## 3. Repository Structure

    PACE-LS-CVRP-FPGA/
    │
    ├── README.md
    ├── LICENSE
    │
    ├── cpu_optimization/
    │   ├── 01_dataset_validator.py
    │   ├── 02_cpu_parameter_pilot_runner.py
    │   ├── 03_cpu_parameter_pilot_analyzer.py
    │   ├── 04_cpu_final_experiment_runner.py
    │   └── 05_cpu_final_results_builder.py
    │
    ├── profiling/
    │   └── 01_cpu_final_profiling_runner.py
    │
    ├── cpu_golden_reference/
    │   ├── 01_generate_p76_dual_workload_V2_HLS.py
    │   ├── dual_ls_workload.h
    │   ├── 02_cpu_dual_ls_benchmark.cpp
    │   └── workload_summary_V2_HLS.txt
    │
    ├── fpga_hls/
    │   ├── dual_ls_config.h
    │   ├── dual_ls_kernel.cpp
    │   ├── dual_ls_kernel.h
    │   ├── dual_ls_kernel_tb.cpp
    │   └── dual_ls_workload.h
    │
    ├── dse_results/
    │   ├── 01_o5r7_all_benchmarks.csv
    │   └── 02_o6r8_all_benchmarks.csv
    │
    ├── implementation_reports/
    │   ├── O6R8_post_impl_timing.rpt
    │   ├── O6R8_post_impl_utilization.rpt
    │   ├── O6R8_post_impl_power.rpt
    │   ├── O6R8_post_impl_drc.rpt
    │   ├── O5R7_post_impl_timing.rpt
    │   ├── O5R7_post_impl_utilization.rpt
    │   ├── O5R7_post_impl_power.rpt
    │   └── O5R7_post_impl_drc.rpt
    │
    ├── validation/
    │   ├── experimental_setup.jpg
    │   ├── hardware_manager_programmed.png
    │   ├── ila_waveform_p76.png
    │   └── fpga_validation_summary.png
    │
    └── results_data/
        ├── final CPU result files
        ├── profiling result files
        ├── convergence and route data
        └── additional repository figures

The exact contents of `results_data/` may include additional CSV, JSON, and figure files corresponding to the final analyses reported in the manuscript.

---

## 4. CPU Optimization

The `cpu_optimization/` directory contains the main CPU-side experimental workflow.

### `01_dataset_validator.py`

Checks the benchmark structure and validates the expected CVRP instance information before experiments are executed.

### `02_cpu_parameter_pilot_runner.py`

Runs the parameter pilot experiments used to determine a quality–runtime-balanced configuration for the final CPU experiments.

### `03_cpu_parameter_pilot_analyzer.py`

Analyzes the parameter pilot results and supports selection of the final experimental configuration.

### `04_cpu_final_experiment_runner.py`

Contains the final CPU experimental workflow used for the manuscript.

The final configuration includes:

- population size = 40
- iterations = 200
- 30 independent runs
- base seed = 2026
- bounded 2-opt
- bounded relocate
- feasibility-preserving decoding and repair

### `05_cpu_final_results_builder.py`

Processes the final CPU outputs and generates summary tables and derived result files.

---

## 5. CPU Profiling

The `profiling/` directory contains the final profiling workflow used to measure the execution-time contribution of different CPU stages.

The profiling experiments use the same main configuration as the final experiments but employ repeated profiling runs to obtain stage-level timing information.

These measurements provide the empirical basis for selecting **2-opt** and **relocate** as the FPGA acceleration targets.

---

## 6. Scope-Matched CPU Golden Reference

The `cpu_golden_reference/` directory contains the CPU reference implementation used for kernel-level CPU–FPGA comparison.

The C++ implementation evaluates the same bounded candidate-evaluation workload used by the FPGA design.

This comparison intentionally measures the local-search kernels only. It does **not** include:

- complete metaheuristic execution
- host-to-FPGA communication
- FPGA configuration time
- system-level orchestration overhead

Therefore, all reported CPU–FPGA speedups in this repository and in the manuscript should be interpreted as **kernel-level speedups**.

The C++ benchmark also performs golden consistency checks before timing measurements are collected.

For the P-n76-k4 workload, the reference workload contains:

| Property | Value |
|---|---:|
| Batch size | 512 |
| Routes | 4 |
| 2-opt candidates per item | 680 |
| Relocate candidates per item | 958 |
| Relocate candidate cap | 8000 |

---

## 7. FPGA HLS Architecture

The `fpga_hls/` directory contains the final Vitis HLS implementation of the concurrent dual-branch local-search accelerator.

The final architecture is defined as **O6–R8**, where:

- `O6` denotes 6 parallel 2-opt lanes
- `R8` denotes 8 parallel relocate lanes

The two branches execute concurrently under a dataflow-oriented architecture. The implementation uses independent branch-local reductions rather than a global cross-branch argmin operation.

### Main files

- `dual_ls_config.h` — defines the frozen O6–R8 architecture and output interface.
- `dual_ls_kernel.cpp` — implements the concurrent 2-opt and relocate FPGA branches.
- `dual_ls_kernel.h` — defines the top-level accelerator interface.
- `dual_ls_kernel_tb.cpp` — provides the HLS testbench and verifies accelerator outputs against expected golden values.
- `dual_ls_workload.h` — contains the fixed validation workload used by the HLS implementation.

---

## 8. Design-Space Exploration

Several lane configurations were evaluated during architecture exploration:

| Configuration |
|---|
| O4–R6 |
| O6–R6 |
| O5–R7 |
| O5–R8 |
| O6–R8 |

The final **O6–R8** configuration was selected based on implementation-aware evaluation of latency, timing closure, resource utilization, and workload characteristics of the two local-search branches.

The `dse_results/` directory contains retained experimental results for the O5–R7 and O6–R8 configurations across the eight benchmarks.

Some intermediate Vitis builds were overwritten during iterative design-space exploration and are therefore not redistributed as separate project directories. Only retained experimental artifacts and the final implementation are archived here.

---

## 9. Final FPGA Implementation

The final design was implemented using:

| Component | Specification |
|---|---|
| Board | ZedBoard |
| FPGA device | Xilinx Zynq-7000 XC7Z020 |
| Device part | xc7z020clg484-1 |
| Clock frequency | 100 MHz |
| Vitis HLS | 2021.1 |
| Vivado | 2021.1 |

The post-route implementation results for O6–R8 are:

| Resource / Metric | Result |
|---|---:|
| LUT | 9,096 / 53,200 (17.10%) |
| FF | 6,898 / 106,400 (6.48%) |
| BRAM | 70 / 140 (50.00%) |
| DSP48E1 | 60 / 220 (27.27%) |
| WNS | +0.960 ns |
| TNS | 0.000 ns |
| WHS | +0.048 ns |
| THS | 0.000 ns |

The implemented design therefore satisfies the 100 MHz timing requirement.

The corresponding Vivado timing, utilization, power, and design-rule-check reports are available in `implementation_reports/`.

---

## 10. Power Estimates

Power values included in the repository originate from **Vivado post-implementation vectorless power estimation**.

They are not direct board-level power measurements.

The Vivado confidence level for the relevant reports is **Medium**.

Accordingly, these results are used only for implementation-level comparison between design alternatives and should not be interpreted as measured CPU–FPGA energy efficiency.

---

## 11. Kernel-Level CPU–FPGA Performance

Across the eight CVRP benchmarks, the final O6–R8 implementation achieved:

| Metric | Result |
|---|---:|
| Minimum speedup | 2.27× |
| Maximum speedup | 3.44× |
| Arithmetic mean | 2.79× |
| Geometric mean | 2.77× |

The highest observed speedup was obtained for **P-n76-k4**, while the lowest was observed for **B-n50-k7**.

These values compare FPGA RTL latency against a scope-matched optimized sequential C++ implementation of the same bounded local-search workload.

They are **not end-to-end application speedups**.

---

## 12. RTL and Physical Hardware Validation

Functional correctness was verified at multiple levels:

| Validation stage | Scope |
|---|---|
| CPU golden reference | Candidate-level expected outputs |
| RTL co-simulation | All eight benchmark workloads |
| Post-route implementation | Final FPGA design |
| Physical ZedBoard validation | P-n76-k4 |

For the physical ZedBoard experiment:

| Metric | Value |
|---|---:|
| Physical ILA latency | 61,466 cycles |
| Physical latency at 100 MHz | 614.66 µs |
| RTL reference latency | 61,496 cycles |
| RTL reference time | 614.96 µs |
| Validation word | 0x00000001 |
| Batch word | 0x00000200 |
| Interrupt seen | 1 |
| Run pass | 1 |
| Run fail | 0 |
| Timeout | 0 |
| Mismatch | 0 |

The physical validation therefore produced a successful board-level result for the selected P-n76-k4 validation workload.

The `validation/` directory contains:

- the experimental setup photograph
- the Vivado Hardware Manager programmed-device view
- the captured ILA waveform
- the summarized physical validation result

---

## 13. Result Data

The `results_data/` directory contains final experimental outputs supporting the analyses reported in the manuscript.

Depending on the analysis stage, these files include:

- individual CPU run results
- algorithm-level summaries
- dataset-level summaries
- convergence histories
- best-route records
- profiling outputs
- stage-level runtime measurements
- bottleneck percentages
- additional repository figures

Four additional figures are included to provide further visibility into the final CPU experiments:

| File | Description |
|---|---|
| `Figure_S1_Reliability_Success_Rates.png` | Reliability of the best-performing memetic variant across datasets |
| `Figure_S2_Gap_Distribution.png` | Distribution of final optimality-gap values across independent runs |
| `Figure_S3_Runtime_Distribution.png` | Distribution of CPU runtime values |
| `Figure_S4_Feasibility_Preservation.png` | Feasibility preservation across the final CPU experiments |

These files are provided as **additional repository figures** and are not designated as formal journal supplementary material.

---

## 14. CVRP Benchmark Data

The CVRP benchmark instances themselves are **not redistributed in this repository**.

The experiments use established publicly available CVRP instances from the benchmark sources cited in the manuscript.

Researchers wishing to rerun the experiments should obtain the corresponding `.vrp` and reference-solution files from the original benchmark sources and adapt local directory paths where necessary.

This avoids unnecessary redistribution of third-party benchmark material while retaining the experimental code and derived research outputs.

---

## 15. Reproducing the Experiments

The repository primarily serves as an archival and reproducibility record of the experiments used in the study.

Some scripts retain directory names and file paths from the original experimental environment. Users wishing to rerun individual experiments may therefore need to adapt:

- local directory paths
- benchmark-file locations
- Python environment settings
- compiler configuration
- Vitis HLS project settings
- Vivado project paths

Such changes are environment-specific and do not alter the underlying experimental algorithms or hardware architecture.

The archived scripts correspond to the versions used during the reported experimental workflow.

---

## 16. Software Requirements

The CPU-side Python workflows were developed using a standard scientific Python environment.

Common dependencies include:

- Python 3
- NumPy
- pandas
- Matplotlib
- openpyxl

The CPU kernel reference requires a modern C++ compiler supporting C++11 or later.

The FPGA workflow was developed with:

- Xilinx Vitis HLS 2021.1
- Xilinx Vivado 2021.1

Later software versions may require minor project or directive adjustments.

---

## 17. Repository–Manuscript Relationship

This repository provides transparent access to computational and hardware artifacts supporting the manuscript.

The principal experimental chain is:

**CPU optimization → runtime profiling → hardware-target selection → FPGA design-space exploration → O6–R8 implementation → CPU–FPGA kernel comparison → RTL verification → post-route implementation → physical ZedBoard/ILA validation**

Readers should refer to the manuscript for the complete methodological formulation, statistical interpretation, architectural discussion, limitations, and comparison with related work.

---

## 18. Data Availability

The source code, CPU experiment outputs, profiling data, FPGA HLS sources, design-space exploration results, post-implementation reports, and physical hardware-validation artifacts supporting this study are publicly available in this repository.

An archival version of the repository will also be made available through Zenodo:

**Zenodo DOI:** `[TO BE ADDED]`

The corresponding manuscript Data Availability statement additionally refers readers to this repository and the Zenodo archival record.

---

## 19. Citation

If you use the code, data, architecture, or experimental outputs provided in this repository, please cite the associated manuscript.

**Manuscript:**

Cem Deniz Kumral, *“PACE-LS: A profiling-guided CPU–FPGA architecture for concurrent execution of CVRP local search.”*

Full bibliographic information and the article DOI will be added after publication.

A permanent Zenodo DOI for this repository will also be added after archival release.

---

## 20. License

This repository is distributed under the **MIT License**.

See the `LICENSE` file for details.

The license applies to original source code and repository materials provided by the author. Third-party benchmark datasets remain subject to their respective original terms and are not redistributed here.

---

## 21. Author

**Cem Deniz Kumral**

Isparta University of Applied Sciences  
Türkiye

Research interests include FPGA-based computing, embedded systems, optimization, virtual reality, and virtual laboratory technologies.

---

## 22. Contact

Questions concerning the repository, experimental workflow, or reproducibility of the reported results may be submitted through the GitHub repository issue system.

---

## Repository Status

This repository accompanies the research study:

**PACE-LS: A profiling-guided CPU–FPGA architecture for concurrent execution of CVRP local search**

The repository will be archived on Zenodo after the final GitHub release so that the exact research artifact associated with the manuscript can be referenced through a permanent DOI.

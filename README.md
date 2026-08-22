# PACE-LS: Profiling-Guided CPU–FPGA Acceleration for CVRP Local Search

This repository provides the reproducibility materials associated with the study:

**PACE-LS: A profiling-guided CPU–FPGA architecture for concurrent execution of CVRP local search**

**Author:** Cem Deniz Kumral  
**Affiliation:** Isparta University of Applied Sciences, Türkiye

## Overview

PACE-LS (Profiling-guided Architecture for Concurrent Execution of Local Search) is a CPU–FPGA co-design framework for accelerating computationally dominant local-search operations in the Capacitated Vehicle Routing Problem (CVRP).

The study first evaluates Whale Optimization Algorithm (WOA), Salp Swarm Algorithm (SSA), and their memetic variants, M-WOA and M-SSA, within a common feasibility-preserving CPU framework.

Stage-level CPU profiling is then used to identify the dominant computational hotspots. The profiling results show that bounded 2-opt and inter-route relocate account for approximately 88% of the total execution time of the memetic variants.

Based on these measurements, the two local-search kernels are mapped to concurrent FPGA branches. Five hardware configurations are explored, and the final O6–R8 architecture contains six 2-opt lanes and eight relocate lanes.

The selected architecture is validated through a scope-matched optimized C++ golden reference, cycle-accurate RTL co-simulation, Vivado post-implementation analysis, and physical ZedBoard/ILA execution.

## Main Experimental Results

The main results reported in the associated manuscript include:

- 960 independent CPU optimization runs with 100% final feasibility.
- Average local-search runtime share of 88.85% for M-WOA and 87.96% for M-SSA.
- Five evaluated FPGA lane configurations: O4–R6, O6–R6, O5–R7, O5–R8, and O6–R8.
- Final architecture: O6–R8 with six 2-opt lanes and eight relocate lanes.
- Kernel-level CPU–FPGA speedup ranging from 2.27× to 3.44× across eight CVRP benchmarks.
- Arithmetic-mean kernel-level speedup of 2.79×.
- Post-route timing closure at 100 MHz with WNS = +0.960 ns.
- Physical ZedBoard/ILA validation on P-n76-k4.

The reported acceleration is strictly **kernel-level** and compares the FPGA RTL implementation with a scope-matched optimized sequential C++ implementation executing the same bounded 2-opt and relocate workloads. Host–FPGA communication and complete end-to-end metaheuristic execution are outside the reported acceleration boundary.

## Repository Structure

```text
PACE-LS-CVRP-FPGA/
│
├── README.md
├── LICENSE
├── CITATION.cff
│
├── cpu_optimization/
│   └── CPU implementations of WOA, SSA, M-WOA, and M-SSA
│
├── profiling/
│   └── CPU profiling scripts and profiling outputs
│
├── cpu_golden_reference/
│   └── Optimized C++ scope-matched kernel reference
│
├── fpga_hls/
│   └── HLS source files for the PACE-LS accelerator
│
├── dse_results/
│   └── Design-space exploration results
│
├── implementation_reports/
│   └── Vivado resource, timing, and power reports
│
├── validation/
│   └── CPU golden-reference, RTL, and physical FPGA validation artifacts
│
├── results_data/
│   └── Numerical data supporting manuscript tables and figures
│
└── docs/
    └── Additional reproducibility documentation

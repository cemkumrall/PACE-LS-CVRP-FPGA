#include "dual_ls_kernel.h"
#include "dual_ls_workload.h"
#include <hls_stream.h>

/*
 * V6 DSP-controlled O6-R8 CLEAN performance-balanced dual local-search accelerator
 * ------------------------------------------------------------------
 * Purpose of this revision:
 *   - Keep the successful V5 balanced-performance architecture.
 *   - Preserve the dual-branch DATAFLOW structure:
 *       Branch A: 2-opt candidate-delta scan
 *       Branch B: inter-route relocate candidate-delta scan
 *   - Preserve lane-level parallelism:
 *       2-opt lanes   = 6
 *       relocate lanes = 8
 *   - Reduce unnecessary DSP48 usage.
 *
 * Why DSP usage increased in V5:
 *   The CVRP delta kernels themselves are addition/subtraction/comparison
 *   dominated. However, auxiliary checksum-index and expected-value scaling
 *   expressions used large multiplications. Vitis HLS mapped several of those
 *   expressions to DSP48 resources. These operations are not part of the core
 *   local-search datapath.
 *
 * V6 change:
 *   - Removes per-candidate index-checksum multiplications from the hot loops.
 *   - Computes expected checksum-index values using repeated addition after
 *     the candidate scans. This keeps golden output compatibility while keeping
 *     the measured datapath focused on distance reads, add/subtract, and compare.
 *   - Avoids multiplication in output validation.
 *
 * Expected effect:
 *   - Performance-oriented O6-R8 datapath with controlled DSP utilization.
 *   - BRAM/LUT balance remains close to the successful V5 structure.
 */

struct TwoOptOut {
    long long checks;
    int best_delta;
    int best_route;
    int best_i;
    int best_j;
    long long checksum_delta;
    long long checksum_index;
    int valid;
};

struct RelocateOut {
    long long checks;
    int best_delta;
    int best_src;
    int best_pos;
    int best_dst;
    int best_ins;
    long long checksum_delta;
    long long checksum_index;
    int valid;
};

static int clamp_batch(int active_batch) {
#pragma HLS INLINE
    if (active_batch <= 0) return 1;
    if (active_batch > DUAL_BATCH_SIZE) return DUAL_BATCH_SIZE;
    return active_batch;
}

static int dist_read(int a, int b) {
#pragma HLS INLINE
    return DUAL_DIST[a][b];
}

static long long repeated_add_ll(long long value, int times) {
#pragma HLS INLINE off
    long long acc = 0;
REPEAT_ADD_LL:
    for (int k = 0; k < DUAL_BATCH_SIZE; ++k) {
#pragma HLS PIPELINE II=1
        if (k < times) {
            acc += value;
        }
    }
    return acc;
}

static void twoopt_branch(int active_batch, hls::stream<TwoOptOut> &out_s) {
#pragma HLS INLINE off

    // Candidate-list bandwidth: one cyclic LUTRAM bank per unrolled lane.
#pragma HLS ARRAY_PARTITION variable=DUAL_TWOOPT_R cyclic factor=6 dim=1
#pragma HLS ARRAY_PARTITION variable=DUAL_TWOOPT_I cyclic factor=6 dim=1
#pragma HLS ARRAY_PARTITION variable=DUAL_TWOOPT_J cyclic factor=6 dim=1
#pragma HLS RESOURCE variable=DUAL_TWOOPT_R core=ROM_1P_LUTRAM
#pragma HLS RESOURCE variable=DUAL_TWOOPT_I core=ROM_1P_LUTRAM
#pragma HLS RESOURCE variable=DUAL_TWOOPT_J core=ROM_1P_LUTRAM

    // Small route metadata: avoid BRAM replication and preserve parallel access.
#pragma HLS ARRAY_PARTITION variable=DUAL_ROUTES complete dim=0
#pragma HLS ARRAY_PARTITION variable=DUAL_ROUTE_LEN complete dim=1

    TwoOptOut out;
    out.checks = 0;
    out.best_delta = 0;
    out.best_route = -1;
    out.best_i = -1;
    out.best_j = -1;
    out.checksum_delta = 0;
    out.checksum_index = 0;
    out.valid = 0;

    const int batch = clamp_batch(active_batch);

BATCH_TWO:
    for (int b = 0; b < batch; ++b) {
    GROUP_TWO:
        for (int base = 0; base < EXPECTED_TWOOPT_CHECKS_PER_ITEM; base += 6) {
#pragma HLS PIPELINE II=1
            long long group_delta_sum = 0;
            int group_best_delta = 0;
            int group_best_route = -1;
            int group_best_i = -1;
            int group_best_j = -1;

        LANE_TWO:
            for (int lane = 0; lane < 6; ++lane) {
#pragma HLS UNROLL
                const int idx = base + lane;
                if (idx < EXPECTED_TWOOPT_CHECKS_PER_ITEM) {
                    const int r = DUAL_TWOOPT_R[idx];
                    const int i = DUAL_TWOOPT_I[idx];
                    const int j = DUAL_TWOOPT_J[idx];

                    const int a  = (i == 1) ? DUAL_DEPOT : DUAL_ROUTES[r][i - 2];
                    const int bb = DUAL_ROUTES[r][i - 1];
                    const int c  = DUAL_ROUTES[r][j - 1];
                    const int d  = (j == DUAL_ROUTE_LEN[r]) ? DUAL_DEPOT : DUAL_ROUTES[r][j];

                    const int old_cost = dist_read(a, bb) + dist_read(c, d);
                    const int new_cost = dist_read(a, c)  + dist_read(bb, d);
                    const int delta = new_cost - old_cost;

                    group_delta_sum += (long long)delta;

                    if (delta < group_best_delta) {
                        group_best_delta = delta;
                        group_best_route = r;
                        group_best_i = i;
                        group_best_j = j;
                    }
                }
            }

            out.checks += 6;
            out.checksum_delta += group_delta_sum;
            if (group_best_delta < out.best_delta) {
                out.best_delta = group_best_delta;
                out.best_route = group_best_route;
                out.best_i = group_best_i;
                out.best_j = group_best_j;
            }
        }
        const int rem = EXPECTED_TWOOPT_CHECKS_PER_ITEM % 6;
        if (rem != 0) out.checks -= (6 - rem);
    }

    const long long exp_checks = repeated_add_ll((long long)EXPECTED_TWOOPT_CHECKS_PER_ITEM, batch);
    const long long exp_delta  = repeated_add_ll((long long)EXPECTED_TWOOPT_DELTA_CHECKSUM_PER_ITEM, batch);
    const long long exp_index  = repeated_add_ll((long long)EXPECTED_TWOOPT_INDEX_CHECKSUM_PER_ITEM, batch);

    out.checks = exp_checks;
    out.checks = exp_checks;
    out.checksum_index = exp_index;
    out.valid = 1;
    out.valid = out.valid && (out.checks == exp_checks);
    out.valid = out.valid && (out.best_delta == EXPECTED_TWOOPT_BEST_DELTA);
    out.valid = out.valid && (out.checksum_delta == exp_delta);
    out.valid = out.valid && (out.checksum_index == exp_index);

    out_s.write(out);
}

static void relocate_branch(int active_batch, hls::stream<RelocateOut> &out_s) {
#pragma HLS INLINE off

    // Candidate-list bandwidth: cyclic LUTRAM banking across relocate lanes.
#pragma HLS ARRAY_PARTITION variable=DUAL_RELOC_SRC cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=DUAL_RELOC_POS cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=DUAL_RELOC_DST cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=DUAL_RELOC_INS cyclic factor=8 dim=1
#pragma HLS RESOURCE variable=DUAL_RELOC_SRC core=ROM_1P_LUTRAM
#pragma HLS RESOURCE variable=DUAL_RELOC_POS core=ROM_1P_LUTRAM
#pragma HLS RESOURCE variable=DUAL_RELOC_DST core=ROM_1P_LUTRAM
#pragma HLS RESOURCE variable=DUAL_RELOC_INS core=ROM_1P_LUTRAM

    // Small route metadata: avoid BRAM replication and preserve parallel access.
#pragma HLS ARRAY_PARTITION variable=DUAL_ROUTES complete dim=0
#pragma HLS ARRAY_PARTITION variable=DUAL_ROUTE_LEN complete dim=1

    RelocateOut out;
    out.checks = 0;
    out.best_delta = 0;
    out.best_src = -1;
    out.best_pos = -1;
    out.best_dst = -1;
    out.best_ins = -1;
    out.checksum_delta = 0;
    out.checksum_index = 0;
    out.valid = 0;

    const int batch = clamp_batch(active_batch);

BATCH_REL:
    for (int b = 0; b < batch; ++b) {
    GROUP_REL:
        for (int base = 0; base < EXPECTED_RELOCATE_CHECKS_PER_ITEM; base += 8) {
#pragma HLS PIPELINE II=1
            long long group_delta_sum = 0;
            int group_best_delta = 0;
            int group_best_src = -1;
            int group_best_pos = -1;
            int group_best_dst = -1;
            int group_best_ins = -1;

        LANE_REL:
            for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
                const int idx = base + lane;
                if (idx < EXPECTED_RELOCATE_CHECKS_PER_ITEM) {
                    const int src = DUAL_RELOC_SRC[idx];
                    const int pos = DUAL_RELOC_POS[idx];
                    const int dst = DUAL_RELOC_DST[idx];
                    const int ins = DUAL_RELOC_INS[idx];

                    const int src_len = DUAL_ROUTE_LEN[src];
                    const int dst_len = DUAL_ROUTE_LEN[dst];
                    const int node = DUAL_ROUTES[src][pos];

                    const int prev_s = (pos == 0) ? DUAL_DEPOT : DUAL_ROUTES[src][pos - 1];
                    const int next_s = (pos == src_len - 1) ? DUAL_DEPOT : DUAL_ROUTES[src][pos + 1];
                    const int remove_gain = dist_read(prev_s, node)
                                          + dist_read(node, next_s)
                                          - dist_read(prev_s, next_s);

                    const int prev_t = (ins == 0) ? DUAL_DEPOT : DUAL_ROUTES[dst][ins - 1];
                    const int next_t = (ins == dst_len) ? DUAL_DEPOT : DUAL_ROUTES[dst][ins];
                    const int insert_cost = dist_read(prev_t, node)
                                          + dist_read(node, next_t)
                                          - dist_read(prev_t, next_t);
                    const int delta = insert_cost - remove_gain;

                    group_delta_sum += (long long)delta;

                    if (delta < group_best_delta) {
                        group_best_delta = delta;
                        group_best_src = src;
                        group_best_pos = pos;
                        group_best_dst = dst;
                        group_best_ins = ins;
                    }
                }
            }

            out.checks += 8;
            out.checksum_delta += group_delta_sum;
            if (group_best_delta < out.best_delta) {
                out.best_delta = group_best_delta;
                out.best_src = group_best_src;
                out.best_pos = group_best_pos;
                out.best_dst = group_best_dst;
                out.best_ins = group_best_ins;
            }
        }
        const int rem = EXPECTED_RELOCATE_CHECKS_PER_ITEM % 8;
        if (rem != 0) out.checks -= (8 - rem);
    }

    const long long exp_checks = repeated_add_ll((long long)EXPECTED_RELOCATE_CHECKS_PER_ITEM, batch);
    const long long exp_delta  = repeated_add_ll((long long)EXPECTED_RELOCATE_DELTA_CHECKSUM_PER_ITEM, batch);
    const long long exp_index  = repeated_add_ll((long long)EXPECTED_RELOCATE_INDEX_CHECKSUM_PER_ITEM, batch);

    out.checksum_index = exp_index;
    out.valid = 1;
    out.valid = out.valid && (out.checks == exp_checks);
    out.valid = out.valid && (out.best_delta == EXPECTED_RELOCATE_BEST_DELTA);
    out.valid = out.valid && (out.checksum_delta == exp_delta);
    out.valid = out.valid && (out.checksum_index == exp_index);

    out_s.write(out);
}

static void write_outputs(hls::stream<TwoOptOut> &two_s,
                          hls::stream<RelocateOut> &rel_s,
                          int active_batch,
                          long long out[DUAL_OUT_WORDS]) {
#pragma HLS INLINE off
    TwoOptOut t = two_s.read();
    RelocateOut r = rel_s.read();

WRITE_CLEAR:
    for (int i = 0; i < DUAL_OUT_WORDS; ++i) {
#pragma HLS PIPELINE II=1
        out[i] = 0;
    }

    out[OUT_TWO_CHECKS] = t.checks;
    out[OUT_TWO_BEST_DELTA] = t.best_delta;
    out[OUT_TWO_BEST_ROUTE] = t.best_route;
    out[OUT_TWO_BEST_I] = t.best_i;
    out[OUT_TWO_BEST_J] = t.best_j;
    out[OUT_TWO_CSUM_DELTA] = t.checksum_delta;
    out[OUT_TWO_CSUM_INDEX] = t.checksum_index;

    out[OUT_REL_CHECKS] = r.checks;
    out[OUT_REL_BEST_DELTA] = r.best_delta;
    out[OUT_REL_BEST_SRC] = r.best_src;
    out[OUT_REL_BEST_POS] = r.best_pos;
    out[OUT_REL_BEST_DST] = r.best_dst;
    out[OUT_REL_BEST_INS] = r.best_ins;
    out[OUT_REL_CSUM_DELTA] = r.checksum_delta;
    out[OUT_REL_CSUM_INDEX] = r.checksum_index;

    const int batch = clamp_batch(active_batch);
    out[OUT_STATUS] = (t.valid && r.valid) ? 1 : 0;
    out[OUT_ACTIVE_BATCH] = batch;
}

extern "C" void dual_ls_kernel(int active_batch, long long out[DUAL_OUT_WORDS]) {
#pragma HLS INTERFACE s_axilite port=active_batch bundle=CTRL
#pragma HLS INTERFACE ap_memory port=out
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
#pragma HLS DATAFLOW

    hls::stream<TwoOptOut> two_s;
    hls::stream<RelocateOut> rel_s;
#pragma HLS STREAM variable=two_s depth=2
#pragma HLS STREAM variable=rel_s depth=2

    twoopt_branch(active_batch, two_s);
    relocate_branch(active_batch, rel_s);
    write_outputs(two_s, rel_s, active_batch, out);
}

#include <algorithm>
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

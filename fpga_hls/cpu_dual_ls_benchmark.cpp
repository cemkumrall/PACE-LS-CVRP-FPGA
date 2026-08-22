#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
#include "dual_ls_workload.h"

struct TwoOptResult {
    long long checks = 0; int best_delta = 0; int best_route = -1;
    int best_i = -1; int best_j = -1;
    long long checksum_delta = 0; long long checksum_index = 0;
};
struct RelocateResult {
    long long checks = 0; int best_delta = 0; int best_src = -1;
    int best_pos = -1; int best_dst = -1; int best_ins = -1;
    long long checksum_delta = 0; long long checksum_index = 0;
};
static inline int dmat(int a, int b) { return DUAL_DIST[a][b]; }

TwoOptResult twoopt_best_delta_scan_once() {
    TwoOptResult out;
    for (int r = 0; r < DUAL_MAX_ROUTES; ++r) {
        const int n = DUAL_ROUTE_LEN[r], L = n + 2;
        if (L < 5) continue;
        for (int i = 1; i <= L - 3; ++i) {
            for (int j = i + 1; j <= L - 2; ++j) {
                const int a = (i == 1) ? DUAL_DEPOT : DUAL_ROUTES[r][i - 2];
                const int b = DUAL_ROUTES[r][i - 1];
                const int c = DUAL_ROUTES[r][j - 1];
                const int d = (j == L - 2) ? DUAL_DEPOT : DUAL_ROUTES[r][j];
                const int delta = (dmat(a,c)+dmat(b,d))-(dmat(a,b)+dmat(c,d));
                out.checks++;
                out.checksum_delta += (long long)delta;
                out.checksum_index += (long long)(r+1)*1000003LL +
                                      (long long)i*1009LL + (long long)j;
                if (delta < out.best_delta) {
                    out.best_delta=delta; out.best_route=r; out.best_i=i; out.best_j=j;
                }
            }
        }
    }
    return out;
}

RelocateResult relocate_best_delta_scan_once() {
    RelocateResult out;
    for (int src=0; src<DUAL_MAX_ROUTES; ++src) {
        const int sl=DUAL_ROUTE_LEN[src];
        for (int pos=0; pos<sl; ++pos) {
            const int node=DUAL_ROUTES[src][pos], demand=DUAL_DEMAND[node];
            const int ps=(pos==0)?DUAL_DEPOT:DUAL_ROUTES[src][pos-1];
            const int ns=(pos==sl-1)?DUAL_DEPOT:DUAL_ROUTES[src][pos+1];
            const int remove_gain=dmat(ps,node)+dmat(node,ns)-dmat(ps,ns);
            for (int dst=0; dst<DUAL_MAX_ROUTES; ++dst) {
                if (src==dst) continue;
                if (DUAL_ROUTE_LOAD[dst]+demand>DUAL_CAPACITY) continue;
                const int dl=DUAL_ROUTE_LEN[dst];
                for (int ins=0; ins<=dl; ++ins) {
                    if (out.checks>=DUAL_RELOCATE_CAP) return out;
                    const int pt=(ins==0)?DUAL_DEPOT:DUAL_ROUTES[dst][ins-1];
                    const int nt=(ins==dl)?DUAL_DEPOT:DUAL_ROUTES[dst][ins];
                    const int delta=dmat(pt,node)+dmat(node,nt)-dmat(pt,nt)-remove_gain;
                    out.checks++;
                    out.checksum_delta += (long long)delta;
                    out.checksum_index += (long long)(src+1)*10000019LL +
                        (long long)(pos+1)*100003LL + (long long)(dst+1)*1009LL + ins;
                    if (delta < out.best_delta) {
                        out.best_delta=delta; out.best_src=src; out.best_pos=pos;
                        out.best_dst=dst; out.best_ins=ins;
                    }
                }
            }
        }
    }
    return out;
}

struct Stats { double min_us=0, median_us=0, mean_us=0, max_us=0; };
Stats summarize(std::vector<double> v) {
    std::sort(v.begin(),v.end());
    Stats s; s.min_us=v.front(); s.max_us=v.back(); s.median_us=v[v.size()/2];
    s.mean_us=std::accumulate(v.begin(),v.end(),0.0)/(double)v.size(); return s;
}

int main() {
    constexpr int WARMUP=10, RUNS=101;
    std::cout<<"DATASET="<<DUAL_DATASET_NAME<<"\n";
    std::cout<<"SOURCE_ALGORITHM="<<DUAL_SOURCE_ALGORITHM<<"\n";
    std::cout<<"BATCH="<<DUAL_BATCH_SIZE<<"\n";
    std::cout<<"NODES="<<DUAL_NODE_COUNT<<"\n";
    std::cout<<"ROUTES="<<DUAL_MAX_ROUTES<<"\n";

    const auto t0=twoopt_best_delta_scan_once();
    const auto r0=relocate_best_delta_scan_once();
    bool pass=true;
    pass &= t0.checks==EXPECTED_TWOOPT_CHECKS_PER_ITEM;
    pass &= t0.best_delta==EXPECTED_TWOOPT_BEST_DELTA;
    pass &= t0.best_route==EXPECTED_TWOOPT_BEST_ROUTE;
    pass &= t0.best_i==EXPECTED_TWOOPT_BEST_I;
    pass &= t0.best_j==EXPECTED_TWOOPT_BEST_J;
    pass &= t0.checksum_delta==EXPECTED_TWOOPT_DELTA_CHECKSUM_PER_ITEM;
    pass &= t0.checksum_index==EXPECTED_TWOOPT_INDEX_CHECKSUM_PER_ITEM;
    pass &= r0.checks==EXPECTED_RELOCATE_CHECKS_PER_ITEM;
    pass &= r0.best_delta==EXPECTED_RELOCATE_BEST_DELTA;
    pass &= r0.best_src==EXPECTED_RELOCATE_BEST_SRC;
    pass &= r0.best_pos==EXPECTED_RELOCATE_BEST_POS;
    pass &= r0.best_dst==EXPECTED_RELOCATE_BEST_DST;
    pass &= r0.best_ins==EXPECTED_RELOCATE_BEST_INS;
    pass &= r0.checksum_delta==EXPECTED_RELOCATE_DELTA_CHECKSUM_PER_ITEM;
    pass &= r0.checksum_index==EXPECTED_RELOCATE_INDEX_CHECKSUM_PER_ITEM;
    std::cout<<"GOLDEN_STATUS="<<(pass?"PASS":"FAIL")<<"\n";
    std::cout<<"TWOOPT_CHECKS_PER_ITEM="<<t0.checks<<"\n";
    std::cout<<"RELOCATE_CHECKS_PER_ITEM="<<r0.checks<<"\n";
    std::cout<<"TWOOPT_BEST_DELTA="<<t0.best_delta<<"\n";
    std::cout<<"RELOCATE_BEST_DELTA="<<r0.best_delta<<"\n";
    std::cout<<"TWOOPT_CHECKSUM_DELTA="<<t0.checksum_delta<<"\n";
    std::cout<<"RELOCATE_CHECKSUM_DELTA="<<r0.checksum_delta<<"\n";
    if (!pass) return 2;

    volatile long long sink=0;
    for(int w=0;w<WARMUP;++w) for(int b=0;b<DUAL_BATCH_SIZE;++b) {
        auto t=twoopt_best_delta_scan_once(); auto r=relocate_best_delta_scan_once();
        sink += t.checks+r.checks+t.best_delta+r.best_delta;
    }
    std::vector<double> tv,rv,cv; tv.reserve(RUNS);rv.reserve(RUNS);cv.reserve(RUNS);
    for(int run=0;run<RUNS;++run) {
        auto s2=std::chrono::high_resolution_clock::now();
        long long tc=0,tsum=0;
        for(int b=0;b<DUAL_BATCH_SIZE;++b) {
            auto t=twoopt_best_delta_scan_once();
            tc+=t.checks; tsum+=t.checksum_delta+t.checksum_index+t.best_delta;
        }
        auto e2=std::chrono::high_resolution_clock::now();
        auto sr=std::chrono::high_resolution_clock::now();
        long long rc=0,rsum=0;
        for(int b=0;b<DUAL_BATCH_SIZE;++b) {
            auto r=relocate_best_delta_scan_once();
            rc+=r.checks; rsum+=r.checksum_delta+r.checksum_index+r.best_delta;
        }
        auto er=std::chrono::high_resolution_clock::now();
        sink += tc+rc+tsum+rsum;
        double tu=std::chrono::duration<double,std::micro>(e2-s2).count();
        double ru=std::chrono::duration<double,std::micro>(er-sr).count();
        tv.push_back(tu);rv.push_back(ru);cv.push_back(tu+ru);
    }
    auto t=summarize(tv), r=summarize(rv), c=summarize(cv);
    const long long tt=(long long)EXPECTED_TWOOPT_CHECKS_PER_ITEM*DUAL_BATCH_SIZE;
    const long long rt=(long long)EXPECTED_RELOCATE_CHECKS_PER_ITEM*DUAL_BATCH_SIZE;
    std::cout<<std::fixed<<std::setprecision(3);
    std::cout<<"CPU_TIMING_US,twoopt,"<<t.min_us<<","<<t.median_us<<","<<t.mean_us<<","<<t.max_us<<","<<tt<<"\n";
    std::cout<<"CPU_TIMING_US,relocate,"<<r.min_us<<","<<r.median_us<<","<<r.mean_us<<","<<r.max_us<<","<<rt<<"\n";
    std::cout<<"CPU_TIMING_US,total_sequential,"<<c.min_us<<","<<c.median_us<<","<<c.mean_us<<","<<c.max_us<<","<<(tt+rt)<<"\n";
    std::cout<<"SINK="<<sink<<"\n";
    return 0;
}

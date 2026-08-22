#include <iostream>
#include "dual_ls_kernel.h"
#include "dual_ls_workload.h"

int main() {
    long long out[DUAL_OUT_WORDS];
    for (int i = 0; i < DUAL_OUT_WORDS; ++i) out[i] = 0;

    dual_ls_kernel(DUAL_BATCH_SIZE, out);

    const long long exp_two_checks = (long long)EXPECTED_TWOOPT_CHECKS_PER_ITEM * DUAL_BATCH_SIZE;
    const long long exp_rel_checks = (long long)EXPECTED_RELOCATE_CHECKS_PER_ITEM * DUAL_BATCH_SIZE;
    const long long exp_two_csum_delta = EXPECTED_TWOOPT_DELTA_CHECKSUM_PER_ITEM * (long long)DUAL_BATCH_SIZE;
    const long long exp_two_csum_index = EXPECTED_TWOOPT_INDEX_CHECKSUM_PER_ITEM * (long long)DUAL_BATCH_SIZE;
    const long long exp_rel_csum_delta = EXPECTED_RELOCATE_DELTA_CHECKSUM_PER_ITEM * (long long)DUAL_BATCH_SIZE;
    const long long exp_rel_csum_index = EXPECTED_RELOCATE_INDEX_CHECKSUM_PER_ITEM * (long long)DUAL_BATCH_SIZE;

    bool pass = true;
    pass = pass && (out[OUT_TWO_CHECKS] == exp_two_checks);
    pass = pass && (out[OUT_TWO_BEST_DELTA] == EXPECTED_TWOOPT_BEST_DELTA);
    pass = pass && (out[OUT_TWO_BEST_ROUTE] == EXPECTED_TWOOPT_BEST_ROUTE);
    pass = pass && (out[OUT_TWO_BEST_I] == EXPECTED_TWOOPT_BEST_I);
    pass = pass && (out[OUT_TWO_BEST_J] == EXPECTED_TWOOPT_BEST_J);
    pass = pass && (out[OUT_TWO_CSUM_DELTA] == exp_two_csum_delta);
    pass = pass && (out[OUT_TWO_CSUM_INDEX] == exp_two_csum_index);

    pass = pass && (out[OUT_REL_CHECKS] == exp_rel_checks);
    pass = pass && (out[OUT_REL_BEST_DELTA] == EXPECTED_RELOCATE_BEST_DELTA);
    pass = pass && (out[OUT_REL_BEST_SRC] == EXPECTED_RELOCATE_BEST_SRC);
    pass = pass && (out[OUT_REL_BEST_POS] == EXPECTED_RELOCATE_BEST_POS);
    pass = pass && (out[OUT_REL_BEST_DST] == EXPECTED_RELOCATE_BEST_DST);
    pass = pass && (out[OUT_REL_BEST_INS] == EXPECTED_RELOCATE_BEST_INS);
    pass = pass && (out[OUT_REL_CSUM_DELTA] == exp_rel_csum_delta);
    pass = pass && (out[OUT_REL_CSUM_INDEX] == exp_rel_csum_index);
    pass = pass && (out[OUT_STATUS] == 1);

    std::cout << "============================================================\n";
    std::cout << "Dual LS kernel testbench\n";
    std::cout << "CONFIG=O6-R8\n";
    std::cout << "Two lanes: " << DUAL_TWOOPT_LANES << ", Rel lanes: " << DUAL_RELOCATE_LANES << "\n";
    std::cout << "two_checks=" << out[OUT_TWO_CHECKS] << " expected=" << exp_two_checks << "\n";
    std::cout << "rel_checks =" << out[OUT_REL_CHECKS] << " expected=" << exp_rel_checks << "\n";
    std::cout << "two_csum_delta=" << out[OUT_TWO_CSUM_DELTA] << " expected=" << exp_two_csum_delta << "\n";
    std::cout << "rel_csum_delta =" << out[OUT_REL_CSUM_DELTA] << " expected=" << exp_rel_csum_delta << "\n";
    std::cout << "status=" << out[OUT_STATUS] << "\n";
    std::cout << "============================================================\n";

    if (!pass) {
        std::cerr << "[FAIL] Dual LS kernel mismatch\n";
        for (int i = 0; i < DUAL_OUT_WORDS; ++i) {
            std::cerr << "out[" << i << "]=" << out[i] << "\n";
        }
        return 1;
    }

    std::cout << "[PASS] Dual LS kernel outputs match golden values\n";
    return 0;
}

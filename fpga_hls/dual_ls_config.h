#ifndef DUAL_LS_CONFIG_H
#define DUAL_LS_CONFIG_H

// Frozen final architecture for the all-dataset experiment.
#define DUAL_TWOOPT_LANES 6
#define DUAL_RELOCATE_LANES 8

#define DUAL_OUT_WORDS 20

#define OUT_TWO_CHECKS       0
#define OUT_TWO_BEST_DELTA   1
#define OUT_TWO_BEST_ROUTE   2
#define OUT_TWO_BEST_I       3
#define OUT_TWO_BEST_J       4
#define OUT_TWO_CSUM_DELTA   5
#define OUT_TWO_CSUM_INDEX   6
#define OUT_REL_CHECKS       8
#define OUT_REL_BEST_DELTA   9
#define OUT_REL_BEST_SRC     10
#define OUT_REL_BEST_POS     11
#define OUT_REL_BEST_DST     12
#define OUT_REL_BEST_INS     13
#define OUT_REL_CSUM_DELTA   14
#define OUT_REL_CSUM_INDEX   15
#define OUT_STATUS           18
#define OUT_ACTIVE_BATCH     19

#endif

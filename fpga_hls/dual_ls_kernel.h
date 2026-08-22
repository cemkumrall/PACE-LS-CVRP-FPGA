#ifndef DUAL_LS_KERNEL_H
#define DUAL_LS_KERNEL_H

#include <stdint.h>
#include "dual_ls_config.h"

#ifdef __cplusplus
extern "C" {
#endif

void dual_ls_kernel(int active_batch, long long out[DUAL_OUT_WORDS]);

#ifdef __cplusplus
}
#endif

#endif

#ifndef TRITON_GENERIC_SWIZZLING_H
#define TRITON_GENERIC_SWIZZLING_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "triton/Tools/LinearLayout.h"
#include <cstdint>

namespace mlir::triton::gpu {
llvm::SmallVector<LinearLayout> optimalSwizzlingCandidates(const LinearLayout &src, const LinearLayout &dst,
                                                          int32_t bitwidth);

LinearLayout optimalSwizzling(const LinearLayout &src, const LinearLayout &dst,
                              int32_t bitwidth);

std::pair<int, int> logBankConflicts(const LinearLayout &src,
                                     const LinearLayout &dst,
                                     const LinearLayout &smem,
                                     int32_t bitwidth);
} // namespace mlir::triton::gpu

#endif // TRITON_GENERIC_SWIZZLING_H

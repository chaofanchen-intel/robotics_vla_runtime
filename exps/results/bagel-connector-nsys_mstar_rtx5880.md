### mstar-i2i-512-srv-steps5-r01 (process 2092343, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 148 | 825 | 817 | 2,698 |
| 1,346 | 2,635 | 2,548 | 8,990 |
| 274 | 2,633 | 2,547 | 8,990 |
| 325 | 2,646 | 2,576 | 8,990 |
| 519 | 2,657 | 2,571 | 8,990 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 816.0 | 168 | `ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_tn` |
| 527.6 | 166 | `ampere_bf16_s1688gemm_bf16_128x64_sliced1x2_ldg8_f2f_tn` |
| 336.5 | 168 | `void flashinfer::BatchPrefillWithPagedKVCacheKernel<(bool)1, flashinfer::KernelTraits<(fla` |
| 317.5 | 86 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_tn` |
| 124.7 | 784 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8` |
| 98.3 | 270 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(` |
| 92.5 | 112 | `ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_relu_f2f_tn` |
| 42.2 | 56 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_relu_f2f_tn` |
| 32.6 | 140 | `triton_poi_fused_mul_silu_split_4` |
| 27.0 | 26 | `void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)96, (int)128, (int)64, (int)4, (` |

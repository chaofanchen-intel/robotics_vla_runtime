### mstar-txt-1k-1tok-r01 (process 2283251, GPU 1)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 328 | 137 | 125 | 2,812 |
| 384 | 138 | 123 | 2,812 |
| 392 | 135 | 126 | 2,812 |
| 339 | 135 | 125 | 2,812 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 64.9 | 192 | `fused_moe_kernel` |
| 35.7 | 288 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` |
| 3.7 | 96 | `void flashinfer::BatchPrefillWithPagedKVCacheKernel<(bool)1, flashinfer::KernelTraits<(fla` |
| 3.4 | 144 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(` |
| 3.3 | 99 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bflo` |
| 3.2 | 48 | `ampere_bf16_s1688gemm_bf16_128x64_sliced1x2_ldg8_f2f_tn` |
| 1.4 | 48 | `_gate_up_kernel` |
| 1.4 | 144 | `moe_sum_reduce_kernel` |
| 1.0 | 288 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_a` |
| 0.8 | 48 | `_down_kernel` |
| 0.7 | 96 | `act_and_mul_kernel` |
| 0.7 | 48 | `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn` |
| 0.7 | 144 | `_qk_norm_rope_kv_kernel` |
| 0.7 | 144 | `void flashinfer::PersistentVariableLengthMergeStatesKernel<(unsigned int)8, (unsigned int)` |

### mstar-txt-1k-1tok-r01 (process 2281943, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 328 | 137 | 125 | 2,812 |
| 384 | 138 | 124 | 2,812 |
| 392 | 136 | 123 | 2,812 |
| 338 | 136 | 123 | 2,812 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 65.0 | 192 | `fused_moe_kernel` |
| 34.0 | 288 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` |
| 3.8 | 96 | `void flashinfer::BatchPrefillWithPagedKVCacheKernel<(bool)1, flashinfer::KernelTraits<(fla` |
| 3.5 | 144 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(` |
| 3.3 | 99 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bflo` |
| 3.3 | 48 | `ampere_bf16_s1688gemm_bf16_128x64_sliced1x2_ldg8_f2f_tn` |
| 1.4 | 48 | `_gate_up_kernel` |
| 1.4 | 144 | `moe_sum_reduce_kernel` |
| 1.0 | 288 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_a` |
| 0.8 | 48 | `_down_kernel` |
| 0.7 | 96 | `act_and_mul_kernel` |
| 0.7 | 48 | `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn` |
| 0.7 | 144 | `_qk_norm_rope_kv_kernel` |
| 0.7 | 144 | `void flashinfer::PersistentVariableLengthMergeStatesKernel<(unsigned int)8, (unsigned int)` |

### mstar-txt-1k-1tok-r01 (process 2283537, GPU 2)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 49 | 209 | 26 | 4,374 |
| 46 | 592 | 75 | 11,967 |
| 44 | 136 | 22 | 2,514 |
| 32 | 2,582 | 319 | 51,642 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 87.4 | 8,040 | `_cp_linear_kernel` |
| 58.3 | 2,000 | `fused_moe_kernel` |
| 44.4 | 3,600 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(` |
| 18.4 | 1,000 | `void flashinfer::BatchPrefillWithPagedKVCacheKernel<(bool)1, flashinfer::KernelTraits<(fla` |
| 11.0 | 1,232 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8` |
| 7.8 | 1,320 | `void flashinfer::PersistentVariableLengthMergeStatesKernel<(unsigned int)8, (unsigned int)` |
| 6.9 | 1,320 | `void flashinfer::BatchQKApplyRotaryPosIdsKernel<(bool)0, (unsigned int)128, (unsigned int)` |
| 6.0 | 320 | `_gate_up_kernel` |
| 6.0 | 2,560 | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 5.9 | 2,480 | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 5.7 | 2,640 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_a` |
| 4.6 | 1,920 | `_cp_attend_kernel` |
| 4.1 | 2,640 | `kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object_at__tensorptrbf16gm` |
| 3.8 | 1,480 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bflo` |

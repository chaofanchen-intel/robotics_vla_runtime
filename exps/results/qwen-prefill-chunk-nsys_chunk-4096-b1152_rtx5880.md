### mstar-txt-1k-1tok-r01 (process 2331304, GPU 1)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 398 | 108 | 91 | 1,766 |
| 457 | 102 | 89 | 1,766 |
| 360 | 103 | 89 | 1,766 |
| 355 | 96 | 90 | 1,766 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 35.2 | 96 | `fused_moe_kernel` |
| 33.0 | 192 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` |
| 3.9 | 48 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(` |
| 3.3 | 48 | `void flashinfer::BatchPrefillWithPagedKVCacheKernel<(bool)1, flashinfer::KernelTraits<(fla` |
| 3.1 | 48 | `ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_tn` |
| 2.6 | 98 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bflo` |
| 1.4 | 48 | `_gate_up_kernel` |
| 1.4 | 96 | `moe_sum_reduce_kernel` |
| 0.8 | 192 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_a` |
| 0.8 | 48 | `_down_kernel` |
| 0.7 | 96 | `_qk_norm_rope_kv_kernel` |
| 0.7 | 48 | `act_and_mul_kernel` |
| 0.5 | 96 | `void flashinfer::PersistentVariableLengthMergeStatesKernel<(unsigned int)8, (unsigned int)` |
| 0.3 | 48 | `void flashinfer::BatchPrefillWithPagedKVCacheKernel<(bool)1, flashinfer::KernelTraits<(fla` |

### mstar-txt-1k-1tok-r01 (process 2330061, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 399 | 108 | 90 | 1,766 |
| 456 | 103 | 90 | 1,766 |
| 359 | 103 | 90 | 1,766 |
| 354 | 97 | 89 | 1,766 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 35.3 | 96 | `fused_moe_kernel` |
| 32.4 | 192 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` |
| 4.0 | 48 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(` |
| 3.3 | 48 | `void flashinfer::BatchPrefillWithPagedKVCacheKernel<(bool)1, flashinfer::KernelTraits<(fla` |
| 3.2 | 48 | `ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_tn` |
| 2.6 | 98 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bflo` |
| 1.4 | 48 | `_gate_up_kernel` |
| 1.4 | 96 | `moe_sum_reduce_kernel` |
| 0.8 | 192 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_a` |
| 0.8 | 48 | `_down_kernel` |
| 0.7 | 48 | `act_and_mul_kernel` |
| 0.7 | 96 | `_qk_norm_rope_kv_kernel` |
| 0.5 | 96 | `void flashinfer::PersistentVariableLengthMergeStatesKernel<(unsigned int)8, (unsigned int)` |
| 0.3 | 48 | `void flashinfer::BatchPrefillWithPagedKVCacheKernel<(bool)1, flashinfer::KernelTraits<(fla` |

### mstar-txt-1k-1tok-r01 (process 2331647, GPU 2)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 49 | 210 | 26 | 4,374 |
| 46 | 590 | 76 | 11,967 |
| 48 | 139 | 22 | 2,514 |
| 32 | 2,610 | 319 | 51,642 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 87.4 | 8,040 | `_cp_linear_kernel` |
| 58.3 | 2,000 | `fused_moe_kernel` |
| 44.5 | 3,600 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(` |
| 18.4 | 1,000 | `void flashinfer::BatchPrefillWithPagedKVCacheKernel<(bool)1, flashinfer::KernelTraits<(fla` |
| 11.0 | 1,232 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8` |
| 7.8 | 1,320 | `void flashinfer::PersistentVariableLengthMergeStatesKernel<(unsigned int)8, (unsigned int)` |
| 7.0 | 1,320 | `void flashinfer::BatchQKApplyRotaryPosIdsKernel<(bool)0, (unsigned int)128, (unsigned int)` |
| 6.0 | 320 | `_gate_up_kernel` |
| 6.0 | 2,560 | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 5.9 | 2,480 | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 5.7 | 2,640 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_a` |
| 4.6 | 1,920 | `_cp_attend_kernel` |
| 4.1 | 2,640 | `kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object_at__tensorptrbf16gm` |
| 3.8 | 1,480 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bflo` |

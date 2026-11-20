### mstar-txt-1k-1tok-r01 (process 2306928, GPU 1)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 323 | 107 | 93 | 1,709 |
| 390 | 111 | 85 | 1,709 |
| 425 | 106 | 85 | 1,709 |
| 373 | 107 | 86 | 1,709 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 34.4 | 96 | `fused_moe_kernel` |
| 31.9 | 192 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` |
| 7.2 | 144 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(` |
| 2.6 | 98 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bflo` |
| 2.4 | 48 | `void flashinfer::BatchPrefillWithPagedKVCacheKernel<(bool)1, flashinfer::KernelTraits<(fla` |
| 1.4 | 48 | `_gate_up_kernel` |
| 1.3 | 96 | `moe_sum_reduce_kernel` |
| 0.8 | 192 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_a` |
| 0.8 | 48 | `_down_kernel` |
| 0.6 | 96 | `_qk_norm_rope_kv_kernel` |
| 0.6 | 48 | `act_and_mul_kernel` |
| 0.3 | 48 | `void flashinfer::BatchPrefillWithPagedKVCacheKernel<(bool)1, flashinfer::KernelTraits<(fla` |
| 0.2 | 96 | `_router_topk_kernel` |
| 0.2 | 48 | `void mstar::moe::moe_align_block_size_kernel<int>(const T1 *, int *, int *, int *, int, in` |

### mstar-txt-1k-1tok-r01 (process 2305618, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 323 | 107 | 85 | 1,709 |
| 389 | 112 | 93 | 1,709 |
| 425 | 105 | 85 | 1,709 |
| 372 | 107 | 90 | 1,709 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 35.5 | 192 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` |
| 34.4 | 96 | `fused_moe_kernel` |
| 7.4 | 144 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(` |
| 2.6 | 98 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bflo` |
| 2.5 | 48 | `void flashinfer::BatchPrefillWithPagedKVCacheKernel<(bool)1, flashinfer::KernelTraits<(fla` |
| 1.4 | 48 | `_gate_up_kernel` |
| 1.3 | 96 | `moe_sum_reduce_kernel` |
| 0.8 | 192 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_a` |
| 0.8 | 48 | `_down_kernel` |
| 0.6 | 96 | `_qk_norm_rope_kv_kernel` |
| 0.6 | 48 | `act_and_mul_kernel` |
| 0.3 | 48 | `void flashinfer::BatchPrefillWithPagedKVCacheKernel<(bool)1, flashinfer::KernelTraits<(fla` |
| 0.2 | 96 | `_router_topk_kernel` |
| 0.2 | 48 | `void mstar::moe::moe_align_block_size_kernel<int>(const T1 *, int *, int *, int *, int, in` |

### mstar-txt-1k-1tok-r01 (process 2307208, GPU 2)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 49 | 211 | 25 | 4,374 |
| 46 | 594 | 75 | 11,967 |
| 44 | 138 | 22 | 2,514 |
| 32 | 2,597 | 319 | 51,642 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 87.4 | 8,040 | `_cp_linear_kernel` |
| 58.2 | 2,000 | `fused_moe_kernel` |
| 44.5 | 3,600 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(` |
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

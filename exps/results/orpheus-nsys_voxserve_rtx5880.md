### voxserve-b1-r01 (process 968666, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 18,851 | 53 | 48 | 516 |
| 1,409 | 2,742 | 2,291 | 140,276 |
| 244 | 2,638 | 2,211 | 133,197 |
| 303 | 5,585 | 4,690 | 284,062 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 3,142.2 | 44,491 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bflo` |
| 955.9 | 58,627 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bflo` |
| 108.7 | 524 | `void flashinfer::sampling::TopPSamplingFromProbKernel<(unsigned int)1024, (cub::BlockScanA` |
| 98.1 | 14,685 | `void flashinfer::BatchQKApplyRotaryPosIdsKernel<(bool)0, (unsigned int)128, (unsigned int)` |
| 78.7 | 14,656 | `void flashinfer::BatchPrefillWithPagedKVCacheKernel<flashinfer::KernelTraits<(flashinfer::` |
| 71.6 | 29,369 | `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_ker` |
| 45.3 | 29,893 | `void flashinfer::norm::RMSNormKernel<(unsigned int)8, __nv_bfloat16>(T2 *, T2 *, T2 *, uns` |
| 29.9 | 168 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(` |
| 26.9 | 29,368 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<c10::BF` |
| 23.2 | 14,684 | `void flashinfer::PersistentVariableLengthMergeStatesKernel<(unsigned int)8, (unsigned int)` |
| 17.6 | 14,684 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::silu_kernel(` |
| 13.8 | 14,684 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<c10::BFlo` |
| 13.7 | 524 | `void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float, c10::BFloat1` |
| 10.5 | 2,698 | `void at::native::<unnamed>::weight_norm_fwd_first_dim_kernel<float, float>(T1 *, T2 *, con` |

### voxserve-b16-r01 (process 980401, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 28 | 54 | 52 | 534 |
| 17 | 58 | 56 | 534 |
| 15 | 5,986 | 4,979 | 303,415 |
| 223 | 8,730 | 7,377 | 399,153 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 3,599.6 | 94,668 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8` |
| 713.5 | 10,099 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bflo` |
| 627.5 | 2,688 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(` |
| 467.9 | 420 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align2` |
| 263.3 | 16,851 | `void flashinfer::BatchPrefillWithPagedKVCacheKernel<flashinfer::KernelTraits<(flashinfer::` |
| 217.9 | 13,308 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bflo` |
| 209.5 | 617 | `void flashinfer::sampling::TopPSamplingFromProbKernel<(unsigned int)1024, (cub::BlockScanA` |
| 187.4 | 448 | `ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_tn` |
| 142.0 | 16 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_bf16_128x256_32x3_tn_align2>(T1:` |
| 129.4 | 17,299 | `void flashinfer::BatchQKApplyRotaryPosIdsKernel<(bool)0, (unsigned int)128, (unsigned int)` |
| 100.4 | 34,598 | `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_ker` |
| 69.0 | 63 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1_tn_align2` |
| 67.2 | 5,096 | `void at::native::<unnamed>::conv_depthwise2d_forward_kernel_generic<float, int>(at::Generi` |
| 64.6 | 11,368 | `fused_add_reciprocal_mul_sin_pow_mul_add` |

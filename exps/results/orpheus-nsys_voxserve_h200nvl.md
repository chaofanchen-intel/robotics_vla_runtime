### voxserve-b1-r01 (process 3534387, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 62,474 | 19 | 15 | 516 |
| 45,931 | 36 | 15 | 522 |
| 264 | 1,624 | 1,138 | 187,243 |
| 63 | 2,899 | 1,982 | 329,514 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 602.7 | 67,546 | `nvjet_tst_64x8_64x16_4x1_v_bz_splitK_TNT` |
| 426.6 | 27,018 | `nvjet_tst_64x8_64x16_2x1_v_bz_TNT` |
| 149.7 | 482 | `void gemv2T_kernel_val<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (int)` |
| 145.4 | 13,537 | `void flashinfer::BatchQKApplyRotaryPosIdsKernel<(bool)0, (unsigned int)128, (unsigned int)` |
| 135.3 | 13,509 | `void flashinfer::BatchPrefillWithPagedKVCacheKernel<flashinfer::KernelTraits<(flashinfer::` |
| 117.8 | 67,545 | `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float, __n` |
| 101.3 | 27,074 | `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_ker` |
| 61.9 | 27,558 | `void flashinfer::norm::RMSNormKernel<(unsigned int)8, __nv_bfloat16>(T2 *, T2 *, T2 *, uns` |
| 42.7 | 483 | `void flashinfer::sampling::TopPSamplingFromProbKernel<(unsigned int)1024, (cub::_V_300304_` |
| 39.3 | 27,074 | `void at::native::vectorized_elementwise_kernel<(int)8, at::native::CUDAFunctor_add<c10::BF` |
| 33.4 | 13,509 | `void flashinfer::PersistentVariableLengthMergeStatesKernel<(unsigned int)8, (unsigned int)` |
| 25.0 | 13,537 | `void at::native::vectorized_elementwise_kernel<(int)8, at::native::<unnamed>::silu_kernel(` |
| 22.1 | 13,537 | `void at::native::vectorized_elementwise_kernel<(int)8, at::native::BinaryFunctor<c10::BFlo` |
| 19.1 | 483 | `void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float, c10::BFloat1` |

### voxserve-b16-r01 (process 3538538, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 1,642 | 34 | 15 | 522 |
| 163 | 4,172 | 2,983 | 437,165 |
| 82 | 4,217 | 2,711 | 390,296 |
| 68 | 4,619 | 3,275 | 482,109 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 493.8 | 53,900 | `nvjet_tst_64x16_64x16_4x1_v_bz_splitK_TNT` |
| 351.6 | 21,560 | `nvjet_tst_64x16_64x16_2x1_v_bz_TNT` |
| 282.0 | 31,535 | `nvjet_tst_64x8_64x16_4x1_v_bz_splitK_TNT` |
| 215.2 | 17,535 | `void flashinfer::BatchQKApplyRotaryPosIdsKernel<(bool)0, (unsigned int)128, (unsigned int)` |
| 205.7 | 17,087 | `void flashinfer::BatchPrefillWithPagedKVCacheKernel<flashinfer::KernelTraits<(flashinfer::` |
| 200.0 | 12,614 | `nvjet_tst_64x8_64x16_2x1_v_bz_TNT` |
| 159.2 | 85,435 | `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float, __n` |
| 151.9 | 35,070 | `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_ker` |
| 137.0 | 385 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_bf16_256x64_64x3_tn_align2>(T1::` |
| 82.0 | 35,697 | `void flashinfer::norm::RMSNormKernel<(unsigned int)8, __nv_bfloat16>(T2 *, T2 *, T2 *, uns` |
| 81.7 | 626 | `void flashinfer::sampling::TopPSamplingFromProbKernel<(unsigned int)1024, (cub::_V_300304_` |
| 76.7 | 11,339 | `fused_add_reciprocal_mul_sin_pow_mul_add` |
| 67.6 | 5,083 | `void at::native::<unnamed>::conv_depthwise2d_forward_kernel_generic<float, int>(at::Generi` |
| 64.2 | 16 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_bf16_128x256_64x3_tn_align2>(T1:` |

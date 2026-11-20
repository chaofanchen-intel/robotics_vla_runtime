### mstar-b1-r01 (process 3531561, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 24 | 153 | 75 | 9,189 |
| 36 | 1,057 | 603 | 73,563 |
| 81 | 870 | 797 | 97,131 |
| 73 | 2,454 | 1,948 | 235,400 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 525.3 | 19,628 | `triton_red_fused_mm_t_1` |
| 329.6 | 19,628 | `triton_red_fused_add_mm_mul_silu_split_t_2` |
| 216.2 | 19,656 | `void flashinfer::BatchQKApplyRotaryPosIdsKernel<(bool)0, (unsigned int)128, (unsigned int)` |
| 200.0 | 19,628 | `triton_red_fused_mm_t_0` |
| 180.1 | 19,628 | `void flashinfer::BatchPrefillWithPagedKVCacheKernel<(bool)1, flashinfer::KernelTraits<(fla` |
| 137.6 | 19,628 | `triton_red_fused_mm_t_view_0` |
| 101.7 | 40,014 | `kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object_at__tensorptrbf16gm` |
| 48.6 | 19,628 | `void flashinfer::PersistentVariableLengthMergeStatesKernel<(unsigned int)8, (unsigned int)` |
| 38.9 | 702 | `void flashinfer::sampling::TopPSamplingFromProbKernel<(unsigned int)1024, (cub::_V_300600_` |
| 32.8 | 702 | `triton_red_fused_mm_slice_t_0` |
| 24.6 | 19,656 | `_kv_scatter_nhd_kernel` |
| 23.0 | 19,656 | `triton_poi_fused_add_0` |
| 16.0 | 702 | `_split_partials_kernel` |
| 13.6 | 1,404 | `void at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<unnamed>::TensorListMet` |

### mstar-b16-r01 (process 3532980, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 23 | 199 | 94 | 11,175 |
| 36 | 3,066 | 1,912 | 221,404 |
| 101 | 2,601 | 1,761 | 201,851 |
| 96 | 2,499 | 2,192 | 251,821 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 424.0 | 15,316 | `triton_tem_fused_mm_t_1` |
| 220.5 | 17,668 | `void flashinfer::BatchPrefillWithPagedKVCacheKernel<(bool)1, flashinfer::KernelTraits<(fla` |
| 212.4 | 17,724 | `void flashinfer::BatchQKApplyRotaryPosIdsKernel<(bool)0, (unsigned int)128, (unsigned int)` |
| 198.7 | 11,396 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bx_splitK_TNT` |
| 155.0 | 14,560 | `triton_tem_fused_mm_t_view_0` |
| 133.3 | 11,396 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` |
| 115.7 | 36,081 | `kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object_at__tensorptrbf16gm` |
| 67.0 | 3,892 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bx_splitK_TNT` |
| 62.7 | 2,380 | `triton_red_fused_mm_t_1` |
| 56.4 | 633 | `void flashinfer::sampling::TopPSamplingFromProbKernel<(unsigned int)1024, (cub::_V_300600_` |
| 48.0 | 17,668 | `void flashinfer::PersistentVariableLengthMergeStatesKernel<(unsigned int)8, (unsigned int)` |
| 40.2 | 16,072 | `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float, __n` |
| 40.2 | 1,740 | `fused_add_reciprocal_mul_sin_pow_mul_add` |
| 39.9 | 2,380 | `triton_red_fused_add_mm_mul_silu_split_t_2` |

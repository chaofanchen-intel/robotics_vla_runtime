### mstar-i2i-512-r01 (process 3730160, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| - | 4,461 | 4,364 | 65,990 |
| 107,438 | 5,111 | 4,792 | 72,571 |
| 146,581 | 5,116 | 4,800 | 72,571 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 1,758.4 | 1,372 | `nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN` |
| 1,049.5 | 2,744 | `nvjet_sm90_tst_168x128_64x5_1x2_h_bz_TNN` |
| 847.1 | 1,428 | `void flashinfer::PrefillWithKVCacheKernel<flashinfer::SparseCollectiveMainloop<PagedParams` |
| 232.8 | 1,372 | `nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_bias_TNN` |
| 114.7 | 1,400 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` |
| 112.7 | 1,400 | `triton_poi_fused_mul_silu_split_4` |
| 59.3 | 1,400 | `nvjet_sm90_tst_256x8_64x6_4x1_v_bz_splitK_TNT` |
| 55.9 | 28 | `nvjet_sm90_tst_192x208_64x4_1x2_h_bz_coopB_TNT` |
| 54.3 | 5,622 | `triton_poi_fused_index_put_3` |
| 43.0 | 1,428 | `void flashinfer::BatchQKApplyRotaryPosIdsHeadParallelismKernel<(bool)0, (unsigned int)128,` |

### vllm-omni-single-stage-i2i-512-r01 (process 3736278, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| - | 410 | 129 | 5,345 |
| 77 | 6,325 | 5,102 | 121,568 |
| 92,700 | 6,687 | 5,545 | 130,455 |
| 140,411 | 6,793 | 5,536 | 130,455 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 1,658.4 | 1,372 | `nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN` |
| 844.3 | 1,372 | `nvjet_sm90_tst_168x128_64x5_1x2_h_bz_TNN` |
| 763.6 | 2,800 | `mot_unified_gemm_kernel` |
| 515.4 | 1,372 | `void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::Col` |
| 375.3 | 2,884 | `triton_poi_fused_mul_silu_slice_0` |
| 181.0 | 5,702 | `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_ker` |
| 156.0 | 19,432 | `void at::native::<unnamed>::CatArrayBatchedCopy_vectorized<at::native::<unnamed>::OpaqueTy` |
| 116.6 | 1,400 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` |
| 89.7 | 14,045 | `void at::native::vectorized_gather_kernel<(int)16, long>(char *, char *, T2 *, int, long, ` |
| 76.6 | 2,968 | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at:` |

### vllm-omni-two-stage-i2i-512-r01 (process 3733020, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| - | 5,886 | 5,124 | 121,568 |
| 106,706 | 6,104 | 5,243 | 124,088 |
| 215,217 | 6,035 | 5,266 | 124,088 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 1,662.0 | 1,372 | `nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN` |
| 848.2 | 1,372 | `nvjet_sm90_tst_168x128_64x5_1x2_h_bz_TNN` |
| 763.4 | 2,744 | `mot_unified_gemm_kernel` |
| 523.2 | 1,372 | `void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::Col` |
| 366.1 | 2,744 | `triton_poi_fused_mul_silu_slice_0` |
| 181.1 | 5,586 | `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_ker` |
| 168.4 | 20,580 | `void at::native::<unnamed>::CatArrayBatchedCopy_vectorized<at::native::<unnamed>::OpaqueTy` |
| 114.6 | 1,372 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` |
| 89.3 | 13,818 | `void at::native::vectorized_gather_kernel<(int)16, long>(char *, char *, T2 *, int, long, ` |
| 74.1 | 2,744 | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at:` |

### vllm-omni-two-stage-i2i-512-r01 (process 3733442, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 590 | 555 | 480 | 6,552 |
| 112,402 | 625 | 495 | 7,016 |
| 45 | 595 | 484 | 6,785 |
| 219,782 | 625 | 496 | 7,016 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 288.2 | 28 | `kernel_unified_attention` |
| 53.4 | 28 | `nvjet_sm90_tst_192x208_64x4_1x2_h_bz_coopB_TNT` |
| 32.8 | 56 | `nvjet_sm90_tst_192x192_64x4_1x2_h_bz_coopB_TNN` |
| 19.5 | 395 | `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_ker` |
| 13.4 | 56 | `triton_poi_fused_mul_silu_slice_0` |
| 11.0 | 28 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` |
| 10.2 | 450 | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 6.9 | 56 | `nvjet_sm90_tst_256x128_64x4_2x4_h_bz_coopA_TNT` |
| 6.2 | 28 | `nvjet_sm90_tst_128x224_64x4_2x1_v_bz_coopA_bias_TNT` |
| 6.2 | 227 | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_noc` |

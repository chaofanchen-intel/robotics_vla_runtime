### mstar-t2i-256-r01 (process 3543642, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 79 | 1,313 | 336 | 42,893 |
| 79 | 1,360 | 336 | 42,893 |
| 78 | 1,389 | 337 | 42,893 |
| 81 | 1,727 | 336 | 42,893 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 94.2 | 3,096 | `nvjet_sm90_tst_192x64_64x6_1x2_h_bz_TNT` |
| 50.2 | 1,548 | `nvjet_sm90_tst_128x64_64x8_1x2_h_bx_splitK_TNT` |
| 46.7 | 3,096 | `nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT` |
| 20.9 | 3,096 | `nvjet_sm90_tst_64x64_64x13_4x2_h_bz_splitK_TNT` |
| 18.2 | 1,800 | `void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::Col` |
| 17.3 | 576 | `nvjet_sm90_tst_128x64_64x8_4x1_v_bz_TNT` |
| 14.7 | 5,601 | `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float, __n` |
| 9.7 | 3,600 | `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0` |
| 7.7 | 252 | `nvjet_sm90_tst_96x64_64x10_2x1_v_bx_splitK_TNN` |
| 7.6 | 576 | `nvjet_sm90_tst_64x32_64x16_1x2_h_bz_TNT` |
| 6.0 | 1,800 | `triton_poi_fused_mul_silu_1` |
| 6.0 | 3,600 | `void at::native::<unnamed>::CatArrayBatchedCopy_vectorized<at::native::<unnamed>::OpaqueTy` |

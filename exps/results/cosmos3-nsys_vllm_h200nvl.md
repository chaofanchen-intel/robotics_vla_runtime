### vllm-omni-t2i-256-r01 (process 3541889, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 86 | 1,698 | 567 | 73,766 |
| 92 | 1,715 | 567 | 73,766 |
| 88 | 1,695 | 566 | 73,766 |
| 89 | 1,716 | 566 | 73,766 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 197.4 | 6,552 | `nvjet_sm90_tst_128x64_64x8_4x1_v_bz_TNT` |
| 100.2 | 3,276 | `nvjet_sm90_tst_96x64_64x10_2x1_v_bz_splitK_TNN` |
| 85.8 | 6,552 | `nvjet_sm90_tst_64x32_64x16_1x2_h_bz_TNT` |
| 42.7 | 6,552 | `nvjet_sm90_tst_64x32_64x16_4x2_h_bz_splitK_TNT` |
| 27.6 | 3,276 | `void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::Col` |
| 23.1 | 10,188 | `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float, __n` |
| 10.0 | 3,276 | `triton_per_fused__flash_attn_forward__fused_rms_norm__unsafe_view_add_cat_mul_neg_slice_vi` |
| 8.2 | 3,276 | `triton_poi_fused__unsafe_view_mul_silu_6` |
| 7.1 | 3,276 | `triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_5` |
| 7.0 | 3,276 | `triton_per_fused__fused_rms_norm__unsafe_view_add_cat_mul_neg_slice_view_2` |
| 6.4 | 3,276 | `triton_red_fused__to_copy_mean_mul_pow_rsqrt_0` |
| 5.5 | 3,276 | `triton_poi_fused__unsafe_view_add_7` |

### vllm-omni-t2i-256-r01 (process 3541586, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|

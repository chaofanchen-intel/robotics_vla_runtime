### mstar-droid-46row-r01 (process 4191338, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 170 | 74 | 34 | 4,339 |
| 148 | 73 | 34 | 4,339 |
| 145 | 76 | 34 | 4,339 |
| 152 | 74 | 34 | 4,339 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 8.4 | 198 | `void flashinfer::PrefillWithKVCacheKernel<flashinfer::SparseCollectiveMainloop<PagedParams` |
| 3.4 | 18 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` |
| 2.3 | 388 | `triton_poi_fused__to_copy_0` |
| 2.3 | 135 | `sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_tilesize128x64x32_warpgroupsize1x1x1_execute_segmen` |
| 2.0 | 370 | `triton_tem_fused__to_copy_addmm_t_2` |
| 1.8 | 18 | `triton_tem_fused_add_gelu_mm_mul_split_t_2` |
| 1.5 | 27 | `fmha_cutlassF_f32_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<float, cu` |
| 1.4 | 198 | `triton_tem_fused_mm_t_view_0` |
| 1.3 | 180 | `triton_tem_fused_mm_t_4` |
| 1.1 | 180 | `nvjet_sm90_tst_32x64_64x16_4x1_v_bz_splitK_TNN` |
| 1.0 | 198 | `void flashinfer::BatchQKApplyRotaryPosIdsKernel<(bool)0, (unsigned int)256, (unsigned int)` |
| 0.9 | 28 | `sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_tilesize128x128x32_warpgroupsize1x1x1_execute_segme` |
| 0.9 | 180 | `triton_tem_fused_mm_t_0` |
| 0.6 | 370 | `_adarms_norm_kernel` |

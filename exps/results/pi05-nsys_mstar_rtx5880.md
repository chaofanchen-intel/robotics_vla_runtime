### mstar-droid-46row-r01 (process 913011, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 315 | 80 | 60 | 4,915 |
| 279 | 80 | 60 | 4,915 |
| 347 | 81 | 58 | 4,915 |
| 365 | 84 | 59 | 4,915 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 11.6 | 18 | `triton_tem_fused__to_copy_add_mean_mm_mul_pow_rsqrt_t_1` |
| 7.2 | 18 | `ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_tn` |
| 5.9 | 388 | `triton_poi_fused__to_copy_0` |
| 5.2 | 540 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(` |
| 3.9 | 180 | `triton_tem_fused_mm_t_4` |
| 3.2 | 198 | `void flashinfer::BatchPrefillWithPagedKVCacheKernel<(bool)1, flashinfer::KernelTraits<(fla` |
| 3.1 | 108 | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4>(T1::Params)` |
| 2.5 | 27 | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_32x3_tn_align4>(T1::Params)` |
| 2.3 | 370 | `triton_tem_fused__to_copy_addmm_t_2` |
| 2.0 | 27 | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_16x5_tn_align4>(T1::Params)` |
| 1.5 | 18 | `triton_poi_fused_gelu_mul_split_2` |
| 1.4 | 27 | `fmha_cutlassF_f32_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<float, cu` |
| 1.2 | 18 | `triton_tem_fused__to_copy_mm_t_1` |
| 1.0 | 18 | `triton_tem_fused_mm_t_view_0` |

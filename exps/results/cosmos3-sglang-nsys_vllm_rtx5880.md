### vllm-omni-t2i-256-r01 (process 3014439, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 463 | 3,724 | 1,815 | 83,587 |
| 382 | 3,693 | 1,814 | 83,587 |
| 284 | 3,696 | 1,815 | 83,587 |
| 398 | 3,690 | 1,813 | 83,587 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 830.5 | 6,552 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_tn` |
| 536.6 | 10,135 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(` |
| 286.7 | 6,552 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_32x6_tn_align8>` |
| 25.4 | 13,392 | `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, __nv_bfloat16, __nv_bfloat16, fl` |
| 21.1 | 3,276 | `void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)32, (int)4, ` |
| 19.3 | 144 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_64x1_tn_align8>` |
| 10.0 | 3,276 | `triton_poi_fused__unsafe_view_mul_silu_1` |
| 9.2 | 6,552 | `triton_poi_fused_cat_0` |
| 7.4 | 91 | `void gemv2T_kernel_val<int, int, float, float, float, float, (int)128, (int)16, (int)4, (i` |
| 6.8 | 3,276 | `triton_red_fused__to_copy_mean_mul_pow_rsqrt_0` |
| 6.6 | 3,276 | `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0` |
| 5.7 | 3,276 | `triton_per_fused__fused_rms_norm__unsafe_view_add_cat_mul_neg_slice_view_0` |

### vllm-omni-t2i-256-r01 (process 3012808, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|

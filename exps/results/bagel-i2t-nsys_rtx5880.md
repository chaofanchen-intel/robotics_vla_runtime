### mstar-i2t-512-r01

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 469 | 2,290 | 2,225 | 49,557 |
| 423 | 2,264 | 2,227 | 49,557 |
| 422 | 2,273 | 2,228 | 49,557 |
| 401 | 2,263 | 2,226 | 49,557 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 1,085.8 | 3,556 | `triton_red_fused_mm_t_1` |
| 550.6 | 3,556 | `triton_red_fused_add_mm_mul_silu_split_t_2` |
| 156.3 | 128 | `triton_red_fused_mm_t_0` |
| 143.9 | 3,584 | `triton_tem_fused_addmm_t_0` |
| 107.5 | 3,556 | `triton_red_fused_mm_t_view_0` |
| 59.4 | 3,556 | `void flashinfer::BatchPrefillWithPagedKVCacheKernel<(bool)1, flashinfer::KernelTraits<(fla` |
| 19.3 | 26 | `void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)96, (int)128, (int)64, (int)4, (` |
| 12.0 | 7,296 | `kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object_at__tensorptrbf16gm` |

### vllm-omni-two-stage-i2t-512-r01

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 366 | 2,939 | 2,922 | 44,001 |
| 440 | 2,931 | 2,917 | 44,001 |
| 245 | 2,929 | 2,911 | 44,001 |
| 302 | 2,931 | 2,916 | 44,001 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 1,647.0 | 7,112 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bflo` |
| 411.9 | 7,240 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bflo` |
| 316.5 | 3,584 | `kernel_unified_attention` |
| 296.6 | 56 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_tn` |
| 142.0 | 28 | `ampere_bf16_s1688gemm_bf16_128x64_sliced1x2_ldg8_f2f_tn` |
| 33.2 | 28 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_relu_f2f_tn` |
| 20.6 | 3,584 | `triton_poi_fused_mul_silu_slice_1` |
| 11.3 | 3,584 | `triton_red_fused_fused_add_rms_norm_2` |

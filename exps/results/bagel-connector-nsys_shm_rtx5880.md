### vllm-omni-two-stage-i2i-512-steps5-r01 (process 2508829, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 120,175 | 1,876 | 1,744 | 10,239 |
| 5,252 | 1,834 | 1,781 | 10,239 |
| 5,285 | 1,838 | 1,783 | 10,239 |
| 5,240 | 1,839 | 1,779 | 10,239 |
| 5,351 | 1,838 | 1,779 | 10,239 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 718.4 | 112 | `ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_tn` |
| 357.4 | 112 | `ampere_bf16_s1688gemm_bf16_128x64_sliced1x2_ldg8_f2f_tn` |
| 246.7 | 112 | `void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)32, (int)4, ` |
| 189.9 | 224 | `mot_unified_gemm_kernel` |
| 64.5 | 224 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8` |
| 46.5 | 224 | `triton_poi_fused_mul_silu_slice_0` |
| 28.4 | 1,680 | `void at::native::<unnamed>::CatArrayBatchedCopy_vectorized<at::native::<unnamed>::OpaqueTy` |
| 17.1 | 456 | `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_ker` |
| 9.5 | 9 | `sm86_xmma_fprop_implicit_gemm_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_tilesize256x128x64_stage2` |
| 8.7 | 254 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<c10::BF` |

### vllm-omni-two-stage-i2i-512-steps5-r01 (process 2512062, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 45,371 | 1,153 | 1,013 | 7,321 |
| 474 | 1,012 | 953 | 6,552 |
| 5,670 | 2,202 | 2,007 | 14,108 |
| 5,010 | 2,116 | 1,969 | 13,800 |
| 4,863 | 851 | 763 | 5,546 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 254.8 | 21 | `kernel_unified_attention` |
| 185.0 | 41 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_tn` |
| 80.5 | 20 | `ampere_bf16_s1688gemm_bf16_128x64_sliced1x2_ldg8_f2f_tn` |
| 51.6 | 40 | `ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_tn` |
| 38.4 | 96 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_relu_f2f_tn` |
| 20.5 | 26 | `void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)96, (int)128, (int)64, (int)4, (` |
| 15.4 | 40 | `triton_poi_fused_mul_silu_slice_0` |
| 15.1 | 291 | `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_ker` |
| 9.7 | 27 | `ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_relu_f2f_tn` |
| 9.5 | 168 | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_noc` |

### vllm-omni-two-stage-i2i-512-steps5-r01: Thinker (process 2512062) and DiT (process 2508829), one row per DiT burst

| DiT burst | to Thinker's first kernel | Thinker first..last | Thinker GPU busy | Thinker launches | Thinker last -> DiT first | of which KV D2H done -> first H2D | first H2D -> first DiT kernel | DiT span | DiT busy | DiT launches |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | - | 5,706 | 2,090 | 15,876 | 915 | - | - | 1,876 | 1,744 | 10,239 |
| 1 | 452 | 3,958 | 2,061 | 15,280 | 842 | - | - | 1,834 | 1,781 | 10,239 |
| 2 | 478 | 3,970 | 2,070 | 15,280 | 838 | - | - | 1,838 | 1,783 | 10,239 |
| 3 | 435 | 851 | 763 | 5,546 | 3,953 | - | - | 1,839 | 1,779 | 10,239 |
| 4 | - | - | - | 0 | - | - | - | 1,838 | 1,779 | 10,239 |

**Truncated Thinker trace**: its last kernel is 12,982 ms before the DiT's last, so the later rows have a missing or partial Thinker window; read only the rows that are followed by a complete one.

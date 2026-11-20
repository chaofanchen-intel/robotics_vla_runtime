### vllm-omni-two-stage-i2i-512-steps5-r01 (process 2535984, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 80,487 | 1,882 | 1,755 | 10,239 |
| 3,051 | 1,823 | 1,769 | 10,239 |
| 3,125 | 1,831 | 1,777 | 10,239 |
| 2,985 | 1,835 | 1,782 | 10,239 |
| 3,099 | 1,823 | 1,769 | 10,239 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 713.3 | 112 | `ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_tn` |
| 355.1 | 112 | `ampere_bf16_s1688gemm_bf16_128x64_sliced1x2_ldg8_f2f_tn` |
| 245.5 | 112 | `void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)32, (int)4, ` |
| 188.8 | 224 | `mot_unified_gemm_kernel` |
| 64.4 | 224 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8` |
| 46.3 | 224 | `triton_poi_fused_mul_silu_slice_0` |
| 28.4 | 1,680 | `void at::native::<unnamed>::CatArrayBatchedCopy_vectorized<at::native::<unnamed>::OpaqueTy` |
| 17.1 | 456 | `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_ker` |
| 9.7 | 9 | `sm86_xmma_fprop_implicit_gemm_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_tilesize256x128x64_stage2` |
| 8.8 | 254 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<c10::BF` |

### vllm-omni-two-stage-i2i-512-steps5-r01 (process 2539302, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 44,263 | 2,463 | 2,132 | 15,876 |
| 2,647 | 2,302 | 2,071 | 15,280 |
| 2,676 | 2,280 | 2,046 | 15,280 |
| 2,618 | 2,203 | 2,015 | 14,972 |
| 2,609 | 1,100 | 1,001 | 7,036 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 331.9 | 27 | `kernel_unified_attention` |
| 256.8 | 54 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_tn` |
| 110.8 | 27 | `ampere_bf16_s1688gemm_bf16_128x64_sliced1x2_ldg8_f2f_tn` |
| 70.6 | 54 | `ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_tn` |
| 46.3 | 108 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_relu_f2f_tn` |
| 21.6 | 26 | `void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)96, (int)128, (int)64, (int)4, (` |
| 20.7 | 54 | `triton_poi_fused_mul_silu_slice_0` |
| 20.1 | 379 | `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_ker` |
| 12.3 | 433 | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 12.2 | 218 | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_noc` |

### vllm-omni-two-stage-i2i-512-steps5-r01: Thinker (process 2539302) and DiT (process 2535984), one row per DiT burst

| DiT burst | to Thinker's first kernel | Thinker first..last | Thinker GPU busy | Thinker launches | Thinker last -> DiT first | of which KV D2H done -> first H2D | first H2D -> first DiT kernel | DiT span | DiT busy | DiT launches |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | - | 2,463 | 2,132 | 15,876 | 257 | 149 | 153 | 1,882 | 1,755 | 10,239 |
| 1 | 508 | 2,302 | 2,071 | 15,280 | 241 | 137 | 147 | 1,823 | 1,769 | 10,239 |
| 2 | 612 | 2,280 | 2,046 | 15,280 | 233 | 130 | 147 | 1,831 | 1,777 | 10,239 |
| 3 | 447 | 2,310 | 2,067 | 15,280 | 228 | 139 | 133 | 1,835 | 1,782 | 10,239 |
| 4 | 546 | 1,100 | 1,001 | 7,036 | 1,453 | - | - | 1,823 | 1,769 | 10,239 |

**Truncated Thinker trace**: its last kernel is 3,276 ms before the DiT's last, so the later rows have a missing or partial Thinker window; read only the rows that are followed by a complete one.

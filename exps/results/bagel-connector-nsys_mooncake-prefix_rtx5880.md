### vllm-omni-two-stage-i2i-512-steps5-r01 (process 124335, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 83,252 | 1,869 | 1,746 | 10,239 |
| 1,973 | 1,806 | 1,750 | 10,239 |
| 1,972 | 1,813 | 1,755 | 10,239 |
| 1,907 | 1,836 | 1,776 | 10,239 |
| 1,980 | 1,826 | 1,771 | 10,239 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 715.7 | 112 | `ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_tn` |
| 356.1 | 112 | `ampere_bf16_s1688gemm_bf16_128x64_sliced1x2_ldg8_f2f_tn` |
| 245.2 | 112 | `void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)32, (int)4, ` |
| 189.2 | 224 | `mot_unified_gemm_kernel` |
| 64.4 | 224 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8` |
| 46.4 | 224 | `triton_poi_fused_mul_silu_slice_0` |
| 28.4 | 1,680 | `void at::native::<unnamed>::CatArrayBatchedCopy_vectorized<at::native::<unnamed>::OpaqueTy` |
| 17.1 | 456 | `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_ker` |
| 9.0 | 9 | `sm86_xmma_fprop_implicit_gemm_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_tilesize256x128x64_stage2` |
| 8.8 | 254 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<c10::BF` |

### vllm-omni-two-stage-i2i-512-steps5-r01 (process 127670, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 45,856 | 1,077 | 967 | 6,986 |
| 1,015 | 1,286 | 1,078 | 8,582 |
| 2,547 | 1,311 | 1,123 | 8,933 |
| 2,584 | 1,213 | 1,064 | 8,625 |
| 2,421 | 928 | 844 | 6,074 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 275.3 | 23 | `kernel_unified_attention` |
| 210.9 | 46 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_tn` |
| 93.2 | 23 | `ampere_bf16_s1688gemm_bf16_128x64_sliced1x2_ldg8_f2f_tn` |
| 59.9 | 46 | `ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_tn` |
| 41.1 | 100 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_relu_f2f_tn` |
| 20.8 | 26 | `void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)96, (int)128, (int)64, (int)4, (` |
| 17.7 | 46 | `triton_poi_fused_mul_silu_slice_0` |
| 16.9 | 323 | `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_ker` |
| 10.3 | 368 | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 10.3 | 185 | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_noc` |

### vllm-omni-two-stage-i2i-512-steps5-r01: Thinker (process 127670) and DiT (process 124335), one row per DiT burst

| DiT burst | to Thinker's first kernel | Thinker first..last | Thinker GPU busy | Thinker launches | Thinker last -> DiT first | of which KV D2H done -> first H2D | first H2D -> first DiT kernel | DiT span | DiT busy | DiT launches |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | - | 3,485 | 2,095 | 15,876 | 258 | 148 | 155 | 1,869 | 1,746 | 10,239 |
| 1 | 420 | 1,311 | 1,123 | 8,933 | 241 | 145 | 139 | 1,806 | 1,750 | 10,239 |
| 2 | 431 | 1,319 | 1,116 | 8,933 | 222 | 125 | 139 | 1,813 | 1,755 | 10,239 |
| 3 | 386 | 928 | 844 | 6,074 | 593 | - | - | 1,836 | 1,776 | 10,239 |
| 4 | - | - | - | 0 | - | - | - | 1,826 | 1,771 | 10,239 |

**Truncated Thinker trace**: its last kernel is 6,235 ms before the DiT's last, so the later rows have a missing or partial Thinker window; read only the rows that are followed by a complete one.

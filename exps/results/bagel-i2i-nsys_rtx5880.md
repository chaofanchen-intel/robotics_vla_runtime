### mstar-i2i-512-r01 (GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| - | 20,293 | 20,226 | 63,358 |
| 1,473 | 22,579 | 22,419 | 69,650 |
| 380 | 22,917 | 22,747 | 69,650 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 9,661.7 | 1,428 | `ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_tn` |
| 4,938.0 | 1,426 | `ampere_bf16_s1688gemm_bf16_128x64_sliced1x2_ldg8_f2f_tn` |
| 3,190.8 | 1,428 | `void flashinfer::BatchPrefillWithPagedKVCacheKernel<(bool)1, flashinfer::KernelTraits<(fla` |
| 1,206.1 | 1,372 | `ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_relu_f2f_tn` |
| 1,077.2 | 1,800 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(` |
| 970.7 | 5,824 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8` |
| 480.3 | 1,400 | `triton_poi_fused_mul_silu_split_4` |
| 330.3 | 86 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_tn` |
| 125.8 | 5,622 | `triton_poi_fused_index_put_3` |
| 83.8 | 1,372 | `triton_poi_fused_add_6` |

### vllm-omni-single-stage-i2i-512-r01 (GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| - | 21,583 | 20,794 | 124,085 |
| 443 | 22,976 | 22,274 | 127,710 |
| 351 | 23,538 | 22,849 | 127,710 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 9,115.9 | 1,428 | `ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_tn` |
| 4,645.1 | 1,400 | `ampere_bf16_s1688gemm_bf16_128x64_sliced1x2_ldg8_f2f_tn` |
| 3,228.9 | 1,428 | `void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)32, (int)4, ` |
| 2,403.3 | 2,800 | `mot_unified_gemm_kernel` |
| 837.0 | 3,024 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8` |
| 598.1 | 2,884 | `triton_poi_fused_mul_silu_slice_0` |
| 318.4 | 56 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_tn` |
| 302.8 | 19,432 | `void at::native::<unnamed>::CatArrayBatchedCopy_vectorized<at::native::<unnamed>::OpaqueTy` |
| 218.5 | 5,702 | `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_ker` |
| 109.9 | 2,968 | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at:` |

### vllm-omni-two-stage-i2i-512-r01 (GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 3,352 | 22,351 | 21,614 | 121,344 |
| 477 | 1,131 | 1,026 | 7,237 |
| 3,677 | 22,720 | 22,102 | 121,344 |
| 5,299 | 22,987 | 22,369 | 121,344 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 9,265.7 | 1,372 | `ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_tn` |
| 4,608.0 | 1,372 | `ampere_bf16_s1688gemm_bf16_128x64_sliced1x2_ldg8_f2f_tn` |
| 3,218.5 | 1,372 | `void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)32, (int)4, ` |
| 2,443.8 | 2,744 | `mot_unified_gemm_kernel` |
| 796.2 | 2,744 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8` |
| 584.2 | 2,744 | `triton_poi_fused_mul_silu_slice_0` |
| 351.2 | 20,580 | `void at::native::<unnamed>::CatArrayBatchedCopy_vectorized<at::native::<unnamed>::OpaqueTy` |
| 220.7 | 5,586 | `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_ker` |
| 107.5 | 2,744 | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at:` |
| 105.6 | 13,818 | `void at::native::vectorized_gather_kernel<(int)16, long>(char *, char *, T2 *, int, long, ` |

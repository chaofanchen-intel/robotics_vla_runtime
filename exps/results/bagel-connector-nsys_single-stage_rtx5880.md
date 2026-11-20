### vllm-omni-single-stage-i2i-512-steps5-r01 (process 2496417, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 45 | 1,919 | 1,739 | 12,980 |
| 360 | 2,742 | 2,638 | 16,605 |
| 653 | 2,760 | 2,648 | 16,605 |
| 376 | 2,752 | 2,644 | 16,605 |
| 391 | 2,767 | 2,660 | 16,605 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 766.7 | 168 | `ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_tn` |
| 488.9 | 140 | `ampere_bf16_s1688gemm_bf16_128x64_sliced1x2_ldg8_f2f_tn` |
| 331.4 | 168 | `void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)32, (int)4, ` |
| 301.8 | 56 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_tn` |
| 198.3 | 280 | `mot_unified_gemm_kernel` |
| 109.8 | 504 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8` |
| 77.8 | 26 | `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:` |
| 68.0 | 364 | `triton_poi_fused_mul_silu_slice_0` |
| 43.3 | 64 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_relu_f2f_tn` |
| 25.4 | 1,792 | `void at::native::<unnamed>::CatArrayBatchedCopy_vectorized<at::native::<unnamed>::OpaqueTy` |

### sglang-omni-txt-1k-1tok-r01 (process 25312, GPU 2)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 17 | 48 | 16 | 1,054 |
| 15 | 46 | 15 | 998 |
| 16 | 2,166 | 383 | 47,238 |
| 91 | 190 | 34 | 4,067 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 3.9 | 76 | `sm80_xmma_fprop_implicit_gemm_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_tilesize64x32x64_stage5_w` |
| 2.6 | 235 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(` |
| 2.4 | 32 | `sm80_xmma_fprop_implicit_gemm_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_tilesize128x128x64_stage3` |
| 2.4 | 348 | `snake_beta_channels_last_kernel` |
| 2.2 | 192 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bflo` |
| 2.2 | 388 | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 2.0 | 28 | `sm80_xmma_dgrad_implicit_gemm_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_tilesize128x64x32_stage5_` |
| 1.4 | 60 | `sm80_xmma_fprop_implicit_gemm_indexed_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_tilesize64x64x64_` |
| 1.3 | 88 | `sm80_xmma_fprop_implicit_gemm_indexed_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_tilesize64x32x64_` |
| 1.2 | 32 | `sm80_xmma_fprop_implicit_gemm_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_tilesize128x32x32_stage4_` |
| 1.1 | 93 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_64x1_tn_align8>` |
| 1.1 | 16 | `sm80_xmma_fprop_implicit_gemm_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_tilesize256x64x32_stage3_` |
| 0.9 | 28 | `sm80_xmma_dgrad_implicit_gemm_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_tilesize64x64x64_stage4_w` |
| 0.9 | 4 | `sm80_xmma_dgrad_implicit_gemm_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_tilesize128x128x32_stage4` |

### sglang-omni-txt-1k-1tok-r01 (process 25323, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 1,850 | 26,841 | 16,805 | 131,590 |
| 1,011 | 161 | 40 | 2,105 |
| 86 | 167 | 14 | 2,134 |
| 88 | 160 | 39 | 2,033 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 28.0 | 194 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` |
| 4.6 | 192 | `fused_moe_kernel` |
| 3.4 | 289 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8` |
| 0.5 | 96 | `void flashinfer::BatchPrefillWithPagedKVCacheKernel<(bool)1, flashinfer::KernelTraits<(fla` |
| 0.5 | 191 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_a` |
| 0.3 | 96 | `_moe_align_small_numel_kernel` |
| 0.3 | 1 | `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` |
| 0.2 | 96 | `_router_triton_kernel` |
| 0.2 | 97 | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 0.2 | 96 | `_triton_mrope_forward_fused` |
| 0.1 | 96 | `void flashinfer::PersistentVariableLengthMergeStatesKernel<(unsigned int)8, (unsigned int)` |
| 0.1 | 96 | `triton_per_fused_copy__mul_0` |
| 0.1 | 96 | `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, __nv_bfloat16, __nv_bfloat16, fl` |
| 0.1 | 96 | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_noc` |

### sglang-omni-txt-1k-1tok-r01 (process 25318, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 1,504 | 26,841 | 15,895 | 131,590 |
| 1,011 | 161 | 16 | 2,105 |
| 87 | 166 | 103 | 2,134 |
| 88 | 159 | 13 | 2,021 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 4.6 | 190 | `fused_moe_kernel` |
| 3.3 | 288 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8` |
| 2.0 | 192 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` |
| 0.5 | 96 | `void flashinfer::BatchPrefillWithPagedKVCacheKernel<(bool)1, flashinfer::KernelTraits<(fla` |
| 0.4 | 190 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_a` |
| 0.3 | 95 | `_moe_align_small_numel_kernel` |
| 0.2 | 95 | `_router_triton_kernel` |
| 0.2 | 97 | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 0.2 | 96 | `_triton_mrope_forward_fused` |
| 0.1 | 96 | `void flashinfer::PersistentVariableLengthMergeStatesKernel<(unsigned int)8, (unsigned int)` |
| 0.1 | 95 | `triton_per_fused_copy__mul_0` |
| 0.1 | 95 | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_noc` |
| 0.1 | 95 | `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, __nv_bfloat16, __nv_bfloat16, fl` |
| 0.1 | 96 | `void sglang::fused_qknorm_warp<(long)128, (bool)0, __nv_bfloat16>(sglang::QKNormParams)` |

### sglang-omni-txt-1k-1tok-r01 (process 25311, GPU 2)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 17 | 36 | 10 | 1,370 |
| 46 | 34 | 2 | 513 |
| 436 | 38 | 8 | 859 |
| 18 | 37 | 10 | 1,364 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 2.5 | 300 | `gemv_add_kernel` |
| 2.4 | 150 | `norm_gate_up_silu_kernel` |
| 2.2 | 150 | `norm_qkv_rope_store_kernel` |
| 0.5 | 71 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bflo` |
| 0.3 | 150 | `decode_attention_kernel` |
| 0.3 | 20 | `void fused_moe::run_global<fused_moe::Fused_Moe_Kernel_sm80<cutlass::bfloat16_t, cutlass::` |
| 0.2 | 60 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bflo` |
| 0.2 | 20 | `void cutlass::Kernel<cutlass::gemm::kernel::MoeFCGemm<cutlass::gemm::threadblock::MmaMulti` |
| 0.1 | 20 | `void flashinfer::BatchPrefillWithPagedKVCacheKernel<(bool)1, flashinfer::KernelTraits<(fla` |
| 0.1 | 20 | `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float, __n` |
| 0.1 | 1 | `triton_red_fused__to_copy_add_argmax_clamp_div_log_neg_2` |
| 0.1 | 40 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_a` |
| 0.1 | 30 | `codebook_step_kernel` |
| 0.1 | 20 | `void tensorrt_llm::kernels::cutlass_kernels::finalizeMoeRoutingKernel<__nv_bfloat16, __nv_` |

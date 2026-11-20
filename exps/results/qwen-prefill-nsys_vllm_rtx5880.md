### vllm-omni-txt-1k-1tok-r01 (process 44003, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 6,012 | 162 | 91 | 1,056 |
| 1,296 | 182 | 20 | 4,619 |
| 223 | 240 | 56 | 4,243 |
| 842 | 30 | 6 | 1,375 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 3.5 | 299 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(` |
| 0.3 | 75 | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)64, (int)64, (` |
| 0.2 | 149 | `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, __nv_bfloat16, __nv_bfloat16, fl` |
| 0.2 | 14 | `void at::native::sbtopk::gatherTopK<c10::BFloat16, unsigned int, (int)2, (bool)0>(at::cuda` |
| 0.2 | 14 | `void <unnamed>::softmax_warp_forward<float, float, float, (int)11, (bool)0, (bool)0, (int)` |
| 0.1 | 14 | `void at::native::radixSortKVInPlace<(int)2, (int)-1, (int)64, (int)32, c10::BFloat16, long` |
| 0.1 | 75 | `triton_per_fused__scaled_dot_product_flash_attention__to_copy__unsafe_view_add_cat_clone_i` |
| 0.1 | 75 | `triton_per_fused__scaled_dot_product_flash_attention__to_copy__unsafe_view_add_cat_clone_i` |
| 0.1 | 14 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8` |
| 0.1 | 75 | `triton_poi_fused__unsafe_view_mul_silu_split_with_sizes_5` |
| 0.1 | 75 | `triton_poi_fused__scaled_dot_product_flash_attention__to_copy__unsafe_view_add_clone_index` |
| 0.1 | 14 | `void at::native::warpMergeSortKVInPlace<(int)2, (int)-1, (int)128, (int)16, c10::BFloat16,` |
| 0.1 | 28 | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso` |
| 0.1 | 14 | `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::A` |

### vllm-omni-txt-1k-1tok-r01 (process 46168, GPU 1)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 364 | 78 | 75 | 1,030 |
| 318 | 77 | 75 | 1,030 |
| 398 | 77 | 75 | 1,030 |
| 359 | 77 | 75 | 1,030 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 34.3 | 96 | `fused_moe_kernel` |
| 25.5 | 98 | `void vllm::cross_device_reduce_1stage<__nv_bfloat16, (int)2>(vllm::RankData *, vllm::RankS` |
| 7.4 | 144 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(` |
| 2.5 | 48 | `void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<(int)128, (int)64, (int)128, ` |
| 1.5 | 48 | `void vllm::moe::moe_sum_vec_kernel<c10::BFloat16, int, (int)8, (bool)0>(T1 *, const T1 *, ` |
| 0.7 | 48 | `void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &vllm::silu_kernel<c10::BFloa` |
| 0.4 | 1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bflo` |
| 0.3 | 44 | `triton_red_fused_fused_add_rms_norm_1` |
| 0.3 | 48 | `triton_red_fused_2` |
| 0.3 | 48 | `triton_red_fused_fused_add_rms_norm_moe_forward_0` |
| 0.2 | 48 | `void vllm::moe::moe_align_block_size_kernel<int>(const T1 *, int *, int *, int *, int *, i` |
| 0.2 | 48 | `void vllm::moe::count_and_sort_expert_tokens_kernel<int>(const T1 *, int *, int *, int *, ` |
| 0.2 | 48 | `_triton_mrope_forward` |
| 0.2 | 48 | `void vllm::moe::topkGating<(int)8, (int)128, (int)4, (int)16, (int)32, int, __nv_bfloat16,` |

### vllm-omni-txt-1k-1tok-r01 (process 46167, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 364 | 77 | 75 | 1,030 |
| 315 | 80 | 78 | 1,030 |
| 398 | 77 | 75 | 1,030 |
| 359 | 77 | 75 | 1,030 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 34.4 | 96 | `fused_moe_kernel` |
| 25.4 | 98 | `void vllm::cross_device_reduce_1stage<__nv_bfloat16, (int)2>(vllm::RankData *, vllm::RankS` |
| 7.5 | 144 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(` |
| 2.5 | 48 | `void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<(int)128, (int)64, (int)128, ` |
| 1.5 | 48 | `void vllm::moe::moe_sum_vec_kernel<c10::BFloat16, int, (int)8, (bool)0>(T1 *, const T1 *, ` |
| 0.7 | 48 | `void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &vllm::silu_kernel<c10::BFloa` |
| 0.4 | 1 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bflo` |
| 0.3 | 44 | `triton_red_fused_fused_add_rms_norm_1` |
| 0.3 | 48 | `triton_red_fused_2` |
| 0.3 | 48 | `triton_red_fused_fused_add_rms_norm_moe_forward_0` |
| 0.2 | 48 | `void vllm::moe::moe_align_block_size_kernel<int>(const T1 *, int *, int *, int *, int *, i` |
| 0.2 | 48 | `void vllm::moe::count_and_sort_expert_tokens_kernel<int>(const T1 *, int *, int *, int *, ` |
| 0.2 | 48 | `_triton_mrope_forward` |
| 0.2 | 48 | `void vllm::moe::topkGating<(int)8, (int)128, (int)4, (int)16, (int)32, int, __nv_bfloat16,` |

### vllm-omni-txt-1k-1tok-r01 (process 52168, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 25 | 20 | 10 | 866 |
| 24 | 31 | 21 | 885 |
| 35 | 61 | 51 | 853 |
| 41 | 79 | 69 | 861 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 18.2 | 4 | `void implicit_convolve_sgemm<__nv_bfloat16, __nv_bfloat16, (int)128, (int)5, (int)5, (int)` |
| 10.4 | 3 | `void implicit_convolve_sgemm<__nv_bfloat16, __nv_bfloat16, (int)128, (int)5, (int)5, (int)` |
| 7.7 | 56 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<c10::BF` |
| 7.5 | 123 | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 3.7 | 50 | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 3.3 | 28 | `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_ten` |
| 3.2 | 7 | `void cutlass__5x_cudnn::Kernel<cutlass_tensorop_bf16_s16816fprop_optimized_bf16_64x64_64x5` |
| 3.1 | 28 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::sin_kernel_cuda(at::Ten` |
| 2.7 | 31 | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 1.3 | 3 | `void cutlass::Kernel2<cutlass_75_tensorop_bf16_s1688gemm_bf16_128x64_nn_align1>(T1::Params` |
| 0.9 | 26 | `void cudnn::engines_precompiled::nchwToNhwcKernel<__nv_bfloat16, __nv_bfloat16, float, (bo` |
| 0.8 | 1 | `sm80_xmma_dgrad_implicit_gemm_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_tilesize128x32x32_stage4_` |
| 0.7 | 40 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<c10::BFloat` |
| 0.7 | 13 | `void cudnn::engines_precompiled::nhwcToNchwKernel<__nv_bfloat16, __nv_bfloat16, float, (bo` |

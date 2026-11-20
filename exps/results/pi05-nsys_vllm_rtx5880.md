### vllm-omni-droid-46row-r01 (process 934775, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 190 | 261 | 103 | 14,762 |
| 213 | 259 | 99 | 14,762 |
| 251 | 259 | 100 | 14,762 |
| 233 | 262 | 100 | 14,762 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 22.1 | 54 | `ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_tn` |
| 15.0 | 165 | `ampere_sgemm_128x64_tn` |
| 14.1 | 1,350 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(` |
| 10.3 | 334 | `ampere_sgemm_64x32_sliced1x4_tn` |
| 6.1 | 370 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, float, float, float, flo` |
| 2.0 | 1,012 | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_noc` |
| 2.0 | 81 | `fmha_cutlassF_f32_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<float, cu` |
| 2.0 | 594 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<c10::BFlo` |
| 1.8 | 180 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_bf16_64x64_64x6_nn_align2>(T1::P` |
| 1.5 | 407 | `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::M` |
| 1.5 | 990 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cu` |
| 1.4 | 198 | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at:` |
| 1.4 | 900 | `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, __nv_bfloat16, __nv_bfloat16, fl` |
| 1.3 | 396 | `void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsigne` |

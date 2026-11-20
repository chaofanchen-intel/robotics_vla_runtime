### sglang-omni-t2i-256-r01 (process 3029761, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 3,031 | 2,409 | 1,868 | 113,327 |
| 3,188 | 2,398 | 1,868 | 113,327 |
| 2,964 | 2,392 | 1,867 | 113,327 |
| 3,101 | 2,067 | 1,618 | 97,716 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 1,130.3 | 5,819 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(` |
| 179.3 | 2,871 | `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn` |
| 126.1 | 2,870 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_32x6_tn_align8>` |
| 43.2 | 23,007 | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 31.5 | 17,385 | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 20.7 | 2,870 | `void cutlass::device_kernel<flash::enable_sm80_to_sm89<flash::FlashAttnFwdSm80<flash::Coll` |
| 15.1 | 5,661 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_a` |
| 13.4 | 5,742 | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 13.0 | 11,909 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<c10::BF` |
| 8.5 | 2,870 | `void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsigne` |
| 6.6 | 2,870 | `void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::ActivationKind)0, (bool)0, (bool)0` |
| 6.5 | 2,870 | `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, __nv_bfloat16, __nv_bfloat16, fl` |

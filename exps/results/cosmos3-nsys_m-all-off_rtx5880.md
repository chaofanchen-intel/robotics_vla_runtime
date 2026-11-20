### mstar-t2i-256-r01

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 199 | 3,962 | 2,150 | 212,300 |
| 277 | 3,854 | 2,150 | 212,300 |
| 302 | 3,816 | 2,149 | 212,300 |
| 249 | 3,907 | 2,151 | 212,300 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 870.2 | 6,696 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_tn` |
| 553.4 | 10,389 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(` |
| 295.6 | 6,696 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_32x6_tn_align8>` |
| 60.4 | 13,629 | `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::M` |
| 59.0 | 27,165 | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 44.9 | 13,629 | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at:` |
| 31.1 | 13,724 | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso` |
| 25.8 | 13,464 | `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, __nv_bfloat16, __nv_bfloat16, fl` |
| 22.1 | 3,348 | `void flashinfer::BatchPrefillWithRaggedKVCacheKernel<flashinfer::KernelTraits<(flashinfer:` |
| 19.3 | 13,536 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<c10::BF` |
| 18.1 | 6,768 | `void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsigne` |
| 17.3 | 23 | `sm86_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize64x32x64_stage4_wa` |

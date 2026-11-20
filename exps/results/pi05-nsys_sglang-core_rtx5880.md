### sglang-core-droid-46row-r01 (process 924762, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 309 | 75 | 39 | 1,700 |
| 21 | 41 | 39 | 7,880 |
| 320 | 57 | 40 | 1,700 |
| 27 | 9 | 8 | 1,651 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 2.8 | 37 | `void cutlass::device_kernel<flash::enable_sm80_to_sm89<flash::FlashAttnFwdSm80<flash::Coll` |
| 2.0 | 148 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(` |
| 1.3 | 76 | `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, float, float, float, flo` |
| 0.2 | 76 | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at:` |
| 0.2 | 76 | `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::M` |
| 0.2 | 111 | `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, __nv_bfloat16, __nv_bfloat16, fl` |
| 0.2 | 81 | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso` |
| 0.2 | 161 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cu` |
| 0.2 | 37 | `void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsigne` |
| 0.1 | 152 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<f` |
| 0.1 | 74 | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 0.1 | 76 | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_noc` |
| 0.1 | 76 | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_noc` |
| 0.1 | 37 | `triton_poi_fused__to_copy_add_cat_mul_neg_slice_unsqueeze_0` |

### sglang-core-droid-46row-r01 (process 3965434, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 73 | 66 | 14 | 1,677 |
| 29 | 31 | 26 | 7,500 |
| 71 | 48 | 14 | 1,659 |
| 20 | 23 | 19 | 5,388 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 3.6 | 129 | `void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::Col` |
| 2.0 | 265 | `void gemv2T_kernel_val<int, int, float, float, float, float, (int)128, (int)16, (int)4, (i` |
| 1.2 | 265 | `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::M` |
| 1.1 | 265 | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at:` |
| 1.0 | 128 | `nvjet_sm90_tst_64x56_64x14_2x1_v_bz_TNT` |
| 0.9 | 129 | `nvjet_sm90_tst_8x64_64x16_2x1_v_bz_TNN` |
| 0.8 | 280 | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso` |
| 0.8 | 128 | `nvjet_sm90_tst_32x64_64x16_4x1_v_bz_splitK_TNN` |
| 0.7 | 552 | `void at::native::vectorized_elementwise_kernel<(int)8, at::native::bfloat16_copy_kernel_cu` |
| 0.7 | 257 | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 0.7 | 129 | `nvjet_sm90_tst_32x64_64x16_4x1_v_bz_TNN` |
| 0.6 | 530 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<f` |
| 0.6 | 129 | `void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsigne` |
| 0.5 | 264 | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_noc` |

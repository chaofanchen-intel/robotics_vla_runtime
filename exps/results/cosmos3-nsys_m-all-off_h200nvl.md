### mstar-t2i-256-r01 (process 3540158, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 98 | 3,601 | 961 | 212,296 |
| 93 | 3,360 | 960 | 212,296 |
| 90 | 3,379 | 962 | 212,296 |
| 91 | 3,376 | 959 | 212,296 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 203.8 | 6,696 | `nvjet_sm90_tst_128x64_64x8_4x1_v_bz_TNT` |
| 103.0 | 3,348 | `nvjet_sm90_tst_96x64_64x10_2x1_v_bz_splitK_TNN` |
| 89.9 | 6,696 | `nvjet_sm90_tst_64x32_64x16_1x2_h_bz_TNT` |
| 85.9 | 13,629 | `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::M` |
| 79.7 | 27,165 | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 62.4 | 13,629 | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at:` |
| 44.6 | 6,696 | `nvjet_sm90_tst_64x32_64x16_4x2_h_bz_splitK_TNT` |
| 44.2 | 13,724 | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso` |
| 32.1 | 3,348 | `void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::Col` |
| 23.5 | 10,152 | `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float, __n` |
| 20.8 | 13,536 | `void at::native::vectorized_elementwise_kernel<(int)8, at::native::CUDAFunctor_add<c10::BF` |
| 19.4 | 6,768 | `void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsigne` |

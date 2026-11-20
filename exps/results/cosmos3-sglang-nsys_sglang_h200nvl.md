### sglang-omni-t2i-256-r01 (process 3547616, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 83 | 1,321 | 484 | 41,688 |
| 76 | 1,314 | 483 | 41,688 |
| 78 | 1,343 | 483 | 41,688 |
| 78 | 536 | 202 | 16,889 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 76.1 | 1,404 | `nvjet_sm90_tst_192x64_64x6_2x1_v_bz_TNT` |
| 43.8 | 1,404 | `nvjet_sm90_tst_96x64_64x10_2x1_v_bz_splitK_TNN` |
| 25.4 | 1,404 | `nvjet_sm90_tst_64x64_64x13_4x1_v_bz_TNT` |
| 18.9 | 1,404 | `nvjet_sm90_tst_64x32_64x16_1x2_h_bz_TNT` |
| 11.5 | 1,404 | `void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::Col` |
| 10.7 | 2,769 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_a` |
| 4.2 | 1,404 | `void sglang::fused_qknorm_rope_warp<(long)128, (long)128, (bool)1, (bool)1, __nv_bfloat16,` |
| 4.1 | 1,404 | `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float, __n` |
| 3.2 | 1,404 | `void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::ActivationKind)0, (bool)1, (bool)0` |
| 0.9 | 39 | `void gemv2T_kernel_val<int, int, float, float, float, float, (int)128, (int)16, (int)4, (i` |
| 0.5 | 129 | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at:` |
| 0.3 | 250 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, fl` |

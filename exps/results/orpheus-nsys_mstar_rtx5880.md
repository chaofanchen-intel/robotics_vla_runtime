### mstar-b1-r01 (process 945889, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 20,364 | 230 | 212 | 9,575 |
| 21 | 1,850 | 1,805 | 82,902 |
| 249 | 2,522 | 2,480 | 113,511 |
| 219 | 4,959 | 4,853 | 221,007 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 2,101.0 | 18,424 | `triton_red_fused_mm_t_1` |
| 1,073.8 | 18,424 | `triton_red_fused_add_mm_mul_silu_split_t_2` |
| 674.4 | 18,424 | `triton_red_fused_mm_t_0` |
| 419.3 | 18,424 | `triton_red_fused_mm_t_view_0` |
| 130.9 | 659 | `triton_red_fused_mm_slice_t_0` |
| 123.0 | 18,452 | `void flashinfer::BatchQKApplyRotaryPosIdsKernel<(bool)0, (unsigned int)128, (unsigned int)` |
| 104.4 | 18,424 | `void flashinfer::BatchPrefillWithPagedKVCacheKernel<(bool)1, flashinfer::KernelTraits<(fla` |
| 61.6 | 37,563 | `kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object_at__tensorptrbf16gm` |
| 30.3 | 659 | `void flashinfer::sampling::TopPSamplingFromProbKernel<(unsigned int)1024, (cub::_V_300600_` |
| 29.1 | 18,452 | `void flashinfer::PersistentVariableLengthMergeStatesKernel<(unsigned int)8, (unsigned int)` |
| 17.6 | 18,452 | `_kv_scatter_nhd_kernel` |
| 16.2 | 18,452 | `triton_poi_fused_add_0` |
| 10.4 | 659 | `_split_write_kernel` |
| 9.0 | 1,318 | `void at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<unnamed>::TensorListMet` |

### mstar-b16-r01 (process 957806, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 15 | 272 | 238 | 10,430 |
| 21 | 5,589 | 5,426 | 244,301 |
| 282 | 4,306 | 4,182 | 188,822 |
| 257 | 5,308 | 5,114 | 227,321 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 1,883.6 | 15,820 | `triton_tem_fused_mm_t_1` |
| 934.3 | 15,820 | `triton_tem_fused_addmm_mul_silu_split_t_3` |
| 592.3 | 15,820 | `triton_tem_fused_mm_t_0` |
| 377.8 | 15,820 | `triton_tem_fused_mm_t_view_0` |
| 300.9 | 16,912 | `void flashinfer::BatchPrefillWithPagedKVCacheKernel<(bool)1, flashinfer::KernelTraits<(fla` |
| 134.1 | 1,176 | `triton_red_fused_mm_t_1` |
| 125.3 | 16,996 | `void flashinfer::BatchQKApplyRotaryPosIdsKernel<(bool)0, (unsigned int)128, (unsigned int)` |
| 117.2 | 564 | `triton_tem_fused_mm_slice_t_0` |
| 73.3 | 34,599 | `kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object_at__tensorptrbf16gm` |
| 68.6 | 1,176 | `triton_red_fused_add_mm_mul_silu_split_t_2` |
| 49.7 | 1,682 | `fused_add_reciprocal_mul_sin_pow_mul_add` |
| 46.4 | 607 | `void flashinfer::sampling::TopPSamplingFromProbKernel<(unsigned int)1024, (cub::_V_300600_` |
| 43.4 | 754 | `void at::native::<unnamed>::conv_depthwise2d_forward_kernel_generic<float, int>(torch::hea` |
| 43.1 | 1,176 | `triton_red_fused_mm_t_0` |

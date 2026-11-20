### mstar-t2i-256-r01 (process 3001926, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 446 | 1,674 | 1,032 | 40,009 |
| 257 | 1,620 | 1,033 | 40,009 |
| 273 | 1,646 | 1,033 | 40,009 |
| 393 | 1,513 | 1,039 | 40,009 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 801.4 | 8,697 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(` |
| 72.9 | 576 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_tn` |
| 40.7 | 3,096 | `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn` |
| 25.2 | 576 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_32x6_tn_align8>` |
| 17.4 | 23 | `sm86_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize64x32x64_stage4_wa` |
| 12.8 | 1,800 | `void flashinfer::BatchPrefillWithRaggedKVCacheKernel<flashinfer::KernelTraits<(flashinfer:` |
| 10.3 | 3,600 | `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0` |
| 9.0 | 1,800 | `triton_poi_fused_mul_silu_1` |
| 7.4 | 4,341 | `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, __nv_bfloat16, __nv_bfloat16, fl` |
| 6.5 | 3,600 | `void at::native::<unnamed>::CatArrayBatchedCopy_vectorized<at::native::<unnamed>::OpaqueTy` |
| 5.7 | 66 | `void cudnn::engines_precompiled::convertTensor_kernel<float, float, float, (cudnnKernelDat` |
| 4.4 | 51 | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4>(T1::Params)` |

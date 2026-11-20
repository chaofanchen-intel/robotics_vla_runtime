### vllm-omni-droid-46row-r01 (process 566, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 71 | 232 | 93 | 14,022 |
| 70 | 392 | 93 | 14,022 |
| 101 | 328 | 93 | 14,022 |
| 71 | 234 | 93 | 14,022 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 18.9 | 81 | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_tn_align1>(T1::Params)` |
| 15.9 | 324 | `sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize32x32x8_stage3_warpsize1x2x1_ffma_aligna4_al` |
| 8.9 | 84 | `sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize64x64x8_stage3_warpsize1x4x1_ffma_aligna4_al` |
| 3.6 | 54 | `nvjet_sm90_tst_128x248_64x4_2x1_v_bz_coopA_TNT` |
| 2.9 | 370 | `void gemv2T_kernel_val<int, int, float, float, float, float, (int)128, (int)16, (int)4, (i` |
| 2.2 | 1,012 | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_noc` |
| 2.2 | 54 | `nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT` |
| 2.2 | 81 | `fmha_cutlassF_f32_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<float, cu` |
| 2.2 | 180 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_nn_align2` |
| 2.1 | 360 | `nvjet_sm90_tst_48x64_64x15_4x1_v_bz_TNN` |
| 1.8 | 407 | `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::M` |
| 1.6 | 360 | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at:` |
| 1.6 | 198 | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at:` |
| 1.6 | 360 | `nvjet_sm90_tst_64x8_64x16_2x4_h_bz_TNT` |

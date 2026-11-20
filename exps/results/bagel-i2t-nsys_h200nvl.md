### mstar-i2t-512-r01 (process 3738669, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 92,608 | 781 | 625 | 49,563 |
| 33,135 | 799 | 625 | 49,563 |
| 33,112 | 775 | 625 | 49,563 |
| 33,119 | 785 | 625 | 49,563 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 234.1 | 3,556 | `triton_red_fused_mm_t_1` |
| 143.4 | 3,556 | `triton_red_fused_add_mm_mul_silu_split_t_2` |
| 42.5 | 3,556 | `nvjet_sm90_tst_64x8_64x16_4x1_v_badd_TNT` |
| 42.4 | 3,556 | `void flashinfer::BatchPrefillWithPagedKVCacheKernel<(bool)1, flashinfer::KernelTraits<(fla` |
| 31.9 | 128 | `triton_red_fused_mm_t_0` |
| 31.2 | 3,556 | `triton_red_fused_mm_t_view_0` |
| 16.7 | 7,296 | `kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object_at__tensorptrbf16gm` |
| 14.3 | 26 | `void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)96, (int)128, (int)64, (int)4, (` |

### vllm-omni-two-stage-i2t-512-r01 (process 3740784, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|

### vllm-omni-two-stage-i2t-512-r01 (process 3741206, GPU 0)

| gap before ms | span ms | GPU busy ms | launches |
|---:|---:|---:|---:|
| 103 | 958 | 937 | 51,113 |
| 92,673 | 963 | 938 | 51,113 |
| 31,097 | 968 | 936 | 51,113 |
| 31,087 | 962 | 937 | 51,113 |

Top kernels of the last burst:

| kernel ms | launches | kernel |
|---:|---:|---|
| 266.7 | 3,556 | `nvjet_sm90_tst_384x8_64x4_4x1_v_bz_TNT` |
| 249.7 | 3,584 | `kernel_unified_attention` |
| 138.6 | 3,556 | `nvjet_sm90_tst_256x8_64x6_4x1_v_bz_splitK_TNT` |
| 53.4 | 28 | `nvjet_sm90_tst_192x208_64x4_1x2_h_bz_coopB_TNT` |
| 43.4 | 3,556 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_bias_TNT` |
| 35.9 | 3,556 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` |
| 34.0 | 128 | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` |
| 31.7 | 56 | `nvjet_sm90_tst_128x216_64x4_2x1_v_bz_coopA_TNT` |

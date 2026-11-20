#!/usr/bin/env bash

# Local copy of exps/tools/mstar_cuda_env.sh, so exps/ does not depend
# on the earlier rounds' trees. Unchanged apart from this header.
#
# Source this before starting M*: its JIT CUDA extensions must build against
# the same CUDA 13.0 toolkit minor version as the cu130 PyTorch build in
# envs/mstar, and the ambient system toolkit is a different one.

if [[ -z "${BASH_SOURCE[0]:-}" ]]; then
    printf '%s\n' 'Source this file from bash: source exps/frameworks/mstar_cuda_env.sh' >&2
    return 2 2>/dev/null || exit 2
fi

MSTAR_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MSTAR_CUDA_HOME="${MSTAR_REPO_ROOT}/envs/mstar/lib/python3.12/site-packages/nvidia/cu13"
MSTAR_MOEVCC="${MSTAR_CUDA_HOME}/bin/nvcc"

if [[ ! -x "${MSTAR_MOEVCC}" ]]; then
    printf 'M* CUDA 13.0 nvcc not found: %s\n' "${MSTAR_MOEVCC}" >&2
    printf '%s\n' 'Install nvidia-cuda-nvcc, nvidia-nvvm, and nvidia-cuda-crt at 13.0.88 in envs/mstar.' >&2
    return 1 2>/dev/null || exit 1
fi

export CUDA_HOME="${MSTAR_CUDA_HOME}"
export CUDACXX="${MSTAR_MOEVCC}"
export PATH="${MSTAR_CUDA_HOME}/bin:${MSTAR_REPO_ROOT}/envs/mstar/bin:${PATH}"
export LD_LIBRARY_PATH="${MSTAR_CUDA_HOME}/lib:${MSTAR_REPO_ROOT}/envs/mstar/lib:${LD_LIBRARY_PATH:-}"
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    MSTAR_GPU_INDEX="${CUDA_VISIBLE_DEVICES%%,*}"
    MSTAR_GPU_INDEX="${MSTAR_GPU_INDEX:-0}"
    MSTAR_COMPUTE_CAPABILITY="$(
        nvidia-smi --id="${MSTAR_GPU_INDEX}" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d '[:space:]'
    )"
    export TORCH_CUDA_ARCH_LIST="${MSTAR_COMPUTE_CAPABILITY:-9.0}"
fi

printf 'M* CUDA toolchain: %s\n' "${MSTAR_CUDA_HOME}"
printf 'M* nvcc: %s\n' "$(${MSTAR_MOEVCC} --version | awk '/release/{print $6; exit}')"
printf 'M* CUDA architecture: %s\n' "${TORCH_CUDA_ARCH_LIST}"
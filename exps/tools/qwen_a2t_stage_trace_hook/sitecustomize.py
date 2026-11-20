"""Enable optional Qwen3-Omni stage timing in benchmark service processes."""
from __future__ import annotations

import os

if os.environ.get("QWEN_A2T_STAGE_TRACE_PATH"):
    if os.environ.get("QWEN_A2T_STAGE_TRACE_FRAMEWORK") == "vllm-omni":
        import torch._dynamo

        # Qwen3's code predictor calls torch.compile outside vLLM's compilation config.
        torch._dynamo.config.disable = True

    from exps.tools.qwen_a2t_stage_trace import install

    install()

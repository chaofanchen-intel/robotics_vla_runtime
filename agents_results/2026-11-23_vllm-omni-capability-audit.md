# vLLM-Omni 能力审计（2026-11-23）

角色：只读研究子代理（派发方：主会话）。未运行服务、未改文件、未占用 GPU。
用途：支撑汇报稿 `share/robotics_vla_report/` 中"框架能力 · vLLM-Omni"四页（12_fw_vllm）以及对照页（14_fw_matrix）的结论。

## 问题（派发时的完整提示）

```
Read-only capability audit. Do not edit files, do not start servers or GPU jobs.

Goal: establish from SOURCE CODE and docs (not memory) what vLLM-Omni can do for: (a) streaming INPUT (video frames, audio chunks, text) into a session while it runs, (b) streaming OUTPUT, (c) full duplex / barge-in / session state, (d) incremental prefill of newly arrived multimodal input, (e) async chunk pipeline between stages, (f) cancellation / abort of an in-flight request, (g) Qwen3-Omni specifically.

Two trees:
- Latest source: /home/chaofan/workspaces/robotics_vla_runtime/external/vllm-omni (commit 2e3ad0a4).
- Pinned source that the experiments actually run: /home/chaofan/workspaces/robotics_vla_runtime/pinned/vllm-omni (commit 5917730 plus a local overlay). Check each claim in the pinned tree too, and say explicitly where latest and pinned differ.

Specific items to verify (with path:line):
1. Streaming Video Input API: docs/serving/video_stream_api.md (and the pinned copy). Is the WebSocket route (reported as /v1/video/chat/stream) present in pinned code? Find the handler. List the events it accepts (session.config, video.frame, audio.chunk, video.query, video.done or whatever the code really accepts). How are frames and audio buffered? Does a video.query issued while a previous query is still generating get queued, rejected, or merged? Which model(s) does it serve?
2. Incremental prefill: find the implementation (grep "incremental" and "prefill"). Is it on by default for the Thinker stage of Qwen3-Omni? Does it apply to multimodal input appended after a generation has already started, or only to context between queries?
3. Realtime APIs: /v1/realtime (turn-based audio) and /v1/realtime?duplex=1 (full duplex). Which models have a duplex plugin in the pinned tree (grep DuplexModelPlugin and the duplex model directories; check docs/models/supported_models.md)? Is Qwen3-Omni one of them, or is only PersonaPlex / MiniCPM-O etc.? What is the barge-in / interruption semantics: is in-flight generation cancelled, and does it stop GPU work? What is session epoch/turn?
4. async_chunk: docs/design/feature/async_chunk.md and the Qwen3-Omni pipeline config (deploy yaml files). Does it stream INPUT to later stages, or only stage-to-stage output? What is the chunk unit?
5. Cancellation: find the abort path for a running AR request (vllm_omni, plus the vLLM engine core abort). Does aborting stop GPU work mid-generation for Qwen3-Omni?
6. Qwen3-Omni limits: max video length or frame rate handling, audio chunk size, any documented limit on appending input while generating.

Answer with a compact report (max ~1400 words):
- A table: capability | pinned verdict (有 / 部分 / 无 / 未核对) | latest differs? | evidence path:line.
- Short paragraphs for items 1-6 with exact semantics you verified.
- A list of what you could NOT verify, explicitly marked.
- Note wherever the public docs claim something the code does not show, or vice versa.
Distinguish VERIFIED (you read the line) from INFERRED.
```

## 范围与版本

- 实验使用的源码：`pinned/vllm-omni`，提交 5917730 加本地覆盖层（overlay）。
- 最新源码：`external/vllm-omni`，提交 2e3ad0a4。
- 覆盖层只看了约 8 个（共 17 个）修改过的文件，没有发现触及视频或双工路径的改动，但未完整审阅。
- 安装的 vLLM 是 0.32.0（`envs/vllm-omni`），vllm_omni 的 dist-info 为 0.32.1.dev0+g5917730a4；vLLM 的版本是否与 pinned 要求一致未核对。

## 结果（子代理原文，未改写）

READ-ONLY AUDIT REPORT (vLLM-Omni). Nothing was edited or run.

Roots: P = /home/chaofan/workspaces/robotics_vla_runtime/pinned/vllm-omni (commit 5917730a plus local overlay). L = /home/chaofan/workspaces/robotics_vla_runtime/external/vllm-omni (commit 2e3ad0a4). All refs below are VERIFIED (I read the line) unless marked INFERRED. Paths are given as P/... or L/... (absolute roots above).

## Table

| Capability | Pinned verdict | Latest differs? | Evidence (pinned unless noted) |
|---|---|---|---|
| (a) Streaming input (frames, audio, text) into a live session | 部分: frames and audio are buffered over WS, but the model only sees them at video.query boundaries | No (diff of video_stream_base.py is one line) | P/vllm_omni/entrypoints/openai/video_stream_base.py:632-700, 753-763, 765-780; P/vllm_omni/entrypoints/openai/api_server.py:1779 |
| (b) Streaming output | 有: text and audio deltas per query on the video endpoint; output events on duplex/realtime | Same for video endpoint | video_stream_base.py:~1030-1110; docs/serving/video_stream_api.md:49-54 |
| (c) Full duplex / barge-in / session state | 部分: Qwen3-Omni has a duplex plugin with supports_barge_in=True, but only when deploy yaml sets session_mode: duplex; no native streaming-input KV | Differs: duplex engine refactored, Haru plugin added only in latest, history policy changed | P/vllm_omni/model_executor/models/qwen3_omni/duplex/plugin.py:335, 370-395 (supports_barge_in at 380); P/vllm_omni/deploy/qwen3_omni_duplex.yaml:2-3 |
| (d) Incremental prefill of multimodal input appended mid-generation | 无 (between queries only); opt-in in pinned | Differs: latest turns it ON by default on CUDA | video_stream_base.py:296-298, 465-478; P/vllm_omni/deploy/qwen3_omni_moe.yaml:32 (false); L/vllm_omni/deploy/qwen3_omni_moe.yaml:40 (true) |
| (e) Async chunk pipeline between stages | 有 for stage-to-stage output; 无 for input streaming | No | P/vllm_omni/deploy/qwen3_omni_moe.yaml:15, 17-22; P/vllm_omni/model_executor/stage_input_processors/qwen3_omni.py:468, 802; P/docs/design/feature/async_chunk.md:11-30, 99-124 |
| (f) Abort of in-flight AR request | 有: abort reaches vLLM core scheduler; GPU stop happens at next engine step (INFERRED) | Same mechanism | Section 5 |
| (g) Qwen3-Omni specifically | 有 for /v1/video/chat/stream (Qwen hooks), duplex plugin, and turn-based realtime (doc says Qwen3-Omni only) | Differs as in (c)/(d) | P/vllm_omni/entrypoints/openai/serving_video_stream.py:51-55; P/docs/serving/realtime_openai_duplex_api.md (Quick Start) |

## 1. Streaming video input API (/v1/video/chat/stream)

- Route: P api_server.py:1779 (latest L api_server.py:1935). Handler: OmniStreamingVideoHandler.handle_session, video_stream_base.py:323. The Qwen subclass (serving_video_stream.py:51-55) returns False from should_trigger_turn, so no auto-query fires. Model: Qwen3-Omni only (the doc says it is "intended for" Qwen3-Omni; the factory returns only the Qwen handler).
- Events the code accepts: session.config (first message only; _receive_config at video_stream_base.py:825-863; a later session.config falls to "Unknown type" error), video.frame (632), audio.chunk (753), video.query (765; also accepts an optional audio_data field, which is undocumented), video.done (781), ping (pong). Undocumented-but-emitted: video.frame.ack (only when frame_id is present, 1484-1500) and video.frames.consumed (1020-1025). The doc table at docs/serving/video_stream_api.md:44-55 omits ack, consumed, ping, and audio_data.
- Buffering: frames are base64 strings in a list plus metadata, filtered by EVS before buffering (video_stream_base.py ~625-645), with a 10 MB base64 cap per frame (61, 636). Eviction at max_frames: legacy drops 1, incremental drops half (658-670). PIL decode runs in background prewarm tasks. Audio is a bytearray with a 4 MiB cap (63); overflow sends "Audio buffer overflow" and clears (757-762). Audio is copied and cleared at each query start (~548-550), so it is NOT retained across queries. At query time it is packed as one 16 kHz WAV (serving_video_stream.py build_engine_prompt; _pcm_to_wav_b64 defaults to 16000 at video_stream_base.py ~1157). No sample-rate check on input.
- video.query while a previous query is generating: neither queued nor merged. It preempts. _start_query_turn (534-541) calls _cancel_active_query (437-447), which sets interrupt_event and cancels the query task. The old response's remaining engine outputs are drained silently (1008-1010) and no response.text.done is sent for it. The new query snapshots the buffer at that moment (559-562). Messages are processed serially by _processor, so the new query waits for the cancel to finish.
- video.done while generating: waits for the in-flight query to finish, then sends session.done (781-787). It does NOT cancel the query.

## 2. Incremental prefill

- Implementation: _stage0_prefix_caching_enabled reads stage-0 cache_config.enable_prefix_caching once at init (video_stream_base.py:84-92). Active iff that is true AND "audio" not in modalities (296-298).
- Default: pinned qwen3_omni_moe.yaml sets stage 0 enable_prefix_caching: false (line 32), so it is OFF by default. The pinned doc says opt-in (docs/serving/video_stream_api.md:19-21). Latest flips it to true (L yaml:40) and its doc says default on.
- Scope: warmup (a max_tokens=1 generate that prefills history plus frames) is gated by "if active_request_id is not None: return" (465-478). Warmup starts only when no query is active, and is re-scheduled after each query or prefill completes. The query snapshot is fixed at turn start, so frames appended during a generation are NOT in the running request. They are warmed after it ends. Audio-output sessions (modalities includes "audio") fall back to the legacy path (doc line 117).

## 3. Realtime APIs and duplex

- /v1/realtime routing (pinned api_server.py ~1815-1835): ?duplex=1 on a server whose duplex handler is absent returns error "VAD realtime is not enabled" and closes with 1008. This CONTRADICTS docs/serving/full_duplex_api.md warning, which says it falls back to the turn-based handler. Bare /v1/realtime goes to the turn-based OpenAIRealtimeHandler (api_server.py:915).
- Duplex mode: _should_serve_duplex (api_server.py ~600-627) requires duplex_plugin on the pipeline and session_mode: duplex in the deploy yaml. Qwen3-Omni pipeline: P/vllm_omni/model_executor/models/qwen3_omni/pipeline.py:24-25 (default_session_mode="turn", duplex_plugin set). The duplex deploy yaml is P/vllm_omni/deploy/qwen3_omni_duplex.yaml:2-3.
- Duplex plugins in pinned: Qwen3-Omni, AuraOmni, JoyAI, Nemotron VoiceChat, MiniCPM-o 4.5, and a PersonaPlex class (personaplex/duplex/plugin.py:89) that is NOT wired (the full_duplex_api.md says PersonaPlex's pipeline declares no duplex_plugin). Latest adds HaruDuplexPlugin (L model_executor/models/haru/duplex/plugin.py:165).
- Qwen3 capabilities (plugin.py:370-395): supports_barge_in=True, supports_core_resumable_request=False, supports_turn_commit_only=True. Each committed utterance starts a new generation request; there is no native streaming-input KV (P/examples/online_serving/qwen3_omni/README.md ~553-556).
- Barge-in semantics (duplex, VERIFIED): overlap_policy.decide() returns barge_in when supports_barge_in and speech is detected under barge_in_on_speech, or on a client override. runner._cancel_active_response (engine/duplex/session/runner.py:2064-2155): ends the response, closes data-plane stream, calls _advance_epoch (941), which calls barge_in (engine_session.py:1903-1911: epoch += 1, clears active request and response ids), invalidates output of the old epoch, then calls _abort_request_background (2191-2193 -> stage_port.abort_requests) on the active request id (2155), cancels the response task, and emits audio.cancelled. Stale-epoch output is dropped. Epoch = interruption fence (advances on cancel, clear, barge-in). turn_id counts committed turns (fence fields, engine_session.py:199, 247; doc realtime_duplex_api.md "Endpoint and transport").
- Turn-based /v1/realtime (realtime/connection.py): input_audio_buffer keeps ALL appended audio until commit (355-392; per-append cap 15 MiB at line 71, buffer cap 64 MiB at line 72). Commit creates a conversation item and generates (477-507, 1083). response.cancel (1429-1470) calls engine.abort, then cancels the response task.

## 4. async_chunk

- Chunk unit: thinker emits per decode step (chunk_size=1), and thinker2talker_async_chunk forwards hidden states (qwen3_omni.py:468). Talker codes accumulate to codec_chunk_frames=25 (initial 4) before code2wav (deploy yaml:17-22; async_chunk.md:21-22, 112). talker2code2wav_async_chunk at qwen3_omni.py:802.
- Input: NOT streamed through the chunk pipeline. Stage 0 prefills the whole multimodal prompt, then only outputs flow stage-to-stage (async_chunk.md:101-107). The engine does support a streaming-input prompt (async_omni.py:210 `streaming_input = isinstance(prompt, AsyncGenerator)`; 310-312), but Qwen3 duplex sets supports_core_resumable_request=False (plugin.py:384). Whether Qwen3 video/turn paths ever use it: INFERRED no.

## 5. Cancellation

- Path (VERIFIED): AsyncOmni.generate catches CancelledError/GeneratorExit and runs a shielded _abort_internal_requests (async_omni.py:381-388) -> AsyncOmni.abort (792-805) -> _abort (async_omni_base.py:507-530) -> engine.abort_async -> orchestrator._abort_request_ids (orchestrator.py:700-711) -> stage_pool.abort_requests / abort_requests_async (stage_pool.py:1347, 1448) -> vLLM.
- vLLM side (installed env, not the pinned tree): AsyncLLM.generate aborts on CancelledError (vllm/v1/engine/async_llm.py:779-781); AsyncLLM.abort -> engine_core.abort_requests_async (911-921); EngineCore.abort_requests -> scheduler.finish_requests(FINISHED_ABORTED) (core.py:554-559); the abort queue is drained before each model step (core.py:678-682, 780-782).
- GPU stop: the request is removed from scheduling at the next engine step; a step already running finishes (INFERRED from scheduler calls; not run). Caveat: installed vLLM is 0.32.0 at /home/chaofan/workspaces/Inference_Engine_Investigation_Repo/envs/vllm-omni/lib/python3.12/site-packages/ with vllm_omni dist-info 0.32.1.dev0+g5917730a4, which suggests it was built from the pinned commit. I did not confirm vLLM's pin from requirements.

## 6. Qwen3-Omni limits

- No documented max video length or fps for /v1/video/chat/stream; frame rate is client-driven. Frames are image_pil items, not native video.
- num_frames 1-128 (default 4, legacy path only); max_frames 1-256 (default 50) (video_stream_base.py:150-195; doc lines 71-72). Constant _MAX_BUFFER_FRAMES=64 (line 62) is DEFINED BUT UNUSED. EVS threshold default 0.95.
- Audio: 4 MiB buffer cap, about 131 s of 16 kHz 16-bit mono (63, 753-763). The example client sends 1 s chunks (examples/online_serving/qwen3_omni/streaming_video_client.py ~38-48).
- Appending while generating: accepted into the buffer, NOT seen by the in-flight query (465-478, 559-562). Doc Known Limitations (video_stream_api.md:139-151) notes a 200 ms idle workaround for back-to-back replies and the overflow error.
- Duplex: 8 camera images and 4 MiB image budget per examples/online_serving/realtime_web/README.md ("Qwen VAD camera input"); limit_mm_per_prompt image 8, audio 4 (P/vllm_omni/deploy/qwen3_omni_duplex.yaml:5-8). Input continues during an active response (full_duplex_api.md lifecycle).

## Latest vs pinned: differences

1. Incremental prefill default: pinned OFF (yaml:32), latest ON on CUDA (L yaml:40, plus model_runner v2 in the same file). Docs match their respective trees.
2. Duplex engine refactored in latest (new files: context_ledger, history_snapshot, memory_runtime, visual_history, perf_ledger; runner.py and engine_session.py changed). Barge-in abort path is the same (L runner.py:2227, 2356).
3. Latest duplex Qwen3 history policy defaults to full_history (no image clipping); pinned uses recent window (MAX_PROMPT_IMAGES=8).
4. Latest /v1/realtime no longer calls wait_for_duplex_warmup (pinned api_server.py ~1816). /v1/duplex unchecked in latest.
5. Latest video handler differs only in sampling-param conversion (to_sampling_params_list, base ~960).
6. Latest qwen3_omni_duplex.yaml uses model_runner v1; yaml comments reference vLLM 0.34 for latest, pinned env has vLLM 0.32.0.

## Doc vs code divergences

- full_duplex_api.md warning (duplex=1 falls back to turn-based) vs code (error + close 1008).
- realtime_duplex_api.md:541-542 names MiniCPM-o 4.5 as the barge-in example and its capability table (836-840) has no Qwen3 column; code says Qwen3-Omni duplex supports_barge_in=True (plugin.py:380). Doc under-claims.
- video_stream_api.md protocol table omits video.frame.ack, video.frames.consumed, ping, video.query audio_data.
- video_stream_api.md:19-21 (opt-in) matches pinned yaml, but not latest.
- Unused _MAX_BUFFER_FRAMES constant (video_stream_base.py:62).

## NOT VERIFIED

- Qwen3-Omni HF processor defaults (fps, max video length, max_pixels): not in this repo.
- Whether stage_vllm_configs[0].cache_config.enable_prefix_caching is populated at runtime as read; I read the code only, ran nothing.
- Whether the thinker prefix cache actually hits for image_pil mm hashes (no run).
- Exact GPU-stop timing after abort beyond "next scheduler step" (INFERRED).
- Body of _prefill_context (video_stream_base.py ~1368-1412) and EVS filter internals.
- Latest duplex session internals beyond the abort path; latest /v1/duplex warmup behaviour.
- Pinned overlay: I saw only ~8 of 17 modified files; none inspected touch video/duplex paths, but the full list was not reviewed.
- vLLM version pin for pinned commit (requirements grep found nothing).
- Tests were not read; no examples were executed.

## 主会话核对

- 汇报稿引用的 vLLM-Omni 行号（video_stream_base.py:437–447、534–562、1008–1010、84–92、296–298、465–478；pinned 与 latest 的 qwen3_omni_moe.yaml:32 与 :40；plugin.py:370–395；runner.py:2064–2155；engine_session.py:1903–1911；async_omni.py:381–388；async_llm.py:779–781）与报告一致。
- 汇报稿写的"pinned 默认关闭增量 prefill，latest 默认开"与报告的"Latest differs"一项一致。
- 文档与代码不一致（`full_duplex_api.md` 写 duplex=1 回退到轮次式，代码返回错误并以 1008 关闭）已写入汇报稿第 12 页与附录。
- 子代理指出 `realtime_duplex_api.md:541–542` 把 MiniCPM-o 作为打断示例、未列 Qwen3 列，而代码中 Qwen3-Omni 插件 `supports_barge_in=True`。汇报稿以代码为准，并在第 12 页注明文档低估了这一点（若要在汇报稿中单独写，需再次确认）。

## 未验证（子代理自己列出的，汇报稿已照此标注）

- Qwen3-Omni 的 HF 处理器默认值（fps、最长视频、max_pixels）：不在本仓库。
- `stage_vllm_configs[0].cache_config.enable_prefix_caching` 在运行时是否如读到的那样生效：只读代码，未运行。
- 思考器（Thinker）的前缀缓存对 image_pil 的多模态哈希是否命中：未运行。
- 中止后 GPU 停止的确切时机，超出"下一个调度步"的部分：推断。
- `_prefill_context`（video_stream_base.py 约 1368–1412）的内部实现与 EVS 过滤细节。
- latest 双工会话内部（中止路径之外），以及 latest 的 /v1/duplex 预热行为。
- 覆盖层的完整文件列表。
- vLLM 对 pinned 提交的版本要求（requirements 中未找到）。
- 测试文件未读，示例未执行。

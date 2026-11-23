# M\* 与 SGLang-Omni 的能力审计（2026-11-23）

角色：只读研究子代理（Explore 类，未运行服务、未改文件、未占用 GPU）。派发方：主会话。
用途：支撑汇报稿 `share/robotics_vla_report/` 中"框架能力"部分（11_fw_mstar、13_fw_sglang、14_fw_matrix）的 M\* 与 SGLang-Omni 结论。

## 问题（派发时的完整提示）

```
Read-only capability audit. Do not edit files, do not start servers or GPU jobs.

Goal: establish from SOURCE CODE and docs (not memory) what M* and SGLang-Omni can do for: (a) streaming INPUT into a running request (client sends more audio/video/text while the request is still generating), (b) streaming OUTPUT (partial text/audio emitted before the request ends), (c) full duplex / barge-in / session state, (d) asynchronous scheduling across components, cancellation of in-flight GPU work, (e) Qwen3-Omni specifically (which streams are declared, whether any client input stream exists for Qwen3-Omni).

Trees to use:
- M* latest (the one our experiments run): /home/chaofan/workspaces/robotics_vla_runtime/external/mstar (commit 2c8ffde8). Key files already known to exist: mstar/streaming/topology.py (CLIENT, ClientStreamLimits, StreamingGraphEdge at ~line 16), mstar/streaming/chunk_policy.py, mstar/conductor/conductor.py, mstar/worker/worker.py, mstar/api_server/data_worker.py, mstar/validation/client_streams.py, mstar/model/qwen3_omni/ (find the Qwen3-Omni model file), mstar/model/nemotron_duplex/, mstar/model/qwen3_tts/, mstar/model/waypoint/, mstar/api_server/openpi.py, rust frontend under mstar/rust/server (look for live call / append / ws handling), docs/clients.rst and docs/mintlify/clients.mdx (the /generate/ws live call "append" protocol, around lines 96-200).
- SGLang-Omni: /home/chaofan/workspaces/robotics_vla_runtime/external/sglang-omni (latest, 92a4c13) and /home/chaofan/workspaces/robotics_vla_runtime/pinned/sglang-omni (the editable checkout that the experiments reference, e8c635e). Also the SGLang core package installed at /home/chaofan/workspaces/robotics_vla_runtime/envs/sglang-omni/lib/python3.12/site-packages/sglang (only if needed for streaming input questions).

Known starting facts (verify, do not just repeat): the M* /generate/ws live call accepts an opening message with "end_input": false and then "append" records carrying text/files with a "stream" name; client streams are declared by models via Connection(from_partition=CLIENT,...) with ClientStreamLimits; the models declaring them appear to be nemotron_duplex, qwen3_tts, waypoint (check whether qwen3-omni declares any). The OpenPI endpoint in mstar/api_server/openpi.py seems to have no append path (verify). The Rust server's OpenPI loop does not read the socket during inference (rust/server/src/ws/openpi.rs ~566-579; verify).

Answer with a compact report (max ~1300 words):
1. A table: capability | M* verdict (有 / 部分 / 无 / 未核对) | SGLang-Omni verdict | evidence path:line for each.
2. For M*: how a running request receives appended input (exact call path from the WS frame to the model hook, e.g. process_stream_chunk), what the Conductor/Worker do when a stream item arrives mid-walk, whether a client can cancel a running request and whether that stops GPU work (cite the code that does or does not stop it), and what Qwen3-Omni declares (partitions, StreamingGraphEdge, client streams, streaming output EMIT).
3. For SGLang-Omni: what streaming exists (output and input), Qwen3-Omni support status, and whether anything is duplex.
4. A short list of things you could NOT verify, marked explicitly.
Every claim: file path and line number. Distinguish VERIFIED (you read the line) from INFERRED.
```

## 范围与版本

- M\*：`external/mstar`，提交 2c8ffde8（最新，实验使用的版本）。
- SGLang-Omni：`pinned/sglang-omni`，提交 e8c635e（实验使用的可编辑检出）；`external/sglang-omni` 的 92a4c132 只做了存在性抽查，未逐行比对。
- SGLang 核心包：未读（只在报告中注明）。

## 结果（子代理原文，未改写）

Read-only audit of M* (commit 2c8ffde8) and SGLang-Omni (pinned e8c635e; latest external 92a4c132 spot-checked only). Nothing was run and nothing was edited. Roots: M* = /home/chaofan/workspaces/robotics_vla_runtime/external/mstar/ ; SGL = /home/chaofan/workspaces/robotics_vla_runtime/pinned/sglang-omni/ . Paths below are relative to these roots. Every cited line was read or matched in grep unless marked INFERRED.

## 1. Verdict table

| Capability | M* | SGLang-Omni | Evidence |
|---|---|---|---|
| (a) input streaming into a running request, generic | 有: `/generate/ws` live call (`end_input:false`, then `append`) | 部分: only `/v1/realtime` session and TTS/diarization WS. No append on chat/generate routes | M*: mstar/api_server/entrypoint.py:2446, 2744, 2683; docs/clients.rst:154-160. SGL: sglang_omni/serve/openai_api.py:1826, 9-18 |
| (a) Qwen3-Omni client input | 无: no client stream declared | 有, session variant only: PCM16 audio plus camera-frame input ports, VAD turns | M*: mstar/model/qwen3_omni/qwen3_omni_model.py:780-807; mstar/api_server/entrypoint.py:600-601. SGL: sglang_omni/models/qwen3_omni/session_node.py:20-27; sglang_omni/models/qwen3_omni/config.py:613-618; docs/cookbook/qwen3_omni.md:126-128 |
| (b) streaming output | 有: text per token (`new_token`, persist), audio chunks (`audio_chunk`) | 有: SSE chat chunks with `audio_b64`; Qwen3 session output PCM_F32 at 24 kHz | M*: qwen3_omni_model.py:535-538, 738-740, 807-809. SGL: openai_api.py:1048-1060, 1166-1240; sglang_omni/client/client.py:348-385; session_node.py:20-27 |
| (c) barge-in / interrupt | M* 无 for control: no interrupt message. `/v1/realtime` refuses `response.cancel` as unsupported | 有, for Qwen3-Omni session: speech start during a reply interrupts it | M*: mstar/api_server/openai/serving_realtime.py:940-945, 1093-1095. SGL: sglang_omni/models/qwen3_omni/session_input.py:262-263, 186-188; sglang_omni/serve/realtime/session_bridge.py:110, 127 |
| (c) full duplex | M* 部分: nemotron_duplex consumes audio continuously; barge-in claimed only in docstring | 部分: Qwen3 session declares `interaction="turn_based"`. Separate duplex models exist (nemotron_voicechat, personaplex, minicpm_o) | M*: mstar/model/nemotron_duplex/nemotron_duplex_model.py:19-21, 499-503. SGL: session_bridge.py:110; sglang_omni/models/nemotron_voicechat/duplex_hooks.py:125 |
| (c) session state | M* 部分: per-call state lives in the model's request state; model is told to stay stateless in `process_stream_chunk` | 有: thinker keeps aborted-unit history so a cut reply is re-fed | M*: docs/mintlify/adding_models.mdx:1686. SGL: sglang_omni/models/qwen3_omni/thinker_session.py:379-391 |
| (d) async scheduling across components | 有: partition workers, conductor, StreamBuffer | 有: stage runtime plus coordinator | M*: mstar/streaming/stream_buffer.py:71; mstar/worker/stream_buffers.py:45. SGL: sglang_omni/pipeline/stage/runtime.py:2533 |
| (d) cancel stops in-flight GPU work | 部分: queued work is dropped; a launched step is not interrupted | 有 (engine abort on close/disconnect); whether the batch stops at once is INFERRED | see sections 2 and 3 |

## 2. M*

**Append path (VERIFIED).** The WS loop in `generate_ws` (entrypoint.py:2446) calls `ws_frames.parse_frame` (ws_frames.py:97 reads `append`). `_ws_is_append` (entrypoint.py:2744) routes to `append_one`, which calls `api_server.stream_input` (entrypoint.py:2683). `EntryPoint.stream_input` (entrypoint.py:1029-1094) checks that the stream is open and the size budget holds, then queues a `ClientAppend` on the preprocess worker (data_worker.py:417). The data worker calls `model.process_stream_chunk` (data_worker.py:1192). The items it returns are routed to the consumer's ranks. The conductor only sends `CLIENT_STREAM_ROUTE` (conductor.py:2318-2323; topology.py:30-33 says the producer sends items directly). Mid-walk, the consumer's `StreamBuffer` (stream_buffer.py:71) is gated by its ChunkPolicy. The partition self-triggers from the stream and the conductor sends its walks none of the streamed input (docs/mintlify/adding_models.mdx:1684). Whether a new walk starts per item is INFERRED.

**Cancel (VERIFIED chain).** Closing the socket runs `abort_submission` for each submitted key (entrypoint.py:2785, which calls entrypoint.py:1945). That calls `preprocess_worker.abort_request` (data_worker.py:423 and forward_abort at 433), which reaches conductor `_abort_request` (conductor.py:1826). A queued request is popped. Otherwise `_remove_request` (conductor.py:1861) drains participants, then REMOVE follows. On the worker, `drains_ready` defers a drain while a step holds the request (request_lifecycle.py:206-213). Backlogged batches are dropped (step_pipeline.py:1001-1005). A step already launched on the GPU is not interrupted by this path. I found no kernel-level cancel by grepping mstar/worker and mstar/conductor for abort/cancel. Net effect: queued work stops, the in-flight step runs to completion, and its output is discarded.

**Qwen3-Omni (VERIFIED).** `get_partitions` lists Thinker, Talker, Code2Wav (qwen3_omni_model.py:761-778). `get_partition_topology` (780-807) has three connections, all partition to partition: Thinker→Talker `thinker_states` and `thinker_mask` (StreamingGraphEdge, which is defined at topology.py:16), and Talker→Code2Wav `codec_tokens`. It declares no `from_partition=CLIENT` connection, so `takes_stream_input` is False (entrypoint.py:600-601). Consequences: an opening with `end_input:false` is treated as one-shot (entrypoint.py:2765), and an append gets a "not taking input" reply (entrypoint.py:2744-2748). The EMIT outputs are `new_token` (text, persist=True, 535-538; more EMIT at 569, 605, 633, 668) and `audio_chunk` (audio, 738-740). Output rate is 24 kHz (807-809).

**Other M* models with client streams (VERIFIED).** nemotron_duplex declares `ClientAudio` 16 kHz audio into Encoder (nemotron_duplex_model.py:499-503). qwen3_tts declares `text_in` (qwen3_tts_model.py:725-729). waypoint declares `actions` (waypoint_model.py:277-280).

**OpenPI (VERIFIED, no append path).** `mstar/api_server/openpi.py:285` receives a frame, then `await _infer` runs inline (293), so no socket read happens during inference. There are no append or end_input references in openpi.py. In `rust/server/src/ws/openpi.rs`, `socket.recv()` is at 566 and `infer(...).await` at 588, inline. openpi.rs has no append references. The Rust generate path does have append (`rust/server/src/ws/generate.rs:813-832`), and closing the socket aborts all tasks (generate.rs:861).

## 3. SGLang-Omni

**Output streaming (VERIFIED).** `chat_stream` emits SSE (openai_api.py:1048-1060, 1166-1240). `completion_stream` yields text and base64 audio per chunk (client.py:348-385).

**Input streaming (VERIFIED).** Only the Session API takes continuous input: `/v1/realtime` (openai_api.py:1826), mounted only with `--enable-realtime` (cli/serve.py:460-470, 554-556). Qwen3-Omni's variant is `Qwen3OmniSessionPipelineConfig` (config.py:613-618, registered as "session" at 696-697). Its ports are PCM16 audio input, camera-frame input, PCM_F32 audio output at 24 kHz, and controls COMMIT, CONFIGURE, INTERRUPT, TRUNCATE (session_node.py:20-27). A VAD node and a `dialog` node cut turns on speech start/stop or client commit (docs/cookbook/qwen3_omni.md:126-128).

**Barge-in (VERIFIED).** `session_input.py:262-263`: SPEECH_STARTED while an answer is running calls `interrupt()`. That emits INTERRUPT and TURN_INTERRUPTED (186-188). The unfinished utterance is re-asked, or the reply is truncated to the audio the client heard. Capabilities are `interaction="turn_based"` (session_bridge.py:110), so Qwen3-Omni is turn-based with barge-in, not simultaneous duplex. `response.cancel` is supported when INTERRUPT is in controls (session_bridge.py:127; docs/developer_reference/full_duplex_guide.md:91).

**Cancel (VERIFIED chain).** A buffered HTTP disconnect calls `client.abort` (serve/speech_to_text.py:235-260). For SSE, `ClosableStreamingResponse` closes the body (serve/streaming.py:25-42). That closes the coordinator's `stream` generator, whose finally calls `self.abort` (pipeline/coordinator.py:497, 534-540). The engine side is `native_chat_scheduler.py:482`, which calls `tokenizer_manager.abort_request`. Session cancel sets an event (scheduling/session.py:558-563), which the Qwen preprocessor checks (session_preprocessing.py:98). Whether the engine halts a running batch at once is INFERRED.

## 4. Not verified

- No server was started. Everything is static reading.
- Rust frontend: only generate.rs and openpi.rs were read. rust/server/src/realtime and bridge.rs were not read, and it is not confirmed which frontend the experiments use.
- GPU level: no CUDA-level interrupt was traced. SGLang core's `tokenizer_manager.abort_request` was not read (envs site-packages).
- M*: the effect of `interrupt_response` in the conversation path (serving_realtime.py:471-493) was not traced. The nemotron barge-in claim (docstring at 19-21) has no implementation in that file. Grep of api_server, conductor, worker and streaming found none.
- SGL latest (92a4c132): only the existence of qwen3_omni session files was checked, not a diff.
- SGL Qwen3 session: whether audio is accepted while a reply is running was only partly traced (session_input.py queue).
- Duplex models in SGL (nemotron_voicechat, personaplex, minicpm_o) were identified by file and flag only, not read in depth.

## 主会话核对

- 汇报稿引用的 M\* 行号（entrypoint.py:2744–2748、2765，openpi.rs:566、588，generate.rs:813–832，qwen3_omni_model.py:535–538、738–740、807–809，data_worker.py:423、433，conductor.py:1826、1861，request_lifecycle.py:206–213，step_pipeline.py:1001–1005，serving_realtime.py:940–945）与子代理报告一致。
- SGLang 的 session_input.py:262–263、186–188，session_bridge.py:110，coordinator.py:534–540，native_chat_scheduler.py:482 与报告一致。
- 汇报稿把 SGLang 的最新版（92a4c132）标为"只抽查"，与本报告的范围一致。
- 子代理的"Qwen3-Omni 没有客户端输入流"结论与汇报稿第 11 页一致。

## 未验证（子代理自己列出的，汇报稿已照此标注）

- 没有启动服务；全部是静态阅读。
- M\* 的 Rust 前端：只读了 `generate.rs` 与 `openpi.rs`，未读 `rust/server/src/realtime` 与 `bridge.rs`；**实验实际使用的前端未被确认**。汇报稿附录与 Pi0.5 结果页写的"M\* 默认 Rust 前端"来自设计文档 §2.6.4，需要单独核对。
- GPU 层面：没有追踪到 CUDA 级的打断；SGLang 核心的 `tokenizer_manager.abort_request` 未读（envs 中的 site-packages）。
- M\*：`interrupt_response` 在会话路径（serving_realtime.py:471–493）中的效果未追踪；nemotron 的 barge-in 只在文档字符串中出现（19–21 行），在 api_server、conductor、worker、streaming 中未找到实现。
- SGLang latest（92a4c132）：只检查了 Qwen3-Omni 会话文件存在，没有 diff。
- SGLang Qwen3 会话：音频在回答进行中是否被接受，只部分追踪（session_input.py 的队列）。
- SGLang 的 duplex 模型（nemotron_voicechat、personaplex、minicpm_o）：只按文件与参数识别，未深读。

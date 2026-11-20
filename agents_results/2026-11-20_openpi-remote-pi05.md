# OpenPI remote serving, reference action-chunk client, and Pi0.5 model

迁移说明（2026-11-20）：本文件从基准仓库 `agents_results/` 迁入 `robotics_vla_runtime/agents_results/`。路径已改为本目录的 `external/`（上游最新快照）和 `docs/design.md`。文中的 `/tmp/...` 是当时的临时克隆，未保留。

- Date: 2026-11-20
- Role: read-only source analyst (research subagent). No edits except this file. No code run, no packages installed, no GPU jobs, no commit.
- Scope: `external/openpi`, HEAD `215abfb217dbac7d5f1273282331b9b1866c0479` (2026-08-25), the revision vLLM-Omni CI pins. Citations are `path:line` relative to `external/openpi/`. Lines marked (inferred) are conclusions from code, not observed behaviour.
- Prior research: `agents_results/` grep for openpi. `2026-11-20_realtime-vllm-omni.md` covers vLLM-Omni's own OpenPI endpoint (a different server, in the vllm-omni repo). `2026-11-14_pi05-third-case-plan.md` was not read for this task. No earlier file answers the upstream questions below.

## Question

1. Wire protocol and server: what the server sends on connect, what a client sends per call and gets back, whether the server serves one request at a time per connection, cross-connection concurrency, and where input transforms run.
2. Reference client action-chunk execution: re-query cadence and how the chunk is consumed.
3. Pi0 / Pi0.5 model: `sample_actions`, denoise loop, prefix KV reuse, prefix attention, Pi0.5 state discretization, default step count.
4. RTC, inpainting, or prefix-action conditioning: does it exist.
5. Server-side state across calls: does it exist.
6. Configs for DROID (`pi05_droid`) and `pi05_base`: horizon, dims, camera keys, state dim, DROID example keys.

## Findings

### 1. Wire protocol and server

- Transport: `websockets` server, msgpack-numpy payloads, compression off and no message size limit (`src/openpi/serving/websocket_policy_server.py:37-46`; client `packages/openpi-client/src/openpi_client/websocket_client_policy.py:37-39`). `GET /healthz` returns `OK` on the same port (`websocket_policy_server.py:86-90`).
- msgpack-numpy: ndarray encoded as dict with `__ndarray__`, raw bytes, dtype, shape (`packages/openpi-client/src/openpi_client/msgpack_numpy.py:21-31`); decoded by `object_hook` (`:43-50`, `:53-57`).
- On connect: the server sends one frame, the packed `policy.metadata` (`websocket_policy_server.py:52`). Metadata = `train_config.policy_metadata` (`scripts/serve_policy.py:101, 115`; `src/openpi/policies/policy.py:54, 109-110`; `src/openpi/policies/policy_config.py:91`). Default is `{}` (`websocket_policy_server.py:31`). `pi05_droid` sets no metadata (`src/openpi/training/config.py:629-642`). ALOHA configs set `{"reset_pose": [0, -1.5, 1.5, 0, 0, 0]}` (`config.py:570, 578, 587, 596`). The client reads it once (`websocket_client_policy.py:32-41`) and retries every 5 s on `ConnectionRefusedError` (`:42-44`). The ALOHA real example uses `reset_pose` (`examples/aloha_real/main.py:31-33`).
- Per call, client: `infer(obs)` packs the dict, then does a blocking `send` and `recv` (`websocket_client_policy.py:47-54`). There is no request id, session id, or timestamp. The keys are whatever the served config's input transform reads. The server does not validate them up front.
  - DROID example (`examples/droid/main.py:119-127`): `observation/exterior_image_1_left` (uint8 224x224x3), `observation/wrist_image_left` (uint8 224x224x3), `observation/joint_position` (7), `observation/gripper_position` (1), `prompt`.
  - LIBERO example (`examples/libero/main.py:130-140`): `observation/image`, `observation/wrist_image`, `observation/state` (8 = eef pos 3 + axis-angle 3 + gripper qpos 2), `prompt`.
  - ALOHA random obs (`examples/simple_client/main.py:153-163`): `state` (14), `images` with `cam_high`, `cam_low`, `cam_left_wrist`, `cam_right_wrist` (CHW), `prompt`.
- Per call, server reply: a dict with `actions` (float32, horizon x action_dim after output transforms), `policy_timing: {infer_ms}` (`policy.py:103-105`), and `server_timing: {infer_ms, prev_total_ms?}` (`websocket_policy_server.py:64-69`). The stock output transforms drop `state` (`droid_policy.py:81`; `libero_policy.py:100`; `aloha_policy.py:100`).
- Server loop (`websocket_policy_server.py:55-83`): `recv` (`:58`), synchronous `self._policy.infer(obs)` (`:61`), `send` (`:71`). Strictly one request at a time per connection. The next message is read only after the reply is sent.
- Cross-connection concurrency: none. One asyncio loop (`:35, 37-46`). No `to_thread`, executor, or threading in `src/openpi/serving` or `openpi_client`, so `infer` runs on the event loop and blocks other connections and the health check while it runs (`:61`). One `Policy` object is shared by all connections (`scripts/serve_policy.py:100, 111-116`).
- Error path: any exception sends the traceback as a text frame, closes with `INTERNAL_ERROR`, and re-raises (`websocket_policy_server.py:77-83`). The client raises `RuntimeError` on a str frame (`websocket_client_policy.py:51-53`). The client has no reconnect logic (`:47-54`) (inferred: a closed socket surfaces as a websockets exception). A missing `prompt` with no default raises in `TokenizePrompt` (`src/openpi/transforms.py:252-254`), which kills the connection.
- Bind: host hard-coded to `0.0.0.0`, port from args (`scripts/serve_policy.py:111-116`). The client sends `Authorization: Api-Key` when `api_key` is set (`websocket_client_policy.py:36`), but the server never reads request headers (`websocket_policy_server.py:48-83`). The header has no effect on this server.
- Where transforms run: on the server. `Policy.infer` applies input transforms to the raw observation (`policy.py:70-71`), batches, runs the model (`:90-95`), then output transforms (`:102`). The chain is built in `policy_config.py:75-89`:
  - Input: repack (empty at inference, `policy_config.py:45`), `InjectDefaultPrompt`, data transform (`DroidInputs`, `LiberoInputs`, or `AlohaInputs`), `Normalize` (quantile for pi05, `config.py:187`), model transforms (`ResizeImages` 224 at `transforms.py:185-191`, `TokenizePrompt` with state discretization at `transforms.py:248-266`, `PadStatesAndActions` to 32 at `transforms.py:328-337`).
  - Output: model output transforms, `Unnormalize`, data output transform (`DroidOutputs` keeps first 8 dims, `droid_policy.py:77-81`), repack.
  - Client side only: resize with pad to 224 and uint8 (`examples/droid/main.py:119-123`; `docs/remote_inference.md:45-48`). The server re-resizes only if the size differs (`transforms.py:189-190`; `packages/openpi-client/src/openpi_client/image_tools.py:27`). Docs say state may be sent unnormalized (`docs/remote_inference.md:48`).
  - Client-side per example: LIBERO 180-degree image rotation (`examples/libero/main.py:115-116`). DROID gripper binarization at 0.5 and clip to [-1, 1] (`examples/droid/main.py:140-149`).
- `reset()`: client no-op (`websocket_client_policy.py:56-58`). The server never calls `reset` (no reference in `websocket_policy_server.py`). No episode boundary on the wire.
- `--record`: wraps the policy in `PolicyRecorder`, which `np.save`s every call to `policy_records/` (`policy.py:113-135`; `scripts/serve_policy.py:104-105`). Disk I/O on every call.
- Timing fields (relevant to benchmarks):
  - `server_timing.infer_ms` wraps all of `Policy.infer`, including host sync at `policy.py:97-100`. Best server-side model latency.
  - `policy_timing.infer_ms` is measured at `policy.py:91-96`, before the host sync at `:97-100`. No `block_until_ready` anywhere in `src/openpi` (grep). (inferred) It measures jit dispatch, so it under-reports device time under JAX async dispatch.
  - `server_timing.prev_total_ms` starts its clock before `recv` (`websocket_policy_server.py:57`) and stops after the previous `send` (`:72`). (inferred) It includes client idle time between requests.

### 2. Reference client action-chunk execution

- No sensor threads, no async. All reference loops are synchronous and count-triggered.
- DROID (`examples/droid/main.py`):
  - 15 Hz control (`:23`). `open_loop_horizon` default 8 (`:40-42`), with the comment "8 ... equals 0.5 seconds" (`:40-41`).
  - Each tick reads the observation (`:104-109`). When `actions_from_chunk_completed == 0 or >= open_loop_horizon` (`:114`), it builds the request from the current observation (`:119-127`) and blocks on `infer` (`:131-133`), with Ctrl-C deferred (`:51-70`; rationale at `:51-53`: interrupting mid-call kills the connection).
  - Then takes `pred_action_chunk[k]` (`:137`), increments `k` (`:138`), post-processes (`:140-149`), calls `env.step` (`:151`), and sleeps to 1/15 s (`:153-156`).
  - Net: execute the first 8 actions of each chunk, re-query at the 9th tick. No `env.step` runs during the call, so the executed action lags the observation by the inference latency (inferred). `examples/droid/README.md:53` says 0.5 to 1 s per chunk is normal.
- LIBERO (`examples/libero/main.py`): `replan_steps=5` (`:29`). A `deque` (`:94`). When empty, query with the current observation (`:127-144`), `extend(action_chunk[:replan_steps])` (`:145-148`), `popleft` per sim step (`:150`). The simulator steps only inside this loop, so it pauses during inference (inferred).
- `ActionChunkBroker` (`packages/openpi-client/src/openpi_client/action_chunk_broker.py:19-44`): when `_last_results` is None, calls the inner policy with the current observation (`:28-30`). Returns index `_cur_step` of every ndarray field (`:32-38`). Clears the buffer when `_cur_step >= action_horizon` (`:41-42`). Re-queries every `action_horizon` calls, regardless of chunk length. If the server returns fewer rows, the index fails (inferred). Observations between queries are ignored. No staleness logic.
- Wiring in examples:
  - `aloha_sim`: `ActionChunkBroker(action_horizon=10)` (`examples/aloha_sim/main.py:21, 36-42`), `Runtime(max_hz=50)` (`:47`).
  - `aloha_real`: `action_horizon=25` (`examples/aloha_real/main.py:18, 35-38`), `max_hz=50` (`:41`).
- `Runtime` (`packages/openpi-client/src/openpi_client/runtime/runtime.py:50-92`): per episode, `environment.reset` and `agent.reset` (`:53-54`). Then loop: `get_observation` → `get_action` → `apply_action` → subscribers → episode-complete check (`:80-92`). Rate limit by sleep when faster than 1/max_hz (`:60-74`). Single thread. `run_in_new_thread` (`:40-44`) only moves the same loop into a thread. `PolicyAgent` forwards each observation to `policy.infer` (`runtime/agents/policy_agent.py:13-15`), so without the broker it re-queries every step.
- Summary: reference Runtime-level pattern = sequential poll, fixed count trigger (N = `open_loop_horizon` / `replan_steps` / `action_horizon`), local FIFO for the chunk, no cancel or replace, no timestamps or age check, no overlap of sensing and inference. It does not implement the team's Runtime-level streaming definition. The harness must add sensor decoupling, time- or staleness-based triggers, and replacement.

### 3. Pi0 / Pi0.5 model

- Config (`src/openpi/models/pi0_config.py:19-41`): dtype bfloat16 (`:20`); PaliGemma `gemma_2b` (`:21`); action expert `gemma_300m` (`:22`); `action_dim` 32 (`:25`); `action_horizon` 50 default (`:26`); `pi05` flag (`:31`); `max_token_len` 200 if pi05 else 48 (`:37-39`); `discrete_state_input` defaults to `pi05` (`:33, 40-41`); `pytorch_compile_mode` "max-autotune" (`:35`).
- Vision: SigLIP So400m/14, `pool_type="none"` (`src/openpi/models/pi0.py:81-89`). 224x224 with patch 14 gives 16x16 = 256 tokens per image (inferred).
- Image slots: always 3 (`base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`, `src/openpi/models/model.py:39-44`). Unused slots are zeros with mask False (`droid_policy.py:50-51`; `aloha_policy.py:56-70`). They still go through SigLIP, since `embed_prefix` loops over all keys (`pi0.py:113-116`) (inferred cost).
- Prefix = image tokens + prompt tokens (`pi0.py:105-137`). All `ar_mask` entries are False (`:125, 133`), so prefix attention is fully bidirectional. The code comments say so (`:124, 132`). PyTorch is identical (`src/openpi/models_pytorch/pi0_pytorch.py:187-236`, `:211, 226`).
- Pi0.5 state: no state token in the suffix (`pi0.py:151` `if not self.pi05`). State enters only as discretized text in the prompt (`src/openpi/models/tokenizer.py:24-28`; `transforms.py:252-266`). Format: `Task: {prompt}, State: {ints};\nAction: `, with 256 bins over [-1, 1] via `np.digitize(state, linspace(-1, 1, 257)[:-1]) - 1` (`tokenizer.py:26-28`).
  - The state fed here is after `Normalize`, which is quantile `(x - q01) / (q99 - q01) * 2 - 1` with no clipping (`transforms.py:141-145`), and before padding to 32 (`config.py:126-138` order: `PadStatesAndActions` runs after `TokenizePrompt`). So DROID's text carries 8 integers.
  - (inferred) Values below q01 map to bin index -1, which appears as the text `-1`.
- Pi0 (non-pi05) state: continuous `state_proj` token in the suffix with `ar=True` (`pi0.py:97, 151-157`). Pi0.5 uses adaRMS time conditioning (`pi0.py:162-169`); Pi0 concatenates action and time (`:170-178`).
- Pi0.5 never reads `Observation.state` unless `discrete_state_input` is on. With `discrete_state_input=False` (as in `pi05_libero`, `config.py:745`), the state reaches neither the prompt nor the suffix (`pi0.py:151`; `tokenizer.py:24`; `transforms.py:256-260`) (inferred from code).
- `sample_actions` (`pi0.py:216-279`):
  - Default `num_steps=10` (`:222`). Noise shape (batch, horizon, 32) (`:228-231`).
  - Prefix forward once, producing the KV cache (`:233-237`).
  - Each step: embed suffix (`:241-242`), attend to cached prefix plus earlier suffix (`:246-267`), `v_t` from the last `action_horizon` suffix tokens (`:269`), Euler update `x_t + dt * v_t` with `dt = -1/num_steps` (`:228, 271`).
  - Loop: `jax.lax.while_loop` with `time >= -dt/2` (`:273-278`), so 10 iterations starting at t = 1.0.
- Suffix attention: first action token `ar=True`, the rest `False` (`pi0.py:182`). Prefix tokens cannot attend to actions. Actions attend to the full prefix and earlier actions (`make_attn_mask` at `:19-44`; full mask at `:249-252`).
- Prefix KV reuse: yes, within one call. JAX: cache passed into each step (`pi0.py:261-267`). PyTorch: `past_key_values` computed once (`pi0_pytorch.py:386-400`) and reused in `denoise_step` (`:422-462`, KV at `:453`). Across calls: none. `Policy.infer` recomputes the prefix on every call (`policy.py:90-95`).
- `Policy` (`policy.py:24-65`): `module_jit` on `sample_actions` (`:64`), RNG key `jax.random.key(0)` (`:65`), split per call (`:75`). Each call gets fresh noise, so the same observation yields different chunks (inferred).
- Model dims:
  - Action expert `gemma_300m`: width 1024, depth 18, MLP 4096, 8 heads, 1 KV head, head_dim 256 (`src/openpi/models/gemma.py:69-78`).
  - PaliGemma `gemma_2b`: width 2048, depth 18, MLP 16384, 8 heads, 1 KV head, head_dim 256 (`gemma.py:79-87`).
  - SigLIP So400m: width 1152 (`src/openpi/models/siglip.py:318`), depth 27 (`:333`).
- Prefix length (inferred from tokenizer padding, `tokenizer.py:35-38`): pi05 with 3 image slots = 3 x 256 + `max_token_len` 200 = 968 positions. pi0 = 3 x 256 + 48 = 816 prefix positions, with state and actions in the suffix.
- PyTorch path (`pi0_pytorch.py`): same structure. `sample_actions` takes `num_steps=10` (`:377`) and is wrapped in `torch.compile(mode=...)` in `__init__` when `pytorch_compile_mode` is set (`:111-113`). The step loop is a Python `while` (`:407-419`). `denoise_step` at `:422-462`.

### 4. RTC, inpainting, prefix-action conditioning

- Does not exist in openpi. Grep for `inpaint`, `rtc`, `real-time chunk`, `prefix_action`, `prev_action`, `guidance`, `soft mask`, `delay` across `*.py`, `*.md`, `*.ipynb`, `*.txt`: no relevant matches.
- The only related hook is the `noise` argument: `sample_actions(..., noise=...)` (`pi0.py:223`; `pi0_pytorch.py:377`) and `Policy.infer(obs, *, noise=...)` (`policy.py:68, 83-88`). The websocket server calls `infer(obs)` without it (`websocket_policy_server.py:61`), so clients cannot set it over the wire. Warm-starting from a previous chunk would need a server change (inferred).

### 5. Server-side state across calls

- Exists only trivially:
  - JAX RNG key `self._rng`, split per call (`policy.py:65, 75`).
  - `PolicyRecorder` step counter (`policy.py:122, 131-132`), only with `--record`.
  - Per-connection `prev_total_time` in the handler (`websocket_policy_server.py:54, 67-72`).
  - JAX module state frozen by `module_jit` (`src/openpi/shared/nnx_utils.py:31-40`).
- Does not exist: session ids, episode boundary, persisted KV cache, server-side `reset` call. The client `reset` is a no-op (`websocket_client_policy.py:56-58`).

### 6. Configs

- `pi05_droid` (`config.py:629-642`): `Pi0Config(action_horizon=15, pi05=True)` (`:631`). `action_dim` 32 (`pi0_config.py:25`). `max_token_len` 200. Discrete state on. Data: `SimpleDataConfig`, `asset_id="droid"` (`config.py:633`), `DroidInputs(PI05)` and `DroidOutputs` (`:634-637`), quantile norm (`:187`). Checkpoint: `gs://openpi-assets/checkpoints/pi05_droid` (`scripts/serve_policy.py:68-71`).
  - Camera keys read: `observation/exterior_image_1_left` to `base_0_rgb`; `observation/wrist_image_left` to `left_wrist_0_rgb`; `right_wrist_0_rgb` is zeros with mask off (`droid_policy.py:44-51, 60-64`).
  - `observation/exterior_image_2_left` is never read in inference. It appears only in the training repack (`config.py:440`).
  - State: `joint_position` (7) + `gripper_position` (1) = 8 (`droid_policy.py:36-40`). Quantile-normalized, tokenized as 8 integers, zero-padded to 32 for the action expert (`transforms.py:328-337`).
  - Output: first 8 of 32 dims (`droid_policy.py:81`). Chunk length 15.
- DROID example observation keys: section 1 (`examples/droid/main.py:119-127`).
  - The example asserts `(10, 8)` (`examples/droid/main.py:134`), and the comment at `:132` says 10. That matches `pi0_droid` (`config.py:602-604`, `action_horizon=10`), not `pi05_droid` (15). The DROID README serves `pi05_droid` (`examples/droid/README.md:22`). (inferred from code, not run) With `pi05_droid` the assert would fail.
- `pi05_base`: not a `TrainConfig`. It is a weight checkpoint, `gs://openpi-assets/checkpoints/pi05_base`. Used as weight loader by `pi05_libero` (`config.py:760`), `pi05_aloha_pen_uncap` (`:824`), `pi05_full_droid_finetune` (`:881`). Also the directory behind `pi05_aloha` in the serve mapping (`scripts/serve_policy.py:60-63`).
  - Architecture defaults from `Pi0Config(pi05=True)`: `action_dim` 32, `action_horizon` 50, `max_token_len` 200, discrete state on (`pi0_config.py:25-26, 37-41`). `pi05_aloha` uses `Pi0Config(pi05=True)` with no horizon override (`config.py:573-577`), so its model output is 50 x 32.
  - ALOHA inputs: 4 cameras. `cam_high` to base, `cam_left_wrist` to left wrist, `cam_right_wrist` to right wrist, `cam_low` accepted but never used (`src/openpi/policies/aloha_policy.py:40, 50, 53, 60-70`). State 14 dims. Output first 14 (`aloha_policy.py:100`).
- `pi05_libero` (`config.py:744-763`): horizon 10, `discrete_state_input=False` (`:745`). Inputs: `observation/image` to base, `observation/wrist_image` to left wrist, right slot masked (`libero_policy.py:56-70`). Output first 7 (`libero_policy.py:100`). Loads `pi05_base` params (`config.py:760`).

## Checked and corrected (premises in the brief)

- "50x32 action chunks" is the `Pi0Config` default (`pi0_config.py:25-26`), not what DROID serving returns. `pi05_droid` = 15 (`config.py:631`). `pi05_libero` = 10 (`config.py:745`). `pi0_droid` = 10 (`config.py:603`).
- "2-3 cameras": DROID reference uses 2 (exterior + wrist), and the third slot is zero-padded and masked (`droid_policy.py:50-51`). The second exterior camera is dropped. ALOHA uses 3 of 4 (`cam_low` unused). LIBERO uses 2.
- `pi05_base` is a weight checkpoint, not a config name.
- Input transforms run on the server, not the client.
- The DROID example's `(10, 8)` assert does not match `pi05_droid`.
- No RTC or prefix-action conditioning in this revision.

## Design implications

### Level 1 harness (runtime-level, no framework change)

1. The protocol is stateless and framework-neutral: connect, read metadata, then obs-to-actions round trips. Any server that speaks it can be driven by the same client loop. vLLM-Omni's endpoint is described in `2026-11-20_realtime-vllm-omni.md`; M* and SGLang were not checked here. Keep payloads identical across frameworks: same keys, 224 resize, uint8, unnormalized state (`docs/remote_inference.md:45-48`).
2. Build the runtime in the harness: sensor threads or snapshots, trigger (count or age), chunk queue, staleness rule. The reference provides only a count trigger and a local FIFO.
3. One outstanding request per connection. No request ids, so an abandoned request leaves its reply in the socket (inferred). The server cannot be asked to abort. Replacing a stale chunk means either waiting and discarding, or closing and reconnecting (new metadata handshake). Record which one each run uses.
4. Multiple client connections queue behind each other on the server's event loop (`websocket_policy_server.py:61`). A multi-stream or multi-robot experiment measures queueing, not parallel inference. Report client wall time per call, `server_timing.infer_ms`, and `prev_total_ms` (which includes idle time). Do not use `policy_timing.infer_ms` as model latency (inferred).
5. Discard warm-up calls. `simple_client` sends 2 (`examples/simple_client/main.py:132-134`). The server does not warm up itself (JIT compile on first call, or `torch.compile` on the PyTorch path).
6. Record the served horizon (15 for `pi05_droid`, 50 for the Pi0.5 default, 10 for `pi05_libero`) and the re-query interval. In the reference DROID loop at 15 Hz with horizon 8, the re-query period is 8 / 15 = 0.53 s. Any call longer than that stalls the loop in the reference client (inferred).
7. Payload: about 150 KB per 224x224x3 uint8 image; about 450 KB per call with 3 cameras. No wire compression (`websocket_policy_server.py:42`; `websocket_client_policy.py:38`).
8. A sensor-only update does not save model compute: every call reruns the full prefix (no cross-call cache). Inference frequency is the only Level 1 knob that changes compute.

### Level 2 (engine-level)

1. Nothing engine-level exists in openpi for streaming: no session, no incremental prefix, no partial-chunk output, no RTC or inpainting, no cancel. Each of these is new work and must be labelled as a deviation from the reference.
2. Incremental prefix is blocked by bidirectional attention (`pi0.py:125, 133`). Any change to any prefix token changes every prefix KV. The only clean partial reuse is per-camera vision embeddings, since SigLIP runs per image (`pi0.py:113-116`) (inferred; not implemented upstream).
3. Pi0.5 puts state in the prefix text (`tokenizer.py:24-28`), so every state update invalidates the prefix KV. Pi0 puts state in the suffix (`pi0.py:151-157`), which is recomputed each denoise step anyway. If the engine streams sensor state, Pi0 is the structurally cheaper case (inferred from masks; not measured).
4. The only in-tree warm-start hook is `noise` (in-process only). Warm-starting from the previous chunk or RTC-style guidance changes the denoise loop (`pi0.py:239-278`). That is an engine-level change and not in the reference.
5. Keep shapes fixed (batch 1, 224 images, `max_token_len`, horizon). JAX `module_jit` compiles per shape (`policy.py:64`).
6. RNG is split per call (`policy.py:75`). Reproducibility needs a controlled seed sequence, or a fixed `noise` input.

## Not verified / could not confirm

- No code was run. Everything is read from source: the `assert` failure on `pi05_droid`, `policy_timing` dispatch semantics, `prev_total_ms` contents, the `pi05_libero` state path, and the `-1` bin case.
- websockets: openpi pins 15.0.1 (`uv.lock`). The source I read is 17.4 from `envs/mstar` (`asyncio/server.py:71-74`: `max_queue=16`, `ping_interval=20`). 15.0.1 defaults were not checked. Whether frames pipelined during `infer` are buffered, and the sync client's keepalive behaviour, were not verified.
- Whether DROID's `RobotEnv` holds the last joint-velocity command during blocked inference. It is an external package, not in this repo.
- Norm stats (keys, q01/q99) for `pi05_droid` and `pi05_base`. They live in GCS assets, not in this repo.
- 256 tokens per image: inferred from patch 14 and `pool_type="none"`. Not checked by running.
- `torch.compile(mode="max-autotune")` behaviour, including CUDA graphs: not checked.
- Not read: `examples/aloha_real/env.py`, `aloha_sim` env, LIBERO env, DROID `RobotEnv`, `msgpack_numpy_test.py`, most of `policy_test.py`, pi0-FAST (out of scope), roboarena and polaris configs, notebooks, docker files.
- Cross-check (not re-verified here): `2026-11-20_realtime-vllm-omni.md` says vLLM-Omni's OpenPI endpoint adds per-connection `session_id` tracking and a `reset` path. Neither exists in this upstream server (`websocket_policy_server.py:48-83`; `websocket_client_policy.py:56-58`).

## Files

- This log: `agents_results/2026-11-20_openpi-remote-pi05.md`.
- `agents_results/README.md` index not updated (read-only scope). The main session should add an entry.

## Main-session spot checks (2026-11-20, `external/openpi` at 215abfb)

- `serving/websocket_policy_server.py:52–72`: one request per loop turn, `server_timing` with `infer_ms` and `prev_total_ms`. Confirmed.
- `training/config.py:629–634`: `pi05_droid` has `action_horizon=15`, `pi05=True`. Confirmed. `config.py:603` (`pi0_droid`) has `action_horizon=10`. Confirmed.
- `models/pi0.py:222`: `num_steps=10` default. Confirmed. `pi0.py:125–133`: image and language tokens use `ar_mask` False, i.e. full attention. Confirmed.
- `policies/droid_policy.py:36–40`: state is joint positions plus gripper, 8 dims. Confirmed.
- Not re-checked: the DROID example assertion at `examples/droid/main.py:134`, and the `ActionChunkBroker` details.

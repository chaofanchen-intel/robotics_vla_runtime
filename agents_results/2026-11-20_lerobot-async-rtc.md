# LeRobot v0.7.0: async/remote inference, RTC, Pi0.5, and timing (source analysis)

迁移说明（2026-11-20）：本文件从基准仓库 `agents_results/` 迁入 `robotics_vla_runtime/agents_results/`。路径已改为本目录的 `external/`（上游最新快照）和 `docs/design.md`。文中的 `/tmp/...` 是当时的临时克隆，未保留。

## Date and role

- Date: 2026-11-20
- Role: read-only source analyst (research subagent). No GPU jobs, no installs, no edits except this file, no commit. Downloaded code was read, never executed.

## Question

1. Async inference: which side decides when to request a chunk (fixed period or queue threshold); config fields and defaults; how observations are timestamped and selected (latest-only or history); how chunks are merged, replaced, or discarded (queue class and methods); server concurrency (one request or several clients); transport; cancellation and staleness handling.
2. RTC: how the committed action prefix constrains the next chunk (inpainting, guidance, averaging, or other); parameters and defaults; whether RTC sits inside the denoise loop of one policy class; which policies support it (is pi05 listed); whether it needs a model forward-path change; quote the code.
3. Pi0.5 in LeRobot: how state becomes prompt text (bins); number of flow/denoise steps; whether prefix KV is reused across denoise steps; whether anything is kept across calls; server vs client preprocessing split and expected input keys.
4. Any timestamp alignment or multi-rate synchronization code; say explicitly if none exists.

## Scope

- Code root: `/home/chaofan/workspaces/robotics_vla_runtime/external/lerobot`, shortened to `LR/` below.
- Version: tag v0.7.0, commit 410c095eacc0011d998a73383eb94abed2715af0 (checked with `git describe --tags --exact-match` and `git rev-parse HEAD`). The clone is shallow with one commit, so no history is available for removed packages.
- Read: `LR/src/lerobot/remote_inference/{README.md,configs.py,client.py,server.py,engine.py,protocol.py,deployment.py,__init__.py,codec.py,chunk_contract.py}`; `LR/src/lerobot/inference/{factory.py,execution.py,policy_runner.py,prediction.py,rtc.py,dispatch.py,chunk_settings.py}`; `LR/src/lerobot/policies/rtc/{configuration_rtc.py,modeling_rtc.py,action_queue.py}`; `LR/src/lerobot/policies/common/flow_matching.py`; `LR/src/lerobot/policies/pi05/{modeling_pi05.py,processor_pi05.py,configuration_pi05.py}`; `LR/src/lerobot/policies/pretrained.py`; `LR/src/lerobot/transport/zenoh.py`; `LR/src/lerobot/rollout/{context.py,robot_wrapper.py}` (parts); `LR/src/lerobot/rollout/strategies/core.py` (parts); `LR/src/lerobot/utils/{cycle_timer.py,action_interpolator.py}` (parts); `LR/src/lerobot/scripts/lerobot_policy_server.py`; `LR/docs/source/{remote_inference.mdx,rtc.mdx,policy_rtc_README.md,pi05.mdx}`; `LR/pyproject.toml` (lines 291, 420).
- Marks: [V] = I read the cited line(s). [I] = inferred from code reading, not run.

## Naming at v0.7.0 (differs from the question's names)

- `lerobot.async_inference` is gone. `LR/src/lerobot/async_inference/` does not exist, and no import or command in `src` references it (grep; the only hits are a test function name and the migration note). The migration note is at `LR/docs/source/remote_inference.mdx:437-441` [V]: the old extra and commands are replaced by `lerobot[remote]`, `lerobot-policy-server`, and `lerobot-rollout --inference.type=remote`.
- The question's `robot_client.py` and `policy_server.py` correspond to `LR/src/lerobot/remote_inference/client.py` and `server.py`, with `engine.py` as the robot-side worker. Entry point: `LR/src/lerobot/scripts/lerobot_policy_server.py` [V].
- `LR/docs/source/async.mdx` does not exist. The equivalent is `LR/docs/source/remote_inference.mdx` (447 lines). The RTC doc is `LR/docs/source/rtc.mdx`.
- `LR/src/lerobot/remote_inference/README.md` is the maintainers' design reference. I used it as a map and checked each claim against code.

## Result

### Q1. Async / remote inference

**Architecture** [V]: The robot side is `RemoteInferenceEngine` (`LR/src/lerobot/remote_inference/engine.py:52`), which runs one worker thread for network exchanges. Scheduling, merging, and the fault latch live in `ChunkRuntime` (`LR/src/lerobot/inference/execution.py:143`). The server is a `PolicyServer` pump (`LR/src/lerobot/remote_inference/server.py:851`) that feeds one `SessionWorker` (`server.py:127`) with a single execution thread (`server.py:635`).

**Who requests, and when** [V]:
- The client decides. The server has no timer and only answers requests.
- The worker polls every 2 ms (`engine.py:760`, `:777`) and calls `ChunkRuntime.should_request()` (`engine.py:759`; `execution.py:357-384`).
- The trigger is a playback threshold, not a fixed period. `playback = self.queue.qsize() * self.interval` and the request fires when `playback <= self.effective_refill + 1e-9` (`execution.py:383-384`).
- `effective_refill = max(refill_seconds, turnaround + interval)` (`execution.py:228-230`). `turnaround` is the maximum over up to 100 steady-state samples (`execution.py:215`, `:222-225`). The first accepted turnaround is the initial floor (`execution.py:448-453`).
- Gates before the threshold: not failed, active, held, or pending (`execution.py:365`). In append mode, at most one accepted successor beyond the executing chunk (`execution.py:375-382`). In aligned mode, a task-version change bypasses the threshold (`execution.py:368-374`).
- Local RTC uses the same runtime with `refill_seconds = queue_threshold / fps` (`LR/src/lerobot/inference/rtc.py:165-168`). The 30-action default is 1.0 s at 30 fps [I, arithmetic].

**Config fields and defaults** [V]:

Client, `RemoteInferenceConfig` (`LR/src/lerobot/inference/factory.py:100-133`):

| field | default | line |
|---|---|---|
| refill_seconds | 0.5 s of remaining playback | :118 |
| chunk_merge | "auto": aligned for mode=chunk, append for RTC | :120, :151-152 |
| blend_steps / blend_weight / blend_components | 0 / 0.5 / [] | :121-123 |
| max_observation_age_s | 5.0 s | :124 |
| handshake_timeout_s | 10.0 s | :125 |
| action_timeout_s | 5.0 s | :126 |
| language_timeout_s | 60.0 s | :127 |
| startup_timeout_s | 10.0 s | :128 |
| action_starvation_grace_s | 1.0 s | :129 |
| encoding / jpeg_quality | raw / 90 | :130-131 |
| zenoh_mode | peer | :133 |

Server, `ExecutionConfig` (`LR/src/lerobot/remote_inference/configs.py:45-55`): supported_modes `["chunk"]` (:48), action_fps 30.0 (:49), action_deadline_s 5.0 (:51), idle_timeout_s 10.0 (:52), warmup_calls 2 (:53), n_action_steps None (:55). `LanguageConfig.deadline_s` 60.0 (:63).

Local RTC, `RTCInferenceConfig` (`factory.py:78-97`): queue_threshold 30 actions (:86), max_observation_age_s 5.0 (:87), action_timeout_s 10.0 (:88), startup_timeout_s 120.0 (:89), language_timeout_s 120.0 (:90), action_starvation_grace_s 1.0 (:91).

Admission check: the client `--fps` must equal the server action rate, or admission fails (`LR/src/lerobot/remote_inference/client.py:257-261`) [V].

**Observation timestamp and selection** [V]:
- Rollout path: `ThreadSafeRobot.get_observation` records `time.monotonic()` before the robot read (`LR/src/lerobot/rollout/robot_wrapper.py:50-59`). One stamp covers the whole read, cameras and motors together. There is no per-sensor timestamp.
- Engine path: `submit_observation(features, capture_time=None)` defaults capture_time to now, which is after the read if the caller reads first (`engine.py:515-518`).
- Selection is latest-only. `self._observation = snapshot` overwrites the slot (`engine.py:521-528`). The worker takes the current slot each loop (`engine.py:686`) and requires `observation is self._observation` at begin (`engine.py:762-766`). There is no history buffer.
- Rollout submission is throttled to ticks where the interpolator needs a new action (`LR/src/lerobot/rollout/strategies/core.py:243-248`). `get_observation` still runs every loop (`core.py:232-233`). At interpolation multiplier 1 this is every tick.
- `begin()` rejects captures older than `max_age` or in the future (`execution.py:391-393`). Aligned mode also requires capture_time >= started_at and strictly newer than the last anchor (`execution.py:405`, `:408-415`).

**Merge, replace, discard** [V]:
- Queue class: `ActionQueue` (`LR/src/lerobot/policies/rtc/action_queue.py:53`), owned by `ChunkRuntime`. Remote chunk mode builds `ActionQueue(RTCConfig(enabled=mode is not CHUNK))` (`execution.py:204`), so `cfg.enabled` is False for chunk mode.
- Append (`chunk_merge=append`): `ActionQueue.merge` (`action_queue.py:236-276`) takes the non-RTC branch (`:273-276`), which calls `_append_actions_queue` (`:308-337`). Remaining actions are kept and the new chunk is concatenated after them.
- Aligned (default for plain chunks): `ChunkRuntime._accept_aligned` (`execution.py:539-582`). The trim is `snapshot.cursor - observation.action_cursor` (`:547`). Future = `actions[trim:]` (`:552`). Optional blend (`:557-576`). Then `ActionQueue.replace_future` (`action_queue.py:154-176`), which fails unless generation and cursor are unchanged (`:169-170`). The cursor counts committed endpoints since the last clear (`action_queue.py:126`, `:151`). The anchor is taken at observation time by `anchor_observation` (`execution.py:232-244`).
- RTC: `ActionQueue._replace_actions_queue` drops the first `delay` steps (`action_queue.py:278-306`, clamp at `:296-298`). `merge` caps delay at `min(real_delay, cursor progress)` (`:264`). `ChunkRuntime.accept` trims by `min(measured, progress)` for RTC (`execution.py:476-484`).
- Discard and fault paths:
  - `invalidate()` bumps the generation, clears the queue and the pending request (`execution.py:246-268`).
  - Silent discard if the generation differs or the request is not current (`execution.py:439`), or if the task version differs (`:442-443`).
  - Aligned: an entirely consumed suffix is discarded, not faulted (`execution.py:548-551`).
  - Terminal fault if a result's source is older than max_age (`execution.py:454-455`), or if a committed endpoint is too old at dispatch (`execution.py:596-597`), or if `trim >= len(actions)` in RTC (`:485-486`).
- Queue methods used: `snapshot`, `merge`, `replace_future`, `get_with_provenance`, `clear`, `qsize`, `empty`, `get_left_over`, `get_processed_left_over`, `_replace_actions_queue`, `_append_actions_queue`, `_check_and_resolve_delays` (`action_queue.py`). `ChunkRuntime` methods: `anchor_observation`, `should_request`, `begin`, `accept`, `_accept_aligned`, `invalidate`, `pop`, `dispatch_allowed`, `check_deadlines`, `activate`, `deactivate`, `release_hold`, `fault` (`execution.py`).

**Server concurrency** [V]:
- One client per server process. A second OPEN while a session exists gets BUSY (`LR/src/lerobot/remote_inference/server.py:292-312`).
- One outstanding inference: `session.busy` returns BUSY (`server.py:358-361`). Sequence numbers must increase (`:363-364`). Generation must match (`:354-355`) and is checked again at execution (`:579-580`).
- One execution thread (`server.py:635-640`). A request that waits longer than `action_deadline_s` gets TIMEOUT and the session is faulted (`server.py:659-662`).
- Pump: observation subscriber capacity 1 (`server.py:880`), outbound reply cap 16 (`:900`, `:923`).

**Transport** [V]:
- Zenoh only. Pub/sub carries observations and actions, query/reply carries open and control, liveliness carries presence (`LR/src/lerobot/remote_inference/README.md:316-324`; `server.py:870-883`).
- Explicit endpoints. Multicast and gossip scouting and shared memory are disabled (`LR/src/lerobot/transport/zenoh.py:86-88`). Publishes use DROP congestion control (`zenoh.py:321-326`). Payload cap 32 MiB (`zenoh.py:58`). Channels are bounded and drop the incoming value when full (`zenoh.py:106-130`).
- Messages: bounded MessagePack, no pickle (`README.md:329`; `LR/pyproject.toml:291`). Protocol version is exactly 1 (`LR/src/lerobot/remote_inference/protocol.py:37`).
- Images: raw RGB by default, optional JPEG (`client.py:410-415`; `factory.py:130-131`).
- gRPC stubs (`LR/src/lerobot/transport/services_pb2*.py`, `services.proto`) remain in the tree. No Python module imports them (grep for `services_pb2` finds nothing) [V]. They are off the inference path. No websocket code exists in the remote, inference, or transport packages.

**Cancellation and staleness** [V]:
- Client cancel: the worker's closure returns true on stop or on `runtime.generation != request_generation` (`engine.py:799-800`). `_exchange` then raises `RequestCancelled` (`client.py:431-432`). Cancelling does not interrupt the server (`README.md:66`).
- Stale replies are discarded by `decode_reply(..., discard_stale=True)` (`client.py:447`).
- A control ACK must echo the requested generation, or the client raises STALE (`client.py:393-397`).
- Timeouts: `_exchange` raises TimeoutError at its deadline (`client.py:430`, `:450`). The worker re-raises (`engine.py:815-821`), and the outer handler faults the engine (`engine.py:843-846`). So any action timeout is terminal.
- Starvation: when the queue runs dry, `_wait_for_actions` invalidates and holds for `action_starvation_grace_s` (`execution.py:270-281`). Resumption needs a fresh capture after the hold ack (`execution.py:283-287`, `:391-398`). Late packets do not extend the grace (`README.md:268`).
- Connection loss is not recovered automatically (`LR/docs/source/remote_inference.mdx:241`; `README.md:304`).

### Q2. RTC

**Where it lives** [V]:
- The guidance math is in a shared class, `RTCProcessor` (`LR/src/lerobot/policies/rtc/modeling_rtc.py:38`). It wraps a policy's velocity function inside a shared loop, `euler_integrate` (`LR/src/lerobot/policies/common/flow_matching.py:256-421`; guided call at `:394-406`).
- Policy loops call it. π0.5: `PI05Pytorch.sample_actions` (`LR/src/lerobot/policies/pi05/modeling_pi05.py:934-1021`; call at `:1004-1021`). π0 calls it at `LR/src/lerobot/policies/pi0/modeling_pi0.py:708`.
- Opt-in per policy via `supports_rtc()`. Base default is False (`LR/src/lerobot/policies/pretrained.py:280-282`). True in source:
  - smolvla (`LR/src/lerobot/policies/smolvla/modeling_smolvla.py:150-151`)
  - groot (`LR/src/lerobot/policies/groot/modeling_groot.py:75-76`)
  - evo1 (`LR/src/lerobot/policies/evo1/modeling_evo1.py:39-40`)
  - pi0 (`LR/src/lerobot/policies/pi0/modeling_pi0.py:772-773`)
  - **pi05** (`LR/src/lerobot/policies/pi05/modeling_pi05.py:1072-1073`)
  - lingbot_vla_v2 (`LR/src/lerobot/policies/lingbot_vla_v2/modeling_lingbot_vla_v2.py:2241-2242`)
  - molmoact2, only when `inference_action_mode == "continuous"` (`LR/src/lerobot/policies/molmoact2/modeling_molmoact2.py:635-636`)
- The call shape must accept `inference_delay` and `prev_chunk_left_over` (`LR/src/lerobot/inference/rtc.py:86-100`; typed kwargs at `LR/src/lerobot/policies/pretrained.py:65-70`).
- Enablement: remote sets `policy_cfg.rtc_config` only when a non-chunk mode is requested (`LR/src/lerobot/remote_inference/deployment.py:142-145`). Local sets `policy.config.rtc_config = cfg.inference.rtc` after the support check (`LR/src/lerobot/rollout/context.py:525-534`).

**How the committed prefix constrains the next chunk** [V unless marked]:
1. The prefix is the unconsumed model-space leftover at request time, `original_queue[last_index:]` (`action_queue.py:129-141`, `:209-222`). It is sent as `model_continuation` with `delay` and `cursor` (`client.py:455-468`). The server truncates it to `execution_horizon` and pads by repeating the last step (`LR/src/lerobot/inference/prediction.py:77-107`; comment at `:105`).
2. `delay = ceil(turnaround / interval)` (`execution.py:120-135`, `:132`). It estimates steps the robot consumes during inference [I].
3. Guided mode (default). Inside each Euler step, `RTCProcessor.denoise_step` gets v_t from the policy and then applies (`modeling_rtc.py:216-233`):
   ```
   v_t = original_denoise_step_partial(x_t)            # :217
   x1_t = x_t - time * v_t                              # :220
   err = (prev_chunk_left_over - x1_t) * weights        # :221
   correction = torch.autograd.grad(x1_t, x_t, grad_outputs, retain_graph=False)[0]  # :223
   guidance_weight = min(c * inv_r2, max_guidance_weight)  # :225-231
   result = v_t - guidance_weight * correction          # :233
   ```
   Weights (`modeling_rtc.py:209-214`, `:255-274`), LINEAR default: 1 for i < delay, linear decay over [delay, execution_horizon), 0 for i >= execution_horizon. This is soft, gradient-based inpainting, not averaging and not an exact constraint. At the first Euler step time=1 so tau=0 and the (1-tau)/tau term is +inf, which `nan_to_num` clamps to max_guidance_weight [I, from `:227-231`].
4. Trained mode (π0.5 only; needs a checkpoint with rtc_training_max_delay > 0). Hard inpainting: `_prepare_trained_rtc_prefix` puts the first `inference_delay` steps of the prefix in place and builds a mask (`modeling_pi05.py:81-128`; mask at `:126-128`). `euler_integrate` overwrites those positions with the prefix before each step and again after the update (`flow_matching.py:370-375`, `:415-416`), and sets their time to clean (`:370-375`). Only the delay prefix is hard-conditioned, not the whole execution horizon (`LR/docs/source/rtc.mdx:108-109`).
5. Merge: the first min(measured, progress) steps of the incoming chunk are discarded (`action_queue.py:264`, `:296-298`; `execution.py:476-484`).
6. The only explicit mix of old and new targets is non-RTC aligned blending: `(1-w)*old + w*new` on the first blend_steps overlapping positions (`execution.py:560-572`; `LR/docs/source/remote_inference.mdx:165`).
7. Non-RTC append has no constraint. The new chunk is queued after the old one (`action_queue.py:308-337`).

**Parameters and defaults** [V]:
- `RTCConfig` (`LR/src/lerobot/policies/rtc/configuration_rtc.py:57-80`): enabled True (:66), mode "guided" (:70), prefix_attention_schedule LINEAR (:74), max_guidance_weight 10.0 (:75), execution_horizon 10 (:76), debug False (:79), debug_maxlen 100 (:80).
- Delay is not a config field. It is a runtime estimate (`execution.py:120-135`) passed per request (`prediction.py:153-154`).
- Trained admission: training_max_delay <= execution_horizon <= prediction_steps - training_max_delay (`configuration_rtc.py:29-54`). Local rollout also requires queue_threshold >= training_max_delay (`LR/src/lerobot/rollout/context.py:120-133`).
- Pi05 config defaults: `rtc_config` None (`LR/src/lerobot/policies/pi05/configuration_pi05.py:77`), `rtc_training_max_delay` 0 (:79). A checkpoint must set rtc_training_max_delay > 0 to get trained RTC; the default config does not.
- Deployment checks: RTC settings must match the requested modes (`LR/src/lerobot/remote_inference/configs.py:95-100`). PolicyRunner requires execution_horizon <= prediction_steps (`LR/src/lerobot/inference/policy_runner.py:103-104`).

**Model forward path** [V/I]:
- Guided: no change to π0.5's `denoise_step`. RTC wraps the velocity call (`flow_matching.py:400-402`). It needs:
  - a gradient with respect to x_t: `RTCProcessor.denoise_step` runs `torch.enable_grad()` and calls autograd (`modeling_rtc.py:216-223`);
  - no inference_mode for guided: `torch.inference_mode(mode is not RTC_GUIDED)` (`LR/src/lerobot/inference/prediction.py:51-59`);
  - `predict_action_chunk` to accept and forward the kwargs. π0.5 forwards `**kwargs` (`modeling_pi05.py:1511-1543`).
  - [I] One VJP per Euler step, so about 10 backward passes per π0.5 chunk. The doc says the same: re-measure turnaround because guidance adds work (`LR/docs/source/remote_inference.mdx:182`).
- Trained: inference only clamps. The checkpoint must have been trained with clean-prefix conditioning (`modeling_pi05.py:1571-1576`; `flow_matching.py:236-243`). This is a checkpoint change, not a forward-path change.
- `PI05Policy.select_action` asserts RTC is disabled (`modeling_pi05.py:1494-1496`). RTC is carried only by `predict_action_chunk`.

**Quoted code** (guided path, `LR/src/lerobot/policies/rtc/modeling_rtc.py:216-233`):
```python
with torch.enable_grad():
    v_t = original_denoise_step_partial(x_t)
    x_t.requires_grad_(True)

    x1_t = x_t - time * v_t  # noqa: N806
    err = (prev_chunk_left_over - x1_t) * weights
    grad_outputs = err.clone().detach()
    correction = torch.autograd.grad(x1_t, x_t, grad_outputs, retain_graph=False)[0]
...
result = v_t - guidance_weight * correction
```

**Doc versus code** [V]: `LR/docs/source/rtc.mdx:53` and `:128` recommend EXP ("recommended for getting started"), but the code default is LINEAR (`configuration_rtc.py:74`).

### Q3. Pi0.5

**State to prompt text** [V]:
- Normalize first. STATE and ACTION use QUANTILES (`LR/src/lerobot/policies/pi05/configuration_pi05.py:95-96`). The formula is `2.0 * (tensor - q01) / denom - 1.0` (`LR/src/lerobot/processor/normalize_processor.py:404`). The branch raises if q01/q99 are missing (`normalize_processor.py:392-395`). There is no clamp.
- Discretize: `np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1` (`LR/src/lerobot/policies/pi05/processor_pi05.py:79`). Output is integers 0 to 255.
- Prompt: `f"Task: {cleaned_text}, State: {state_str};\nAction: "`, with state integers joined by spaces (`processor_pi05.py:85-88`). Task text is stripped and underscores and newlines replaced (`:83`).
- The state goes into the prompt only when `use_proprioceptive_memory` is False (`processor_pi05.py:54`, `:155`).
- Stock path has no separate state embedding. State is None unless memory is on (`modeling_pi05.py:1440-1442`), and the state token path is memory-only (`:795-798`). Padding to max_state_dim happens only in the memory path (`:1443`). `max_state_dim` is not referenced in the step's `__call__` (`processor_pi05.py:56-94`) [V].
- Tokenizer: PaliGemma, max_length 200 (`configuration_pi05.py:89`), right padding (`processor_pi05.py:157-162`).
- Edge [I, not run]: the state is not clamped before digitize. Values below -1 get bin index -1 (text "-1") because `np.digitize` returns 0. Values at or above about 0.9922 saturate at 255. The comment at `processor_pi05.py:75` assumes values are already in [-1, 1].
- LeRobot's pi05 doc agrees on the discretization (`LR/docs/source/pi05.mdx:65`).

**Steps and chunk** [V]:
- chunk_size 50 (`configuration_pi05.py:38`), n_action_steps 50 (:39), num_inference_steps 10 (:61).
- `sample_actions` uses `num_steps = self.config.num_inference_steps` (`modeling_pi05.py:947-948`) and calls `euler_integrate` with it (`:1004-1012`). Uniform dt is -1/N with times 1.0 down to 0.1 (`flow_matching.py:336-348`).
- Output is cropped to the original action dimension (`modeling_pi05.py:1546-1549`). Internal width is max_action_dim 32 (`configuration_pi05.py:58`). So "50x32" is internal padding, and the emitted width is the dataset's action dimension.

**Prefix KV reuse** [V]:
- The prefix is prefilled once per call with `use_cache=True` (`modeling_pi05.py:964-980`). The cache is passed to every denoise step (`:1005-1010`). `denoise_step` clones the cache each step (`:1047`) and runs only the suffix with `use_cache=False` (`:1051-1053`).
- Across calls: no cache is stored on `self` (no matches for `self._past`, `self.past_key`, `self._kv`, `self._cache`). `PI05Policy.reset` clears only action and memory queues (`modeling_pi05.py:1316-1341`).
- Other state kept across calls on the server:
  - `PolicyRunner._model_action_dim` (`LR/src/lerobot/inference/policy_runner.py:83`, `:262`);
  - stateful saved processors, such as `RelativeActionsProcessorStep` caching state (`prediction.py:99`; queue unbound at `policy_runner.py:142-143`). Relative actions are off by default (`configuration_pi05.py:70`);
  - memory queues, which chunk serving forbids (`LR/src/lerobot/policies/pretrained.py:336-339`);
  - the RTC continuation is supplied by the client per request, so the server holds none (`client.py:455-468`).
- Warmup runs on zero observations, then `reset(full=True)` (`LR/src/lerobot/remote_inference/deployment.py:182-218`).

**Server vs client preprocessing and expected keys** [V]:
- Client: capture (`robot_wrapper.py:50-59`); rename local camera names to checkpoint names (`engine.py:497`); drop non-admitted features (`engine.py:501-502`); optional JPEG (`client.py:410-415`). No normalization or tokenization on the client.
- Server expected keys: exactly the admitted `features` list (`ServerConfig.features`, `LR/src/lerobot/remote_inference/configs.py:80`). Each request's key set, shape, and dtype must match (`server.py:463-476`). Each request also carries `task` and `task_version` (`server.py:455-458`).
- Doc example: `observation.state` float32 [6] with component names, and `observation.images.front` RGB uint8 [480, 640, 3] (`LR/docs/source/remote_inference.mdx:298-317`).
- `PolicyRunner._batch`: checks shape and dtype against the admitted spec (`LR/src/lerobot/inference/policy_runner.py:212-217`), converts RGB uint8 to float32 CHW divided by 255 (`:221-222`), adds `task` and `robot_type` (`:224-225`).
- Saved preprocessor, with rename map forced empty (`LR/src/lerobot/remote_inference/deployment.py:158-165`): add batch, relative (off), normalize, pi05 state prompt, tokenize, to device (`processor_pi05.py:146-164`).
- Model: resize with pad to 224x224 (`modeling_pi05.py:1408-1412`; default `configuration_pi05.py:81-84`). The client sends native camera resolution, and the deployment must admit that shape.
- Postprocess: unnormalize, to CPU (`processor_pi05.py:166-170`). `predict_chunk` slices to n_action_steps (`prediction.py:185-187`).

**Status in LeRobot docs** [V]: π0.5 is a "candidate". The README says "No remote HW acceptance established here" (`LR/src/lerobot/remote_inference/README.md:158`). The hardware-validated list is ACT, SmolVLA, XVLA, EVO1, LaWAM (`LR/docs/source/remote_inference.mdx:52`). The README lists π0.5 as RTC-declared (`README.md:168`).

### Q4. Timestamp alignment and multi-rate sync

**None exists** in the inference, remote, RTC, or rollout path at v0.7.0 [V]. There is no sensor-to-sensor timestamp matching, no resampling, and no camera-to-motor synchronization. What does exist:
- One monotonic stamp per robot read (`robot_wrapper.py:50-59`).
- Client-local age bounds at begin, accept, and dispatch (`execution.py:391-393`, `:454-455`, `:596-597`). The docs say clocks need not be synchronized (`LR/docs/source/remote_inference.mdx:332`; `README.md:204`).
- Logical alignment through the action cursor (`execution.py:240-244`, `:547`; `client.py:469-470`; `server.py:489-496`). The server treats capture_time as opaque (`server.py:461-462`).
- Control cadence: `CycleTimer` paces ticks at 1/fps with precise sleep (`LR/src/lerobot/utils/cycle_timer.py:104-108`, `:218`, `:429`). Policy requests are not on that clock.
- Motor interpolation between consecutive policy actions (`LR/src/lerobot/utils/action_interpolator.py:24-52`). This is temporal interpolation, not synchronization.
- Dataset side, for training and offline use only [I]: `check_delta_timestamps` requires deltas to be multiples of 1/fps within tolerance (`LR/src/lerobot/datasets/feature_utils.py:174-215`). Query windows carry padding masks (`LR/src/lerobot/datasets/dataset_reader.py:298-311`). Streaming decode tolerances are in `LR/src/lerobot/streaming/episode_cache.py:71`, `:292-312`.

## Notes for the team (code observations, not run)

1. `submit_observation` without capture_time stamps "now", after the read (`engine.py:515-518`). Pass a pre-read timestamp.
2. Most staleness failures are terminal faults, not silent drops (`engine.py:815-846`; `execution.py:454-455`, `:596-597`). Only generation and task mismatches are silently discarded.
3. The default `chunk_merge=auto` is aligned (`factory.py:151-152`). A baseline must set `--inference.chunk_merge=append` explicitly.
4. Local RTC queue_threshold is in actions, while remote refill is in seconds (`factory.py:86`, `:118`; `rtc.py:168`).
5. `RTCInferenceConfig.queue_threshold` is not validated for sign (`factory.py:93-97`). A negative value would make the worker never request (`rtc.py:572`) [I, not run].
6. The RTC doc example uses EXP while the code default is LINEAR (see Q2).
7. Chunk mode still sends delay and continuation on the wire (`client.py:455-468`; validated at `server.py:497-517`). `predict_chunk` ignores them in chunk mode (`prediction.py:133-134`). Size is small (up to 50x32 float32 per request) [I].

## Checked and corrected

- Main session, 2026-11-20 (read the lines again in `external/lerobot`, v0.7.0):
  - `src/lerobot/async_inference/` does not exist; `src/lerobot/remote_inference/` does (`client.py`, `server.py`, `engine.py`, `transport/`). Confirmed rename.
  - `policies/pi05/modeling_pi05.py:1072–1073` `supports_rtc()` returns True. Confirmed.
  - `policies/rtc/configuration_rtc.py:66–76`: mode `guided`, `prefix_attention_schedule = LINEAR`, `max_guidance_weight = 10.0`, `execution_horizon = 10`, with the source comment "Todo change to exp". Confirmed: the code default is LINEAR, and the docs' EXP recommendation disagrees with it.
  - `inference/factory.py:118–129`: `refill_seconds 0.5`, `chunk_merge "auto"`, `max_observation_age_s 5.0`, `action_timeout_s 5.0`, `action_starvation_grace_s 1.0`. Confirmed.
  - `remote_inference/README.md:66`: cancellation "does not interrupt a GPU call". Confirmed.
  - `remote_inference/server.py:292–312` rejects a second session with BUSY while one is owned. This is consistent with the one-client claim.

## Unverified / could not confirm

- Nothing was executed. No tests, no runtime timing, no wire-size measurement.
- Not read: `LR/src/lerobot/transport/protocol/codec.py` and `decode_reply` in full; `LR/tests/remote_inference/*`; `LR/examples/remote_inference/server.yaml`; `LR/docs/source/inference.mdx`; the rollout main loop beyond `notify_observation` and the return-to-home loop; server presence and expiry paths beyond the lines cited.
- Git history: the clone is shallow with one commit, so I could not find when `async_inference` was removed.
- Whether any published π0.5 checkpoint has rtc_training_max_delay > 0. The repo names none. `LR/docs/source/pi05.mdx:179-227` describes training one.
- Whether guided RTC works under torch.compile or CUDA graphs. This is reasoned from the autograd call (`modeling_rtc.py:216-223`), not tested.
- numpy `digitize` behavior for out-of-range state: from numpy semantics, not executed.
- Quality difference between LINEAR and EXP schedules: not measured.
- Whether the rollout core submits on every tick at multiplier 1: only `core.py:232-248` was read.

## Design implications

### Level 1: runtime harness (no framework change)

- Scheduling: use a playback threshold with measured turnaround, not a fixed period. Mirror `effective_refill = max(refill, p_max_turnaround + interval)` (`execution.py:228-230`). Log configured and effective values.
- Observations: latest-only slot, one request in flight, and a pre-read capture stamp (`engine.py:515-528`; `robot_wrapper.py:52-59`). LeRobot does not measure cross-sensor skew, so the harness must add its own timing if it needs that.
- Alignment: trimming a returned chunk needs the number of actions consumed at capture time. LeRobot's aligned mode needs the commitment cursor and generation echoed (`client.py:469-470`; `server.py:489-496`). Without them, only append is available, with the backward-bump behavior the doc describes (`remote_inference.mdx:277`).
- Staleness: LeRobot faults on stale results. For a benchmark, either match that and report the fault, or record stale events as data without aborting. Pick one and state it.
- Metrics to copy from LeRobot's events: playback at submission, observation age at submission, turnaround, trimmed and usable steps, source age at dispatch (`engine.py:779-842`; `README.md:347`).
- Transport: LeRobot uses Zenoh. Other frameworks' transports add overhead the harness must measure separately. Raw RGB at 640x480x3 is about 0.92 MB per camera frame [I, arithmetic].
- Timing: use the client's monotonic clock only. The server treats stamps as opaque (`server.py:461-462`), so no clock sync is needed.
- π0.5 request shape for Level 1 (no RTC): chunk mode, chunk 50, n_action_steps 50 (`configuration_pi05.py:38-39`), 10 denoise steps, memory off (`pretrained.py:336-339`).
- Keep the continuation on the runtime side. Chunk requests need no server-side state beyond the session. The client already holds the continuation and sends it per request (`client.py:455-468`), which keeps Level 1 and Level 2 separable.

### Level 2: engine-level (RTC or other denoise changes)

- Minimal engine contract, matching LeRobot's shared predict path (`prediction.py:110-191`): `predict(obs, inference_delay: int, prefix: [<= execution_horizon, action_dim]) -> chunk`. The runtime keeps the leftover and computes delay. The engine should not own the queue (`README.md:64-66`).
- Guided RTC needs a gradient through the velocity at each denoise step (`modeling_rtc.py:216-223`) and must not run under inference mode (`prediction.py:58`). Compiled or CUDA-graph denoise loops would need re-capture with autograd, or an analytic correction. Expect added per-step cost (`remote_inference.mdx:182`).
- Trained RTC is a clamp of the first `delay` steps after each Euler update (`flow_matching.py:370-375`, `:415-416`), which is cheap at inference. It needs a checkpoint trained with clean-prefix conditioning (`modeling_pi05.py:1571-1576`). If no such checkpoint can be found, the cross-framework RTC comparison is guided-only.
- Prefix sizing: prefix length = min(execution_horizon, leftover). π0.5 default execution_horizon is 10 (`configuration_rtc.py:76`). A short prefix is padded by repeating the last step (`prediction.py:104-106`). The three frameworks must make the same choice.
- KV reuse: the π0.5 prefix cache lives only inside one call (`modeling_pi05.py:964-1053`). The M* plan notes a frozen prefix read by action_gen (`agents_results/2026-11-14_pi05-third-case-plan.md:25`). Frameworks must keep the prefix per call, and copy or clone it if the suffix forward mutates it [I].
- Fair comparison: LeRobot's default for plain chunks is aligned, not RTC. Compare RTC arms only with RTC arms, and aligned or append only with the same merge mode. The results table should name the mode.

## Main-session correction (2026-11-20, from the full-history check)

- The removal point is known now: `async_inference` exists at tag v0.6.1 and is removed in commit `ae2b57f12` (#4937, "add remote inference engine"), which is contained in v0.7.0. The statement "removal point unknown" above is resolved.
- v0.6.1 `async_inference`: the server is not serialized (gRPC pool of 4 threads, `policy_server.py:426`, no inference lock, `policy_server.py:324`). The "one in-flight request" statement in Q1 applies to the v0.7.0 `remote_inference` session admission only.

# Verify the unverified upstream items (design doc section 7)

迁移说明（2026-11-20）：本文件从基准仓库 `agents_results/` 迁入 `robotics_vla_runtime/agents_results/`。路径已改为本目录的 `external/`（上游最新快照）和 `docs/design.md`。文中的 `/tmp/...` 是当时的临时克隆，未保留。

## Date and role

- Date: 2026-11-20.
- Role: read-only verifier. No GPU jobs, no servers, no package installs, no commits. The only file edited is this log. Downloaded code was read, never executed. Numpy snippets ran with `-I` on an existing project interpreter (`envs/mstar/bin/python3`, numpy 2.5.4); nothing was installed.

## Question

Verify items A–J from `docs/design.md` section 7 (unverified items). Each item is marked VERIFIED, REFUTED, or NOT VERIFIABLE HERE.

- A. LeRobot history: newest tag with `src/lerobot/async_inference/`; robot client and policy server paths; send trigger; chunk merge; server concurrency.
- B. LeRobot `ActionQueue` (v0.7.0): lock, methods that take it, callbacks under lock.
- C. OpenPI `websockets` pin.
- D. OpenPI `examples/droid/main.py` action-shape assertion versus `pi0_droid` and `pi05_droid`.
- E. OpenPI `ActionChunkBroker`: when it re-queries and what it does with observations in between.
- F. DROID `RobotEnv`: whether a blocking policy call stalls the loop; whether observations come from a thread.
- G. `lerobot/pi05_base` cache: normalization stats; image tokens per camera.
- H. LeRobot pi05 state discretization: bin edges and out-of-range behavior.
- I. vLLM-Omni Pi0.5: `session_id` and `reset` handling.
- J. ROS 2 `message_filters`: whether ROS 2 is installed here.

## Scope

- LeRobot full history: `/tmp/lerobot_full` (fresh clone of `https://github.com/huggingface/lerobot.git`, 1972 commits, main HEAD `83883ae60`). Tags read: v0.6.1 (`7e241bd6`, newest tag with `async_inference`) and v0.7.0 (`410c095e`).
- `external/lerobot` checkout: v0.7.0 (`410c095`), for items B and H.
- `external/openpi`: main `215abfb`. Paths: `pyproject.toml`, `uv.lock`, `packages/openpi-client/pyproject.toml`, `examples/droid/main.py`, `examples/droid/README.md`, `src/openpi/training/config.py`, `src/openpi/models/pi0.py`, `src/openpi/policies/policy.py`, `src/openpi/policies/droid_policy.py`, `packages/openpi-client/src/openpi_client/action_chunk_broker.py`, `src/openpi/models/siglip.py`.
- `external/vllm-omni`: `2e3ad0a4f`. Paths: `vllm_omni/entrypoints/openpi/{connection.py,serving.py}`, `vllm_omni/diffusion/models/pi05/{pipeline_pi05.py,config.py}`.
- DROID env: `https://github.com/droid-dataset/droid` (HEAD `33ae6a67`, 2025-09-15), cloned to `/tmp/droid_env/repo`. `git ls-remote` matched this HEAD.
- Model cache: `models/cache/models--lerobot--pi05_base`, snapshot `22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0`. Only the JSON header of `model.safetensors` was read; no tensor data was loaded.
- `external/message_filters`: tag 8.0.5 (`995b9b2`). Read for source only.
- Existing same-day logs checked for overlap: `agents_results/2026-11-20_lerobot-async-rtc.md`, `2026-11-20_openpi-remote-pi05.md`, `2026-11-20_ros2-message-filters.md`, `2026-11-20_realtime-vllm-omni.md`.

Path convention below: `lerobot@TAG:path` means the file at that tag in `/tmp/lerobot_full`. Other paths are repo-relative, under the stated `external/` checkout.

## Result

### A. LeRobot async_inference history: VERIFIED (with qualifications)

- Tags are sorted by creation date, newest first. v0.7.0 has no `src/lerobot/async_inference/`; the newest tag that has it is **v0.6.1**. `git ls-tree` at v0.6.1 lists `src/lerobot/async_inference/robot_client.py` and `src/lerobot/async_inference/policy_server.py` (plus `configs.py`, `constants.py`, `helpers.py`). Both claimed paths exist.
- On main the directory was deleted by commit `ae2b57f12` (2026-10-13, "feat(rollout): add remote inference engine (#4937)"). v0.7.0 replaces it with `src/lerobot/remote_inference/` and `src/lerobot/scripts/lerobot_policy_server.py`, which imports `lerobot.remote_inference` and `lerobot.transport.ZenohTransport`.
- Observation send trigger (client, v0.6.1). It is a queue threshold, not a fixed period. The control loop runs every tick (period `1/fps`; `fps` default 30: `lerobot@v0.6.1:src/lerobot/async_inference/constants.py:17`, `configs.py:138`; sleep at `robot_client.py:479`). Each tick it calls `_ready_to_send_observation` (`robot_client.py:403-406`): `qsize / action_chunk_size <= chunk_size_threshold`. `chunk_size_threshold` defaults to 0.5 (`configs.py:137`; range check `configs.py:173-174`). `action_chunk_size` starts at -1 (`robot_client.py:121`) and becomes the largest chunk length received (`robot_client.py:304`). Also `must_go` is set when the queue is empty (`robot_client.py:428-429`); the server bypasses its similarity filter for `must_go` (`policy_server.py:290-291`).
- Inferred from code, not run: before the first chunk, `0 / -1 <= 0.5` is true, so an observation is sent on every tick until the first chunk arrives.
- Sending is synchronous on the control thread: `self.stub.SendObservations(...)` (`robot_client.py:207`), called from `control_loop_observation` (`robot_client.py:432`).
- Chunk merge (client): `_aggregate_action_queues` (`robot_client.py:224-267`). Actions with timestep `<= latest_action` are dropped (`:245-247`). New timesteps are added (`:250-252`). Timesteps already queued are blended with `aggregate_fn(old, new)` (`:254-264`). The default is `weighted_average` = `0.3*old + 0.7*new` (`configs.py:30`, default name `:141-143`). The queue is swapped as a whole (`:266-267`). The new queue is built only from incoming actions, so queued actions whose timestep is missing from the incoming chunk are dropped.
- Race (read from code, not run): the snapshot of the old queue (`:236-239`) and the swap (`:266-267`) take the lock separately. The control thread can pop an action between them (`:375-378`), and `latest_action` is only updated later (`:385`).
- Server (v0.6.1): observation buffer is `Queue(maxsize=1)` (`policy_server.py:75`). A new observation evicts the oldest unprocessed one (`:301-304`). Inference runs inside the `GetActions` handler (`:214-232`, predict at `:232`). The client's receive thread polls `GetActions` in a loop (`robot_client.py:275-278`). `GetActions` waits up to `obs_queue_timeout` (default 2 s: `constants.py:23`, `policy_server.py:223`) and returns Empty on timeout. `inference_latency` (default `1/30` s: `constants.py:20`, `configs.py:59-61`) pads the reply (`policy_server.py:254-256`).
- Concurrency: **the server does not process one request at a time by design.** gRPC runs a `ThreadPoolExecutor(max_workers=4)` (`policy_server.py:426`), and nothing locks `predict_action_chunk` (`:324`, `:357`). One-at-a-time holds per client only because the client has a single receive thread (`robot_client.py:275-278`, started at `:498`) and a single sender (the control thread). `Ready` resets server state (`policy_server.py:99-114`), which assumes one client.
- Similarity filter: `observations_similar` compares the joint-state norm with `atol=1` (`helpers.py:276-278`, `:281-298`). The server skips similar observations unless `must_go` (`policy_server.py:268-294`).
- Correction to `2026-11-20_lerobot-async-rtc.md:245`, which says the removal point could not be found in the shallow clone. The full clone resolves it (last tag v0.6.1; removal `ae2b57f12`).

### B. LeRobot ActionQueue lock (v0.7.0): VERIFIED

- `from threading import Lock` (`lerobot@v0.7.0:src/lerobot/policies/rtc/action_queue.py:26`). `self.lock = Lock()` (`:88`) is a non-reentrant `threading.Lock`.
- Methods that take it (`with self.lock:`): `get_with_provenance` (`:109`), `snapshot` (`:131`), `clear` (`:145`), `replace_future` (`:168`), `qsize` (`:184`), `empty` (`:195`), `get_action_index` (`:206`), `get_left_over` (`:219`), `get_processed_left_over` (`:231`), `merge` (`:260`). `get` and `get_with_task` (`:99`, `:104`) reach the lock only through `get_with_provenance`.
- Private helpers `_replace_actions_queue` (`:278`), `_append_actions_queue` (`:308`), and `_check_and_resolve_delays` (`:339`) do not take the lock. They are called from `merge` while the lock is already held (`:266`, `:269`, `:274`). No nested acquisition exists in the module, so there is no self-deadlock. Any future locked call inside `merge` would deadlock.
- Callbacks: none. The module takes and stores no callables (no "callback" or "callable" in the file). Under the lock it only logs (`:301-303`, `:359-363`).
- Caller context: the engine holds an `RLock` (`lerobot@v0.7.0:src/lerobot/inference/execution.py:203`). For example, `execution.py:603` opens `with self.lock:` and calls `self.queue.get_with_provenance()` at `:606`. The lock order is engine RLock, then queue lock. The queue never calls back into the engine, so no inversion exists.

### C. OpenPI websockets pin: VERIFIED (with a discrepancy)

- `packages/openpi-client/pyproject.toml:10`: `"websockets>=11.0"` (floor only).
- `uv.lock:5160-5161`: `name = "websockets"`, `version = "15.0.1"`. This is the only websockets entry in the lock.
- Root `pyproject.toml:73-74` sets `[tool.uv.workspace] members = ["packages/*"]`, so openpi-client is a workspace member. `uv.lock:3200` and `:3214` record websockets as its dependency. The root `pyproject.toml` has no direct websockets pin.
- Discrepancy: `examples/aloha_sim/requirements.txt:131` and `examples/aloha_real/requirements.txt:153` pin `websockets==14.1`. The workspace lock says 15.0.1.
- The design doc's note that a subagent read 17.4 source is not version-matched; not checked here.

### D. OpenPI DROID example assertion: VERIFIED by code reading (not executed)

- `examples/droid/main.py:134`: `assert pred_action_chunk.shape == (10, 8)`. The comment at `:132` says "10 joint velocity actions (7) + gripper position (1)". The client loads no config; it only calls `policy_client.infer` (`:133`). The server config is chosen in `examples/droid/README.md:22` (`--policy.config=pi05_droid`) and at `:66` (`pi0_droid` for baselines).
- `src/openpi/training/config.py:602-603`: `name="pi0_droid"`, `Pi0Config(action_horizon=10)`. This matches the assertion.
- `src/openpi/training/config.py:630-631`: `name="pi05_droid"`, `Pi0Config(action_horizon=15, pi05=True)`. This does not match.
- Chain for `pi05_droid`: `src/openpi/models/pi0.py:231` makes noise of shape `(batch, action_horizon, action_dim)`; `pi0.py:279` returns `x_0`. `src/openpi/policies/policy.py:94` puts it in `outputs["actions"]`; `policy.py:98`/`:100` take `x[0, ...]`. Neither `policy_config.py` nor `scripts/serve_policy.py` changes the horizon (grep). `src/openpi/policies/droid_policy.py:81` slices `data["actions"][..., :8]`. The reply is therefore `(15, 8)`, and the assertion at `main.py:134` would raise. Not run (no GPU or server allowed).
- Same conclusion as `2026-11-20_openpi-remote-pi05.md:109`.

### E. OpenPI ActionChunkBroker: VERIFIED

- `packages/openpi-client/src/openpi_client/action_chunk_broker.py:24`: `_last_results` starts as `None`.
- `:28-30`: `if self._last_results is None: self._last_results = self._policy.infer(obs); self._cur_step = 0`. This is the only place `obs` is used.
- `:32-39`: returns `x[self._cur_step, ...]` for each ndarray field and increments `_cur_step`.
- `:41-42`: `if self._cur_step >= self._action_horizon: self._last_results = None`. The next call re-queries the policy.
- Observations received in between are ignored: the `obs` argument is not stored, queued, or used. The broker is synchronous and never runs the policy in the background.
- Users: `examples/aloha_sim/main.py:36` and `examples/aloha_real/main.py:35`. The DROID example does not use it; it has its own counter (`examples/droid/main.py:92`, `:114-115`, `:138`).
- Inferred edge case: if the inner chunk is shorter than `action_horizon`, the slice raises IndexError at that step.

### F. DROID RobotEnv: client-side loop VERIFIED; robot-side hold NOT VERIFIABLE HERE

- Source: `droid-dataset/droid` at `33ae6a67`, cloned to `/tmp/droid_env/repo`. The example imports it at `examples/droid/main.py:16` (`from droid.robot_env import RobotEnv`).
- Blocking: the example is one thread. Each tick reads the observation (`main.py:32-37`), optionally calls the policy (`:133`, only when `actions_from_chunk_completed` hits 0 or `open_loop_horizon`=8; `:42`, `:114`), and calls `env.step(action)` (`:151`). Sleep to 15 Hz at `:154-156` (`DROID_CONTROL_FREQUENCY = 15`, `:23`). While `infer` blocks, no observation is read and no step is sent. A call longer than the 8-step chunk period (8/15 = 0.53 s) delays the next step.
- Observations are not read in a separate thread. `RobotEnv.get_observation` (`robot_env.py:108-133`) is synchronous: `get_state` (`:112`, zerorpc `get_robot_state` at `droid/misc/server_interface.py:76-77`) and `read_cameras` (`:117`, `:87-88`). A grep over `droid/` finds no threads in `camera_utils`; threads appear only in the GUI, oculus, trajectory-writer, and subprocess helpers.
- `step`: `robot_env.py:49-63` calls `update_robot` (`:75-82`) with the default `blocking=False`. This becomes zerorpc `update_command` (`server_interface.py:45-47`), then `franka/robot.py:46-51`. For the non-blocking joint path, the robot command runs in `run_threaded_command(helper_non_blocking)` (`franka/robot.py:115`), so `step` does not wait for motion. The RPC call itself is synchronous.
- NOT VERIFIABLE HERE: whether the arm holds the last setpoint during a stall. That behavior lives in the robot process and controller, which were not checked.

### G. pi05_base cache: stats absent; 256 image tokens per camera: VERIFIED

- Cache layout: the snapshot holds `config.json`, `model.safetensors`, `policy_preprocessor.json`, `policy_postprocessor.json`, and `README.md`, all symlinks into `blobs/`. There is no processor-state safetensors (for example `policy_preprocessor_step_N_normalizer_processor.safetensors`, named by `lerobot@v0.7.0:src/lerobot/processor/pipeline.py:391`).
- `policy_preprocessor.json:28` and `policy_postprocessor.json:8`: `"features": {}`. `policy_preprocessor.json:29-32` and `policy_postprocessor.json:9-12`: STATE and ACTION use `QUANTILES`. So the processor JSON carries no statistics.
- `model.safetensors`: header parsed (8-byte length, then JSON). 812 tensors. Metadata is only a tied-weight map. Zero keys match `normaliz|quantile|q01|q99|q10|q50|q90|stats|buffer_|mean|std|min|max`. Prefix groups: `action_in_proj` (2), `action_out_proj` (2), `time_mlp_in`/`time_mlp_out` (4), `paligemma_with_expert.paligemma` (603), `paligemma_with_expert.gemma_expert` (201). The code expects `q01` and `q99` for QUANTILES (`lerobot@v0.7.0:src/lerobot/processor/normalize_processor.py:389-392`). Result: quantile stats are not present in the local cache. Upstream stats were not fetched.
- Patch size 14: `src/openpi/models/pi0.py:84` `variant="So400m/14"`, decoded by `src/openpi/models/siglip.py:306` into `patch_size=(14,14)`. The safetensors header confirms it: `paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight` is F32 `[1152, 3, 14, 14]`, and `position_embedding.weight` is `[256, 1152]`.
- Image size 224: `config.json:63-66` (`image_resolution [224, 224]`). Three camera inputs at `[3, 224, 224]` (`config.json:4-28`).
- Tokens per camera: (224/14)^2 = 16 x 16 = 256 (the position table has no CLS entry). Three camera slots give 768 image tokens. The patch size is not stated in the pi05_base `config.json`; it comes from the openpi source and the weight shapes.

### H. Out-of-range state bin: VERIFIED (numpy run)

- `external/lerobot/src/lerobot/policies/pi05/processor_pi05.py:75-79`: "State should already be normalized to [-1, 1]..." and `discretized_states = np.asarray(np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1])) - 1`. The same expression appears in openpi `src/openpi/models/tokenizer.py:26`, `:70`, `:183`.
- Bin edges: 256 edges, `-1 + i/128` for `i` = 0..255. Checked: `edge[0] = -1.0`, `edge[1] = -0.9921875`, `edge[128] = 0.0`, `edge[255] = 0.9921875`.
- Snippet: copied the expression verbatim, run via stdin with `envs/mstar/bin/python3 -I` (numpy 2.5.4). Input float32 (as torch float32 would give) and float64.
- Results: -1.5 → **-1**; -1.0 → 0; 0.0 → 128; 0.999 → 255; 1.5 → 255. Float32 and float64 gave identical results.
- Behavior: the low side is **not clamped**. Values below -1 become -1, outside the 0..255 range. That is written into the prompt as the string `-1` (`processor_pi05.py:87-88`, `" ".join(map(str, discretized_states[i]))`), so it behaves like a sentinel. The high side **clamps** into bin 255 because the top bin is open-ended. Nothing wraps.

### I. vLLM-Omni Pi0.5 session handling: VERIFIED

- Reset endpoint: `connection.py:262-265`: `if endpoint == "reset": self.reset(); self.serving.reset(obs); ... "reset successful"`. `connection.py:180-182` `reset()` clears only connection bookkeeping (`_current_session_id`, `_seen_sessions`). `serving.py:169-170` `reset(obs)` is a no-op: "Compatibility hook; per-connection state lives in RobotRealtimeConnection".
- Session id: `connection.py:267`: `session_id = str(obs.get("session_id") or self._current_session_id or "default")`. `connection.py:277`: `reset = self._mark_session_seen(session_id)`, which is True on the first sighting of that id on the connection (`:184-202`; `return False` at `:192`, `return True` at `:202`).
- Passed on: `serving.py:194-198` (`infer`), then `_build_request` (`:254-304`) puts `reset`, `session_id`, and `robot_obs` into `extra_args` (`:287-291`). `request_id` embeds `session_id` (`:251-252`).
- Pi0.5 pipeline: `vllm_omni/diffusion/models/pi05/pipeline_pi05.py:13` says "`session_id` / `reset` from the OpenPI protocol are accepted but ignored", and line 12 says the model is stateless across calls. The code reads only `extra_args.get("robot_obs")` (`:269-270`). A grep of `vllm_omni/diffusion/models/pi05/` finds no read of `reset` or `session_id` from `extra_args`. `config.py:428` rejects `use_visual_memory` because "this path does not keep" per-session history.
- Answer: an observation with a different `session_id` does **not** change the Pi0.5 computation. It changes connection bookkeeping (the first-observation `reset` flag) and `request_id`.
- Answer: reset clears **no** Pi0.5 state. `drop_session` (`serving.py:172-183`) is reached from `serving_rl_rollout.py:324` (REST rollout path), not from the WebSocket connection. The connection has no drop call; disconnect only breaks the loop (`connection.py:243-244`).

### J. ROS 2 message_filters: NOT VERIFIABLE HERE (ROS 2 not installed)

- `ls /opt/ros`: "No such file or directory" (exit 2). `ls /usr/include | grep -i -E "rclcpp|message_filters|rcl"`: no matches (exit 1). `/usr/include/ros` absent. `dpkg-query` for `ros-*`, `rclcpp`, `message-filters`: 0 packages. `ros2` and `rclcpp` not on PATH. `ROS_DISTRO` unset. `/opt` contains only `containerd` and `intel`.
- Only the source is present: `external/message_filters` at tag 8.0.5 (`995b9b2`). It was not built. Runtime claims about message_filters cannot be checked on this machine.

## Checked and corrected

- `2026-11-20_lerobot-async-rtc.md:245` said the removal of `async_inference` could not be dated with a shallow clone. The full clone shows the last tag with it is v0.6.1 and main removed it in `ae2b57f12` (2026-10-13).
- `2026-11-20_openpi-remote-pi05.md:109` (DROID assertion versus `pi05_droid`) and `:52` (no `env.step` during the call) agree with this check.
- `2026-11-20_ros2-message-filters.md` is source-only. This check confirms ROS 2 is not installed, so its runtime claims stay unverified.
- `agents_results/README.md` index was not updated, because only this log file may be edited. The main session should add an entry.

## Not verified

- No runtime test of any LeRobot, openpi, or vLLM-Omni code path. The v0.6.1 client/server race in A is read from code only.
- D (`pi05_droid` would fail the `(10, 8)` assertion) is a code-path conclusion, not a run.
- F: the robot-side behavior during a stall (whether the arm holds the last setpoint) was not checked.
- G: upstream normalization stats (Hugging Face `lerobot/pi05_base` or openpi `assets`) were not fetched. No tensor data was loaded.
- C: the websockets 17.4 source read by an earlier subagent was not version-matched to 15.0.1.
- I: diffusion-engine behavior for `reset` or `drop_session` below the pipeline level was not read.
- J: no ROS runtime, build, or headers exist, so nothing in message_filters was executed.

## Commands run

- `git clone https://github.com/huggingface/lerobot.git /tmp/lerobot_full` (full history, 1972 commits). Then `git tag --sort=-creatordate`, `git cat-file -e`, `git ls-tree`, `git show <tag>:<path>`, `git log --diff-filter=A/D -- src/lerobot/async_inference`, `git grep`.
- Scratch copies written only under `/tmp` (not in the repo): `/tmp/lerobot_v061_*.txt`, `/tmp/rtc_action_queue_v070.py.txt`.
- `git ls-remote https://github.com/droid-dataset/droid`; `git clone --quiet https://github.com/droid-dataset/droid /tmp/droid_env/repo`.
- `grep` and `sed -n` reads over `external/openpi`, `external/vllm-omni`, `external/lerobot`, and `models/cache/models--lerobot--pi05_base`.
- Safetensors header: read 8 bytes plus the JSON header with `python3 -I` (stdin script). No tensor data loaded.
- Numpy digitize snippet (H): `envs/mstar/bin/python3 -I` (stdin), numpy 2.5.4. `python3` on the default path has no numpy.
- ROS checks: `ls /opt/ros`, `ls /usr/include | grep -i ...`, `ls /usr/include/ros`, `ls /opt`, `dpkg-query -W`, `command -v ros2 rclcpp`, `echo $ROS_DISTRO`, `find` for `message_filters*`/`rclcpp*` outside the repo.
- No GPU job, server, pip or conda install, or commit.

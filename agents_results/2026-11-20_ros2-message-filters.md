# ROS 2 message_filters: source analysis (tag 8.0.5)

迁移说明（2026-11-20）：本文件从基准仓库 `agents_results/` 迁入 `robotics_vla_runtime/agents_results/`。路径已改为本目录的 `external/`（上游最新快照）和 `docs/design.md`。文中的 `/tmp/...` 是当时的临时克隆，未保留。

- Date: 2026-11-20
- Role: read-only source analyst. No builds, tests, installs, or edits (other than this log). Third-party code was read, never executed.
- Scope: `external/message_filters` = ros2/message_filters tag 8.0.5, commit 995b9b20e55e641197d7d69059e5066b3ca2d7b2. Citations are `path:line` relative to that package root (`<pkg>`) unless a path is absolute. Tags: **[verified]** = I read the line; **[inferred]** = derived from control flow by reading, not executed.
- Prior research check: `grep -ril` for message_filters / ApproximateTime / "message filters" in `agents_results/` found nothing. This is the first record.

## Questions

1. Subscriber and Synchronizer: delivery, thread model, need for node/executor, standalone use, header-only vs rclcpp.
2. Cache: storage, query methods, boundary semantics, "latest per stream at or before t".
3. ApproximateTime: parameters, pivot, selection algorithm, out-of-order, gaps, drops.
4. ExactTime: matching rule and behavior with unequal rates.
5. Time source (header.stamp vs arrival), skew and latency handling.
6. Periodic trigger or snapshot-at-tick operation.
7. Dependence on ROS message types.

## Findings

### Q1. Subscriber, Synchronizer, thread model, rclcpp dependency

- **[verified]** `Subscriber<M>` wraps `rclcpp::create_subscription<M>` and forwards each message via `this->signalMessage(e)` (`include/message_filters/subscriber.hpp:283-288`, `358-361`). It needs `NodeInterfaces` or a node to subscribe (`subscriber.hpp:151-156`, `227-231`).
- **[verified]** message_filters creates no threads. `grep` for `std::thread`, `std::async`, `Thread(` over `include/` and `src/` finds nothing. Callbacks run on whatever thread rclcpp's executor uses. rclcpp is not in this repo, so that behavior is unverified (see Not verified).
- **[verified]** Every sync policy guards its state with a `std::mutex` and calls the user callback while the lock is held: `sync_policies/approximate_time.hpp:165` (lock) and `:386` (signal via `publishCandidate`); `exact_time.hpp:97` and `:145`; `approximate_epsilon_time.hpp:99` and `:255`; `latest_time.hpp:153` and `:207`; `input_aligner.hpp:231`/`:351` and `:407`.
- **[verified]** Contrast: C++ `TimeSequencer::dispatch` releases its lock before calling callbacks (`time_sequencer.hpp:196-215`).
- **[verified]** `Signal9::call` copies the callback list under its own mutex and invokes callbacks outside that mutex (`signal9.hpp:138-152`). The policy lock above still encloses it.
- **[verified]** A Synchronizer is usable without a node: construct it from a policy (`synchronizer.hpp:75-79`) and feed it with `add<i>(shared_ptr)` (`synchronizer.hpp:154-158`). The unit tests do exactly this, with no node: `test/test_approximate_time_policy.cpp:99-135` builds `sync_(queue_size)` and calls `sync_.add<0>(p)` / `sync_.add<1>(q)`.
- **[verified]** Standalone use still links rclcpp. `MessageEvent(ConstMessagePtr)` stamps receipt time with `systemClockNow()` (`message_event.hpp:114-119`), defined as `rclcpp::Clock().now()` in `src/message_event.cpp:31`, `36-39`. The library target links `rclcpp::rclcpp`, `rcutils::rcutils`, `std_msgs::std_msgs` (`CMakeLists.txt:31-35`), and `package.xml:24-26` declares them.
- **[verified]** Header-only vs compiled:
  - Compiled: only `src/connection.cpp` (no rclcpp use, `:34-48`) and `src/message_event.cpp` (rclcpp::Clock).
  - Templates, header-only: all of `sync_policies/*.hpp`, `synchronizer.hpp`, `cache.hpp`, `signal1/9.hpp`, `simple_filter.hpp`, `time_synchronizer.hpp`, `chain.hpp`, `delta_filter.hpp`, `pass_through.hpp`, `input_aligner.hpp`, `time_sequencer.hpp`.
  - Every SimpleFilter and Synchronizer includes `message_event.hpp`, which includes `rclcpp/time.hpp` (`message_event.hpp:40`; `simple_filter.hpp:38`; `synchronizer.hpp:41`). So the whole core needs the rclcpp time types.
  - Timers need rclcpp (`time_sequencer.hpp:41`, `222-228`; `input_aligner.hpp:217`). Logging uses rcutils (`approximate_time.hpp:34`, `147-156`).
- **[verified]** Python: the whole package imports `rclpy` at module load (`src/message_filters/__init__.py:39-51`, `simple_filter.py:30`, `message_traits.py:32-33`). The Python logic cannot be imported without rclpy.

### Q2. Cache

- **[verified]** C++ `Cache<M, TimeGetter>` is a count-bounded store (`cache.hpp:81-83` default size 1; `setCacheSize` at `:143-151` ignores 0).
- **[verified]** Storage: a `std::deque<MessageEvent>` kept sorted by stamp. `add` pops the oldest-by-stamp when full (`:173-175`), then inserts at the `lower_bound` position (`:177-184`), then signals outside the lock (`:187`).
- **[verified]** Timestamp: `header.stamp` via `DefaultTimeGetter` (`cache.hpp:357-360`; `message_traits.hpp:96-119`). For a headerless type the constructor throws unless `allow_headerless` is set, in which case receipt time is used (`cache.hpp:91-97`, `117-123`, `362-365`).
- **[verified]** Query methods (all lock `cache_lock_`, so each call is atomic):
  - `getInterval(start,end)`: inclusive at both ends, copies `shared_ptr`s (`:198-224`).
  - `getSurroundingInterval(start,end)`: one element strictly before `start` if any, all elements in `[start,end]`, and the first element at or after `end` (`:233-268`). Derived from the iterator arithmetic.
  - `getElemBeforeTime(t)`: strictly before `t`. The `lower_bound` then `--it` path is at `:275-291`. The test asserts an exact match is NOT returned (`test/msg_cache_unittest.cpp:300-305`).
  - `getElemAfterTime(t)`: strictly after `t` (`upper_bound`, `:298-313`; test `msg_cache_unittest.cpp:315-320`).
  - `getLatestTime` / `getOldestTime`: `back()` / `front()` stamps; a default-constructed `rclcpp::Time` if empty (`:318-349`).
- **[verified]** Answer to "latest per stream at or before t": yes, one call per stream, `getElemBeforeTime(t + 1 ns)` gives at-or-before in C++ [inferred from the strict semantics]. There is no maximum-age check; a stalled stream returns its last sample indefinitely while it remains in the ring.
- **[verified]** Python `Cache` (`src/message_filters/__init__.py:122-224`) differs:
  - `getElemBeforeTime` and `getElemAfterTime` are inclusive (`<=` at `:201-207`, `>=` at `:193-199`). This is the opposite of C++ at the boundary.
  - `add` appends without sorting, despite the comment "Insert sorted" at `:174-176`.
  - Eviction is by arrival order (`:179-181`), and `getLatestTime` returns the last arrival (`:209-213`).
  - Consequence [inferred]: the Python Cache is order-sensitive and should not be treated as equivalent to the C++ one.

### Q3. ApproximateTime (C++ `sync_policies/approximate_time.hpp`)

**Parameters** [verified]. There is no `pivot` parameter. `pivot_` is an internal stream index set at `:536` (member at `:627`). The in-repo docs do not describe these parameters (grep of `doc/`, `README.md`, `CHANGELOG.rst` finds no `age_penalty`, `pivot`, or `max_interval`).

| Parameter | Where | Default | Effect |
|---|---|---|---|
| `queue_size` | ctor `:69`, assert `:82` | none (required) | Per stream, `deque.size() + past.size()` is capped (`:181-189`). Comment at `:80-81` says at least 2 is recommended. |
| `age_penalty` | `setAgePenalty` `:200-206` | 0.1 (`:75`) | Replacement test `:543-544`, see algorithm. |
| inter-message lower bound, per stream | `setInterMessageLowerBound` `:208-214` | 0 (`:77`) | Used only in the virtual proof `:449-456`. Violations trigger one warning (`:117-159`); the bound is not enforced. |
| `max_interval_duration` | `setMaxIntervalDuration` `:216-222` | about 68 years (`:74`) | Rejects candidates whose span exceeds it (`:522-526`). Effectively unbounded by default. |

Python `ApproximateTimeSynchronizer.__init__` (`src/message_filters/__init__.py:381-396`) takes `queue_size`, `slop`, `queue_offset` (static per-input ns offsets), `allow_headerless`, `sync_arrival_time`, `time_getter`.

**Algorithm (C++)** [verified line by line]:

1. `add<i>` pushes the message at the back of stream i's deque, with no sorting (`:162-168`). `process()` runs only when all streams have at least one message (`:172-175`), and is also called after an overflow drop (`:195`). Nothing else calls `process()` (grep).
2. `process()` (`:502-612`) loops while every deque is non-empty. It computes `start` = earliest head and `end` = latest head, with their stream indices (`:398-437`).
3. **No candidate yet** (`:518-538`): reject the head set if `end - start > max_interval` (`:522-526`) or if the end stream has a recent drop flag (`:527-531`); on rejection the start head is deleted. Otherwise accept: copy the current heads as the candidate and clear `past_` (`makeCandidate` `:303-311`); set `pivot = end stream` and `pivot_time = end` (`:536-537`); move the start head to `past_` (`:538`).
4. **Candidate exists** (`:539-556`): the new head set replaces the candidate iff `(start - cand_start) > (end - cand_end) * (1 + age_penalty)` (`:543-544`). With `age_penalty = 0` this is "smaller span wins". Replacement calls `makeCandidate`, which discards `past_` (`:550`, `:310`). In either case the start head moves to `past_` (`:547`, `:553`).
5. **Publish** (`:559-610`): when the start stream is the pivot (`:560-562`); or when even the best possible future candidate cannot beat the current one (`:563-569`); or when the virtual proof with the inter-message bounds succeeds (`:570-588`). Otherwise wait.
6. `publishCandidate` (`:383-394`) signals the tuple, removes the candidate's messages from each deque, and restores the rest of `past_` to the front of the deques.

**Selection** [verified from code; the description is mine]: this is a sliding minimum-span search over stream heads. The window is anchored at the latest head (the pivot). It is not "closest to a reference time".

**Consequences** [inferred from the algorithm, not run]:

- Each publish consumes one message from every stream, so the output rate is at most the slowest stream's rate (about 30 Hz if cameras are included).
- After a publish, each stream's remaining buffered messages are newer than the published one. Per-stream output timestamps therefore increase when input is in order.
- Publication waits for the stream whose head was consumed at candidate creation to deliver its next message, unless a rate bound proves optimality. Without bounds, the virtual proof fails (`:589-601`). The test diagram `test/test_approximate_time_policy.cpp:478-487` (EarlyPublish) shows output at t=7, although all four messages arrived by t=3. The harness asserts only the output sequence (`:134`, `:204`), not timing, so this timing claim is inferred.

**Out-of-order, gaps, drops**:

- Out-of-order: appended as-is. `checkInterMessageBound` (`:117-159`) warns once and does not reorder or reject. Result is undefined by design [inferred].
- Gaps: `process()` does not run while any stream is empty (`:505`). There is no timeout or timer in this header (grep confirms). Other streams are capped at `queue_size` (`:182-189`). When the gap ends, the older backlog and the new message can pair, and the default span limit is unbounded.
- Drops are silent (no callback in ApproximateTime) in four places:
  1. overflow of a stream past `queue_size` (`:182-189`);
  2. start heads rejected by `max_interval` or the drop flag (`:522-531`);
  3. messages in `past_` discarded when a better candidate replaces the current one (`:310`);
  4. ExactTime has `registerDropCallback` (`exact_time.hpp:105-127`), ApproximateTime does not.
- Tests (names and bodies read): `ImperfectMatch` (`test_approximate_time_policy.cpp:272`), `Acceleration` (`:298-321`), `DroppedMessages` (`:324`, queue size 1 vs 2), `DoublePublish` (`:406`), `EarlyPublish` (`:478`), `RateBound` (`:506`; `setInterMessageLowerBound(0, 1.5 s)` changes the output).

**Python `ApproximateTimeSynchronizer.add`** (`src/message_filters/__init__.py:398-455`) is a different algorithm [verified by reading]:

- Stores the message in its queue keyed by ns, and trims the oldest over `queue_size` (`:417-419`).
- For each other queue, collects stamps within `slop` of the new message, sorted by distance (`:428-441`), then walks the `itertools.product` of those lists in order (`:442`). The first combination with span `< slop` and all entries present is emitted (`:446-454`), and only those entries are deleted (`:452-453`). Older unmatched entries are not pruned (contrast the exact `TimeSynchronizer` pruning at `:354-358`).
- Returns without signaling when some queue has no candidate within `slop` (`:437-439`).
- **Suspected bug** [inferred, not executed]: `queue_offset` is subtracted from the stored key (`:413-417`), but the candidate tuple uses the raw `stamp` for the new message's own index (`:446`), and the span and membership checks use that raw value (`:448-449`). If the offset stream arrives second, the span check fails (the offset is 0.5 s in the tests, `test/test_approxsync.py:134`, `157`, `178`, and `slop` is 0.1 s), and the membership check fails too. The tests only send the offset stream first (`test/test_approxsync.py:137-144`, `:160-165`).

### Q4. ExactTime (`sync_policies/exact_time.hpp`; `TimeSynchronizer` is a thin wrapper at `time_synchronizer.hpp:73-94`)

- **[verified]** Keyed by `std::map<rclcpp::Time, Tuple>`, using the stamp of the arriving message (`:99`, `:171`). A tuple completes only when every stream has a message with the same stamp (`isFull` `:131-137`, `checkTuple` `:140-145`). Matching is nanosecond equality.
- **[verified]** On completion it signals (`:145`), sets `last_signal_time_` to stream 0's stamp (`:148`), erases that tuple (`:150`), and drops every tuple with key `<= last_signal_time_` via `clearOldTuples` (`:164-168`), firing drop callbacks (`:174-183`).
- **[verified]** Overflow: if `tuples_.size() > queue_size`, the oldest tuples are dropped (`:155-159`). `queue_size = 0` disables the bound (`:155` guard), so the map can grow without limit.
- **[verified]** A second message from the same stream with the same stamp silently overwrites the first (`:100`), with no drop callback.
- **[inferred]** Unequal rates: a 30 Hz camera and a ~500 Hz joint stream on independent clocks almost never share a stamp. Partial tuples accumulate until overflow, and each completion discards the older partials. Output is effectively zero unless the streams are hardware-synchronized to identical stamps.
- **[verified]** Python `TimeSynchronizer.add` (`__init__.py:328-360`) has the same exact-match rule. It keys a dict by ns, prunes older stamps after a signal (`:354-358`), and raises `ValueError` on headerless input (`:330-334`).

### Q5. Time source, skew, latency

- **[verified]** Default timestamp: `header.stamp` as `rclcpp::Time(stamp, RCL_ROS_TIME)` (`message_traits.hpp:96-103`, `112-119`). Types with no header yield `Time(0,0)` silently (`:89-93`). The policies have no header check; only `Cache` rejects headerless types (`cache.hpp:91-97`) [inferred from the absence of checks in the policies].
- **[verified]** Receipt/arrival time: the `MessageEvent` constructor used by `Subscriber` stamps receipt time with `systemClockNow()` (`message_event.hpp:114-119`, `subscriber.hpp:286-287`). It is used only by `Cache` when `allow_headerless` is set (`cache.hpp:362-365`).
- **[verified]** `LatestTime` ignores header stamps. It uses `ros_clock_->now()` at arrival (`latest_time.hpp:167`, `195-200`). Its default constructor creates a standalone `RCL_ROS_TIME` clock (`:102-104`). The header comment recommends passing the node's clock to get sim time (`:46-52`).
- **[verified]** Skew and latency: none in the C++ headers (grep for `offset|skew|latency` under `include/` finds nothing except the InputAligner timeout). The only offset is the Python static `queue_offset` (`__init__.py:371-372`, `:414-415`), which is not estimated.
- **[verified]** Clock-domain mixing: C++ `TimeSequencer` releases a message when `stamp + delay <= node_clock.now()` (`time_sequencer.hpp:199-204`); Python does the same (`__init__.py:557-568`). Header stamps and the node clock must share a time base.
- **[verified]** `rclcpp::Time` comparisons across different clock types throw (`message_event.hpp:200-202` comment; `latest_time.hpp:252-258` catches it).

### Q6. Periodic trigger / snapshot-at-tick

- **[verified] Answer: no.** There is no snapshot-at-tick operation in message_filters.
- Event-driven only: the policies run only from `add()` (`approximate_time.hpp:174`, `195`; the exact and epsilon policies are likewise triggered by arrivals).
- Timers exist in exactly two places:
  - `TimeSequencer`: C++ `time_sequencer.hpp:218-228`, Python `__init__.py:508-510`. It releases each message after a fixed delay, one message at a time.
  - `InputAligner`: C++ `input_aligner.hpp:214-221` (`create_wall_timer` calling `dispatchMessages`), Python `input_aligner.py:200-202`. `dispatchMessages` drains every dispatchable message in timestamp order (`input_aligner.hpp:227-241`). It forwards every message once, rather than picking one per tick.
- `LatestTime` triggers on arrival of the fastest non-late stream (`latest_time.hpp:169-171`, `find_pivot` `:334-372`), not on a clock.

### Q7. Dependence on ROS message types

- **[verified]** Policies and `Synchronizer` are templates over `Ms...` (`synchronizer.hpp:47`; `approximate_time.hpp:54-55`; `exact_time.hpp:52-54`), with a template-template `TimeGetter` constrained by a C++20 concept (`message_traits.hpp:137-140`).
- **[verified]** ROS types appear at the Subscriber layer: `rclcpp::create_subscription<M>` (`subscriber.hpp:283-288`), plus the type-adapter trait (`:115-131`).
- **[verified]** `message_traits.hpp:40` includes `std_msgs/msg/header.hpp`, and `HasHeader` uses `std_msgs::msg::Header` (`:59-60`). Every policy includes `message_traits.hpp` (`approximate_time.hpp:45`, `exact_time.hpp:43`, `approximate_epsilon_time.hpp:42`). So std_msgs headers are needed at compile time even with a custom `TimeGetter`.
- **[verified]** Python: message types are accessed only by attribute (`src/message_filters/message_traits.py:36-50`).

### Other policies

- **LatestTime** (`latest_time.hpp`): keeps only the latest message per stream (`events_`, `:166`, `:375`; no queue). Publishes on arrival of the fastest non-late stream, once every stream has arrived once (`:155-171`). Rate estimated from arrival times with EMA (`compute_hz` `:246-289`; defaults `:212-214`). Stale values are held indefinitely (no staleness check, `:166`, `:375`) [inferred from the absence of a timeout].
- **ApproximateEpsilonTime** (`approximate_epsilon_time.hpp`): per-stream FIFO (`:94-111`, overflow drops oldest at `:107-110`). Emits when all heads are within `epsilon` of the oldest head (`:168-171`, `:264-273`). Otherwise it discards every head within `epsilon` of the oldest (`:227-233`, `:268-273`). Whether this can discard a valid match is unverified.
- **InputAligner** (`input_aligner.hpp`): a timestamp-ordered merge (multiset per input, `:257-258`). Drops arrivals older than the last output (`:355-359`). Timeout is data-driven: an empty input is skipped when `last_in_ts - predicted_next >= timeout` (`:410-413`, `firstTimeStamp` `:265-277`). With no period hint, an empty active input blocks the merge until data or timeout arrives [inferred].
- **TimeSequencer** (`time_sequencer.hpp:151-216`): sorted buffer. Drops arrivals older than the last release (`:154-156`) and overflow (`:160-162`). Releases after `delay` on the node clock (`:199-204`).

## Checked and corrected

- The Python `Cache` indentation was ambiguous in a first pass. Re-read with exact indentation (`__init__.py:158-184`).
- An early listing of `time_sequencer.hpp` had a line-number offset. All cited line numbers were re-read with correct numbering.
- The question assumed a `pivot` parameter. There is none; `pivot_` is internal (`approximate_time.hpp:627`, set at `:536`).
- The EarlyPublish and Acceleration test diagrams are illustrative. The harness asserts only the output sequence (`test_approximate_time_policy.cpp:121-135`).

## Unverified / could not confirm

- rclcpp is not installed or vendored here (`ls /opt/ros/*/include/rclcpp` returns nothing). Not checked: the default `rclcpp::Time` value, the default `rclcpp::Clock()` type (`message_event.cpp:38`), `rclcpp::Time` comparison throwing behavior, executor threading and callback groups, and QoS-depth queueing in the rclcpp subscription.
- No tests were run. All test expectations are as written in the test files.
- Inferred and not executed: ApproximateTime publication latency and per-stream monotonicity; the Python `queue_offset` asymmetry; the Python `ApproximateTimeSynchronizer` first-match order; callback re-entry deadlock in the C++ policies (non-recursive mutex under the callback, `approximate_time.hpp:165`/`:386`); Python lock not released if a callback raises (`__init__.py:335-360`, `:416-455`, no try/finally; `simple_filter.py:54-56` does not catch).
- ApproximateEpsilon head discard (`:227-233`): whether it can lose a valid match was not exhaustively checked.
- Python `Cache` behavior under out-of-order arrivals: inferred only.
- Not read in detail: `chain.hpp`, `delta_filter.hpp`, `pass_through.hpp`, `parameter_adapter.hpp`, `null_types.hpp`, the Python `input_aligner.py` and `delta_filter.py` internals, `test/test_custom_time_getter.cpp`, and `test/test_fuzz.cpp` (excluded from CI by `CMakeLists.txt:105` `SKIP_TEST`).
- The ros-token and line counts below are a rough grep, not a parse.

## Design implications for a Level 1 runtime

Level 1 needs, for each Pi0.5 call at tick t, the latest observation of each stream at or before t, with an explicit age per stream.

**A. Per-stream latest-value snapshot at tick (recommended).**

- message_filters has no tick. Write the tick loop yourself. Per stream, keep a time-sorted ring (size = rate x retention window; e.g., joints at 500 Hz for 0.5 s is about 250 entries, 30 Hz camera for 1 s is about 30) [inferred arithmetic].
- For each stream, take the newest message with `stamp <= t`. In C++ use `getElemBeforeTime(t + 1 ns)`; in Python use `getElemBeforeTime(t)`, which is inclusive. Do not use the Python `Cache` as the reference for the C++ boundary semantics. [inferred for the +1 ns]
- Enforce a maximum age per stream yourself. `Cache` returns stale messages with no limit.
- Pin the returned pointers for the tick. Each `Cache` query locks separately, and a late message with `stamp <= t` can change a later re-query.
- Record per-stream age and staleness as observation metadata. Skew between streams is bounded by each stream's age, with no cross-stream compensation in the library.

**B. ApproximateTime (only for matched multi-sensor frames).**

- Event-driven, not tick-driven. Output rate is at most the slowest stream's rate (about 30 Hz if the camera is in the set), so 10 Hz inference needs decimation.
- Publication latency can add up to one period of the stream whose head was consumed [inferred].
- A stalled stream halts all output (no timeout). The default span is unbounded, so set `setMaxIntervalDuration` (`approximate_time.hpp:216-222`).
- Inter-message bounds are assumptions, not enforced (`:117-159`).

**C. ExactTime.**

- Only useful if all streams carry hardware-synchronized identical stamps. With independent 30 Hz and ~500 Hz streams the expected yield is near zero [inferred], and each match discards older partials.

**LatestTime, InputAligner, TimeSequencer.**

- LatestTime uses arrival time, not header stamps, fires at the fastest stream's rate (joints, so roughly 500 Hz rather than 10 Hz), and holds stale values indefinitely. Not a fit for a header-aligned snapshot.
- InputAligner and TimeSequencer give a time-ordered per-message stream with delay or timeout. Useful as an ordering stage feeding option A, not as the snapshot.

**Recommendation.** For Level 1, use option A. Do not adopt the synchronizer policies. Consider vendoring or reimplementing the 219-line `Cache` logic, not depending on message_filters. Define one clock domain for header stamps: the library has no skew or latency handling.

**Share of reusable pure C++** (line counts, `include/` and `src/`, non-blank non-comment lines; rough, from a grep and a file-level classification):

- Total C++: 3,133 lines.
- Generic template logic whose only ROS coupling is `rclcpp::Time`/`Duration` (and `rclcpp/time.hpp` via `message_event.hpp`), plus rcutils logging in ApproximateTime: 2,050 lines (about 65%). The kernels that matter for Level 1 (ApproximateTime 466, ExactTime 130, ApproximateEpsilonTime 224, Cache 219) total 1,039 lines (about 33%).
- Clock- or timer-coupled: InputAligner 280, TimeSequencer 141, LatestTime 259; total 680 (about 22%).
- ROS glue (Subscriber 195, message_traits 61, message_event 138 + .cpp 9): 403 (about 13%).
- Porting cost is the time-type swap, not the line count. The swap must preserve `rclcpp::Time` semantics (clock types, `Duration * double` at `approximate_time.hpp:543-544`), replace `RCUTILS_LOG_WARN_ONCE` (`:147-156`), drop the `std_msgs` include (`message_traits.hpp:40`), and replace `systemClockNow` (`message_event.hpp:114-119`).
- Python: about 1,150 lines, all behind `import rclpy` (`__init__.py:39-51`). Reuse means copying the logic, not importing it.

## Main-session spot checks (2026-11-20, `external/message_filters` at 995b9b2)

- `include/message_filters/sync_policies/approximate_time.hpp:543–544`: the code is written as `(end − candEnd)·(1+p) >= (start − candStart)` for "not better". Its negation is the replacement rule in the log. Confirmed, equivalent form.
- `include/message_filters/cache.hpp:275–291`: `getElemBeforeTime` uses `lower_bound` with `<`, so the result is strictly before the query time. Confirmed.
- `include/message_filters/synchronizer.hpp:154–158`: `add<i>(shared_ptr)` path exists. Confirmed.
- `include/message_filters/latest_time.hpp:166–171`: `publish()` runs when the arriving stream is the pivot (`find_pivot`) and `is_full()`. Confirmed: fires on the fastest stream.
- Not re-checked: the rclcpp-dependent parts (no rclcpp here) and the Python `__init__.py` line numbers.

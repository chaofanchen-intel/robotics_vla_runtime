# Frozen inputs

Every input path consumed by the current harness resolves under `exps/inputs/`;
no runner, task, fixture, or configuration opens a file from a prior-round
directory. Keep fixture bytes frozen: changing a prompt, row order, resize,
audio file, or manifest hash creates a new fixture version and a new comparison
condition.

The media and dataset-derived fixtures are intentionally Git-ignored because
of size and licensing. A runnable workspace must provision this directory from
the approved local fixture bundle; current tasks must not fall back to another
directory or silently download a replacement. Fixture manifests identify the
source dataset/item and hashes, while active paths point to their materialized
files here. `exps.doctor` checks required fixture presence. Model
weights/tokenizers and runtime environments are separate workspace-level
dependencies under `models/cache/` and `envs/`, not prior-round directories.

| Directory | Active contents | Local paths / provenance |
|---|---|---|
| `BAGEL/` | `bagel_shared_v1.json`, plus three pre-resized source images | `images/shared_source_{256,512,1024}.png`; prompt and source-item identifiers are in the manifest. |
| `SEED-TTS/` | EN 100 / ZH 200 local voice-clone subsets and prompt WAVs | `seedtts_en_local_v1.jsonl`, `seedtts_zh_local_v1.jsonl`, `prompt-wavs/{en,zh}/`; `localisation_manifest.json` records counts and hashes. |
| `ORPHEUS/` | 256 frozen Seed-TTS EN target texts | `seedtts_en_targets_v1.jsonl`; `fixture_manifest.json` records selection order and source-list digest. |
| `QW-A2T/`, `QW-TXT/` | FLEURS audio and fixed 1K-token text fixture | Paths and hashes are local to these directories. |
| `PI05/`, `VJ-AC/` | DROID 46-row fixture and V-JEPA DROID subset | Image/audio payloads are fixture-local; model checkpoints remain under `models/cache/`. |
| `COS-T2I/` | Five fixed Cosmos3 prompts | `cosmos3_t2i_prompts_v1.jsonl`. |

`seedtts_en_selection_100_v1.jsonl` and `seedtts_zh_selection_200_v1.jsonl`
preserve the frozen row selections used to produce the local T2S subsets; the
current Qwen T2S task reads only `*_local_v1.jsonl`. Their prompt-audio paths
also resolve to `exps/inputs/SEED-TTS/prompt-wavs/`. One-off fixture-generation
scripts are not runtime requirements; restore missing assets from the approved
fixture bundle and verify the recorded hashes.

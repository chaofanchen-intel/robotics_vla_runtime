#!/usr/bin/env bash
# Verify or apply managed overlays to the already-pinned runtime checkouts.
# This script never changes checkout refs or discards uncommitted changes.
set -euo pipefail
cd "$(dirname "$0")/../.."

mstar=external/mstar
sglang=external/sglang-omni
vllm=external/vllm-omni
patch_dir=exps/patches
pinned_mstar=9c6f1ad757bcc1c19ea47e98f18010199bc098dd
pinned_vllm=5917730a4b02aedb44d374173cdc1b10484ccd52
pinned_sglang=e8c635ecb9c68bcf2fbc32ceb3c40f85f3590657
patches=(
  mstar-01-vjepa-max-rollout-horizon-32.diff
  mstar-02-orpheus-offline-tokenizer.diff
  mstar-04-bagel-vllm-request-parity.diff
  mstar-05-code-predictor-smem-fit.diff
)
sglang_patches=(sglang-01-qwen-length-controls.diff)
vllm_patches=(vllm-01-bagel-i2t-length-controls.diff vllm-02-stage-omni-kv-recv-timeout.diff vllm-03-worker-connector-kv-transfer-port.diff)
vllm_agent_skill_overlays=(
  .claude/skills/3d-animation-short-generator
  .claude/skills/brand-promo-video-generator
  .claude/skills/co-op-game-intro-generator
  .claude/skills/h3-prompt-writing
  .claude/skills/handdrawn-live-video-generator
  .claude/skills/minimalist-product-ad-generator
  .claude/skills/music-video-subtitle-generator
  .claude/skills/paper-collage-explainer-generator
  .claude/skills/papercraft-stop-motion-explainer
)

mode=--verify
for option in "$@"; do
  case "$option" in
    --verify|--confirm|--revert) mode="$option" ;;
    *) echo "Unknown option: $option" >&2; exit 2 ;;
  esac
done
all_patches=("${patches[@]}" "${sglang_patches[@]}" "${vllm_patches[@]}")

head_of() { git -C "$1" rev-parse HEAD; }
ensure_pinned_others() {
  local repo expected actual
  while read -r repo expected; do
    actual=$(head_of "external/$repo")
    if [[ "$actual" != "$expected" ]]; then
      echo "external/$repo HEAD $actual differs from pinned $expected; stopping" >&2
      exit 1
    fi
  done <<REFS
vllm-omni $pinned_vllm
sglang-omni $pinned_sglang
vjepa2 204698b45b3712590f06245fbfba32d3be539812
vox-serve ce212de19e5df651ad9a44e5e3719b5e8cdb8ee3
REFS
}
ensure_pinned_others
verify_patch() {
  local patch file
  for patch in "${patches[@]}"; do
    git -C "$mstar" apply --ignore-space-change --reverse --check "../../$patch_dir/$patch" 2>/dev/null || {
      echo "Missing or altered patch: $patch" >&2; exit 1;
    }
  done
  while read -r file; do
    case "$file" in
      mstar/model/vjepa2/vjepa2_model.py|mstar/model/orpheus/orpheus_model.py|benchmark/request.py|mstar/model/components/code_predictor_kernels.py) ;;
      *) echo "Unexpected M* edit: $file" >&2; exit 1 ;;
    esac
  done < <(git -C "$mstar" diff --name-only)
  [[ -z "$(git -C "$mstar" ls-files --others --exclude-standard)" ]] || {
    echo "M* has untracked files; stopping" >&2; exit 1;
  }
}
verify_sglang_patch() {
  local patch file
  for patch in "${sglang_patches[@]}"; do
    git -C "$sglang" apply --ignore-space-change --reverse --check "../../$patch_dir/$patch" 2>/dev/null || {
      echo "Missing or altered patch: $patch" >&2; exit 1;
    }
  done
  while read -r file; do
    case "$file" in
      sglang_omni/client/types.py|sglang_omni/models/qwen3_omni/request_builders.py|sglang_omni/serve/openai_api.py|sglang_omni/serve/protocol.py|tests/unit_test/qwen3_omni/test_request_builder_text_only.py|tests/unit_test/serve/test_openai_api.py) ;;
      # This checkout locally replaces upstream's CLAUDE.md -> AGENTS.md symlink
      # with workspace instructions; it is documentation, not a runtime patch.
      CLAUDE.md)
        [[ -f "$sglang/CLAUDE.md" && ! -L "$sglang/CLAUDE.md" &&
           "$(git -C "$sglang" show HEAD:CLAUDE.md)" == "AGENTS.md" ]] || {
          echo "Unexpected SGLang-Omni CLAUDE.md edit (expected the local instructions overlay)" >&2
          exit 1
        } ;;
      *) echo "Unexpected SGLang-Omni edit: $file" >&2; exit 1 ;;
    esac
  done < <(git -C "$sglang" diff --name-only)
  [[ -z "$(git -C "$sglang" ls-files --others --exclude-standard)" ]] || {
    echo "SGLang-Omni has untracked files; stopping" >&2; exit 1;
  }
}
is_local_vllm_agent_skill_overlay_root() {
  local path="$1" overlay entry
  for overlay in "${vllm_agent_skill_overlays[@]}"; do
    [[ "$path" == "$overlay" ]] || continue
    [[ -d "$vllm/$overlay" && ! -L "$vllm/$overlay" ]] || return 1
    entry=$(git -C "$vllm" ls-tree HEAD -- "$overlay")
    [[ "${entry%% *}" == "120000" ]] || return 1
    return 0
  done
  return 1
}
is_local_vllm_agent_skill_file() {
  local path="$1" overlay
  case "$path" in
    *.md|*.yaml|*.yml) ;;
    *) return 1 ;;
  esac
  for overlay in "${vllm_agent_skill_overlays[@]}"; do
    [[ "$path" == "$overlay/"* ]] || continue
    is_local_vllm_agent_skill_overlay_root "$overlay"
    return
  done
  return 1
}
verify_vllm_patch() {
  local patch file
  for patch in "${vllm_patches[@]}"; do
    git -C "$vllm" apply --ignore-space-change --reverse --check "../../$patch_dir/$patch" 2>/dev/null || {
      echo "Missing or altered patch: $patch" >&2; exit 1;
    }
  done
  while read -r file; do
    case "$file" in
      vllm_omni/config/stage_config.py|vllm_omni/config/omni_config.py|tests/engine/test_stage_omni_kv_config.py) ;;
      vllm_omni/diffusion/models/bagel/bagel_transformer.py|vllm_omni/diffusion/models/bagel/pipeline_bagel.py|vllm_omni/entrypoints/openai/serving_chat.py|tests/diffusion/models/bagel/test_step_execution.py|tests/entrypoints/openai_api/test_serving_chat_sampling_params.py) ;;
      vllm_omni/distributed/omni_connectors/model_runner/omni_connector_runtime.py) ;;
      .claude/skills/*)
        is_local_vllm_agent_skill_overlay_root "$file" || {
          echo "Unexpected vLLM-Omni agent skill edit: $file" >&2; exit 1;
        } ;;
      *) echo "Unexpected vLLM-Omni edit: $file" >&2; exit 1 ;;
    esac
  done < <(git -C "$vllm" diff --name-only)
  while IFS= read -r file; do
    [[ -n "$file" ]] || continue
    [[ "$file" == tests/engine/test_stage_omni_kv_config.py ]] && continue  # added by vllm-02, untracked until the patch is applied
    is_local_vllm_agent_skill_file "$file" || {
      echo "Unexpected untracked vLLM-Omni file: $file" >&2; exit 1;
    }
  done < <(git -C "$vllm" ls-files --others --exclude-standard)
}
write_ledger() {
  envs/mstar/bin/python - "$patch_dir" "${all_patches[@]}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
patches = [
    {"file": name, "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()}
    for name in sys.argv[2:]
]
(root / "applied.json").write_text(json.dumps({"applied": patches}, indent=2) + "\n")
PY
}

if [[ "$mode" == --verify ]]; then
  [[ "$(head_of "$mstar")" == "$pinned_mstar" ]] || {
    echo "M* HEAD differs from the recorded main ref; stopping" >&2; exit 1;
  }
  [[ "$(git -C "$mstar" branch --show-current)" == main ]] || {
    echo "M* checkout is not on branch main; stopping" >&2; exit 1;
  }
  verify_patch
  verify_sglang_patch
  verify_vllm_patch
  [[ -f "$patch_dir/applied.json" ]] || {
    echo "Patch ledger missing" >&2; exit 1;
  }
  envs/mstar/bin/python - "$patch_dir" "${all_patches[@]}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
actual = json.loads((root / "applied.json").read_text())["applied"]
expected = [
    {"file": name, "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()}
    for name in sys.argv[2:]
]
if actual != expected:
    raise SystemExit(f"Patch ledger mismatch: {actual!r} != {expected!r}")
PY
  echo "Pinned baseline and patch ledger verified"
  exit 0
fi

[[ "$(head_of "$mstar")" == "$pinned_mstar" ]] || {
  echo "M* HEAD differs from the recorded main ref; stopping" >&2
  exit 1
}
[[ "$(git -C "$mstar" branch --show-current)" == main ]] || {
  echo "M* checkout is not on branch main; stopping" >&2
  exit 1
}

if [[ "$mode" == --revert ]]; then
  [[ "$(head_of "$mstar")" == "$pinned_mstar" ]] || {
    echo "Cannot revert patches on a non-pinned M* HEAD" >&2; exit 1;
  }
  for patch in "${patches[@]}"; do
    if git -C "$mstar" apply --ignore-space-change --reverse --check "../../$patch_dir/$patch" 2>/dev/null; then
      git -C "$mstar" apply --ignore-space-change --reverse "../../$patch_dir/$patch"
    elif ! git -C "$mstar" apply --ignore-space-change --check "../../$patch_dir/$patch"; then
      echo "Patch $patch is neither applied nor cleanly reverted; stopping" >&2
      exit 1
    fi
  done
  if [[ -n "$(git -C "$sglang" status --porcelain=v1 -uall)" ]]; then
    verify_sglang_patch
  fi
  for patch in "${sglang_patches[@]}"; do
    if git -C "$sglang" apply --ignore-space-change --reverse --check "../../$patch_dir/$patch" 2>/dev/null; then
      git -C "$sglang" apply --ignore-space-change --reverse "../../$patch_dir/$patch"
    elif ! git -C "$sglang" apply --ignore-space-change --check "../../$patch_dir/$patch"; then
      echo "Patch $patch is neither applied nor cleanly reverted; stopping" >&2
      exit 1
    fi
  done
  if [[ -n "$(git -C "$vllm" status --porcelain=v1 -uall)" ]]; then
    verify_vllm_patch
  fi
  for patch in "${vllm_patches[@]}"; do
    if git -C "$vllm" apply --ignore-space-change --reverse --check "../../$patch_dir/$patch" 2>/dev/null; then
      git -C "$vllm" apply --ignore-space-change --reverse "../../$patch_dir/$patch"
    elif ! git -C "$vllm" apply --ignore-space-change --check "../../$patch_dir/$patch"; then
      echo "Patch $patch is neither applied nor cleanly reverted; stopping" >&2
      exit 1
    fi
  done
  envs/mstar/bin/python - "$patch_dir" <<'PY'
import json
import sys
from pathlib import Path
Path(sys.argv[1], "applied.json").write_text(json.dumps({"applied": []}, indent=2) + "\n")
PY
  echo "Selected patches reverted; no source checkout was reset"
  exit 0
fi

for patch in "${patches[@]}"; do
  if git -C "$mstar" apply --ignore-space-change --reverse --check "../../$patch_dir/$patch" 2>/dev/null; then
    continue
  fi
  git -C "$mstar" apply --ignore-space-change --check "../../$patch_dir/$patch"
  git -C "$mstar" apply --ignore-space-change "../../$patch_dir/$patch"
done
for patch in "${sglang_patches[@]}"; do
  if git -C "$sglang" apply --ignore-space-change --reverse --check "../../$patch_dir/$patch" 2>/dev/null; then
    continue
  fi
  git -C "$sglang" apply --ignore-space-change --check "../../$patch_dir/$patch"
  git -C "$sglang" apply --ignore-space-change "../../$patch_dir/$patch"
done
for patch in "${vllm_patches[@]}"; do
  if git -C "$vllm" apply --ignore-space-change --reverse --check "../../$patch_dir/$patch" 2>/dev/null; then
    continue
  fi
  git -C "$vllm" apply --ignore-space-change --check "../../$patch_dir/$patch"
  git -C "$vllm" apply --ignore-space-change "../../$patch_dir/$patch"
done
verify_patch
verify_sglang_patch
verify_vllm_patch
write_ledger
echo "Pinned baseline patched; run bash $patch_dir/bootstrap.sh --verify"

#!/usr/bin/env bash
# The engine: gr8 at a fixed commit with patches/gr8 applied, built on this Spark in containers (no compiler is
# installed on the host), then copied to the other three. Sourced after scripts/lib.sh and cluster.sh.
[[ "${_ML4_ENGINE_LOADED:-0}" == 1 ]] && return 0
_ML4_ENGINE_LOADED=1

SRC="$WORK/src/gr8"
ENGINE_BIN="$WORK/bin/g6m4"
FRONT_BIN="$WORK/bin/g6front-rs"
SHIM="$WORK/bin/g6s2shim.py"

# engine_stamp: changes when the commit, a patch or a build flag changes; a build with the same stamp is not repeated.
engine_stamp() {
  { echo "$GR8_COMMIT $ENGINE_FLAGS $BASE_IMAGE $RUST_IMAGE"; cat "$REPO_ROOT"/patches/gr8/*.patch; } | sha256sum | cut -c1-16
}

# --- the images --------------------------------------------------------------------------------------------------------
_pull_rank() {
  [[ "${P_IMAGE[$1]}" == ok ]] && return 0
  on "$1" docker pull -q "$BASE_IMAGE" >/dev/null || { echo "Spark $1 ($(rname "$1")): docker pull failed"; return 1; }
}
ensure_image() {
  local r missing=() pid
  for r in 0 1 2 3; do [[ "${P_IMAGE[r]}" == ok ]] || missing+=("$r"); done
  if (( ${#missing[@]} == 0 )); then ok 'The container image is on all four Sparks.'; return 0; fi
  if dry; then note "Would pull the container image (about 24 GB) on Spark(s) ${missing[*]}."; return 0; fi
  ( EACH_RANKS="${missing[*]}" each_rank _pull_rank || { show_each; exit 1; } ) & pid=$!
  spin "$pid" "Downloading the container image (about 24 GB) on Spark(s) ${missing[*]}" \
    || fail E_IMAGE "Could not download $BASE_IMAGE on every Spark. Check the internet connection, then run ./start.sh again."
  for r in "${missing[@]}"; do P_IMAGE[r]=ok; done
  ok 'The container image is now on all four Sparks.'
}

# --- source ------------------------------------------------------------------------------------------------------------
fetch_source() {
  local p
  if [[ "$(cat "$SRC/.ml4-stamp" 2>/dev/null)" == "$1" ]]; then return 0; fi
  rm -rf -- "$SRC"; mkdir -p -- "$(dirname -- "$SRC")"
  git clone -q --filter=blob:none "$GR8_REPO" "$SRC" || return 1
  git -C "$SRC" checkout -q "$GR8_COMMIT" || return 1
  for p in "$REPO_ROOT"/patches/gr8/*.patch; do
    git -C "$SRC" apply --index "$p" || { echo "patch failed: $p"; return 1; }
  done
  printf '%s\n' "$1" > "$SRC/.ml4-stamp"
}

# --- build -------------------------------------------------------------------------------------------------------------
# In the Rust image: the settings table, three libraries the engine links, and the API front end.
build_rust() {
  docker run --rm -v "$SRC":/g -w /g "$RUST_IMAGE" bash -c "
    set -e
    cargo run --quiet --release --manifest-path engine/knobs-rs/Cargo.toml --bin knobs -- gen
    (cd engine/knobs-rs && cargo build --quiet --release --features capi)
    (cd engine && cargo build --quiet --release)
    (cd engine/loader-rs && cargo build --quiet --release)
    # The front end's grammar library (xgrammar) is C++: its build needs cmake and libclang.
    export DEBIAN_FRONTEND=noninteractive
    { apt-get update -qq && apt-get install -y -qq cmake clang libclang-dev; } > /tmp/apt.log 2>&1 || { cat /tmp/apt.log; exit 1; }
    (cd frontend/g6front-rs && cargo build --quiet --release)
    chown -R $(id -u):$(id -g) engine docs frontend"
}
# In the NVIDIA image: the engine itself, for the Spark's GPU (sm_121a).
build_engine() {
  docker run --rm --gpus all -v "$WORK":/w -w /w/src/gr8/engine/runtime "$BASE_IMAGE" bash -c "
    set -e
    GR8_NVCC_CACHE=0 OUT=/w/build WIDTHS=4 EXTRA='$ENGINE_FLAGS' JOBS=$BUILD_JOBS ./build_m4.sh 121a g6
    chown -R $(id -u):$(id -g) /w/build"
}
_push_bin_rank() {
  local r=$1 dst; dst="$(work_of "$r")/bin"
  on "$r" mkdir -p "$dst" || return 1
  # shellcheck disable=SC2086
  rsync -a -e "ssh $SSH_OPTS" "$ENGINE_BIN" "$WORK/bin/.stamp" "${RANKS[r]}:$dst/" || { echo "Spark $r: copy failed"; return 1; }
}
ensure_engine() {
  local stamp logf pid r
  stamp=$(engine_stamp)
  if [[ "$(cat "$WORK/bin/.stamp" 2>/dev/null)" == "$stamp" && -x "$ENGINE_BIN" && -x "$FRONT_BIN" && -f "$SHIM" ]]; then
    ok "The engine is built (gr8 ${GR8_COMMIT:0:8} + $(ls "$REPO_ROOT"/patches/gr8/*.patch | wc -l | xargs) patches, build $stamp)."
  else
    if dry; then note "Would fetch gr8 ${GR8_COMMIT:0:8}, apply patches/gr8 and build the engine (about 6 minutes)."; return 0; fi
    need_cmd git
    mkdir -p -- "$WORK/bin" "$LOG_DIR"; logf="$LOG_DIR/build.log"; : > "$logf"
    ( fetch_source "$stamp" ) >> "$logf" 2>&1 & pid=$!
    spin "$pid" "Fetching the engine source (gr8 ${GR8_COMMIT:0:8}) and applying the patches" \
      || fail E_SOURCE "Could not fetch or patch the engine source. Log: $logf"
    ok "Engine source ready: $SRC"
    ( build_rust ) >> "$logf" 2>&1 & pid=$!
    spin "$pid" 'Building the Rust parts (libraries and the API front end)' "$logf" \
      || { tail -n 20 "$logf" | sed 's/^/    /' >&2; fail E_BUILD "The Rust build failed. Log: $logf"; }
    ok 'Rust parts built.'
    ( build_engine ) >> "$logf" 2>&1 & pid=$!
    spin "$pid" 'Building the engine for the GPU (the longest step, about 5 minutes)' "$logf" \
      || { grep -n -E ' error|error:' "$logf" | tail -n 20 | sed 's/^/    /' >&2; fail E_BUILD "The engine build failed. Log: $logf"; }
    cp -- "$WORK/build/g6m4" "$ENGINE_BIN.new" && mv -- "$ENGINE_BIN.new" "$ENGINE_BIN"
    cp -- "$SRC/frontend/g6front-rs/target/release/g6front-rs" "$FRONT_BIN.new" && mv -- "$FRONT_BIN.new" "$FRONT_BIN"
    cp -- "$SRC/frontend/g6s2shim.py" "$SHIM"
    [[ -f "$SRC/frontend/wire.py" ]] && cp -- "$SRC/frontend/wire.py" "$WORK/bin/wire.py"
    printf '%s\n' "$stamp" > "$WORK/bin/.stamp"
    ok "Engine built: $ENGINE_BIN ($(sha256sum "$ENGINE_BIN" | cut -c1-12))."
  fi
  dry && return 0
  # The same binary on the other three Sparks.
  EACH_RANKS='1 2 3' each_rank _push_bin_rank || { show_each; fail E_BUILD 'Could not copy the engine to every Spark.'; }
  ok 'The engine is on all four Sparks.'
}

#!/usr/bin/env bash
# The server: one engine container on each Spark (a rank), the OpenAI API in front of rank 0, and the memory guard.
# Sourced after scripts/lib.sh and cluster.sh.
[[ "${_ML4_SERVE_LOADED:-0}" == 1 ]] && return 0
_ML4_SERVE_LOADED=1

URL="http://127.0.0.1:$PORT"
RUN="$WORK/run"          # this Spark's run directory: s2/ (the engine's request files), rank0.err, rank0.out
api_up() { curl -fsS --max-time 5 "$URL/v1/models" 2>/dev/null | grep -q "\"$SERVED_NAME\""; }
engine_up() { docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "${CONTAINER_PREFIX}0" && [[ -f "$RUN/s2/ready.s2" ]]; }

# The engine's settings. These are the measured ones (docs/TUNING.md); ENGINE_ENV / ENGINE_ARGS add to them.
engine_env() {
  printf '%s ' -e GR8_CP=1 -e GR8_CP_R=8192 -e GR8_DEC_SPLIT=3 -e GR8_DEC_SHARED_SCRATCH=1 -e GR8_PF_ATTN_ROWS=1024 \
    -e GR8_PF_IXSPLIT=1 -e GR8_IDXQ_SPLIT=1 -e GR8_CP_ONECHAIN=1 -e GR8_DFL_REUSE=1 -e GR8_DFL_REUSE_MIN=0
  local kv; for kv in $ENGINE_ENV; do printf '%s ' -e "$kv"; done
}
engine_args() {
  printf '%s ' --image --layers all --cap "$CONTEXT" --chunk 1 --temp 0 --graph --wide --reuse-prefix \
    --prefill "$PREFILL_ROWS" --pfv2 --pf-lanes "$PREFILL_LANES" --mtp 0 --dflash-k 3 --spec-adaptive --spec-levels 1,3 \
    --serve /w/run/s2 --s2-port "$ENGINE_PORT" --s2-sessions "$SESSIONS" --s2-epoch "$1" $ENGINE_ARGS
}

# --- stop --------------------------------------------------------------------------------------------------------------
_stop_rank() { on "$1" docker rm -f "$CONTAINER_PREFIX$1" >/dev/null 2>&1 || true; }
stop_front() {
  local pid
  pid=$(read_state front.pid); pid_alive "$pid" && kill "$pid" 2>/dev/null
  # The request bridge gets SIGKILL: on SIGTERM it tells the engine to stop, and the engine is stopped separately.
  pid=$(read_state shim.pid); pid_alive "$pid" && kill -9 "$pid" 2>/dev/null
  rm -f -- "$STATE_DIR/front.pid" "$STATE_DIR/shim.pid"
}
stop_guard() { local pid; pid=$(read_state memguard.pid); pid_alive "$pid" && kill "$pid" 2>/dev/null; rm -f -- "$STATE_DIR/memguard.pid"; }
stop_all() { stop_guard; stop_front; each_rank _stop_rank || true; }

# --- start the ranks ---------------------------------------------------------------------------------------------------
_launch_rank() {
  local r=$1 epoch=$2 w ips
  w=$(work_of "$r"); ips=$(IFS=,; for i in 0 1 2 3; do printf '%s,' "$(rhost "$i")"; done); ips=${ips%,}
  # A new run directory each start (its files belong to root, so a container removes it).
  on "$r" docker run --rm -v "$w":/w --entrypoint /bin/rm "$BASE_IMAGE" -rf /w/run/s2 "/w/run/rank$r.out" "/w/run/rank$r.err" >/dev/null 2>&1
  on "$r" mkdir -p "$w/run/s2" || return 1
  # shellcheck disable=SC2046
  on "$r" docker run -d --name "$CONTAINER_PREFIX$r" --gpus all --network host --ipc=host --device /dev/infiniband \
    --cap-add IPC_LOCK --ulimit memlock=-1 -v "$w":/w $(engine_env) "$BASE_IMAGE" bash -c \
    "/w/bin/g6m4 --rank $r --ips $ips --ring --manifest /w/pack/main/rank$r.manifest.json --pack /w/pack/main/rank$r.pack --dflash-manifest /w/pack/draft/rank$r.manifest.json --dflash-pack /w/pack/draft/rank$r.pack $(engine_args "$epoch") > /w/run/rank$r.out 2> /w/run/rank$r.err" >/dev/null
}
launch_ranks() {
  if dry; then note "Would start one engine container on each Spark (${CONTAINER_PREFIX}0..3) with $(engine_args EPOCH)"; return 0; fi
  each_rank _stop_rank || true
  each_rank _launch_rank "$(date +%s)" || { show_each; fail E_START 'Could not start the engine container on every Spark (see above).'; }
  ok 'Engine containers started on all four Sparks.'
}

# Memory guard: every Spark is sampled each MEMGUARD_INTERVAL_S. When one stays under MEMGUARD_FLOOR_GIB for
# MEMGUARD_HOLD_S, the guard stops the server. A Spark that runs out of memory stops answering; this prevents that.
_mem_rank() { on_fn "$1" mem_counting 'mem_counting | cut -d" " -f1'; }
start_memguard() {
  [[ "$MEMGUARD" == 1 ]] || return 0
  dry && return 0
  # 9>&- : the guard must not keep start.sh's lock (file descriptor 9) open.
  ( exec </dev/null >/dev/null 2>&1 9>&-
    trap '' HUP INT; trap 'exit 0' TERM
    declare -a under=(0 0 0 0)
    while sleep "$MEMGUARD_INTERVAL_S"; do
      docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_PREFIX}0" || exit 0
      each_rank _mem_rank || true
      for r in 0 1 2 3; do
        g=${EACH_OUT[r]:-}
        if is_number "$g" && ! at_least "$g" "$MEMGUARD_FLOOR_GIB"; then under[r]=$(( under[r] + MEMGUARD_INTERVAL_S )); else under[r]=0; fi
        if (( under[r] >= MEMGUARD_HOLD_S )); then
          printf '%s memory guard: Spark %s (%s) has %s GiB available, under %s GiB for %ss. The server is stopped.\n' \
            "$(date -u +%FT%TZ)" "$r" "$(rhost "$r")" "$g" "$MEMGUARD_FLOOR_GIB" "${under[r]}" >> "$LOG_DIR/memguard.log"
          stop_front; each_rank _stop_rank || true
          exit 0
        fi
      done
    done ) &
  write_state memguard.pid "$!"
  ok "Memory guard is on: it stops the server if a Spark stays under $MEMGUARD_FLOOR_GIB GiB for ${MEMGUARD_HOLD_S}s."
}

# wait_engine: until rank 0 writes its ready file. Shows the engine's own boot stages. Fails when a container exits.
wait_engine() {
  local t0=$SECONDS i=0 stage frames='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏' r last=-30
  nowrap
  while ! [[ -f "$RUN/s2/ready.s2" ]]; do
    (( SECONDS - t0 < READY_TIMEOUT_S )) || fail E_TIMEOUT "The engine is not ready after ${READY_TIMEOUT_S}s. Look at $RUN/rank0.err, then run ./stop.sh."
    if (( (SECONDS - t0) % 5 == 4 )); then
      each_rank _alive_rank || true
      for r in 0 1 2 3; do
        [[ "${EACH_OUT[r]}" == up ]] && continue
        (( TTY )) && printf '\r\033[2K'
        wrap
        on "$r" tail -n 6 "$(work_of "$r")/run/rank$r.err" 2>/dev/null | sed 's/^/    /' >&2
        stop_all
        fail E_START "The engine on Spark $r ($(rname "$r")) stopped during the start (its last lines are above; full log: ~/$ML4_DIR/run/rank$r.err on that Spark)."
      done
    fi
    stage=$(grep -o 'boot [0-9.]* s .*' "$RUN/rank0.err" 2>/dev/null | tail -n 1 | sed 's/^boot [0-9.]* s //') || true
    if (( TTY )); then
      printf '\r\033[2K  %s%s%s Loading the model on four Sparks %s(%s)%s %s' "$C_CYAN" "${frames:i++%10:1}" "$C_RESET" "$C_DIM" "$(hms $(( SECONDS - t0 )))" "$C_RESET" "${stage:-starting}"
      sleep 0.25
    else
      (( SECONDS - last >= 30 )) && { printf '  ... loading the model (%s) %s\n' "$(hms $(( SECONDS - t0 )))" "${stage:-starting}"; last=$SECONDS; }
      sleep 1
    fi
  done
  (( TTY )) && printf '\r\033[2K'
  wrap
  ok "The model is loaded on all four Sparks ($(hms $(( SECONDS - t0 ))))."
}
_alive_rank() { if on "$1" docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_PREFIX$1"; then echo up; else echo down; fi; }

# The API: the engine's request bridge (Python, no packages) and the Rust front end, both on this Spark.
start_front() {
  local sock="$STATE_DIR/engine.sock" i front pid
  dry && { note "Would start the API on $BIND_HOST:$PORT."; return 0; }
  stop_front; rm -f -- "$sock"; mkdir -p -- "$LOG_DIR"
  ( cd "$WORK/bin" && exec setsid python3 "$SHIM" --dir "$RUN/s2" --sock "$sock" --sessions "$SESSIONS" --max-model-len "$CONTEXT" \
      --model-name "$SERVED_NAME" --vocab-size 204800 --eos-ids 2 ) > "$LOG_DIR/bridge.log" 2>&1 < /dev/null 9>&- &
  write_state shim.pid "$!"
  for (( i = 0; i < 50; i++ )); do [[ -S "$sock" ]] && break; sleep 0.2; done
  [[ -S "$sock" ]] || { tail -n 5 "$LOG_DIR/bridge.log" | sed 's/^/    /' >&2; fail E_API "The request bridge did not start. Log: $LOG_DIR/bridge.log"; }
  setsid "$FRONT_BIN" --model-dir "$WORK/model" --engine-sock "$sock" --host "$BIND_HOST" --port "$PORT" --dialect ml4 \
    --enable-prompt-tokens-details --served-model-name "$SERVED_NAME" --default-reasoning-effort "$REASONING_EFFORT" \
    --max-model-len "$CONTEXT" > "$LOG_DIR/api.log" 2>&1 < /dev/null 9>&- &
  front=$!; write_state front.pid "$front"
  # The front end reads the tokenizer before it listens. Measured: 40 seconds on the first start after an install,
  # 1 second on later starts.
  ( t0=$SECONDS
    until api_up; do
      kill -0 "$front" 2>/dev/null || exit 1
      (( SECONDS - t0 < API_TIMEOUT_S )) || exit 1
      sleep 0.5
    done ) & pid=$!
  spin "$pid" 'Starting the API (the first start reads the tokenizer: up to one minute)' \
    || { tail -n 8 "$LOG_DIR/api.log" | cut -c1-300 | sed 's/^/    /' >&2; fail E_API "The API does not answer on port $PORT after ${API_TIMEOUT_S}s, or it stopped. Log: $LOG_DIR/api.log"; }
  ok "The API answers: $(public_url)/v1 (model $SERVED_NAME)."
}

smoke_test() {
  local t0=$SECONDS
  curl -fsS --max-time 120 "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly one word: ready\"}],\"max_tokens\":256,\"temperature\":0}" \
    > "$LOG_DIR/smoke.json" || fail E_SMOKE "The test request failed. Logs: $LOG_DIR/api.log, $RUN/rank0.err"
  python3 - "$LOG_DIR/smoke.json" <<'PY' || fail E_SMOKE "The test reply is empty: $LOG_DIR/smoke.json"
import json, sys
m = json.load(open(sys.argv[1]))["choices"][0]["message"]
text = (m.get("content") or m.get("reasoning_content") or m.get("reasoning") or "").strip()
sys.exit(0 if text and len(set(text)) > 2 else 1)
PY
  ok "Test request answered in $(( SECONDS - t0 ))s."
}

#!/usr/bin/env bash
# The weights: one download on this Spark, then one conversion for each Spark into the engine's format (no
# requantization: the same numbers in a different layout), each checked against the download and copied to its Spark
# over the cables. Sourced after scripts/lib.sh, cluster.sh and engine.sh.
[[ "${_ML4_PACK_LOADED:-0}" == 1 ]] && return 0
_ML4_PACK_LOADED=1

# --- the download ------------------------------------------------------------------------------------------------------
find_hf() {
  local c
  for c in "$(command -v hf 2>/dev/null)" "$HOME/.local/bin/hf"; do [[ -n "$c" && -x "$c" ]] && { printf '%s' "$c"; return 0; }; done
  return 1
}
ensure_hf() {
  HF=$(find_hf) && return 0
  command -v pipx >/dev/null || fail E_DEPENDENCY 'The Hugging Face tool (hf) is not installed, and pipx is not installed. Run: sudo apt install pipx && pipx install "huggingface_hub[hf_xet]"'
  log 'Installing the Hugging Face tool (pipx install huggingface_hub) ...'
  pipx install 'huggingface_hub[hf_xet]' >/dev/null || fail E_DEPENDENCY 'pipx install huggingface_hub failed.'
  HF=$(find_hf) || fail E_DEPENDENCY 'hf is not found after the installation (is ~/.local/bin on PATH?).'
}
_dir_bytes() { du -sb "$1" 2>/dev/null | cut -f1; }
_shard_bytes() { find "$MODEL_DIR" -maxdepth 1 -name '*.safetensors' -printf '%s\n' 2>/dev/null | awk '{ s += $1 } END { printf "%d", s }'; }
pack_complete() {
  local f
  for f in config.json tokenizer.json tokenizer_config.json chat_template.jinja; do [[ -f "$MODEL_DIR/$f" ]] || return 1; done
  [[ "$(find "$MODEL_DIR" -maxdepth 1 -name '*.safetensors' | wc -l | xargs)" == "$EXPECTED_SHARDS" ]] || return 1
  [[ "$(_shard_bytes)" == "$EXPECTED_WEIGHT_BYTES" ]]
}
_download_bar() { bar_line 'Mistral Large 4 EXL3' "$(_dir_bytes "$MODEL_DIR")" "$EXPECTED_WEIGHT_BYTES" "$DL_START" $(( SECONDS - DL_T0 )); }
ensure_pack() {
  local need free have pid
  if pack_complete; then ok "The model download is complete: $MODEL_DIR ($EXPECTED_SHARDS files, $(( EXPECTED_WEIGHT_BYTES / 1000000000 )) GB)."; return 0; fi
  if all_packs_complete && [[ -f "$WORK/model/tokenizer.json" ]]; then ok 'Every Spark has its converted weights; the download is not necessary.'; return 0; fi
  need=$(( EXPECTED_WEIGHT_BYTES / 1073741824 + DISK_SLACK_GIB ))
  if dry; then note "Would download $PACK_ID (403 GB) to $MODEL_DIR."; return 0; fi
  mkdir -p -- "$MODEL_DIR" || fail E_DISK "Cannot create $MODEL_DIR."
  free=$(df -Pk "$MODEL_DIR" | awk 'END { printf "%d", $4 / 1048576 }')
  have=$(( $(_dir_bytes "$MODEL_DIR") / 1073741824 ))
  (( free + have >= need )) || fail E_DISK "The download needs about $need GiB in $MODEL_DIR; $free GiB is free (and $have GiB is already there)."
  ensure_hf
  log "Downloading $PACK_ID (403 GB) to $MODEL_DIR. If it stops, run ./start.sh again: it continues where it stopped."
  DL_START=$(_dir_bytes "$MODEL_DIR"); DL_T0=$SECONDS
  HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 "$HF" download "$PACK_ID" --revision "$PACK_REVISION" --local-dir "$MODEL_DIR" \
    > "$LOG_DIR/download.log" 2>&1 & pid=$!
  watch_bars "$pid" -- 1 _download_bar
  wait "$pid" || { tail -n 5 "$LOG_DIR/download.log" | sed 's/^/    /' >&2; fail E_DOWNLOAD "The download stopped. Run ./start.sh again to continue. Log: $LOG_DIR/download.log"; }
  pack_complete || fail E_DOWNLOAD "The download ended, but $MODEL_DIR is not complete ($EXPECTED_SHARDS model files of $EXPECTED_WEIGHT_BYTES bytes in total are necessary). Run ./start.sh again."
  ok "Downloaded in $(hms $(( SECONDS - DL_T0 )))."
}

# The small files the API needs (tokenizer, chat template, configuration), kept beside the converted weights.
ensure_model_files() {
  local f
  dry && return 0
  [[ -f "$MODEL_DIR/tokenizer.json" ]] || { [[ -f "$WORK/model/tokenizer.json" ]] && return 0; fail E_WEIGHTS "No tokenizer in $MODEL_DIR."; }
  mkdir -p -- "$WORK/model"
  for f in "$MODEL_DIR"/*.json "$MODEL_DIR"/*.jinja; do [[ -f "$f" ]] && cp -u -- "$f" "$WORK/model/"; done
}

# The draft head: one file from Mistral's own release, checked by its SHA-256.
ensure_draft() {
  local f="$MODEL_DIR/draft/$DSPARK_FILE" pid
  if all_packs_complete && [[ -f "$WORK/model/tokenizer.json" ]]; then return 0; fi
  if [[ "$(stat -c %s "$f" 2>/dev/null)" == "$DSPARK_BYTES" && "$(read_state draft-verified)" == "$DSPARK_SHA256" ]]; then
    ok 'The draft head is downloaded and verified.'; return 0
  fi
  if dry; then
    if [[ "$(stat -c %s "$f" 2>/dev/null)" == "$DSPARK_BYTES" ]]; then note 'Would check the draft head (it is downloaded; SHA-256).'
    else note "Would download the draft head ($DSPARK_REPO, 4.2 GB)."; fi
    return 0
  fi
  if [[ "$(stat -c %s "$f" 2>/dev/null)" != "$DSPARK_BYTES" ]]; then
    ensure_hf
    HF_HUB_DISABLE_PROGRESS_BARS=1 "$HF" download "$DSPARK_REPO" "$DSPARK_FILE" --revision "$DSPARK_REVISION" --local-dir "$MODEL_DIR/draft" \
      > "$LOG_DIR/download-draft.log" 2>&1 & pid=$!
    spin "$pid" 'Downloading the draft head from Mistral (4.2 GB)' \
      || fail E_DOWNLOAD "Could not download $DSPARK_FILE from $DSPARK_REPO. If Hugging Face asks you to accept the model's licence, do that on its page, then set HF_TOKEN and run ./start.sh again. Log: $LOG_DIR/download-draft.log"
  fi
  ( sha256sum "$f" | grep -q "^$DSPARK_SHA256 " ) & pid=$!
  spin "$pid" 'Checking the draft head (SHA-256)' || fail E_DOWNLOAD "$f does not have the expected SHA-256. Delete it and run ./start.sh again."
  write_state draft-verified "$DSPARK_SHA256"
  ok 'The draft head is downloaded and verified.'
}

# --- conversion and copy -----------------------------------------------------------------------------------------------
# pack_stamp: changes when the download, the draft head or the converter changes.
pack_stamp() { { echo "$PACK_REVISION $DSPARK_SHA256 $GR8_COMMIT"; cat "$REPO_ROOT"/scripts/convert/*.py; } | sha256sum | cut -c1-16; }
# _pack_state R: "<stamp or -> <bytes of the rank's main pack>" on Spark R.
_pack_state() {
  on "$1" sh -c 'd="$1/pack"; echo "$(cat "$d/.complete-$2" 2>/dev/null || echo -) $(stat -c %s "$d/main/rank$2.pack" 2>/dev/null || echo 0) $(stat -c %s "$d/draft/rank$2.pack" 2>/dev/null || echo 0)"' sh "$(work_of "$1")" "$1"
}
declare -a PACK_DONE=()
read_pack_states() {
  local r stamp; stamp=$(pack_stamp)
  each_rank _pack_state || { show_each; fail E_WEIGHTS 'Could not read the state of the weights on every Spark.'; }
  for r in 0 1 2 3; do
    read -r s b d <<< "${EACH_OUT[r]}"
    if [[ "$s" == "$stamp" && "$b" == "$RANK_PACK_BYTES" && "$d" -gt 0 ]]; then PACK_DONE[r]=1; else PACK_DONE[r]=0; fi
  done
}
all_packs_complete() { [[ "${PACK_DONE[*]:-}" == '1 1 1 1' ]]; }

# convert_rank R: Spark R's quarter of the model and of the draft head, written on this Spark and compared with the
# download, tensor by tensor.
convert_rank() {
  local r=$1 layers; layers=$(seq -s, 0 71)
  docker run --rm --gpus all --ipc=host -e CHECK_RANKS="$r" -v "$MODEL_DIR":/pack:ro -v "$WORK":/w \
    -v "$REPO_ROOT/scripts/convert":/c:ro "$BASE_IMAGE" bash -c "
      set -e
      pip install -q transformers==$TRANSFORMERS_VERSION 2>&1 | tail -1
      rm -f /w/pack/main/rank$r.* /w/pack/draft/rank$r.*
      python3 -I /c/ml4_pack.py /pack /w/src/gr8 /c 4 /w/pack/main --layers $layers --ranks $r
      echo CHECKING
      python3 -I /c/ml4_pack_check.py /pack /w/pack/main 4 /c
      python3 -I /c/ml4_dspark_pack.py /pack /w/src/gr8 4 /w/pack/draft --ranks $r
      python3 -I /c/ml4_dspark_check.py /pack /w/pack/draft 4 $r
      chown -R $(id -u):$(id -g) /w/pack"
}
# One progress line for each Spark of the batch that is converted (CONV_RANKS).
_convert_bars() {
  local r label
  for r in $CONV_RANKS; do
    label="Spark $r: converting"
    grep -q CHECKING "$LOG_DIR/convert-$r.log" 2>/dev/null && label="Spark $r: checking"
    bar_line "$label" "$(stat -c %s "$WORK/pack/main/rank$r.pack" 2>/dev/null || echo 0)" "$RANK_PACK_BYTES" 0 $(( SECONDS - CONV_T0 ))
  done
}

# copy_routes T: this Spark's routes to Spark T, one per line: "<jump address-rank or -> <destination address>".
# A neighbour: over the cable. The Spark across the ring: through a neighbour, as an ssh jump, so the bytes use two
# cables and only this Spark's ssh key is necessary.
copy_routes() {
  local t=$1 n ip
  if [[ -n "${VIA[0 $t]:-}" ]]; then for ip in ${VIA[0 $t]//,/ }; do echo "- $ip"; done; return 0; fi
  for n in 1 3; do
    [[ -n "${VIA[0 $n]:-}" && -n "${VIA[$n $t]:-}" ]] || continue
    for ip in ${VIA[$n $t]//,/ }; do echo "${VIA[0 $n]%%,*}-$n $ip"; done
  done
}
ssh_user() { [[ "${RANKS[$1]}" == *@* ]] && printf '%s@' "${RANKS[$1]%%@*}"; return 0; }
copy_ssh() {   # copy_ssh JUMP: the ssh command for rsync -e
  local cmd="ssh $COPY_SSH_OPTS"
  [[ "$1" == - ]] || cmd+=" -o ProxyCommand=\"ssh $COPY_SSH_OPTS -W %h:%p $(ssh_user "${1##*-}")${1%-*}\""
  printf '%s' "$cmd"
}
_route_ok() { eval "$(copy_ssh "$1")" "$(ssh_user "$2")$3" true </dev/null >/dev/null 2>&1; }
_copy_size() { on "$1" sh -c 'du -sb "$1" 2>/dev/null | cut -f1' sh "$(work_of "$1")/pack/main"; }
_copy_bars() {
  local r
  EACH_RANKS="$COPY_RANKS" each_rank _copy_size || true
  for r in $COPY_RANKS; do bar_line "Spark $r: copying" "${EACH_OUT[r]:-0}" "$RANK_PACK_BYTES" 0 $(( SECONDS - COPY_T0 )); done
}
# push_rank R: Spark R's converted files from this Spark to Spark R (runs in the background; its output is the log).
push_rank() {
  local r=$1 jump dest route='' dst try
  dst="$(work_of "$r")/pack"
  while read -r jump dest; do
    [[ -n "$dest" ]] || continue
    if _route_ok "$jump" "$r" "$dest"; then route="$jump $dest"; break; fi
  done < <(copy_routes "$r")
  if [[ -z "$route" ]]; then
    echo "ssh over the cables does not answer; copying over the shared network (slower)"
    route="- $(rhost "$r")"
  fi
  read -r jump dest <<< "$route"
  echo "route: jump $jump, destination $dest"
  on "$r" mkdir -p "$dst/main" "$dst/draft" || return 1
  on "$r" rm -f "$dst/.complete-$r"
  for try in 1 2 3 4 5; do
    rsync -a --whole-file --inplace -e "$(copy_ssh "$jump")" "$WORK/pack/main/rank$r.pack" "$WORK/pack/main/rank$r.manifest.json" "$(ssh_user "$r")$dest:$dst/main/" \
      && rsync -a --whole-file --inplace -e "$(copy_ssh "$jump")" "$WORK/pack/draft/rank$r.pack" "$WORK/pack/draft/rank$r.manifest.json" "$(ssh_user "$r")$dest:$dst/draft/" && return 0
    sleep $(( try * 3 ))
  done
  return 1
}
# _batch RANKS: convert these Sparks' quarters at the same time, then copy them at the same time.
_batch() {
  local r dst t0=$SECONDS n=0; local -a pids=() cp=()
  for r in "$@"; do
    ( convert_rank "$r" ) > "$LOG_DIR/convert-$r.log" 2>&1 & pids+=("$!"); n=$(( n + 1 ))
  done
  CONV_RANKS="$*" CONV_T0=$SECONDS
  watch_bars "${pids[@]}" -- "$n" _convert_bars
  n=0
  for r in "$@"; do
    wait "${pids[n]}" || { tail -n 8 "$LOG_DIR/convert-$r.log" | sed 's/^/    /' >&2; fail E_CONVERT "The conversion for Spark $r failed. Log: $LOG_DIR/convert-$r.log"; }
    grep -q MISMATCH "$LOG_DIR/convert-$r.log" && fail E_CONVERT "The converted weights for Spark $r are not equal to the download. Log: $LOG_DIR/convert-$r.log"
    n=$(( n + 1 ))
  done
  ok "Spark$( (( $# > 1 )) && echo s) $(IFS=,; echo "$*" | sed 's/,/, /g'): weights converted and checked in $(hms $(( SECONDS - t0 )))."
  pids=()
  for r in "$@"; do
    if (( r == 0 )); then printf '%s\n' "$(pack_stamp)" > "$WORK/pack/.complete-0"; continue; fi
    ( push_rank "$r" ) > "$LOG_DIR/copy-$r.log" 2>&1 & pids+=("$!"); cp+=("$r")
  done
  (( ${#cp[@]} )) || return 0
  t0=$SECONDS; COPY_RANKS="${cp[*]}" COPY_T0=$SECONDS
  watch_bars "${pids[@]}" -- "${#cp[@]}" _copy_bars
  n=0
  for r in "${cp[@]}"; do
    dst="$(work_of "$r")/pack"
    wait "${pids[n]}" || fail E_WEIGHTS "The copy to Spark $r failed. Run ./start.sh again to continue. Log: $LOG_DIR/copy-$r.log"
    [[ "$(on "$r" stat -c %s "$dst/main/rank$r.pack")" == "$RANK_PACK_BYTES" ]] || fail E_WEIGHTS "Spark $r: the copied file does not have the expected size."
    on "$r" sh -c 'printf "%s\n" "$2" > "$1"' sh "$dst/.complete-$r" "$(pack_stamp)"
    rm -f -- "$WORK/pack/main/rank$r".* "$WORK/pack/draft/rank$r".*
    n=$(( n + 1 ))
  done
  ok "Spark$( (( ${#cp[@]} > 1 )) && echo s) $(IFS=,; echo "${cp[*]}" | sed 's/,/, /g'): weights copied over the cables in $(hms $(( SECONDS - t0 )))."
}
ensure_rank_packs() {
  local r need free; local -a todo=() batch=()
  read_pack_states
  if all_packs_complete; then ok "Every Spark has its converted weights ($(( RANK_PACK_BYTES / 1000000000 )) GB each, checked)."; return 0; fi
  if dry; then note 'Would convert the weights for the four Sparks (about 15 minutes) and copy them over the cables (about 5 minutes).'; return 0; fi
  [[ "$CONVERT_JOBS" =~ ^[1-4]$ ]] || fail E_CONFIG "CONVERT_JOBS must be 1, 2, 3 or 4 (it is '$CONVERT_JOBS')."
  need=$(( RANK_PACK_BYTES / 1073741824 + DISK_SLACK_GIB ))
  for r in 1 2 3 0; do
    if (( PACK_DONE[r] )); then ok "Spark $r ($(rname "$r")): weights already there."; continue; fi
    (( P_DISK[r] >= need )) || fail E_DISK "Spark $r ($(rname "$r")) has ${P_DISK[r]} GiB of free disk; its weights need $need GiB."
    todo+=("$r")
  done
  # This Spark holds the files of one batch during the conversion. Its own file is in the last batch.
  free=$(df -Pk "$WORK" 2>/dev/null | awk 'END { printf "%d", $4 / 1048576 }')
  r=$(( ${#todo[@]} < CONVERT_JOBS ? ${#todo[@]} : CONVERT_JOBS ))
  (( ${free:-0} >= need * r )) || fail E_DISK "This Spark needs $(( need * r )) GiB of free disk for the conversion; ${free:-0} GiB is free. With less disk, set CONVERT_JOBS=1 (slower)."
  mkdir -p -- "$WORK/pack/main" "$WORK/pack/draft"
  log "Converting the weights: one quarter for each Spark, compared with the download, then copied over the cables."
  for r in "${todo[@]}"; do
    batch+=("$r")
    if (( ${#batch[@]} == CONVERT_JOBS )); then _batch "${batch[@]}"; batch=(); fi
  done
  (( ${#batch[@]} == 0 )) || _batch "${batch[@]}"
}

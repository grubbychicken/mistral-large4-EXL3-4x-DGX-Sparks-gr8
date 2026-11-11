#!/usr/bin/env bash
# One command on one Spark: find the four Sparks, check them and the ring of cables, build the engine, download and
# convert the weights, start the server, and test it. A second run skips every step that is already done.
# Usage: ./start.sh [restart]        ./stop.sh stops the server; ./status.sh shows its state
#   DRY_RUN=1 ./start.sh   checks the cluster and shows what a start would do; it changes nothing
# Settings: scripts/config.sh (defaults) < config.env < exported environment.
set -Eeuo pipefail
# A command that fails where no message handles it: say where, so the script never stops in silence. Only for the
# script itself: its background helpers (for example the probes of computers that are not Sparks) may fail quietly.
trap 'rc=$?; if [[ $BASHPID == "$$" ]]; then printf "\n  ✘ start.sh stopped at line %s (exit %s). Run ./status.sh, then ./start.sh again.\n" "$LINENO" "$rc" >&2; fi' ERR
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./scripts/lib.sh
source ./scripts/cluster.sh
source ./scripts/engine.sh
source ./scripts/pack.sh
source ./scripts/serve.sh

MODE=start
case "${1:-}" in
  '') ;;
  restart) MODE=restart ;;
  help|-h|--help) sed -n '2,6s/^# \{0,1\}//p' "$0"; exit 0 ;;
  *) fail E_USAGE 'Usage: ./start.sh [restart]' ;;
esac
TOTAL=6
for c in python3 ssh rsync curl ip ss sha256sum flock docker; do need_cmd "$c"; done
mkdir -p -- "$STATE_DIR" "$LOG_DIR"

banner
dry && printf '\n  %s%s%s\n' "$C_YELLOW" 'DRY_RUN=1: checks only. Nothing is changed.' "$C_RESET"
if ! dry; then
  exec 9>"$STATE_DIR/lock"; flock -n 9 || fail E_LOCK 'Another ./start.sh or ./stop.sh is running.'
fi

step 1 "$TOTAL" 'Cluster: find the four Sparks and check each one'
validate_config
probe_cluster
if engine_up && api_up && [[ "$MODE" == start ]]; then
  live "The server is already running: $(public_url)/v1 (model $SERVED_NAME)."
  note './start.sh restart starts it again with new settings. ./stop.sh stops it.'
  exit 0
fi
if [[ -n "$(printf '%s' "${P_OURS[@]}")" ]]; then
  if dry; then note 'A server from an earlier start is there; a real start stops it first.'
  else log 'Stopping the server from an earlier start ...'; stop_all; sleep 2; probe_cluster; fi
fi
check_health
if dry && [[ -n "$(printf '%s' "${P_OURS[@]}")" ]]; then note 'Ports are not checked: the running server holds them.'; else check_ports; fi

step 2 "$TOTAL" 'Network: check the ring of cables'
detect_ring
check_links

step 3 "$TOTAL" 'Engine: container image, source, build'
ensure_image
ensure_engine

step 4 "$TOTAL" 'Weights: download, convert for each Spark, copy'
read_pack_states
ensure_pack
ensure_model_files
ensure_draft
ensure_rank_packs

step 5 "$TOTAL" 'Start: free memory, load the model on four Sparks'
free_page_cache
launch_ranks
if dry; then
  step 6 "$TOTAL" 'API: start it and test it'
  start_front
  live 'DRY RUN complete. No problem found: the cluster is ready for ./start.sh.'
  exit 0
fi
trap 'printf "\n"; warn W_DETACHED "Stopped waiting. The engine continues in the background: ./status.sh shows it, ./stop.sh stops it."; exit 130' INT TERM
start_memguard
wait_engine

step 6 "$TOTAL" 'API: start it and test it'
start_front
smoke_test
if [[ "$SANITY" == 1 ]]; then
  if python3 tools/sanity.py 127.0.0.1 "$PORT" "$SERVED_NAME" > "$LOG_DIR/sanity.txt" 2>&1; then ok "Sanity checks passed ($(grep -c PASS "$LOG_DIR/sanity.txt") of $(grep -cE 'PASS|FAIL' "$LOG_DIR/sanity.txt"))."
  else warn W_SANITY "A sanity check failed ($LOG_DIR/sanity.txt). The server stays up."; fi
fi
trap - INT TERM

live 'Mistral Large 4 is LIVE on four DGX Sparks'
printf '  %sEndpoint%s  %s/v1\n' "$C_BOLD" "$C_RESET" "$(public_url)"
printf '  %sModel%s     %s   (%s tokens of context, %s requests at a time)\n' "$C_BOLD" "$C_RESET" "$SERVED_NAME" "$CONTEXT" "$SESSIONS"
printf '  %sLogs%s      %s    %sStatus%s ./status.sh    %sStop%s ./stop.sh\n' "$C_BOLD" "$C_RESET" "$LOG_DIR" "$C_BOLD" "$C_RESET" "$C_BOLD" "$C_RESET"
if [[ "$BIND_HOST" == 127.0.0.1 ]]; then
  note "From another computer: ssh -N -L $PORT:127.0.0.1:$PORT $(whoami)@$(rhost 0)   then use http://127.0.0.1:$PORT/v1"
else
  note "From another computer on your network: http://$(rhost 0):$PORT/v1   (the API has no password)"
fi

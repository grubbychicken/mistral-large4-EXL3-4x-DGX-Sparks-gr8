#!/usr/bin/env bash
# Show the server's state: the engine container and the free memory on each Spark, the memory guard, the API.
# Usage: ./status.sh
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./scripts/lib.sh
source ./scripts/cluster.sh
source ./scripts/serve.sh
(( ${#RANKS[@]} == 4 )) || fail E_CONFIG 'The Sparks are not known yet. Run ./start.sh first.'
_status_rank() {
  local s; s=$(on "$1" docker ps -a --filter "name=^${CONTAINER_PREFIX}$1\$" --format '{{.Status}}' 2>/dev/null)
  printf '%s|%s|%s\n' "$(on "$1" hostname)" "${s:-not running}" "$(on_fn "$1" mem_counting 'mem_counting | cut -d" " -f1')"
}
each_rank _status_rank || true
up=0
for r in 0 1 2 3; do
  IFS='|' read -r name state mem <<< "${EACH_OUT[r]:-?|no answer|?}"
  if [[ "$state" == Up* ]]; then up=$(( up + 1 )); ok "$(printf 'Spark %s  %-16s engine %-22s %s GiB free' "$r" "$name" "$state" "$mem")"
  else warn W_DOWN "$(printf 'Spark %s  %-16s engine %-22s %s GiB free' "$r" "$name" "$state" "$mem")"; fi
done
if pid_alive "$(read_state memguard.pid)"; then ok "Memory guard: on (floor $MEMGUARD_FLOOR_GIB GiB)."; else note 'Memory guard: off.'; fi
[[ -s "$LOG_DIR/memguard.log" ]] && note "Last guard action: $(tail -n 1 "$LOG_DIR/memguard.log")"
if api_up; then
  live "LIVE: $(public_url)/v1 (model $SERVED_NAME)"
elif (( up == 4 )) && ! [[ -f "$RUN/s2/ready.s2" ]]; then
  log "The model is loading: $(grep -o 'boot [0-9.]* s .*' "$RUN/rank0.err" 2>/dev/null | tail -n 1)"
elif (( up == 4 )); then
  warn W_API "The engine runs, but the API does not answer on port $PORT. Run ./start.sh restart."
else
  log 'The server is not running. Start it with ./start.sh'
fi

#!/usr/bin/env bash
# Stop the server: the memory guard, the API, and the engine container on each Spark. The weights and the build stay.
# Usage: ./stop.sh
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./scripts/lib.sh
source ./scripts/cluster.sh
source ./scripts/serve.sh
(( ${#RANKS[@]} == 4 )) || fail E_CONFIG 'The Sparks are not known yet (no SPARKS setting and no .state/cluster.env). Nothing was started from here.'
mkdir -p -- "$STATE_DIR"
exec 9>"$STATE_DIR/lock"; flock -w 30 9 || fail E_LOCK 'Another ./start.sh or ./stop.sh is running.'
stop_all 9>&-
each_rank _alive_rank || true
for r in 0 1 2 3; do [[ "${EACH_OUT[r]:-}" == up ]] && fail E_STOP "The engine container on Spark $r ($(rhost "$r")) is still there. Run on that Spark: docker rm -f $CONTAINER_PREFIX$r"; done
ok 'Stopped: the API and the engine on all four Sparks. Memory is free again.'

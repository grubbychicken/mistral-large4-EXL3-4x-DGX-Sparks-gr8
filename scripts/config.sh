#!/usr/bin/env bash
# Every setting and its default. Order of precedence: exported environment > config.env > the defaults below.
# config.env (optional, in the repository root) holds literal KEY=value lines; it is read, never executed.
# Sourced by scripts/lib.sh. Runs no ssh, Docker, GPU or network command.
[[ "${_ML4_CONFIG_LOADED:-0}" == 1 ]] && return 0
_ML4_CONFIG_LOADED=1
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)

if [[ -f "$REPO_ROOT/config.env" ]]; then
  while IFS= read -r _cfg_line || [[ -n "$_cfg_line" ]]; do
    [[ "$_cfg_line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    _cfg_key=${BASH_REMATCH[2]}; _cfg_value=${BASH_REMATCH[3]}
    if [[ "$_cfg_value" =~ ^\"([^\"]*)\"[[:space:]]*(#.*)?$ || "$_cfg_value" =~ ^\'([^\']*)\'[[:space:]]*(#.*)?$ ]]; then
      _cfg_value=${BASH_REMATCH[1]}
    else
      _cfg_value=${_cfg_value%%#*}; _cfg_value=${_cfg_value%"${_cfg_value##*[![:space:]]}"}
    fi
    [[ -n "${!_cfg_key+set}" ]] || export "$_cfg_key=$_cfg_value"
  done < "$REPO_ROOT/config.env"
fi
unset _cfg_key _cfg_line _cfg_value

# --- the four Sparks ---------------------------------------------------------------------------------------------------
# Empty: ./start.sh finds the other three Sparks through the cables and puts them in the ring's order (it saves the
# result in .state/cluster.env). To set them yourself: four ssh targets, this Spark first, then around the ring.
SPARKS="${SPARKS:-}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o StrictHostKeyChecking=accept-new -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-ml4-%C -o ControlPersist=300}"
# ssh for the large copies: one connection per stream, a fast cipher, no compression.
COPY_SSH_OPTS="${COPY_SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=10 -o ControlMaster=no -o ControlPath=none -o Compression=no -o StrictHostKeyChecking=accept-new -c aes128-gcm@openssh.com}"

# --- where things live on every Spark --------------------------------------------------------------------------------
ML4_DIR="${ML4_DIR:-ml4-gr8}"                # under each Spark's $HOME: src/ (engine source), bin/, pack/, run/

# --- the weights: the EXL3 3.0 bpw quant of Mistral Large 4 ----------------------------------------------------------
PACK_ID="${PACK_ID:-piotrkosecki/Mistral-Large-4-1T-A52B-EXL3-3.0bpw}"
PACK_REVISION="${PACK_REVISION:-b01606641f9db4dbdbafc5d14304ad4b87397d68}"
MODEL_DIR="${MODEL_DIR:-$HOME/.cache/huggingface/local/Mistral-Large-4-1T-A52B-EXL3-3.0bpw}"   # the download, on this Spark only
EXPECTED_SHARDS="${EXPECTED_SHARDS:-76}"
EXPECTED_WEIGHT_BYTES="${EXPECTED_WEIGHT_BYTES:-403284753715}"
RANK_PACK_BYTES="${RANK_PACK_BYTES:-108662579200}"   # one Spark's quarter of the model after conversion
# Mistral's draft head (the last shard of the original checkpoint, 4.2 GB). It proposes tokens; the model checks them.
DSPARK_REPO="${DSPARK_REPO:-mistralai/Mistral-Large-4-1T-A52B}"
DSPARK_REVISION="${DSPARK_REVISION:-af26e7869af601441d5cdf75a1272469ddc3d059}"
DSPARK_FILE="${DSPARK_FILE:-consolidated-00054-of-00054.safetensors}"
DSPARK_BYTES="${DSPARK_BYTES:-4199942840}"
DSPARK_SHA256="${DSPARK_SHA256:-4e369d0365bb15e42d5ccd4701e326f10514c3e1cd63812161f743a8911cde79}"

# --- the engine: gr8 at a fixed commit, plus patches/gr8 -------------------------------------------------------------
GR8_REPO="${GR8_REPO:-https://github.com/kindlingai/gr8}"
GR8_COMMIT="${GR8_COMMIT:-069ec213fc72b5ed18ac4f6bf1c1a296dae1fb16}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c}"
RUST_IMAGE="${RUST_IMAGE:-rust@sha256:e6f2cff35d0f309acf8433b4f05878ed77092602c5065cc81801177da0bb3a9c}"
ENGINE_FLAGS="${ENGINE_FLAGS:--DG6_NR=4 -DDEC_S2 -DG6_CP=1 -DG6_MODEL_ML4=1 -DML4_BENCH_NEXP=129 -DPF_FC2GL=6 -DPF_ISGRT=1 -DDEC_SLOT=2 -DDEC_AVP=1 -DSEL32 -DDEC_IDX31=1}"
BUILD_JOBS="${BUILD_JOBS:-8}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.22.2}"   # the converter reads the model's position tables from it

# --- serving -----------------------------------------------------------------------------------------------------------
CONTEXT="${CONTEXT:-65536}"                  # tokens a request can hold, prompt and reply together (4096-65536)
SESSIONS="${SESSIONS:-4}"                    # requests that can run at the same time (more wait in a queue)
PORT="${PORT:-8000}"                         # the API's port on this Spark
BIND_HOST="${BIND_HOST:-127.0.0.1}"          # 127.0.0.1: this Spark only. 0.0.0.0: your network (the API has no password)
SERVED_NAME="${SERVED_NAME:-mistral-large-4}"
REASONING_EFFORT="${REASONING_EFFORT:-low}"  # the default when a request does not set reasoning_effort
PREFILL_ROWS="${PREFILL_ROWS:-2048}"         # prompt tokens the engine processes in one pass
PREFILL_LANES="${PREFILL_LANES:-2}"          # passes that run at the same time
ENGINE_PORT="${ENGINE_PORT:-18100}"          # the ranks' control connection
ENGINE_ENV="${ENGINE_ENV:-}"                 # more KEY=VALUE words for every rank's container
ENGINE_ARGS="${ENGINE_ARGS:-}"               # more engine arguments
READY_TIMEOUT_S="${READY_TIMEOUT_S:-600}"
API_TIMEOUT_S="${API_TIMEOUT_S:-180}"            # the API front end reads the tokenizer first: 40 s on the first start, 1 s later
SANITY="${SANITY:-1}"                        # 1: run tools/sanity.py (four short checks) after the start

# --- memory (one pool of 128 GB on each Spark, for the system and the GPU together) ------------------------------------
MIN_AVAIL_GIB="${MIN_AVAIL_GIB:-117}"        # each Spark needs this much memory available before a start
MEMGUARD="${MEMGUARD:-1}"                    # 1: a guard stops the server before a Spark runs out of memory
MEMGUARD_FLOOR_GIB="${MEMGUARD_FLOOR_GIB:-3}"
MEMGUARD_HOLD_S="${MEMGUARD_HOLD_S:-10}"
MEMGUARD_INTERVAL_S="${MEMGUARD_INTERVAL_S:-5}"
DISK_SLACK_GIB="${DISK_SLACK_GIB:-20}"
CONVERT_JOBS="${CONVERT_JOBS:-4}"                # the quarters converted at the same time on this Spark (1 to 4); 1 needs the least disk


# --- state and logs ----------------------------------------------------------------------------------------------------
STATE_DIR="${STATE_DIR:-$REPO_ROOT/.state}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs}"
DRY_RUN="${DRY_RUN:-0}"
CONTAINER_PREFIX=ml4-gr8-rank

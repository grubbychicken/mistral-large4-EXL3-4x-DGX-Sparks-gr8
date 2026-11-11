#!/usr/bin/env bash
# Helpers every script shares: settings, colours and messages, the header, progress bars, ssh to the four Sparks.
# Sourced; defines functions only.
[[ "${_ML4_LIB_LOADED:-0}" == 1 ]] && return 0
_ML4_LIB_LOADED=1
# shellcheck source=scripts/config.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/config.sh"
# shellcheck source=scripts/memavail.sh
source "$REPO_ROOT/scripts/memavail.sh"
# shellcheck source=scripts/logo.sh
source "$REPO_ROOT/scripts/logo.sh"

# --- colours: on for a terminal; NO_COLOR (even empty) or TERM=dumb turns them off, FORCE_COLOR=1 forces them -------
_colour_ok() { [[ -n "${FORCE_COLOR:-}" ]] || { [[ -t "$1" && -z "${NO_COLOR+x}" && "${TERM:-}" != dumb ]]; }; }
if _colour_ok 1; then
  C_RESET=$'\033[0m' C_BOLD=$'\033[1m' C_DIM=$'\033[2m'
  C_RED=$'\033[31m' C_GREEN=$'\033[32m' C_YELLOW=$'\033[33m' C_BLUE=$'\033[34m' C_MAGENTA=$'\033[35m' C_CYAN=$'\033[36m'
  TTY=1
else
  C_RESET='' C_BOLD='' C_DIM='' C_RED='' C_GREEN='' C_YELLOW='' C_BLUE='' C_MAGENTA='' C_CYAN=''
  TTY=0
fi
if _colour_ok 2; then E_RED=$'\033[1;31m' E_YELLOW=$'\033[33m' E_RESET=$'\033[0m'; else E_RED='' E_YELLOW='' E_RESET=''; fi

# --- messages ----------------------------------------------------------------------------------------------------------
#   step N TOTAL MSG   a phase header (cyan)      ok MSG     a check passed (green ✔)
#   log MSG            plain progress             warn C M   a warning (yellow !)
#   note MSG           detail (dim)               fail C M   an error (red ✘), then exit 1
STEP_STARTED=$SECONDS
step() {
  local took=''
  (( $1 > 1 && SECONDS - STEP_STARTED >= 2 )) && took=" ${C_DIM}(previous step: $(hms $(( SECONDS - STEP_STARTED ))))${C_RESET}"
  STEP_STARTED=$SECONDS
  printf '\n%s━━ [%s/%s] %s%s%s\n' "$C_BOLD$C_CYAN" "$1" "$2" "$3" "$C_RESET" "$took"
}
log()  { printf '  %s\n' "$*"; }
note() { printf '  %s%s%s\n' "$C_DIM" "$*" "$C_RESET"; }
ok()   { printf '  %s✔%s %s\n' "$C_GREEN" "$C_RESET" "$*"; }
warn() { printf '  %s! [%s]%s %s\n' "$E_YELLOW" "$1" "$E_RESET" "$2" >&2; }
fail() { [[ -t 1 ]] && printf '\033[?7h'; printf '\n  %s✘ [%s]%s %s\n' "$E_RED" "$1" "$E_RESET" "$2" >&2; exit 1; }
live() { printf '\n%s%s%s\n' "$C_BOLD$C_GREEN" "$*" "$C_RESET"; }
hms()  { if (( $1 >= 3600 )); then printf '%dh%02dm' $(( $1 / 3600 )) $(( $1 % 3600 / 60 )); elif (( $1 >= 60 )); then printf '%dm%02ds' $(( $1 / 60 )) $(( $1 % 60 )); else printf '%ds' "$1"; fi; }

# shq: words as a shell reads them back. plan: a command printed, not run (DRY_RUN). run: plan or do.
shq() { local a; for a in "$@"; do if [[ "$a" =~ ^[A-Za-z0-9_./:=,@%+-]+$ ]]; then printf '%s ' "$a"; else printf "'%s' " "${a//\'/\'\\\'\'}"; fi; done; }
plan() { printf '    %s$ %s%s\n' "$C_DIM" "$(shq "$@")" "$C_RESET"; }
dry() { [[ "$DRY_RUN" == 1 ]]; }

is_number() { [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]; }
at_least() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a >= b) }'; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || fail E_DEPENDENCY "This Spark does not have the command: $1."; }

# --- progress ----------------------------------------------------------------------------------------------------------
# bar_line LABEL GOT TOTAL START SECONDS: one progress bar line (bytes): bar, percent, GiB, speed, time left.
# A line that is redrawn in place must stay on one terminal row. If the terminal is narrow, a long line would continue
# on the next row and each redraw would leave a copy behind. nowrap makes the terminal cut such a line at its right
# edge; wrap puts the usual behaviour back (also when the script ends).
nowrap() { (( TTY )) && printf '\033[?7l'; return 0; }
wrap() { (( TTY )) && printf '\033[?7h'; return 0; }
trap wrap EXIT
bar_line() {
  awk -v label="$1" -v got="$2" -v tot="$3" -v start="$4" -v secs="$5" -v cyan="$C_CYAN" -v green="$C_GREEN" \
      -v dim="$C_DIM" -v reset="$C_RESET" 'BEGIN {
    if (got !~ /^[0-9]+$/) got = 0
    p = tot > 0 ? got * 100 / tot : 0; if (p > 100) p = 100
    f = int(p * 30 / 100); s = ""; t = ""
    for (i = 1; i <= f; i++) s = s "█"
    for (; i <= 30; i++) t = t "░"
    rate = secs > 0 ? (got - start) / secs : 0
    left = rate > 0 ? (tot - got) / rate : -1
    if (got >= tot) eta = green "done" reset
    else if (left < 0) eta = "starting"
    else if (left >= 3600) eta = sprintf("%dh%02dm left", left / 3600, (left % 3600) / 60)
    else eta = sprintf("%dm%02ds left", left / 60, left % 60)
    printf "    %-22s %s%s%s%s%s%s %3d%%  %6.1f / %6.1f GiB  %5.0f MB/s  %s\n", label, (got >= tot ? green : cyan), s, reset, dim, t, reset, p, got / 1073741824, tot / 1073741824, rate / 1e6, eta
  }'
}
# watch_bars PIDS -- N FN: until every pid in PIDS ends, call FN (it prints N bar lines) and redraw the lines in place
# on a terminal (every 2 s), or print them once a minute when the output is not a terminal.
watch_bars() {
  local -a pids=()
  while [[ "$1" != -- ]]; do pids+=("$1"); shift; done
  shift
  local n=$1 fn=$2 drawn=0 running p last=-60
  nowrap
  while :; do
    running=0
    for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && running=1; done
    if (( TTY )) || (( SECONDS - last >= 60 )) || (( ! running )); then
      (( TTY && drawn )) && printf '\033[%dA' "$n"
      if (( TTY )); then "$fn" | sed $'s/^/\033[2K/'; else "$fn"; fi
      drawn=1 last=$SECONDS
    fi
    (( running )) || break
    sleep 2
  done
  wrap
}
# spin PID MSG [LOGFILE]: a spinner with the elapsed time while PID runs; with LOGFILE, also its last line.
spin() {
  local pid=$1 msg=$2 logf=${3:-} t0=$SECONDS i=0 tail_line='' frames='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏' last=-60 n0=0
  [[ -n "$logf" ]] && n0=$(wc -l < "$logf" 2>/dev/null || echo 0)     # show only log lines that this step writes
  nowrap
  while kill -0 "$pid" 2>/dev/null; do
    if (( TTY )); then
      [[ -n "$logf" ]] && tail_line=$(tail -n "+$(( n0 + 1 ))" "$logf" 2>/dev/null | tail -n 1 | tr -d '\r' | cut -c1-70)
      printf '\r\033[2K  %s%s%s %s %s(%s)%s %s%s%s' "$C_CYAN" "${frames:i++%10:1}" "$C_RESET" "$msg" "$C_DIM" "$(hms $(( SECONDS - t0 )))" "$C_RESET" "$C_DIM" "$tail_line" "$C_RESET"
      sleep 0.2
    else
      (( SECONDS - last >= 60 )) && { printf '  ... %s (%s)\n' "$msg" "$(hms $(( SECONDS - t0 )))"; last=$SECONDS; }
      sleep 5
    fi
  done
  (( TTY )) && printf '\r\033[2K'
  wrap
  wait "$pid"
}

# --- the four Sparks ---------------------------------------------------------------------------------------------------
# RANKS: the ssh targets, rank 0 (this Spark) first. Filled from SPARKS, or by find_sparks (scripts/cluster.sh).
declare -a RANKS=() NODE_NAME=()
set_ranks() { read -r -a RANKS <<< "$SPARKS"; }
[[ -n "$SPARKS" ]] || { [[ -f "$STATE_DIR/cluster.env" ]] && SPARKS=$(sed -n 's/^SPARKS=//p' "$STATE_DIR/cluster.env"); }
set_ranks
rhost() { local t=${RANKS[$1]}; printf '%s' "${t#*@}"; }
rname() { printf '%s' "${NODE_NAME[$1]:-$(rhost "$1")}"; }

# on R WORDS...: a command on rank R, words quoted for the remote shell. Rank 0 runs here.
on() {
  local r=$1; shift
  if (( r == 0 )); then "$@"; return; fi
  # shellcheck disable=SC2086
  ssh $SSH_OPTS -- "${RANKS[$r]}" "$(shq "$@")"
}
# on_sh R SCRIPT: a bash script on rank R.
on_sh() {
  local r=$1; shift
  if (( r == 0 )); then bash -c "$1"; return; fi
  # shellcheck disable=SC2086
  ssh $SSH_OPTS -- "${RANKS[$r]}" bash -s <<< "$1"
}
# on_fn R 'FN...' CALL: CALL on rank R with the named functions defined there first.
on_fn() { on_sh "$1" "$(declare -f $2); $3"; }

# each_rank FN [ARGS...]: FN R ARGS on the ranks in EACH_RANKS (default: all four) at the same time. Output goes to
# EACH_OUT[r], the exit status to EACH_RC[r]; returns 1 when a rank failed.
declare -a EACH_OUT=() EACH_RC=()
each_rank() {
  local fn=$1 r tmp bad=0; shift
  local -a ranks=(${EACH_RANKS:-0 1 2 3}) pids=()
  tmp=$(mktemp -d)
  for r in "${ranks[@]}"; do ( "$fn" "$r" "$@" > "$tmp/$r.out" 2>&1; echo $? > "$tmp/$r.rc" ) & pids+=("$!"); done
  wait "${pids[@]}"
  EACH_OUT=() EACH_RC=()
  for r in "${ranks[@]}"; do
    EACH_OUT[r]=$(cat "$tmp/$r.out"); EACH_RC[r]=$(cat "$tmp/$r.rc")
    [[ "${EACH_RC[r]}" == 0 ]] || bad=1
  done
  rm -rf -- "$tmp"
  return "$bad"
}
show_each() { local r; for r in "${!EACH_OUT[@]}"; do [[ -z "${EACH_OUT[r]}" ]] || printf '%s\n' "${EACH_OUT[r]}" | sed 's/^/    /'; done; }

# --- state -------------------------------------------------------------------------------------------------------------
write_state() { mkdir -p -- "$STATE_DIR"; printf '%s\n' "$2" > "$STATE_DIR/.$1.$$" && mv -- "$STATE_DIR/.$1.$$" "$STATE_DIR/$1"; }
read_state() { cat "$STATE_DIR/$1" 2>/dev/null || true; }
pid_alive() { [[ -n "$1" ]] && kill -0 "$1" 2>/dev/null; }
home_of() { printf '%s' "${P_HOME[$1]:-$HOME}"; }          # rank R's $HOME (after probe_cluster)
work_of() { printf '%s/%s' "$(home_of "$1")" "$ML4_DIR"; }  # rank R's working directory
WORK="$HOME/$ML4_DIR"                                       # this Spark's working directory
public_url() { if [[ "$BIND_HOST" == 127.0.0.1 ]]; then printf 'http://127.0.0.1:%s' "$PORT"; else printf 'http://%s:%s' "$(rhost 0)" "$PORT"; fi; }

# The header: the logo, what this recipe is, and who made it.
# The gr8 mark (a chip with a campfire in it, after github.com/kindlingai), 28 x 28 points: '.' nothing, o orange,
# y yellow, k dark. A character shows two points, one over the other, so 14 lines draw it.
GR8_PIX=(
  ............................
  ..........oo....oo..........
  ..........oo....oo..........
  ..........oo....oo..........
  ..........oo....oo..........
  ........oooooooooooo........
  .......oooooooooooooo.......
  ......oooooooooooooooo......
  .....oooooooooooooooooo.....
  .....oooooooookoooooooo.....
  .oooooooooooookoooooooooooo.
  .oooooooooookokkooooooooooo.
  .....oooooookkkkkoooooo.....
  .....ooooookkyykkoooooo.....
  .....ooooookkyykkoooooo.....
  .....ooooookkyykkoooooo.....
  .oooooooooookyykooooooooooo.
  .ooooooookkkkkkkkkkoooooooo.
  .....oooookkkkkkkkooooo.....
  .....ooookkkkookkkkoooo.....
  ......oooooooooooooooo......
  .......oooooooooooooo.......
  ........oooooooooooo........
  ..........oo....oo..........
  ..........oo....oo..........
  ..........oo....oo..........
  ..........oo....oo..........
  ............................
)
declare -A GR8_RGB=([o]='255;107;26' [y]='255;196;64' [k]='20;17;15')
# gr8_line N: line N (0-13) of the mark. Without colours: only the orange and yellow points, as half blocks.
gr8_line() {
  local top=${GR8_PIX[2 * $1]} bot=${GR8_PIX[2 * $1 + 1]} x t b out=''
  for (( x = 0; x < ${#top}; x++ )); do
    t=${top:x:1}; b=${bot:x:1}
    if (( TTY )); then
      if [[ $t == . && $b == . ]]; then out+=' '
      elif [[ $b == . ]]; then out+=$'\033[38;2;'"${GR8_RGB[$t]}m▀"$'\033[0m'
      elif [[ $t == . ]]; then out+=$'\033[38;2;'"${GR8_RGB[$b]}m▄"$'\033[0m'
      else out+=$'\033[38;2;'"${GR8_RGB[$t]}"$'m\033[48;2;'"${GR8_RGB[$b]}m▀"$'\033[0m'; fi
    else
      [[ $t == [oy] ]] && t=1 || t=0; [[ $b == [oy] ]] && b=1 || b=0
      case $t$b in 11) out+='█' ;; 10) out+='▀' ;; 01) out+='▄' ;; *) out+=' ' ;; esac
    fi
  done
  printf '%s' "$out"
}
# banner: the causal-mask logo (scripts/logo.sh, full size with its gaps) and the gr8 mark on its right.
banner() {
  local -a L=(); local i k r w
  mapfile -t L < <(LOGO_COLOR=$TTY print_logo large)
  for (( i = 0; i < ${#L[@]}; i++ )); do
    k=$(( i - 1 )); r=$(( k / 3 )); w=0
    (( k >= 0 && k % 3 != 2 && r <= 4 )) && w=$(( 6 * r + 6 ))      # the visible width of this logo line
    if (( i >= 1 && i <= 14 )); then printf '%s%*s%s\n' "${L[i]}" $(( 38 - w )) '' "$(gr8_line $(( i - 1 )))"; elif (( i == 0 )); then echo; fi
  done
  # the credits, each under its logo
  printf '\n  %s%-36s%s%s\n' "$C_CYAN" '@grubbychicken' "$C_RESET" 'gr8 engine'
  printf '  %s%-36s%s%s\n' "$C_DIM" 'x.com/grubbychicken' 'github.com/kindlingai/gr8' "$C_RESET"
  printf '  %s%s%s\n\n' "$C_DIM" 'github.com/grubbychicken' "$C_RESET"
  printf '  %s%s%s\n' "$C_BOLD$C_MAGENTA" 'Mistral Large 4 · EXL3 3.0 bpw · gr8 engine · 4× DGX Spark' "$C_RESET"
  printf '  %s%s%s\n' "$C_DIM" "$CONTEXT tokens of context · $SESSIONS requests at a time · API on $BIND_HOST:$PORT" "$C_RESET"
}

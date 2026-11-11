#!/usr/bin/env bash
# The four Sparks: find them through the cables, check their health, check the ring of cables.
# Sourced after scripts/lib.sh.
#
# The ring inventory (node_inventory, pair_links) follows the 4x DGX Spark recipes of the MiaAI-Lab (scripts/tp4.sh,
# scripts/nodes.sh; Copyright 2026 Mia's AI Lab, Apache License 2.0; changed for this recipe, see NOTICE.md): every
# cable goes port to port and has its own subnet; no cable reaches the Spark across the ring.
[[ "${_ML4_CLUSTER_LOADED:-0}" == 1 ]] && return 0
_ML4_CLUSTER_LOADED=1

# --- find the Sparks -------------------------------------------------------------------------------------------------
# link_probe: this node's name, its address on the shared network, and for each cable port that is up the link-local
# addresses that answer on that cable (the Spark at the other end). Read-only. Sent to the other Sparks with declare -f.
link_probe() {
  local n dev u own peers
  echo "name $(hostname)"
  echo "lan $(ip -o -4 route show default 2>/dev/null | sed -n 's/.* src \([^ ]*\).*/\1/p' | head -1)"
  own=$(ip -o -6 addr show scope link 2>/dev/null | awk '{print $4}' | cut -d/ -f1)
  for n in $(for n in /sys/class/net/*; do
               [[ -d $n/device/infiniband ]] || continue
               u=$(ls "$n/device/infiniband_verbs" 2>/dev/null | head -1); u=${u#uverbs}
               [[ "$u" =~ ^[0-9]+$ ]] || u=9999; echo "$u $n"
             done | sort -n | cut -d' ' -f2); do
    dev=${n##*/}
    [[ "$(cat "$n/operstate" 2>/dev/null)" == up ]] || continue
    peers=$(ping -6 -c 2 -W 1 -I "$dev" ff02::1 2>/dev/null | grep -o 'from fe80[^ ]*' | cut -d' ' -f2 | cut -d% -f1 | sed 's/:$//' | sort -u | grep -vxF -e "$own" | tr '\n' ' ')
    echo "dev $dev $peers"
  done
  echo "own $(tr '\n' ' ' <<< "$own")"
}
# find_sparks: sets SPARKS (four addresses on the shared network, in the ring's order) with no input from the user.
# Each cable shows the link-local addresses of the Spark at its other end. The Sparks are then found on the shared
# network: every machine there that answers ssh runs link_probe, and the one that owns those addresses is the neighbour.
_lan_probe() { ssh -o BatchMode=yes -o ConnectTimeout=4 -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR -- "$1" bash -s <<< "$(declare -f link_probe); link_probe" 2>/dev/null; }
find_sparks() {
  local me lan cidr tmp h order
  mkdir -p -- "$STATE_DIR"
  me=$(link_probe)
  lan=$(sed -n 's/^lan //p' <<< "$me")
  [[ -n "$lan" ]] || fail E_NETWORK 'This Spark has no default route. Connect it to your network (Wi-Fi or Ethernet) first.'
  grep -q '^dev [^ ]* fe80' <<< "$me" || fail E_FABRIC 'No Spark answers on a cable from this one. Connect the QSFP cables (each Spark to two others) and check that the ports are up: ip -br link.'
  cidr=$(ip -o -4 addr show | awk -v a="$lan" '{ split($4, x, "/"); if (x[1] == a) print $4 }' | head -1)
  tmp=$(mktemp -d)
  printf '%s\n' "$me" > "$tmp/probe.$lan"
  # Machines on the shared network that answer a ping, then link_probe on each one that accepts our ssh key.
  for h in $(python3 -c '
import ipaddress, sys
n = ipaddress.ip_interface(sys.argv[1]).network
if n.num_addresses <= 1024:
    print(" ".join(str(h) for h in n.hosts() if str(h) != sys.argv[2]))' "$cidr" "$lan"); do
    ( ping -c 1 -W 1 "$h" >/dev/null 2>&1 && _lan_probe "$h" > "$tmp/probe.$h" ) &
  done
  wait
  order=$(python3 - "$tmp" "$lan" <<'PY'
import glob, os, sys
tmp, me = sys.argv[1], sys.argv[2]
nodes = {}
for f in glob.glob(os.path.join(tmp, "probe.*")):
    n = {"devs": [], "own": set(), "name": "", "lan": f.rsplit("probe.", 1)[1]}
    for line in open(f):
        w = line.split()
        if not w: continue
        if w[0] == "name" and len(w) > 1: n["name"] = w[1]
        elif w[0] == "dev": n["devs"].append((w[1], set(w[2:])))
        elif w[0] == "own": n["own"] = set(w[1:])
    if n["name"] and n["own"]: nodes[n["lan"]] = n
def owner(ll, skip):
    return next((k for k, n in nodes.items() if k not in skip and ll in n["own"]), None)
def neighbours(k):
    out = []
    for dev, peers in nodes[k]["devs"]:
        for ll in sorted(peers):
            o = owner(ll, {k})
            if o and o not in out: out.append(o)
    return out
if me not in nodes: sys.exit("E this Spark has no cable ports that are up")
nb = neighbours(me)
if len(nb) < 2: sys.exit("E %d" % len(nb))
a, b = nb[0], nb[1]                      # a: the Spark on this one's first cable port
far = [x for x in neighbours(a) if x not in (me, a, b)]
if not far or far[0] not in neighbours(b): sys.exit("E far")
print(" ".join([me, a, far[0], b]))
print(" ".join(nodes[k]["name"] for k in [me, a, far[0], b]))
PY
  ) || { rm -rf -- "$tmp"; fail E_DISCOVERY "Could not find four Sparks in a ring. Check that (1) all four Sparks are on your network, (2) this Spark can log in to the other three with an ssh key and no password (ssh-copy-id $(whoami)@<address>), (3) each Spark has cables to two others. Or set SPARKS in config.env (see config.env.example)."; }
  rm -rf -- "$tmp"
  SPARKS=$(sed -n 1p <<< "$order"); set_ranks
  printf 'SPARKS=%s\n' "$SPARKS" > "$STATE_DIR/cluster.env"
  ok "Found four Sparks through the cables: $(sed -n 2p <<< "$order" | sed 's/ / → /g')"
  note "Saved to .state/cluster.env: SPARKS=\"$SPARKS\" (delete the file to search again)."
}

# --- configuration -----------------------------------------------------------------------------------------------------
validate_config() {
  local key r
  if (( ${#RANKS[@]} == 0 )); then log 'Looking for the other Sparks on the cables ...'; find_sparks; fi
  (( ${#RANKS[@]} == 4 )) || fail E_CONFIG "SPARKS must list four Sparks, this one first; it lists ${#RANKS[@]}."
  for r in 0 1 2 3; do
    [[ "${RANKS[r]}" =~ ^[A-Za-z0-9_.@:-]+$ ]] || fail E_CONFIG "SPARKS entry '${RANKS[r]}' is not an ssh target."
  done
  [[ "$(printf '%s\n' "${RANKS[@]}" | sort -u | wc -l)" == 4 ]] || fail E_CONFIG 'SPARKS lists a Spark twice.'
  [[ "$PACK_REVISION" =~ ^[0-9a-f]{40}$ && "$GR8_COMMIT" =~ ^[0-9a-f]{40}$ ]] || fail E_CONFIG 'PACK_REVISION and GR8_COMMIT must be full 40-character commit ids.'
  is_number "$CONTEXT" && (( CONTEXT >= 4096 && CONTEXT <= 65536 )) || fail E_CONFIG 'CONTEXT must be a number of tokens from 4096 to 65536.'
  [[ "$MODEL_DIR" == /* ]] || fail E_CONFIG 'MODEL_DIR must be an absolute path.'
  for key in PORT ENGINE_PORT SESSIONS READY_TIMEOUT_S MEMGUARD_HOLD_S MEMGUARD_INTERVAL_S EXPECTED_SHARDS PREFILL_ROWS PREFILL_LANES BUILD_JOBS; do
    [[ "${!key}" =~ ^[1-9][0-9]*$ ]] || fail E_CONFIG "$key must be a positive whole number, not '${!key}'."
  done
  (( SESSIONS <= 8 )) || fail E_CONFIG 'SESSIONS can be 8 at most.'
  for key in MIN_AVAIL_GIB MEMGUARD_FLOOR_GIB DISK_SLACK_GIB; do is_number "${!key}" || fail E_CONFIG "$key must be a number of GiB."; done
  for key in DRY_RUN MEMGUARD SANITY; do [[ "${!key}" =~ ^[01]$ ]] || fail E_CONFIG "$key must be 0 or 1."; done
  is_this_host "$(rhost 0)" || fail E_CONFIG "The first Spark in SPARKS ($(rhost 0)) is not this machine ($(hostname)). Run ./start.sh on the first Spark, or put this one first."
}
is_this_host() {
  [[ "$1" == "$(hostname)" || "$1" == "$(hostname -s)" || "$1" == localhost ]] && return 0
  ip -o -4 addr show 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | grep -qxF -- "$1"
}

# --- health: one probe on each Spark, all four at the same time --------------------------------------------------------
node_probe() {
  local rt
  echo "name=$(hostname)"
  echo "arch=$(uname -m)"
  echo "home=$HOME"
  if docker info >/dev/null 2>&1; then echo docker=ok; else echo docker=no; fi
  rt=$(docker info --format '{{json .Runtimes}}' 2>/dev/null)
  if [[ "$rt" == *nvidia* || -x /usr/bin/nvidia-container-runtime-hook || -x /usr/bin/nvidia-ctk ]]; then echo nvrt=ok; else echo nvrt=no; fi
  echo "gpu=$(nvidia-smi -L 2>/dev/null | head -1 | sed 's/ (UUID.*//')"
  echo "apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | tr '\n' ';')"
  echo "mem=$(mem_counting 2>/dev/null | cut -d' ' -f1)"
  echo "total=$(awk '$1 == "MemTotal:" { printf "%.1f", $2 / 1048576 }' /proc/meminfo)"
  echo "disk=$(df -Pk "$HOME" | awk 'END { printf "%d", $4 / 1048576 }')"
  if docker image inspect "$1" >/dev/null 2>&1; then echo image=ok; else echo image=no; fi
  echo "ours=$(docker ps -a --filter "name=$2" --format '{{.Names}}:{{.State}}' 2>/dev/null | tr '\n' ' ')"
  echo "ib=$(ls /dev/infiniband 2>/dev/null | grep -c uverbs)"
}
_probe_rank() { on_fn "$1" 'node_probe mem_counting' "node_probe $(shq "$BASE_IMAGE" "$CONTAINER_PREFIX")"; }
declare -a P_NAME=() P_HOME=() P_MEM=() P_TOTAL=() P_DISK=() P_IMAGE=() P_OURS=() P_GPU=() P_APPS=() PROBE_RAW=()
probe_cluster() {
  local r out
  each_rank _probe_rank || true
  for r in 0 1 2 3; do
    out=${EACH_OUT[r]}
    [[ "${EACH_RC[r]}" == 0 && "$out" == *name=* ]] \
      || fail E_SSH "Cannot reach Spark $r (${RANKS[r]}) with ssh: ${out:-no answer}. This Spark must log in to it with a key and without a password: ssh-copy-id ${RANKS[r]}"
    P_NAME[r]=$(sed -n 's/^name=//p' <<< "$out"); NODE_NAME[r]=${P_NAME[r]}
    P_HOME[r]=$(sed -n 's/^home=//p' <<< "$out")
    P_MEM[r]=$(sed -n 's/^mem=//p' <<< "$out")
    P_TOTAL[r]=$(sed -n 's/^total=//p' <<< "$out")
    P_DISK[r]=$(sed -n 's/^disk=//p' <<< "$out")
    P_IMAGE[r]=$(sed -n 's/^image=//p' <<< "$out")
    P_OURS[r]=$(sed -n 's/^ours=//p' <<< "$out" | xargs)
    P_GPU[r]=$(sed -n 's/^gpu=//p' <<< "$out")
    P_APPS[r]=$(sed -n 's/^apps=//p' <<< "$out")
    PROBE_RAW[r]=$out
  done
}
# check_health: the platform checks on every Spark (after probe_cluster).
check_health() {
  local r out who bad=0
  for r in 0 1 2 3; do
    out=${PROBE_RAW[r]}; who="Spark $r ($(rname "$r"))"
    grep -qx 'arch=aarch64' <<< "$out" || fail E_PLATFORM "$who is not an aarch64 machine (a DGX Spark)."
    grep -qx 'docker=ok' <<< "$out" || fail E_DOCKER "$who: this user cannot use Docker. Add the user to the docker group: sudo usermod -aG docker \$USER (then log in again)."
    grep -qx 'nvrt=ok' <<< "$out" || fail E_NVIDIA_RUNTIME "$who: Docker has no NVIDIA runtime (install the NVIDIA Container Toolkit)."
    [[ "$(sed -n 's/^ib=//p' <<< "$out")" -gt 0 ]] || fail E_FABRIC "$who has no /dev/infiniband/uverbs* devices (the ConnectX-7 driver)."
    [[ "${P_GPU[r]}" == *GB10* ]] || warn W_GPU "$who: the GPU is '${P_GPU[r]:-none}', not a GB10."
    if [[ -n "${P_APPS[r]}" && -z "${P_OURS[r]}" ]]; then
      warn W_GPU "$who has other GPU programs running: ${P_APPS[r]%;}. They use memory that the model needs."
    fi
    if [[ -z "${P_MEM[r]}" ]]; then
      warn W_RAM "$who: cannot read the available memory."
    elif ! at_least "${P_MEM[r]}" "$MIN_AVAIL_GIB" && [[ -z "${P_OURS[r]}" ]]; then
      bad=1
      warn W_RAM "$who: ${P_MEM[r]} GiB of memory is available; a start needs $MIN_AVAIL_GIB GiB (MIN_AVAIL_GIB). Stop other programs (docker ps; nvidia-smi). If the total (${P_TOTAL[r]} GiB) is low, set Display Reserved Memory to 0.5 GB (README, Requirements)."
    fi
    ok "$(printf 'Spark %s  %-16s %-15s %s · %s of %s GiB free · %s GiB disk' "$r" "${P_NAME[r]}" "$(rhost "$r")" "${P_GPU[r]#GPU 0: }" "${P_MEM[r]:-?}" "${P_TOTAL[r]:-?}" "${P_DISK[r]:-?}")"
  done
  (( bad == 0 )) || { dry || fail E_MEMORY 'Not enough memory is available on every Spark (see above).'; }
}

# free_page_cache: drop the file cache on every Spark. On a DGX Spark the GPU and the system share one pool of memory;
# files that Linux keeps in memory after a download or a copy take space that the model needs. No sudo: a privileged
# container can write vm.drop_caches.
_drop_rank() {
  on "$1" sync
  on "$1" docker run --rm --privileged --network none --pull=never --entrypoint /bin/sh "$BASE_IMAGE" \
    -c 'echo 3 > /proc/sys/vm/drop_caches' || { echo "Spark $1 $(rname "$1"): could not drop the file cache"; return 1; }
  on_fn "$1" mem_counting 'mem_counting | cut -d" " -f1'
}
free_page_cache() {
  local r gib bad=0 summary=''
  if dry; then note 'Would drop the file cache on every Spark, then check the available memory.'; return 0; fi
  each_rank _drop_rank || { show_each; fail E_MEMORY 'Could not drop the file cache on every Spark (see above).'; }
  for r in 0 1 2 3; do
    gib=${EACH_OUT[r]##*$'\n'}
    summary+="$(rname "$r") $gib · "
    if ! is_number "$gib" || ! at_least "$gib" "$MIN_AVAIL_GIB"; then
      bad=1; warn W_RAM "Spark $r ($(rname "$r")): ${gib:-?} GiB available after the file cache was dropped; the model needs $MIN_AVAIL_GIB. Stop other programs (docker ps; nvidia-smi)."
    fi
  done
  (( bad == 0 )) || fail E_MEMORY 'Not enough memory is available on every Spark (see above).'
  ok "File cache dropped. Memory available (GiB): ${summary% · }."
}

# check_ports: the API port and the engine's control port are free on this Spark.
check_ports() {
  local p
  for p in "$PORT" "$ENGINE_PORT"; do
    if ss -Htln "sport = :$p" 2>/dev/null | grep -q .; then
      fail E_PORT "Port $p is in use on this Spark ($(ss -Htlnp "sport = :$p" 2>/dev/null | awk '{print $NF}' | head -1)). Stop that program, or change PORT / ENGINE_PORT in config.env."
    fi
  done
  ok "Ports are free on this Spark: API $PORT, engine control $ENGINE_PORT."
}

# --- the ring ----------------------------------------------------------------------------------------------------------
# node_inventory: this node's default-route netdev and one line per RoCE netdev that is up with an IPv4:
# "<netdev> <rdma device> <address/prefix> <RoCE v2 GID index of that address or -> <mtu> <port state>". The netdevs
# come in uverbs order. Read-only; sent to each worker with declare -f.
node_inventory() {
  local n dev hca cidr want idx f u
  echo "default $(ip -o -4 route show default 2>/dev/null | sed -n 's/.* dev \([^ ]*\).*/\1/p' | head -1)"
  for n in $(for n in /sys/class/net/*; do
               [[ -d $n/device/infiniband ]] || continue
               u=$(ls "$n/device/infiniband_verbs" 2>/dev/null | head -1); u=${u#uverbs}
               [[ "$u" =~ ^[0-9]+$ ]] || u=9999; echo "$u $n"
             done | sort -n | cut -d' ' -f2); do
    dev=${n##*/}
    [[ "$(cat "$n/operstate" 2>/dev/null)" == up ]] || continue
    hca=$(ls "$n/device/infiniband" | head -1)
    for cidr in $(ip -o -4 addr show dev "$dev" | awk '{print $4}'); do
      # shellcheck disable=SC2046
      want=$(printf '0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x' $(tr '.' ' ' <<< "${cidr%/*}"))
      idx=-
      for f in /sys/class/infiniband/"$hca"/ports/1/gids/*; do
        [[ "$(cat "$f" 2>/dev/null)" == "$want" ]] || continue
        [[ "$(cat /sys/class/infiniband/"$hca"/ports/1/gid_attrs/types/"${f##*/}" 2>/dev/null)" == *v2* ]] && { idx=${f##*/}; break; }
      done
      echo "$dev $hca $cidr $idx $(cat "$n/mtu") $(awk '{print $2}' /sys/class/infiniband/"$hca"/ports/1/state 2>/dev/null)"
    done
  done
}

# pair_links (stdin: inventory lines prefixed by rank): two ranks are linked where they hold addresses in one subnet.
# Prints, per rank: "rank R <default netdev> <hcas toward peers> <GID index or ->"; per linked pair: "toward R S <hcas>",
# "ping R <src> <dst> <mtu>" for every address pair on a shared subnet, "via R S <S's address on that subnet>", and from rank 0 "peer S <S's lowest address on
# its link to 0>"; "missing R S" for unlinked pairs; "ring 0 a b c" when the links form one ring; "bad R <netdev> <why>".
pair_links() {
  python3 -c '
import ipaddress, sys
nodes, default = {}, {}
for line in sys.stdin:
    f = line.split()
    if len(f) == 3 and f[1] == "default":
        default[int(f[0])] = f[2]
    elif len(f) >= 7:
        nodes.setdefault(int(f[0]), []).append((f[1], f[2], ipaddress.ip_interface(f[3]), f[4], int(f[5]), f[6]))
ranks = [0, 1, 2, 3]
linked = lambda e, others: any(e[2].network == o[2].network for o in others)
links = {r: set() for r in ranks}
for r in ranks:
    used = set()
    for s in ranks:
        if s == r:
            continue
        mine = [e for e in nodes.get(r, []) if linked(e, nodes.get(s, []))]
        if not mine:
            if r < s:
                print("missing", r, s)
            continue
        links[r].add(s)
        used.update(id(e) for e in mine)
        print("toward", r, s, ",".join(dict.fromkeys(e[1] for e in mine)))
        for e in mine:
            for o in nodes[s]:
                if o[2].network == e[2].network:
                    print("ping", r, e[2].ip, o[2].ip, min(e[4], o[4]), e[0])
                    print("via", r, s, o[2].ip)
            if e[5] != "ACTIVE":
                print("bad", r, e[0], "RDMA port " + e[1] + " is " + e[5])
            if e[3] == "-":
                print("bad", r, e[0], "no RoCE v2 GID holds " + str(e[2].ip))
        if r == 0:
            print("peer", s, min(o[2].ip for o in nodes[s] if linked(o, mine)))
    hcas, gids = [], set()
    for e in nodes.get(r, []):
        if id(e) in used and e[1] not in hcas:
            hcas.append(e[1]); gids.add(e[3])
    print("rank", r, default.get(r) or "-", ",".join(hcas) or "-", gids.pop() if len(gids) == 1 else "-")
def splits(n):
    for r in ranks:
        for s in links[r]:
            mine = [e for e in nodes.get(r, []) if linked(e, nodes.get(s, []))]
            theirs = [o for o in nodes.get(s, []) if linked(o, mine)]
            nets = [ipaddress.ip_interface(f"{e[2].ip}/{n}").network for e in mine]
            if len(set(nets)) != len(nets) or any(sum(o[2].ip in net for o in theirs) != 1 for net in nets):
                return False
    return True
own = max((e[2].network.prefixlen for r in ranks for e in nodes.get(r, [])), default=32)
pre = next((n for n in range(own, 31) if splits(n)), None)
if pre is not None:
    print("prefix", pre)
if all(len(links[r]) == 2 for r in ranks):
    order = [0, min(links[0])]
    while len(order) < 4:
        nxt = (links[order[-1]] - {order[-2]}).pop()
        if nxt == 0:
            break
        order.append(nxt)
    if len(order) == 4 and 0 in links[order[-1]]:
        print("ring", *order)
'
}

# Filled by detect_ring.
declare -A TOWARD=() VIA=()   # VIA["R S"]: S's addresses on the cables between R and S, comma-separated
declare -a PINGS=()
_inventory_rank() { on_fn "$1" node_inventory node_inventory; }
detect_ring() {
  local r kind a b c d e f pairs missing='' ring='' bad='' nxt
  each_rank _inventory_rank || fail E_FABRIC "Could not read the ConnectX-7 ports of every Spark."
  pairs=$(for r in 0 1 2 3; do sed "s/^/$r /" <<< "${EACH_OUT[r]}"; done | pair_links) \
    || fail E_FABRIC 'Could not match the cables between the Sparks (python3 is necessary on this Spark).'
  PINGS=(); TOWARD=(); VIA=()
  while read -r kind a b c d e f; do
    case "$kind" in
      missing) missing+=" $a-$b" ;;
      ring) ring="$a $b $c $d" ;;
      toward) TOWARD["$a $b"]=$c ;;
      ping) PINGS+=("$a $b $c $d $e") ;;
      via) [[ ",${VIA[$a $b]:-}," == *",$c,"* ]] || VIA["$a $b"]+="${VIA[$a $b]:+,}$c" ;;
      bad) bad+=$'\n'"    Spark $a ($(rname "$a")) $b: $c $d $e $f" ;;
    esac
  done <<< "$pairs"
  [[ -z "$bad" ]] || fail E_FABRIC "Some cable ports are not usable:$bad"$'\n'"  Look on that Spark: rdma link; ip -br address (docs/NETWORK.md)."
  [[ -n "$missing" ]] || fail E_FABRIC 'Every pair of Sparks shares a cable subnet (a switch). This recipe is for a ring of direct cables (docs/NETWORK.md).'
  [[ -n "$ring" ]] || fail E_FABRIC "The cables do not make one ring through all four Sparks (pairs with no cable:$missing). Connect each Spark to two others; each cable needs its own subnet (docs/NETWORK.md)."
  if [[ "$ring" != '0 1 2 3' ]]; then
    local fix=''
    for r in $ring; do fix+=" ${RANKS[r]}"; done
    [[ "${_RING_FIXED:-0}" == 0 ]] || fail E_FABRIC "Could not put the Sparks in the ring's order. Set in config.env: SPARKS=\"${fix# }\""
    _RING_FIXED=1
    warn W_ORDER "SPARKS was not in the ring's order. Now using: ${fix# }"
    SPARKS=${fix# }; set_ranks
    printf 'SPARKS=%s\n' "$SPARKS" > "$STATE_DIR/cluster.env"
    probe_cluster; detect_ring; return
  fi
  # The engine sends to the next Spark from each Spark's first cable port (f0) and receives on its second (f1).
  if [[ "${TOWARD[0 1]}" != *f0* && "${TOWARD[0 3]}" == *f0* && "${_RING_FIXED:-0}" == 0 ]]; then
    _RING_FIXED=1
    SPARKS="${RANKS[0]} ${RANKS[3]} ${RANKS[2]} ${RANKS[1]}"; set_ranks
    printf 'SPARKS=%s\n' "$SPARKS" > "$STATE_DIR/cluster.env"
    warn W_ORDER "The ring goes the other way around. Now using: $SPARKS"
    probe_cluster; detect_ring; return
  fi
  for r in 0 1 2 3; do
    nxt=$(( (r + 1) % 4 ))
    [[ "${TOWARD[$r $nxt]}" == *f0* ]] || fail E_FABRIC "Spark $r ($(rname "$r")) reaches the next Spark ($(rname "$nxt")) through ${TOWARD[$r $nxt]}, not through its first port. Cable the ring in one direction: port 0 of each Spark to port 1 of the next (docs/NETWORK.md)."
  done
  ok "Ring: $(rname 0) → $(rname 1) → $(rname 2) → $(rname 3) → back to $(rname 0)."
}

# check_links: a full-MTU, unfragmentable ping across every cable address pair, from both ends, all ranks at once.
_ping_rank() {
  local r=$1 line a src dst mtu dev script=''
  for line in "${PINGS[@]}"; do
    read -r a src dst mtu dev <<< "$line"
    [[ "$a" == "$r" ]] || continue
    script+="ping -q -c 2 -i 0.2 -W 1 -M do -s $(( mtu - 28 )) -I $src $dst >/dev/null 2>&1 && echo 'ok $dev $src $dst $mtu' || echo 'FAIL $dev $src $dst $mtu' &"$'\n'
  done
  on_sh "$r" "$script wait"
}
check_links() {
  local r line n=0 bad=''
  each_rank _ping_rank || true
  for r in 0 1 2 3; do
    while read -r line; do
      [[ -n "$line" ]] || continue
      n=$(( n + 1 ))
      [[ "$line" == ok* ]] || bad+=$'\n'"    Spark $r ($(rname "$r")): ${line#FAIL } (port, from, to, MTU)"
    done <<< "${EACH_OUT[r]}"
  done
  (( n > 0 )) || fail E_FABRIC 'No cable addresses to ping.'
  [[ -z "$bad" ]] || fail E_FABRIC "Full-size pings failed on these cables:$bad"$'\n'"  Check the cable, the MTU at the two ends (ip link) and the addresses (ip -br address)."
  ok "All $n cable paths answer full-size pings (every port, in the two directions)."
}


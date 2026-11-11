# The memory a Spark can give to a starting engine, as one number.
#
# A DGX Spark has one pool of 128 GB for the system and the GPU together, so a memory check reads the host. This
# recipe counts the available memory as MemAvailable (/proc/meminfo) plus the free pages that the kernel keeps on its
# per-CPU lists, which MemAvailable does not include. After a large free (a build ends, a copy ends) those lists hold
# several GiB for a while. The rule and most of the code come from the MiaAI-Lab 4x DGX Spark recipes
# (scripts/memavail.sh; Copyright 2026 Mia's AI Lab, Apache License 2.0). Changed for this recipe: see NOTICE.md.
#
# The rule: in /proc/zoneinfo, under each zone's `pagesets` line, add every `count:` (pages on a CPU's list) and every
# `high_min:` (the resting level of that list). parked = count - high_min pages (or count less 512 MiB when the kernel
# prints no high_min); never below 0. available = MemAvailable + parked, at most MemTotal.
#
# The functions use only awk and getconf and read only /proc, so they can be sent to another Spark over ssh.

# mem_counting [meminfo] [zoneinfo] [page bytes]: "<available GiB> <parked GiB> <parked bytes>" (GiB
# with one decimal), from /proc/meminfo, /proc/zoneinfo and getconf PAGESIZE by default. A missing
# zoneinfo counts no parked pages. Prints nothing (and fails) when the meminfo has no MemAvailable.
mem_counting() {
  local mi=${1:-/proc/meminfo} zi=${2:-/proc/zoneinfo} pg=${3:-}
  [ -n "$pg" ] || pg=$(getconf PAGESIZE 2>/dev/null)
  case "$pg" in ''|*[!0-9]*) pg=4096 ;; esac
  [ -r "$zi" ] || zi=/dev/null
  awk -v page="$pg" -v rest_default=536870912 -v zfile="$zi" '
    FILENAME != zfile {
      if ($1 == "MemTotal:") total = $2 * 1024
      else if ($1 == "MemAvailable:") { avail = $2 * 1024; seen = 1 }
      next
    }
    {
      row = $0
      gsub(/\r/, " ", row)
      t = row; sub(/^[ \t]+/, "", t); sub(/[ \t]+$/, "", t)
      if (t == "pagesets") { listed = 1; next }
      if (listed && substr($0, 1, 4) != "    " && substr($0, 1, 10) != "  vm stats") { listed = 0; next }
      if (!listed) next
      n = split(t, w, /[ \t]+/)
      if (w[1] != "count:" && w[1] != "high_min:") next
      if (n < 2 || w[2] !~ /^[0-9]+$/) { bad = 1; next }
      if (w[1] == "count:") count += w[2]; else rest += w[2]
    }
    END {
      if (!seen) exit 1
      parked = 0
      if (!bad && count > 0) {
        resting = rest > 0 ? rest * page : rest_default
        parked = count * page - resting
        if (parked < 0) parked = 0
      }
      a = avail + parked
      if (total > 0 && a > total) a = total
      printf "%.1f %.1f %.0f\n", a / 1073741824, parked / 1073741824, parked
    }' "$mi" "$zi"
}

# mem_avail_gb: "<GiB available, one decimal> <GiB parked>" on this node, for the memory checks.
# Falls back to `free`'s available column when /proc/meminfo has no MemAvailable.
mem_avail_gb() {
  local m
  if m=$(mem_counting 2>/dev/null) && [ -n "$m" ]; then
    echo "$m" | awk '{ printf "%s %s\n", $1, $2 }'
  else
    free -g 2>/dev/null | awk '/^Mem:/ { printf "%s.0 0.0\n", $7 }'
  fi
}

# mem_total_gb: MemTotal in GiB, one decimal (on a 128 GB GB10 Spark this is what Linux sees; the
# pack author's test machines exposed about 121.6 GiB — their measurement, not this recipe's).
mem_total_gb() {
  awk '$1 == "MemTotal:" { printf "%.1f\n", $2 * 1024 / 1073741824 }' /proc/meminfo 2>/dev/null
}

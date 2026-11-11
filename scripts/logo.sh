#!/usr/bin/env bash
# Causal-mask logo for the terminal.
#   ./logo.sh            large (default)
#   ./logo.sh small      compact
# Or source it and call:  print_logo [large|small]
#
# Uses 24-bit colour. Falls back to shaded blocks when NO_COLOR is set
# or output isn't a terminal.

print_logo() {
  local size="${1:-large}"

  # Brand colours
  local fg=(46 230 255) # cyan    #2EE6FF
  local ac=(255 61 180) # magenta #FF3DB4
  local bg=(10 13 20)   # ground  #0A0D14 (opacity is blended against this)

  # Attention weights (%) for the lower-triangular mask, row by row.
  local -a rows=("100" "35 100" "80 25 60" "30 90 20 50" "100 25 45 30 70")

  local w h gapx gapy pad
  if [[ "$size" == "small" ]]; then
    w=2
    h=1
    gapx=1
    gapy=0
    pad="  "
  else
    w=4
    h=2
    gapx=2
    gapy=1
    pad="  "
  fi

  local color=1
  if [[ -n "${NO_COLOR:-}" || ! -t 1 ]]; then color=0; fi
  [[ "${LOGO_COLOR:-}" == 1 ]] && color=1   # the caller knows that the output goes to a terminal

  local block="" spaces="" gap="" i
  for ((i = 0; i < w; i++)); do
    block+="█"
    spaces+=" "
  done
  for ((i = 0; i < gapx; i++)); do gap+=" "; done

  local r line cell c j k pct base r_ g_ b_ ch
  echo
  for ((r = 0; r < ${#rows[@]}; r++)); do
    read -r -a cell <<<"${rows[r]}"
    line="$pad"
    for ((j = 0; j < ${#cell[@]}; j++)); do
      pct=${cell[j]}
      if ((color)); then
        if ((j == r)); then base=("${ac[@]}"); else base=("${fg[@]}"); fi
        r_=$(((base[0] * pct + bg[0] * (100 - pct)) / 100))
        g_=$(((base[1] * pct + bg[1] * (100 - pct)) / 100))
        b_=$(((base[2] * pct + bg[2] * (100 - pct)) / 100))
        line+=$'\e[38;2;'"${r_};${g_};${b_}m${block}"$'\e[0m'
      else
        if ((pct >= 80)); then
          ch="█"
        elif ((pct >= 50)); then
          ch="▓"
        elif ((pct >= 30)); then
          ch="▒"
        else
          ch="░"
        fi
        for ((k = 0; k < w; k++)); do line+="$ch"; done
      fi
      ((j < ${#cell[@]} - 1)) && line+="$gap"
    done
    for ((k = 0; k < h; k++)); do printf '%s\n' "$line"; done
    if ((r < ${#rows[@]} - 1)); then
      for ((k = 0; k < gapy; k++)); do echo; done
    fi
  done
  echo
}

# Run directly (not when sourced)
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  print_logo "${1:-large}"
fi

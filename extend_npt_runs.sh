#!/bin/bash
# Extend the production of the constant-pressure runs and (re)submit them.
#
#   ./extend_npt_runs.sh                       # prepare + submit (on the cluster)
#   DRY=1 ./extend_npt_runs.sh                 # only report what it would do
#   SERIES="nptaniso" PROD=20000 ./extend_npt_runs.sh
#
# For every run cif_runs/<system>/<series>_<T> of SERIES (default
# npt3b nptaniso nptanisoimp) it
#   * writes PROD (default 30000 production steps = 30 ps) into prod_steps,
#     so production ends at step 1000 + PROD;
#   * refreshes run.sh from run_cif_heat.sh with --time=WALLTIME (default
#     1000 h: the slowest case, Co_12dMIm npt3b, needs ~145 s/step);
#   * submits it under the usual name h<T>_<tag>_<series>, unless a job of that
#     name is queued or running.
# run.sh continues from prod/restart.npz (positions, velocities, cell) -- no new
# warm-up; the trajectory files are appended to.  Runs whose last attempt ended
# in a spurious SCF solution are resubmitted too: the Langevin noise is not
# seeded, so the resumed run follows a different path from the checkpoint on.
# If a job hits the wall-clock limit, ./resubmit_cif_heat.sh picks it up (it
# reads prod_steps).
#
# Run from the project root on a node that has `sbatch`.
set -eu
cd "$(dirname "$0")"

BASE=${BASE:-cif_runs}
SERIES=${SERIES:-"npt3b nptaniso nptanisoimp"}
PROD=${PROD:-30000}
WALLTIME=${WALLTIME:-1000:00:00}

last_step() {   # highest step in energy.dat, -1 if none
  awk '!/^#/ && NF == 7 { s = $1 } END { print (s == "" ? -1 : s) }' \
      "$1" 2>/dev/null || echo -1
}
queued=$(squeue -h -u "$USER" -o '%.100j' 2>/dev/null | sed 's/^ *//' || true)
n_sub=0

for ser in $SERIES; do
  for d in "$BASE"/*/"${ser}"_[0-9]*; do
    [ -f "$d/temperature" ] || continue
    T=$(cat "$d/temperature")
    case "$d" in
      */Co_1MIm/*)    tag=1MIm ;;
      */Co_12dMIm/*)  tag=12dMIm ;;
      */CoSCN2bIm2/*) tag=bIm2 ;;
    esac
    name="h${T}_${tag}_${ser}"
    pstep=$(last_step "$d/prod/energy.dat")
    old=$(cat "$d/prod_steps" 2>/dev/null || echo 10000)

    if [ "$pstep" -ge $((1000 + PROD)) ]; then
      printf '%-36s prod step %-6s done\n' "$d" "$pstep"; continue
    fi
    if grep -qx "$name" <<< "$queued"; then
      printf '%-36s prod step %-6s already queued/running (%s), not touched\n' "$d" "$pstep" "$name"; continue
    fi
    printf '%-36s prod step %-6s prod_steps %s -> %s, submit as %s\n' "$d" "$pstep" "$old" "$PROD" "$name"
    [ -n "${DRY:-}" ] && continue

    echo "$PROD" > "$d/prod_steps"
    sed -e "s/^#SBATCH --time=.*/#SBATCH --time=${WALLTIME}/" run_cif_heat.sh > "$d/run.sh.new"
    mv "$d/run.sh.new" "$d/run.sh"; chmod +x "$d/run.sh"
    ( cd "$d" && sbatch -J "$name" run.sh )
    n_sub=$((n_sub + 1))
  done
done

echo "submitted $n_sub job(s)"

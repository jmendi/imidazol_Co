#!/bin/bash
# Resubmit every CIF-cell temperature run that has not finished.
#
#   ./resubmit_cif_heat.sh              # resubmit all unfinished runs
#   DRY=1 ./resubmit_cif_heat.sh        # only report what it would do
#   UPDATE=1 ./resubmit_cif_heat.sh     # also refresh run.sh from run_cif_heat.sh
#   FORCE=1 ./resubmit_cif_heat.sh      # also runs that failed for physical reasons
#   EXCLUDE='' ./resubmit_cif_heat.sh   # do not skip the 2x2x1 supercells
#
# Covers the run directories created by send_cif_heat.sh
# (cif_runs/<system>/md_<T>{,_corrected,_restrained}) and by send_sc_heat.sh
# (cif_runs/<system>/sc<rep>_md_<T>...).  A run is skipped when
#   * its production has reached the final step, or
#   * a job with its name is already queued or running.
#   * its last attempt ended in a PHYSICS failure (spurious SCF solution,
#     --fmax-start exceeded, thermostat divergence): resubmitting repeats it --
#     unless FORCE=1;
#   * its path matches EXCLUDE (an extended regex, default 'sc221_': the 2x2x1
#     supercells, ~3000 s/step, cannot finish) -- set EXCLUDE='' to include them.
# Anything else -- never started, or killed by the wall-clock limit, a node or
# scancel -- is resubmitted; run.sh itself resumes from the last checkpoint.
#
# Run from the project root on a node that has `sbatch`.
set -eu
cd "$(dirname "$0")"

BASE=${BASE:-cif_runs}
TOTAL=${TOTAL:-11000}      # warm-up 1000 + production 10000

last_step() {   # highest step in energy.dat, -1 if none
  awk '!/^#/ && NF == 7 { s = $1 } END { print (s == "" ? -1 : s) }' \
      "$1" 2>/dev/null || echo -1
}

# full job names (squeue truncates %j to a few characters by default)
queued=$(squeue -h -u "$USER" -o '%.100j' 2>/dev/null | sed 's/^ *//' || true)
EXCLUDE=${EXCLUDE-sc221_}
physics_failure() {   # last md.out of the run ended in a failure that a resume would repeat
  local f
  f=$(ls -t "$1"/prod/md.out "$1"/warmup/md.out 2>/dev/null | head -1)
  # only the latest attempt: md.out collects every attempt, each starting with "... atoms, T = ..."
  [ -n "$f" ] && awk '/ atoms, T = /{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}' "$f" \
    | grep -qE 'spurious solution|exceeds --fmax-start|thermostat diverged'
}
n_sub=0

for d in "$BASE"/Co_1MIm/md_*_corrected "$BASE"/Co_12dMIm/md_[0-9]* \
         "$BASE"/CoSCN2bIm2/md_*_restrained "$BASE"/*/sc[0-9][0-9][0-9]_md_* \
         "$BASE"/*/npt_[0-9]* "$BASE"/*/npt3b_[0-9]* "$BASE"/*/nptaniso_[0-9]* "$BASE"/*/nptanisoimp_[0-9]*; do
  [ -f "$d/temperature" ] || continue         # not a send_cif_heat.sh run
  T=$(cat "$d/temperature")
  case "$d" in
    */Co_1MIm/*)    tag=1MIm ;;
    */Co_12dMIm/*)  tag=12dMIm ;;
    */CoSCN2bIm2/*) tag=bIm2 ;;
  esac
  name="h${T}_${tag}"
  case "$d" in */sc[0-9][0-9][0-9]_md_*)            # supercell (send_sc_heat.sh)
    sc=${d##*/sc}; name="${name}_${sc%%_*}" ;; esac
  case "$d" in */npt_*) name="${name}_npt" ;; */npt3b_*) name="${name}_npt3b" ;;
                */nptaniso_*) name="${name}_nptaniso" ;; */nptanisoimp_*) name="${name}_nptanisoimp" ;; esac
  # runs may carry their own production length (supercells: 5000)
  total=$TOTAL
  [ -f "$d/prod_steps" ] && total=$((1000 + $(cat "$d/prod_steps")))

  wstep=$(last_step "$d/warmup/energy.dat")
  pstep=$(last_step "$d/prod/energy.dat")
  if [ "$pstep" -ge 0 ]; then state="prod step $pstep"
  elif [ "$wstep" -ge 0 ]; then state="warm-up step $wstep"
  else state="not started"; fi

  if [ "$pstep" -ge "$total" ]; then
    printf '%-40s %-18s done\n' "$d" "$state"; continue
  fi
  if [ -n "$EXCLUDE" ] && [[ "$d" =~ $EXCLUDE ]]; then
    printf '%-40s %-18s skipped (EXCLUDE=%s)\n' "$d" "$state" "$EXCLUDE"; continue
  fi
  if [ -z "${FORCE:-}" ] && physics_failure "$d"; then
    printf '%-40s %-18s skipped: last attempt failed for physical reasons (FORCE=1 to resubmit)\n' "$d" "$state"; continue
  fi
  if grep -qx "$name" <<< "$queued"; then
    printf '%-40s %-18s already queued/running (%s)\n' "$d" "$state" "$name"; continue
  fi

  printf '%-40s %-18s -> resubmit as %s\n' "$d" "$state" "$name"
  [ -n "${DRY:-}" ] && continue
  if [ -n "${UPDATE:-}" ]; then     # refresh, keeping this run's CPUs, memory and time
    cpus=$(sed -n 's/^#SBATCH --cpus-per-task=//p' "$d/run.sh")
    mem=$(sed -n 's/^#SBATCH --mem=//p' "$d/run.sh")
    wall=$(sed -n 's/^#SBATCH --time=//p' "$d/run.sh")
    sed -e "s/^#SBATCH --cpus-per-task=.*/#SBATCH --cpus-per-task=${cpus:-1}/" \
        -e "s/^#SBATCH --mem=.*/#SBATCH --mem=${mem:-8G}/" \
        -e "s/^#SBATCH --time=.*/#SBATCH --time=${wall:-500:00:00}/" \
        run_cif_heat.sh > "$d/run.sh.new" && mv "$d/run.sh.new" "$d/run.sh"
    chmod +x "$d/run.sh"
  fi
  ( cd "$d" && sbatch -J "$name" run.sh )
  n_sub=$((n_sub + 1))
done

echo "submitted $n_sub job(s)"

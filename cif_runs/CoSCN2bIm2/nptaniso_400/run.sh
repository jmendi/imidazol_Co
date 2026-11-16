#!/bin/bash -l
# ===========================================
# Slurm Header - CIF-cell NVT MD: warm-up + production at one temperature
# ===========================================
#SBATCH --partition=CLUSTER
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --get-user-env
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.error
#SBATCH --time=1000:00:00
#SBATCH --mem=8G
#SBATCH --exclude=compute-0-2   # no /usr/local/software/ips2019 there (MKL ImportError, 2026-09-25)
# ===========================================
. startjob      # Do not remove this line!
# ===========================================
#
# Submitted by send_cif_heat.sh from inside the run directory, which holds
#
#   warmup/   relax.xyz (last frame of the 100 K run) + input.lvs
#   prod/     created here once the warm-up has finished
#
# Stage 1  warmup/  steps     0 ->  1000, fresh Maxwell-Boltzmann velocities
# Stage 2  prod/    steps  1000 -> 11000, continues warm-up positions AND
#                   velocities (warmup/restart.npz at step 1000)
#
# Both stages: Langevin 0.01 fs^-1, dt 1 fs, mixer johnson beta 0.1 order 6 --
# the same settings as the 100 K CIF runs.  Extra md_fireballpy.py options for
# both stages come from the file md_opts in the run directory, if present
# (send_npt_heat.sh: --d3 periodic --barostat crescale ...).
#
# The script is idempotent: if the job hits the wall-clock limit or is killed,
# resubmit it unchanged (`sbatch run.sh` in this directory).  A finished stage
# is skipped, an interrupted one is trimmed back to its checkpoint and resumed.

ulimit -s unlimited
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Intel 2019 MKL: fireballpy's Fortran kernel is linked against it
module load mpi/4.1.2-IPS2019
# Python environment: micromamba env fireballpy unless the run directory names
# another one in the file py_env, as "micromamba:<env>" or "conda:<env>"
# (send_npt_heat.sh writes conda:fireballpy-stress for the anisotropic series).
# ~/.bashrc returns early in a batch job, so neither conda nor micromamba is
# initialised there; hook them in by hand, as the bashrc aliases do.  (The
# miniconda fireballpy env carries a 2018 libgfortran that breaks DFT-D3.)
PY_ENV=$(cat "${SLURM_SUBMIT_DIR:-$PWD}/py_env" 2>/dev/null) || PY_ENV=micromamba:fireballpy
case "$PY_ENV" in
  micromamba:*)
    export MAMBA_ROOT_PREFIX=~/micromamba
    eval "$(~/.local/bin/micromamba shell hook --shell bash)"
    micromamba activate "${PY_ENV#micromamba:}" ;;
  conda:*)
    source ~/miniconda3/bin/activate
    conda activate "${PY_ENV#conda:}" ;;
  *) echo "py_env must be micromamba:<env> or conda:<env>, got '$PY_ENV'" >&2; exit 1 ;;
esac
# _fireball.so is linked against the IPS2019 MKL by absolute path, but Python
# loads it RTLD_LOCAL; MKL then dlopen()s its CPU kernel (libmkl_avx*.so), which
# cannot see libmkl_core's symbols and aborts with "Cannot load libmkl_avx.so or
# libmkl_def.so" at the first BLAS call.  The miniconda env hid this because its
# numpy was MKL-built; the micromamba numpy uses OpenBLAS.  Preloading the MKL
# layers makes their symbols global.  libiomp5 must come first: intel_thread
# calls omp_* but does not list libiomp5 as NEEDED.  Only python gets the
# preload -- exported, it broke the rsync/date calls inside endjob.
IPS=/usr/local/software/ips2019/compilers_and_libraries_2019.5.281/linux
MKLLIB=$IPS/mkl/lib/intel64_lin
MKL_PRELOAD="$IPS/compiler/lib/intel64_lin/libiomp5.so:$MKLLIB/libmkl_core.so:$MKLLIB/libmkl_intel_thread.so:$MKLLIB/libmkl_intel_lp64.so"
python() { LD_PRELOAD="$MKL_PRELOAD${LD_PRELOAD:+:$LD_PRELOAD}" command python "$@"; }
python -c 'import fireballpy' || { echo "fireballpy not importable by $(type -P python)" >&2; exit 1; }

# Work directly in the submit directory, not the node scratch copy made by
# startjob, so checkpoints land on /home as they are written.
RUN=${SLURM_SUBMIT_DIR:-$PWD}
# But endjob -- run by us at the end AND by startjob's signal trap on scancel or
# the wall-clock limit -- rsyncs the scratch copy back over RUN, and that copy
# is frozen at job start: on 2026-09-24 a cancelled job had its energy.dat,
# md.out and run.sh reverted that way.  So before every endjob, mirror RUN into
# scratch; the copy-back then changes nothing.
SCR=$PWD                        # startjob leaves us in the scratch copy
sync_scratch() {
  [ "$SCR" != "$RUN" ] && [ -d "$SCR" ] && rsync -a --delete "$RUN"/ "$SCR"/
}
# fireballpy rewrites info.py.dat in the FData directory on every load; with
# all jobs sharing one directory, two that start together can read it truncated
# (forrtl severe (24) end-of-file, unit 12).  Give each job its own directory of
# symlinks to the FData files, minus info.py.dat, which it then writes privately.
# Kept outside the scratch copy, which sync_scratch mirrors with --delete.
FDATA_SRC=${FDATA_SRC:-$HOME/fireball_basis/Fdata_bio_S_Co/coutput}
export FIREBALL_FDATA=${TMPDIR:-/tmp}/fdata_${SLURM_JOB_ID:-$$}
rm -rf "$FIREBALL_FDATA"; mkdir -p "$FIREBALL_FDATA"
cp -s "$FDATA_SRC"/* "$FIREBALL_FDATA"/ 2>/dev/null || true   # files (skips dirs)
for d in "$FDATA_SRC"/*/; do ln -s "${d%/}" "$FIREBALL_FDATA"/; done
rm -f "$FIREBALL_FDATA/info.py.dat"                  # the link, not the file
[ -f "$FIREBALL_FDATA/info.dat" ] && [ "$(ls "$FIREBALL_FDATA" | wc -l)" -eq "$(ls "$FDATA_SRC" | grep -cvx info.py.dat)" ] ||
  { echo "could not build private FData dir $FIREBALL_FDATA" >&2; exit 1; }
finish() { rm -rf "$FIREBALL_FDATA"; sync_scratch; . endjob; }
for sig in EXIT HUP INT TERM USR1 USR2 XCPU; do
  h=$(trap -p "$sig")
  [ -n "$h" ] || continue
  h=${h#trap -- \'}; h=${h%\' "$sig"}; h=${h%\' SIG"$sig"}
  trap "rm -rf \"\$FIREBALL_FDATA\"; sync_scratch; $h" "$sig"
done
# run directories sit at <project>/cif_runs/<system>/<run>; the project is
# /home/jesus/imidazol_Co on the cluster, ~/fast05/imidazol_Co on the workstation
PROJ=$(cd "$RUN/../../.." && pwd)
# target temperature is written into the run directory by send_cif_heat.sh, so
# a plain resubmission picks it up (and a stray $TEMP in the environment cannot)
TEMP=$(cat "$RUN/temperature" 2>/dev/null) || true
[ -n "$TEMP" ] || { echo "no $RUN/temperature file" >&2; exit 1; }
WARM_STEPS=${WARM_STEPS:-1000}
# a run directory may carry its own production length (the supercells use 5000)
PROD_STEPS=$(cat "$RUN/prod_steps" 2>/dev/null) || PROD_STEPS=${PROD_STEPS:-10000}
# A thermal 100 K snapshot carries ~1-3 eV/A of force; 5 eV/A still rejects a
# broken geometry (see run_heat.sh).  Only applies to the fresh warm-up start.
FMAX_START=${FMAX_START:-5.0}
# must divide WARM_STEPS, so the warm-up leaves a checkpoint at its last step
CHECKPOINT=${CHECKPOINT:-20}

MD_OPTS=$(cat "$RUN/md_opts" 2>/dev/null) || MD_OPTS=
# MD driver: md_fireballpy.py unless the run directory names another in md_script
# (md_fireballpy_aniso.py for the flexible-cell series)
MD_SCRIPT=$(cat "$RUN/md_script" 2>/dev/null) || MD_SCRIPT=md_fireballpy.py
COMMON="--temp ${TEMP} --dt 1 --thermostat langevin --friction 0.01
        --mixer johnson --beta 0.1 --mix-order 6 --checkpoint ${CHECKPOINT} ${MD_OPTS}"

log() { echo "$(date '+%F %T')  $*"; }

last_step() {   # highest step in energy.dat, -1 if none
  awk '!/^#/ && NF == 7 { s = $1 } END { print (s == "" ? -1 : s) }' \
      "$1/energy.dat" 2>/dev/null || echo -1
}
ckpt_step() {   # step stored in restart.npz, -1 if none
  python -c "import numpy as np, sys; print(int(np.load(sys.argv[1])['step']))" \
      "$1/restart.npz" 2>/dev/null || echo -1
}

# run_stage DIR FIRST_STEP LAST_STEP
#   brings DIR from wherever it is to LAST_STEP; FIRST_STEP is its step origin
run_stage() {
  local d=$1 first=$2 last=$3 step cs
  step=$(last_step "$d")
  if [ "$step" -ge "$last" ]; then
    log "$(basename "$d"): already complete (step $step)"
    return 0
  fi
  cs=$(ckpt_step "$d")
  if [ "$step" -lt 0 ] && [ "$cs" -lt 0 ]; then
    log "$(basename "$d"): fresh start, $first -> $last"
    python -u "${PROJ}/${MD_SCRIPT}" "$d" $COMMON \
        --steps $((last - first)) --fmax-start "$FMAX_START" >> "$d/md.out" 2>&1
  else
    [ "$cs" -ge 0 ] || { log "ERROR: $d has energy.dat but no restart.npz"; return 1; }
    # drop everything from the checkpoint on: frames written after it would be
    # duplicated, and the resumed run re-records the checkpoint frame itself
    python "${PROJ}/trim_trajectory.py" "$d" --step $((cs - 1))
    log "$(basename "$d"): resuming from step $cs, $((last - cs)) steps to go"
    python -u "${PROJ}/${MD_SCRIPT}" "$d" $COMMON \
        --steps $((last - cs)) --restart >> "$d/md.out" 2>&1
  fi
  local rc=$?
  log "$(basename "$d"): exit status $rc, last step $(last_step "$d")"
  return $rc
}

log "=================================================================="
log "$RUN  T = $TEMP K  on $(hostname), $OMP_NUM_THREADS threads, job ${SLURM_JOB_ID:-?}"
[ -n "$MD_OPTS" ] && log "extra options: $MD_OPTS"
log "python env $PY_ENV, MD driver $MD_SCRIPT"
log "=================================================================="

W="$RUN/warmup"
P="$RUN/prod"
TOTAL=$((WARM_STEPS + PROD_STEPS))

run_stage "$W" 0 "$WARM_STEPS" || { log "warm-up failed, see $W/md.out"; finish; exit 1; }

# seed production from the warm-up's final state (positions + velocities)
if [ ! -f "$P/energy.dat" ]; then
  cs=$(ckpt_step "$W")
  if [ "$cs" -ne "$WARM_STEPS" ]; then
    log "ERROR: warmup/restart.npz is at step $cs, expected $WARM_STEPS"
    finish; exit 1
  fi
  mkdir -p "$P"
  cp "$W/relax.xyz" "$W/input.lvs" "$W/restart.npz" "$P/"
  printf '#   step      time_fs         Epot_eV         Ekin_eV        Etot_eV      T_inst_K   walltime_s\n' \
      > "$P/energy.dat"
  log "prod: seeded from warmup/restart.npz (step $cs)"
fi

run_stage "$P" "$WARM_STEPS" "$TOTAL" || { log "production failed, see $P/md.out"; finish; exit 1; }

log "all done: production steps $WARM_STEPS -> $(last_step "$P") in $P/answer.xyz"

# ===========================================
finish          # sync_scratch + endjob.  Do not remove this line!
#--------------------------------------------

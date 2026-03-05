\
#!/usr/bin/env bash
set -euo pipefail

# Worldsheet Polyakov → K → DPH (run helper)
#
# Quick:
#   ./run.sh                            # run run0001
#   ./run.sh run0002                    # run run0002 (expects params.json there)
#   ./run.sh --init run0002             # create run0002 params.json from run0001
#   ./run.sh --sweep run0001            # same as run (sweeps are in params.json)
#   ./run.sh --clean run0001            # remove artifacts for run0001 (keeps params.json)
#   ./run.sh --clean-all                # remove artifacts + reset docs/project_log.md
#   ./run.sh --venv                     # create .venv + install requirements
#
# Env overrides:
#   PYTHON=python3 PARAMS=... ./run.sh

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

usage() {
  sed -n '1,120p' "$0" | sed 's/^# \{0,1\}//'
}

run_params() {
  local params="$1"
  if [[ ! -f "$params" ]]; then
    echo "Missing params.json at: $params" >&2
    exit 2
  fi
  echo "[run.sh] $PYTHON code/worldsheet_polyakov/compute_K_dph.py $params"
  exec "$PYTHON" "code/worldsheet_polyakov/compute_K_dph.py" "$params"
}

init_run() {
  local run="$1"
  local dir="outputs/worldsheet_polyakov/$run"
  mkdir -p "$dir"
  if [[ -f "$dir/params.json" ]]; then
    echo "[run.sh] params.json already exists: $dir/params.json"
    return 0
  fi
  echo "[run.sh] seeding $dir/params.json from run0001"
  cp "outputs/worldsheet_polyakov/run0001/params.json" "$dir/params.json"
  "$PYTHON" - <<PY
import json, pathlib
p = pathlib.Path("$dir/params.json")
d = json.loads(p.read_text())
d["run_id"] = "$run"
d["io"]["output_dir"] = "outputs/worldsheet_polyakov/$run"
p.write_text(json.dumps(d, indent=2))
PY
}

clean_run() {
  local run="$1"
  local dir="outputs/worldsheet_polyakov/$run"
  if [[ ! -d "$dir" ]]; then
    echo "[run.sh] no such run dir: $dir" >&2
    exit 2
  fi
  echo "[run.sh] cleaning artifacts in $dir (keeping params.json)"
  find "$dir" -type f \( \
    -name "K_W*.npy" -o \
    -name "K_abs_W*.npy" -o \
    -name "K.npy" -o \
    -name "K_abs.npy" -o \
    -name "samples.npy" -o \
    -name "dph_plot*.png" -o \
    -name "sweeps_summary*.json" \
  \) -print -delete 2>/dev/null || true
}

clean_all() {
  echo "[run.sh] cleaning all artifacts under outputs/worldsheet_polyakov"
  find outputs/worldsheet_polyakov -type f \( \
    -name "K_W*.npy" -o \
    -name "K_abs_W*.npy" -o \
    -name "K.npy" -o \
    -name "K_abs.npy" -o \
    -name "samples.npy" -o \
    -name "dph_plot*.png" -o \
    -name "sweeps_summary*.json" \
  \) -print -delete 2>/dev/null || true
  echo "[run.sh] resetting docs/project_log.md"
  mkdir -p docs
  printf "# Project Log\n\n" > docs/project_log.md
}

make_venv() {
  echo "[run.sh] creating venv at $VENV_DIR"
  "$PYTHON" -m venv "$VENV_DIR"
  echo "[run.sh] installing requirements.txt"
  "$VENV_DIR/bin/python" -m pip install --upgrade pip
  "$VENV_DIR/bin/python" -m pip install -r requirements.txt
  echo "[run.sh] done. Activate with: source $VENV_DIR/bin/activate"
}

main() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
  fi

  local cmd="${1:-}"
  case "$cmd" in
    --init)
      [[ -n "${2:-}" ]] || { echo "Missing run id for --init"; exit 2; }
      init_run "$2"
      ;;
    --clean)
      [[ -n "${2:-}" ]] || { echo "Missing run id for --clean"; exit 2; }
      clean_run "$2"
      ;;
    --clean-all)
      clean_all
      ;;
    --venv)
      make_venv
      ;;
    --sweep)
      # semantic alias for run
      local run="${2:-run0001}"
      local params="outputs/worldsheet_polyakov/$run/params.json"
      run_params "${PARAMS:-$params}"
      ;;
    "" )
      local params="${PARAMS:-outputs/worldsheet_polyakov/run0001/params.json}"
      run_params "$params"
      ;;
    run* )
      local params="outputs/worldsheet_polyakov/$cmd/params.json"
      run_params "${PARAMS:-$params}"
      ;;
    * )
      # if user passed a path to params.json
      if [[ -f "$cmd" && "$cmd" == *.json ]]; then
        run_params "$cmd"
      else
        echo "Unknown command: $cmd" >&2
        usage >&2
        exit 2
      fi
      ;;
  esac
}

main "$@"

# ---- Safe helper (MemoryMax cap) ----
safe_run() {
  local params="$1"
  local mem="${MEM_MAX:-8G}"
  if [[ ! -f "$params" ]]; then
    echo "Missing params.json at: $params" >&2
    exit 2
  fi
  echo "[run.sh] SAFE MemoryMax=$mem"
  systemd-run --user --scope -p MemoryMax="$mem" env MPLBACKEND=Agg \
    nice -n 10 ionice -c2 -n7 \
    "$PYTHON" -u "code/worldsheet_polyakov/compute_K_dph.py" "$params"
}

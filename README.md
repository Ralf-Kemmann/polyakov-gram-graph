\
Worldsheet Polyakov → K → DPH (Skeleton)
======================================

Quick start (from this folder):
  python code/worldsheet_polyakov/compute_K_dph.py outputs/worldsheet_polyakov/run0001/params.json

Notes:
- Default params include pragmatics.max_samples_for_K=200 to keep the first run manageable.
  For a full run, remove or increase that value.
- This is Euclidean + conformal gauge, baseline only (no Virasoro constraints).
- Outputs land in outputs/worldsheet_polyakov/run0001/

One-command runs
----------------
Option A (Make):
  make venv
  make run0001
  make init-run RUN=run0002
  make run RUN=run0002
  make clean-run RUN=run0001
  make clean-all

Option B (run.sh):
  ./run.sh
  ./run.sh --init run0002
  ./run.sh run0002
  ./run.sh --clean run0001
  ./run.sh --clean-all
  ./run.sh --venv

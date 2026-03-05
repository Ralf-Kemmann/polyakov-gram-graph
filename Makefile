\
# Worldsheet Polyakov → K → DPH (Make targets)
#
# Quick:
#   make venv          # create .venv + install deps
#   make run0001       # 1-command run
#   make init-run RUN=run0002  # copy run0001 params.json into new run dir (if missing)
#   make run RUN=run0002       # run run0002
#   make clean-run RUN=run0001 # remove generated artifacts for one run (keeps params.json)
#   make clean-all     # remove artifacts + reset docs/project_log.md
#
# You can also run without venv by setting PYTHON=python3.

SHELL := /usr/bin/env bash
PYTHON ?= python3
VENV_DIR ?= .venv
VENV_PY := $(VENV_DIR)/bin/python

MODULE_DIR := code/worldsheet_polyakov
DEFAULT_RUN := run0001

RUN ?= $(DEFAULT_RUN)
PARAMS ?= outputs/worldsheet_polyakov/$(RUN)/params.json

# ---------- helpers ----------
.PHONY: help check-params mkdir-run init-run venv run run0001 sweep clean-outputs clean-run clean clean-all

help:
	@echo "Targets:"
	@echo "  make venv                 Create $(VENV_DIR) and install requirements"
	@echo "  make run                  Run pipeline using PARAMS (default: $(PARAMS))"
	@echo "  make run0001              Alias for make run RUN=run0001"
	@echo "  make sweep                Alias for make run (sweeps are defined in params.json)"
	@echo "  make init-run RUN=run0002 Create new run dir and seed params.json from run0001"
	@echo "  make clean-run RUN=...    Remove generated artifacts for one run (keeps params.json)"
	@echo "  make clean                Remove generated artifacts for all runs (keeps params.json files)"
	@echo "  make clean-all            Clean + reset docs/project_log.md"
	@echo ""
	@echo "Variables:"
	@echo "  PYTHON=python3 | RUN=run0001 | PARAMS=outputs/worldsheet_polyakov/run0001/params.json"
	@echo "  VENV_DIR=.venv"

check-params:
	@test -f "$(PARAMS)" || (echo "Missing params.json at: $(PARAMS)"; exit 2)

mkdir-run:
	@mkdir -p "$(dir $(PARAMS))"

# Seed a new run directory by copying run0001 params.json (only if missing)
init-run:
	@mkdir -p "outputs/worldsheet_polyakov/$(RUN)"
	@if [[ -f "outputs/worldsheet_polyakov/$(RUN)/params.json" ]]; then \
		echo "[make] params.json already exists for $(RUN)"; \
	else \
		echo "[make] seeding outputs/worldsheet_polyakov/$(RUN)/params.json from run0001"; \
		cp "outputs/worldsheet_polyakov/run0001/params.json" "outputs/worldsheet_polyakov/$(RUN)/params.json"; \
		$(PYTHON) - <<'PY' \
import json, pathlib; \
p=pathlib.Path("outputs/worldsheet_polyakov/$(RUN)/params.json"); \
d=json.loads(p.read_text()); \
d["run_id"]="$(RUN)"; \
d["io"]["output_dir"]=f"outputs/worldsheet_polyakov/$(RUN)"; \
p.write_text(json.dumps(d, indent=2)); \
PY \
	; \
	fi

# Virtualenv convenience
venv:
	@echo "[make] creating venv at $(VENV_DIR)"
	@$(PYTHON) -m venv "$(VENV_DIR)"
	@echo "[make] installing requirements.txt"
	@"$(VENV_PY)" -m pip install --upgrade pip
	@"$(VENV_PY)" -m pip install -r requirements.txt
	@echo "[make] done. Use: source $(VENV_DIR)/bin/activate"

run: check-params
	@echo "[make] running: $(PYTHON) $(MODULE_DIR)/compute_K_dph.py $(PARAMS)"
	@$(PYTHON) "$(MODULE_DIR)/compute_K_dph.py" "$(PARAMS)"

# 1-command default run
run0001:
	@$(MAKE) run RUN=run0001

# Sweeps are controlled by params.json; this is just a semantic alias
sweep:
	@$(MAKE) run

# ---------- cleaning ----------
clean-outputs:
	@find outputs/worldsheet_polyakov -type f \( \
		-name "K_W*.npy" -o \
		-name "K_abs_W*.npy" -o \
		-name "K.npy" -o \
		-name "K_abs.npy" -o \
		-name "samples.npy" -o \
		-name "dph_plot*.png" -o \
		-name "sweeps_summary*.json" \
	\) -print -delete 2>/dev/null || true

clean-run:
	@if [[ ! -d "outputs/worldsheet_polyakov/$(RUN)" ]]; then \
		echo "[make] no such run dir: outputs/worldsheet_polyakov/$(RUN)"; \
		exit 2; \
	fi
	@echo "[make] cleaning generated artifacts for $(RUN) (keeping params.json)"
	@find "outputs/worldsheet_polyakov/$(RUN)" -type f \( \
		-name "K_W*.npy" -o \
		-name "K_abs_W*.npy" -o \
		-name "K.npy" -o \
		-name "K_abs.npy" -o \
		-name "samples.npy" -o \
		-name "dph_plot*.png" -o \
		-name "sweeps_summary*.json" \
	\) -print -delete 2>/dev/null || true

clean: clean-outputs
	@echo "[make] done."

clean-all: clean
	@echo "[make] resetting docs/project_log.md"
	@mkdir -p docs
	@echo "# Project Log" > docs/project_log.md
	@echo "" >> docs/project_log.md
	@echo "[make] done."

# ---- Safe run (systemd memory cap) ----
MEM_MAX ?= 8G
VENV_PY := .venv/bin/python
RUN_PY := $(if $(wildcard $(VENV_PY)),$(VENV_PY),$(PYTHON))

.PHONY: run-safe run0001-safe

run-safe: check-params
	@echo "[make] running (safe): systemd-run MemoryMax=$(MEM_MAX) $(RUN_PY) $(MODULE_DIR)/compute_K_dph.py $(PARAMS)"
	@systemd-run --user --scope -p MemoryMax=$(MEM_MAX) env MPLBACKEND=Agg \
		nice -n 10 ionice -c2 -n7 \
		$(RUN_PY) -u "$(MODULE_DIR)/compute_K_dph.py" "$(PARAMS)"

run0001-safe:
	@$(MAKE) run-safe RUN=run0001

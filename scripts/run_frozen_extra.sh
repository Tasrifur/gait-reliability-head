#!/usr/bin/env bash
# Plug-in head + energy prior on frozen GaitPart and GaitGL (GPU machine). Resumable: completed
# embedding variants/chunks are skipped; rerun the script after an interruption.
# Usage (from the project root): nohup bash scripts/run_frozen_extra.sh > logs/frozen_extra.out 2>&1 &
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=.venv/bin/python
mkdir -p logs results/p5_frozen results/p5_plugin
log() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a logs/frozen_extra.log; }

for bb in ${@:-gaitpart gaitgl}; do
    log "embed $bb"
    $PY -W ignore -m gaitlab.frozen_hybrid embed $bb >> logs/frozen_extra_$bb.log 2>&1 || { log "embed $bb FAILED"; exit 1; }
    log "repro $bb"
    $PY -W ignore -m gaitlab.frozen_hybrid repro $bb >> logs/frozen_extra_$bb.log 2>&1 || { log "repro $bb FAILED"; exit 1; }
    log "eval $bb"
    $PY -W ignore -m gaitlab.frozen_hybrid eval $bb >> logs/frozen_extra_$bb.log 2>&1 || { log "eval $bb FAILED"; exit 1; }
    log "final $bb"
    $PY -W ignore -m gaitlab.plugin_head final cuda $bb >> logs/frozen_extra_$bb.log 2>&1 || { log "final $bb FAILED"; exit 1; }
done
log 'all done'

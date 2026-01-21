#!/usr/bin/env bash
# =============================================================================
# Mamba Stateful Training Pipeline Runner
# =============================================================================
# This script orchestrates the complete training pipeline:
#   1. dagster_prep_families → Generate feature caches (via Dagster)
#   2. stage_a_selector.py   → Produce best family weights
#   3. run_stage_b_stateful_phase2.py → Train Mamba model
#
# Usage:
#   ./run_training_pipeline.sh                    # Run with defaults
#   ./run_training_pipeline.sh AAPL,MSFT,NVDA     # Specific symbols
#   ./run_training_pipeline.sh AAPL 63 100        # Symbol, horizon, trials
#
# Can be scheduled via cron:
#   0 2 * * * /home/rohan/Main\ mamba\ statefull/tools/run_training_pipeline.sh >> /var/log/mamba_training.log 2>&1
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"

# Python environment
PYTHON="${WORKSPACE_ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="$(which python3)"
fi

# Default parameters (can be overridden via args or env)
SYMBOLS="${1:-${MAMBA_SYMBOLS:-AAPL,MSFT,NVDA}}"
HORIZON="${2:-${MAMBA_HORIZON:-63}}"
OPTUNA_TRIALS="${3:-${MAMBA_OPTUNA_TRIALS:-100}}"

# Walk-forward settings
WF_START="${MAMBA_WF_START:-2010-07-02}"
WF_END="${MAMBA_WF_END:-2025-07-01}"
WF_TRAIN_YEARS="${MAMBA_WF_TRAIN_YEARS:-5}"
WF_STEP_DAYS="${MAMBA_WF_STEP_DAYS:-126}"

# Paths
CACHE_FEATURES="${WORKSPACE_ROOT}/cache/features"
ARTIFACTS_STAGE_A="${WORKSPACE_ROOT}/artifacts/stage_a"
ARTIFACTS_OPTUNA="${WORKSPACE_ROOT}/artifacts/optuna_studies"
ARTIFACTS_BACKTESTS="${WORKSPACE_ROOT}/artifacts/backtests"
LOG_DIR="${WORKSPACE_ROOT}/logs"

# Run ID for this execution
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/pipeline_${RUN_ID}.log"

# =============================================================================
# Helper Functions
# =============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" | tee -a "$LOG_FILE" >&2
}

ensure_dirs() {
    mkdir -p "$CACHE_FEATURES" "$ARTIFACTS_STAGE_A" "$ARTIFACTS_OPTUNA" "$ARTIFACTS_BACKTESTS" "$LOG_DIR"
}

# =============================================================================
# Pipeline Steps
# =============================================================================

run_dagster_prep_families() {
    local symbol="$1"
    log "=== STEP 1: dagster_prep_families for ${symbol} ==="
    
    local output_file="${CACHE_FEATURES}/${symbol}_h${HORIZON}_merged.parquet"
    local partition_key="symbol=${symbol}|horizon=${HORIZON}"
    
    # Run via Dagster job execute
    if DAGSTER_PREP_WF_START="$WF_START" \
       DAGSTER_PREP_WF_END="$WF_END" \
       DAGSTER_PREP_WF_TRAIN_YEARS="$WF_TRAIN_YEARS" \
       DAGSTER_PREP_WF_STEP_DAYS="$WF_STEP_DAYS" \
       DAGSTER_PREP_WRITE_MERGED="yes" \
       DAGSTER_PREP_MODE="stage-b" \
       "$PYTHON" -m dagster job execute \
        -m dagster_prep_families.definitions \
        -j prep_families_job \
        --partition "$partition_key" 2>&1 | tee -a "$LOG_FILE"; then
        
        if [[ -f "$output_file" ]]; then
            log "✓ dagster_prep_families SUCCESS: $output_file"
            return 0
        else
            log_error "dagster_prep_families ran but output not found: $output_file"
            return 1
        fi
    else
        log_error "dagster_prep_families FAILED for $symbol"
        return 1
    fi
}

run_stage_a_selector() {
    local symbol="$1"
    log "=== STEP 2: stage_a_selector.py for ${symbol} ==="
    
    local outdir="${ARTIFACTS_STAGE_A}/${symbol}_h${HORIZON}"
    local weights_file="${outdir}/family_weights_best.json"
    
    mkdir -p "$outdir"
    
    if "$PYTHON" "${WORKSPACE_ROOT}/tools/stage_a_selector.py" \
        --symbol "$symbol" \
        --horizon "$HORIZON" \
        --start "$WF_START" \
        --end "$WF_END" \
        --wf-start "$WF_START" \
        --wf-end "$WF_END" \
        --wf-train-years "$WF_TRAIN_YEARS" \
        --outdir "$outdir" \
        --universe-mode global \
        --selector-method legacy_optuna \
        --proxy-time-adaptive 2>&1 | tee -a "$LOG_FILE"; then
        
        if [[ -f "$weights_file" ]]; then
            log "✓ stage_a_selector SUCCESS: $weights_file"
            echo "$weights_file"
            return 0
        else
            log_error "stage_a_selector ran but weights not found: $weights_file"
            return 1
        fi
    else
        log_error "stage_a_selector FAILED for $symbol"
        return 1
    fi
}

run_stage_b_phase2() {
    local symbol="$1"
    local weights_path="$2"
    log "=== STEP 3: run_stage_b_stateful_phase2.py for ${symbol} ==="
    
    local study_name="phase2_v16_full_h${HORIZON}_${symbol}"
    local study_db="${ARTIFACTS_OPTUNA}/${study_name}__label_base.db"
    
    if "$PYTHON" "${WORKSPACE_ROOT}/tools/run_stage_b_stateful_phase2.py" \
        --symbol "$symbol" \
        --horizon "$HORIZON" \
        --search-mode full \
        --n-trials "$OPTUNA_TRIALS" \
        --study-name "$study_name" \
        --study-db "$study_db" \
        --phase2-family-weights-path "$weights_path" \
        --universe-mode full \
        --preset research \
        --train-start "$WF_START" \
        --train-end "$WF_END" 2>&1 | tee -a "$LOG_FILE"; then
        
        log "✓ stage_b_phase2 SUCCESS: $study_db"
        return 0
    else
        log_error "stage_b_phase2 FAILED for $symbol"
        return 1
    fi
}

# =============================================================================
# Main Pipeline
# =============================================================================

main() {
    log "========================================================"
    log "MAMBA STATEFUL TRAINING PIPELINE"
    log "========================================================"
    log "Run ID:      $RUN_ID"
    log "Symbols:     $SYMBOLS"
    log "Horizon:     $HORIZON"
    log "Trials:      $OPTUNA_TRIALS"
    log "WF Window:   $WF_START to $WF_END (train=${WF_TRAIN_YEARS}y, step=${WF_STEP_DAYS}d)"
    log "Python:      $PYTHON"
    log "Workspace:   $WORKSPACE_ROOT"
    log "Log File:    $LOG_FILE"
    log "========================================================"
    
    ensure_dirs
    
    # Track results
    declare -A RESULTS
    local total=0
    local success=0
    local failed=0
    
    # Process each symbol
    IFS=',' read -ra SYMBOL_ARRAY <<< "$SYMBOLS"
    for symbol in "${SYMBOL_ARRAY[@]}"; do
        symbol="$(echo "$symbol" | tr '[:lower:]' '[:upper:]' | xargs)"
        total=$((total + 1))
        
        log ""
        log "========================================================"
        log "Processing: $symbol ($total of ${#SYMBOL_ARRAY[@]})"
        log "========================================================"
        
        # Step 1: Dagster Prep Families
        if ! run_dagster_prep_families "$symbol"; then
            RESULTS[$symbol]="FAILED at dagster_prep_families"
            failed=$((failed + 1))
            continue
        fi
        
        # Step 2: Stage-A Selector
        weights_path=$(run_stage_a_selector "$symbol")
        if [[ $? -ne 0 ]] || [[ -z "$weights_path" ]]; then
            RESULTS[$symbol]="FAILED at stage_a_selector"
            failed=$((failed + 1))
            continue
        fi
        
        # Step 3: Stage-B Phase2 Training
        if ! run_stage_b_phase2 "$symbol" "$weights_path"; then
            RESULTS[$symbol]="FAILED at stage_b_phase2"
            failed=$((failed + 1))
            continue
        fi
        
        RESULTS[$symbol]="SUCCESS"
        success=$((success + 1))
    done
    
    # Summary
    log ""
    log "========================================================"
    log "PIPELINE COMPLETE"
    log "========================================================"
    log "Total:   $total"
    log "Success: $success"
    log "Failed:  $failed"
    log ""
    log "Results:"
    for symbol in "${!RESULTS[@]}"; do
        log "  $symbol: ${RESULTS[$symbol]}"
    done
    log "========================================================"
    
    # Return exit code based on results
    if [[ $failed -gt 0 ]]; then
        return 1
    fi
    return 0
}

# Run main
main "$@"

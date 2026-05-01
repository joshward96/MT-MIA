#!/usr/bin/env bash
# Generate RelDiff synthetic data for california, 3 seeds in parallel.
# Run from MT-MIA/ root:  bash scripts/generate_reldiff_california.sh

# Verify enough GPUs are visible
num_visible=$(nvidia-smi -L | wc -l)
echo "GPUs visible: $num_visible"
nvidia-smi -L

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RELDIFF_DIR="$ROOT_DIR/RelDiff"
DATASET="california"
SEEDS=(42 43 44)
GPUS=(0 1 2)

if [ "${#SEEDS[@]}" -ne "${#GPUS[@]}" ]; then
    echo "ERROR: SEEDS (${#SEEDS[@]}) and GPUS (${#GPUS[@]}) must match in length" >&2
    exit 1
fi

if [ "$num_visible" -lt "${#GPUS[@]}" ]; then
    echo "ERROR: requested ${#GPUS[@]} GPUs but only $num_visible visible" >&2
    exit 1
fi

echo "=== Preparing RelDiff input data ==="
cd "$ROOT_DIR"
python3 prepare_reldiff_data.py

cd "$RELDIFF_DIR"
export WANDB_MODE=offline
mkdir -p "$ROOT_DIR/logs"

train_and_sample() {
    set -euo pipefail
    local seed=$1
    local gpu=$2
    local run_id="_seed_${seed}"
    local ckpt_dir="ckpt/${DATASET}/multi${run_id}"
    local log="$ROOT_DIR/logs/${DATASET}_seed_${seed}.log"

    {
        echo "=== [seed=${seed} gpu=${gpu}] Training ==="
        date
        CUDA_VISIBLE_DEVICES=${gpu} python3 src/scripts/train_joint_diffusion.py ${DATASET} \
            --num-epochs 60000 \
            --batch-size 256 \
            --run-id "$run_id" \
            --config-path src/reldiff/configs/reldiff_config.toml

        echo "=== [seed=${seed}] Sampling best (EMA) checkpoint ==="
        CUDA_VISIBLE_DEVICES=${gpu} python3 src/scripts/sample_joint_diffusion.py ${DATASET} \
            --run-id "$run_id" \
            --num-samples 1

        out_best="$ROOT_DIR/synth_data/reldiff_synth/${DATASET}/seed_${seed}"
        mkdir -p "$out_best"
        cp "$RELDIFF_DIR/data/synthetic/${DATASET}/RelDiff/${run_id}/sample1"/*.csv "$out_best/"
        echo "Saved best -> $out_best"

        echo "=== [seed=${seed}] Sampling final checkpoint ==="
        local final_ckpt
        final_ckpt=$(ls -t "$ckpt_dir"/model_*.pt 2>/dev/null | head -1 || true)
        if [ -z "${final_ckpt:-}" ]; then
            echo "WARNING: no model_*.pt found in $ckpt_dir, skipping final"
        else
            CUDA_VISIBLE_DEVICES=${gpu} python3 src/scripts/sample_joint_diffusion.py ${DATASET} \
                --run-id "${run_id}_final" \
                --num-samples 1 \
                --ckpt "$final_ckpt"
            out_final="$ROOT_DIR/synth_data/reldiff_synth/${DATASET}/final/seed_${seed}"
            mkdir -p "$out_final"
            cp "$RELDIFF_DIR/data/synthetic/${DATASET}/RelDiff/${run_id}_final/sample1"/*.csv "$out_final/"
            echo "Saved final -> $out_final"
        fi

        echo "=== [seed=${seed}] Done at $(date) ==="
    } >> "$log" 2>&1
}

# Launch all seeds in parallel, one per GPU
pids=()
seeds_by_pid=()
for i in "${!SEEDS[@]}"; do
    seed="${SEEDS[$i]}"
    gpu="${GPUS[$i]}"
    train_and_sample "$seed" "$gpu" &
    pid=$!
    pids+=("$pid")
    seeds_by_pid+=("$seed")
    echo "Launched seed=$seed on GPU=$gpu (pid=$pid)"
done

# Wait for each job and report failures
fail=0
for idx in "${!pids[@]}"; do
    pid="${pids[$idx]}"
    seed="${seeds_by_pid[$idx]}"
    if wait "$pid"; then
        echo "Seed ${seed} completed successfully"
    else
        echo "ERROR: seed ${seed} (pid ${pid}) failed — see logs/${DATASET}_seed_${seed}.log" >&2
        fail=1
    fi
done

echo ""
if [ "$fail" -ne 0 ]; then
    echo "=== One or more seeds failed ==="
    exit 1
fi

echo "=== All seeds complete. Outputs in $ROOT_DIR/synth_data/reldiff_synth/${DATASET}/{best,final}/ ==="
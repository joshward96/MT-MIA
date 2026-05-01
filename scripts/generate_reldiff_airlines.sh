#!/usr/bin/env bash
# Generate RelDiff synthetic data for airlines — 3 seeds in parallel on 3 GPUs.
# Run from MT-MIA/ root:  bash scripts/generate_reldiff_airlines.sh

# Verify we have enough GPUs
NUM_GPUS=$(python3 -c 'import torch; print(torch.cuda.device_count())')
echo "✓ CUDA check passed: detected ${NUM_GPUS} GPU(s)"
for i in $(seq 0 $((NUM_GPUS - 1))); do
    echo "    GPU ${i}: $(python3 -c "import torch; print(torch.cuda.get_device_name(${i}))")"
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RELDIFF_DIR="$ROOT_DIR/RelDiff"
DATASET="airlines"
SEEDS=(42 43 44)
GPUS=(0 1 2)

if [ "${NUM_GPUS}" -lt "${#SEEDS[@]}" ]; then
    echo "ERROR: need ${#SEEDS[@]} GPUs, only have ${NUM_GPUS}." >&2
    exit 1
fi

echo "=== Preparing RelDiff input data ==="
cd "$ROOT_DIR"
python3 data_gen/prepare_reldiff_data.py

cd "$RELDIFF_DIR"
export WANDB_MODE=online
mkdir -p "$ROOT_DIR/logs"

train_and_sample() {
    local seed=$1
    local gpu=$2
    local run_id="_seed_${seed}"
    local ckpt_dir="ckpt/${DATASET}/multi${run_id}"
    local log="$ROOT_DIR/logs/${DATASET}_seed__final${seed}.log"

    {
        echo "=== [seed=${seed} gpu=${gpu}] Training ==="
        CUDA_VISIBLE_DEVICES=${gpu} python3 src/scripts/train_joint_diffusion.py ${DATASET} \
            --num-epochs 30000 \
            --batch-size 256 \
            --run-id "$run_id" \
            --config-path src/reldiff/configs/reldiff_config.toml

        echo "=== [seed=${seed}] Sampling best (EMA) checkpoint ==="
        CUDA_VISIBLE_DEVICES=${gpu} python3 src/scripts/sample_joint_diffusion.py ${DATASET} \
            --run-id "$run_id" \
            --num-samples 1 \

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

        echo "=== [seed=${seed}] Done ==="
    } >> "$log" 2>&1
}

# Launch all seeds concurrently, one per GPU
declare -A pids
gpu=0
for seed in "${SEEDS[@]}"; do
    echo ">>> Starting seed $seed on GPU $gpu (log: logs/${DATASET}_seed__final${seed}.log)"
    train_and_sample "$seed" "$gpu" &
    pids[$seed]=$!
    gpu=$((gpu + 1))
done

# Wait for all and collect exit codes
fail=0
for seed in "${SEEDS[@]}"; do
    if wait "${pids[$seed]}"; then
        echo ">>> Seed $seed completed successfully"
    else
        echo ">>> ERROR: seed $seed failed — see logs/${DATASET}_seed__final${seed}.log" >&2
        fail=1
    fi
done

echo ""
if [ "$fail" -ne 0 ]; then
    echo "=== One or more seeds failed ==="
    exit 1
fi

echo "=== All seeds complete. Outputs in $ROOT_DIR/synth_data/reldiff_synth/${DATASET}/{best,final}/ ==="
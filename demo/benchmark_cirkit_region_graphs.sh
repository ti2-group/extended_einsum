#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

OUTPUT=${OUTPUT:-results/cirkit_region_graph_benchmark_v3.csv}
REFERENCE_RESULTS=${REFERENCE_RESULTS:-results/cirkit_region_graph_benchmark_v2.csv}
DEVICE=${DEVICE:-auto}
DATASET=${DATASET:-mnist}
DATA_DIR=${DATA_DIR:-datasets}
NUM_SAMPLES=${NUM_SAMPLES:-0}
DRY_RUN=${DRY_RUN:-0}

readonly EPOCHS=15
readonly WARMUP_EPOCHS=5
readonly SHUFFLE_SEED=20260715

SEEDS=(0 1 2)
REGION_GRAPHS=(quad-tree-2 quad-graph)
BATCH_SIZES=(256 512)
CP_UNIT_SIZES=(64 128 256 512)
TUCKER_UNIT_SIZES=(32 64 128)

# Cirkit implements lse-sum. XE is measured with both of its stable modes.
BACKEND_CONFIGS=(
    "xe:lse-sum"
    "xe:scaled-max"
    "cirkit:lse-sum"
)

export PYTHONUNBUFFERED=1

TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

# CP and Cirkit are unchanged. Seed a new output with those existing rows, but
# deliberately exclude every XE Tucker row: both stable XE Tucker code paths
# changed and must be remeasured, including configurations that previously OOMed.
# Existing rows in OUTPUT always take precedence.
if [[ "$DRY_RUN" != "1" && -s "$REFERENCE_RESULTS" && "$REFERENCE_RESULTS" != "$OUTPUT" ]]; then
    uv run python - "$REFERENCE_RESULTS" "$OUTPUT" <<'PY'
import csv
import sys
from pathlib import Path

reference_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])

with reference_path.open(newline="") as reference_file:
    reference_reader = csv.DictReader(reference_file)
    fieldnames = reference_reader.fieldnames
    if not fieldnames:
        raise SystemExit(f"Reference CSV has no header: {reference_path}")
    reference_rows = [
        row
        for row in reference_reader
        if row["backend_type"] == "cirkit" or row["sum_product_layer"] == "cp"
    ]

output_rows = []
output_has_header = False
if output_path.exists() and output_path.stat().st_size:
    with output_path.open(newline="") as output_file:
        output_reader = csv.DictReader(output_file)
        if output_reader.fieldnames != fieldnames:
            raise SystemExit(f"Cannot reuse {reference_path}: its header differs from {output_path}")
        output_has_header = True
        output_rows = list(output_reader)

key_fields = (
    "backend_type",
    "region_graph",
    "sum_product_layer",
    "units",
    "batch_size",
    "semiring",
    "seed",
    "torch_compile",
    "status",
    "epoch",
)
existing_keys = {tuple(row[field] for field in key_fields) for row in output_rows}
rows_to_add = [row for row in reference_rows if tuple(row[field] for field in key_fields) not in existing_keys]

if rows_to_add:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        if not output_has_header:
            writer.writeheader()
        writer.writerows(rows_to_add)
    print(f"reused {len(rows_to_add)} unchanged CP/Cirkit rows from {reference_path}")
PY
fi

declare -A COMPLETED_CONFIGS=()
declare -A OOM_CONFIGS=()

if [[ -s "$OUTPUT" ]]; then
    uv run python - "$OUTPUT" "$EPOCHS" >"$TEMP_DIR/completed-configs" <<'PY'
import csv
import sys
from collections import defaultdict

output_path = sys.argv[1]
num_epochs = int(sys.argv[2])
required_fields = {
    "backend_type",
    "region_graph",
    "sum_product_layer",
    "units",
    "batch_size",
    "semiring",
    "seed",
    "torch_compile",
    "status",
    "epoch",
}

with open(output_path, newline="") as output_file:
    reader = csv.DictReader(output_file)
    missing_fields = required_fields - set(reader.fieldnames or ())
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise SystemExit(f"Cannot resume {output_path}: missing CSV fields: {missing}")

    epochs_by_configuration = defaultdict(set)
    for row in reader:
        if row["status"] != "ok" or row["torch_compile"].lower() not in {"1", "true"}:
            continue
        key = "|".join(
            (
                row["seed"],
                row["region_graph"],
                row["backend_type"],
                row["semiring"],
                row["sum_product_layer"],
                row["units"],
                row["batch_size"],
            )
        )
        epochs_by_configuration[key].add(int(row["epoch"]))

expected_epochs = set(range(num_epochs))
for key, epochs in epochs_by_configuration.items():
    if expected_epochs <= epochs:
        print(key)
PY

    while IFS= read -r key; do
        [[ -n "$key" ]] && COMPLETED_CONFIGS["$key"]=1
    done <"$TEMP_DIR/completed-configs"

    uv run python - "$OUTPUT" >"$TEMP_DIR/oom-configs" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as output_file:
    for row in csv.DictReader(output_file):
        if row["status"] == "ok" or "out of memory" not in row["error"].lower():
            continue
        print(
            "|".join(
                (
                    row["region_graph"],
                    row["backend_type"],
                    row["semiring"],
                    row["sum_product_layer"],
                    row["units"],
                    row["batch_size"],
                )
            )
        )
PY

    while IFS= read -r key; do
        [[ -n "$key" ]] && OOM_CONFIGS["$key"]=1
    done <"$TEMP_DIR/oom-configs"
fi

CONFIGURATIONS=()
for seed in "${SEEDS[@]}"; do
    for region_graph in "${REGION_GRAPHS[@]}"; do
        for backend_config in "${BACKEND_CONFIGS[@]}"; do
            backend=${backend_config%%:*}
            semiring=${backend_config#*:}
            for batch_size in "${BATCH_SIZES[@]}"; do
                for units in "${CP_UNIT_SIZES[@]}"; do
                    CONFIGURATIONS+=("$seed|$region_graph|$backend|$semiring|cp|$units|$batch_size")
                done
                for units in "${TUCKER_UNIT_SIZES[@]}"; do
                    CONFIGURATIONS+=("$seed|$region_graph|$backend|$semiring|tucker|$units|$batch_size")
                done
            done
        done
    done
done

# Use a fixed randomized order to reduce thermal and temporal bias while keeping
# the benchmark exactly reproducible and resumable.
printf '%s\n' "${CONFIGURATIONS[@]}" >"$TEMP_DIR/configurations"
uv run python - "$TEMP_DIR/configurations" "$SHUFFLE_SEED" >"$TEMP_DIR/shuffled-configurations" <<'PY'
import random
import sys

with open(sys.argv[1]) as configurations_file:
    configurations = [line.rstrip("\n") for line in configurations_file if line.strip()]
random.Random(int(sys.argv[2])).shuffle(configurations)
print("\n".join(configurations))
PY
mapfile -t CONFIGURATIONS <"$TEMP_DIR/shuffled-configurations"

total=${#CONFIGURATIONS[@]}
completed=${#COMPLETED_CONFIGS[@]}
known_oom=${#OOM_CONFIGS[@]}
echo "benchmark configurations: $total ($completed already complete)"
echo "known OOM configurations skipped for every seed: $known_oom"
echo "output: $OUTPUT"
echo "measured epochs: $EPOCHS; warmup epochs: $WARMUP_EPOCHS; torch.compile: always enabled"

progress=0
for key in "${CONFIGURATIONS[@]}"; do
    progress=$((progress + 1))
    if [[ -n "${COMPLETED_CONFIGS[$key]:-}" ]]; then
        echo "[$progress/$total] skipping completed $key"
        continue
    fi

    IFS='|' read -r seed region_graph backend semiring layer units batch_size <<<"$key"
    oom_key="$region_graph|$backend|$semiring|$layer|$units|$batch_size"
    if [[ -n "${OOM_CONFIGS[$oom_key]:-}" ]]; then
        echo "[$progress/$total] skipping known OOM $key"
        continue
    fi
    run_output="$TEMP_DIR/run-$progress.csv"
    command=(
        uv run python demo/cirkit.py
        --train
        --dataset "$DATASET"
        --data-dir "$DATA_DIR"
        --num-samples "$NUM_SAMPLES"
        --device "$DEVICE"
        --region-graph "$region_graph"
        --sum-product-layer "$layer"
        --unit-sizes "$units"
        --batch-sizes "$batch_size"
        --backends "$backend"
        --semiring "$semiring"
        --seed "$seed"
        --warmup-epochs "$WARMUP_EPOCHS"
        --epochs "$EPOCHS"
        --torch-compile
        --stop-on-error
        --verbose-errors
        --output "$run_output"
    )

    printf '[%d/%d] ' "$progress" "$total"
    printf '%q ' "${command[@]}"
    printf '\n'
    if [[ "$DRY_RUN" == "1" ]]; then
        continue
    fi

    set +e
    "${command[@]}"
    run_status=$?
    set -e

    if [[ -s "$run_output" ]]; then
        mkdir -p "$(dirname -- "$OUTPUT")"
        if [[ ! -s "$OUTPUT" ]]; then
            cp "$run_output" "$OUTPUT"
        else
            tail -n +2 "$run_output" >>"$OUTPUT"
        fi
    fi

    if [[ -s "$run_output" ]] && rg -qi 'out of memory' "$run_output"; then
        OOM_CONFIGS["$oom_key"]=1
        echo "[$progress/$total] recorded new OOM signature; remaining seeds will be skipped: $oom_key"
    fi

    if [[ "$run_status" -eq 0 ]]; then
        COMPLETED_CONFIGS["$key"]=1
    else
        echo "[$progress/$total] failed with exit status $run_status; continuing" >&2
    fi
done

echo "benchmark sweep finished; results are in $OUTPUT"

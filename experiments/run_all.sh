#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PUBLICATION_UV_CACHE="${EXTENDED_EINSUM_UV_CACHE:-/tmp/extended-einsum-uv-cache}"
PUBLICATION_MPL_CACHE="${EXTENDED_EINSUM_MPL_CACHE:-/tmp/extended-einsum-publication-matplotlib}"
UV_EXECUTABLE="${EXTENDED_EINSUM_UV_BIN:-/home/christoph/.local/bin/uv}"
mkdir -p "$PUBLICATION_MPL_CACHE"
export MPLCONFIGDIR="$PUBLICATION_MPL_CACHE"

"$UV_EXECUTABLE" --cache-dir "$PUBLICATION_UV_CACHE" run --group demo python experiments/speedup.py
"$UV_EXECUTABLE" --cache-dir "$PUBLICATION_UV_CACHE" run --group demo python experiments/ablation.py
"$UV_EXECUTABLE" --cache-dir "$PUBLICATION_UV_CACHE" run --group demo --with pyjuice==2.6.1 python experiments/pyjuice_cp_t/benchmark.py
"$UV_EXECUTABLE" --cache-dir "$PUBLICATION_UV_CACHE" run --group demo python experiments/monarch/benchmark.py
"$UV_EXECUTABLE" --cache-dir "$PUBLICATION_UV_CACHE" run --group demo python experiments/plot_speedup.py
"$UV_EXECUTABLE" --cache-dir "$PUBLICATION_UV_CACHE" run --group demo python experiments/plot_ablation.py
"$UV_EXECUTABLE" --cache-dir "$PUBLICATION_UV_CACHE" run --group demo python experiments/pyjuice_cp_t/plot.py
"$UV_EXECUTABLE" --cache-dir "$PUBLICATION_UV_CACHE" run --group demo python experiments/monarch/table.py

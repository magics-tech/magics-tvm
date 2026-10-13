#!/usr/bin/env bash
# Builds the magics-tvm wheel into dist/, running magics/build-wheel.sh in a
# podman or docker container (CONTAINER_ENGINE picks one).
#   magics/build.sh [--debug]    JOBS=N caps ninja; delete build/ to start clean
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=magics-tvm-build

engine=${CONTAINER_ENGINE:-$(basename "$(command -v podman || command -v docker || echo none)")}
[[ $engine == none ]] && { echo "build.sh needs podman or docker" >&2; exit 1; }

# Only the submodules the build reads; a recursive checkout is 528 MB.
git -C "$ROOT" submodule update --init --depth 1 \
  3rdparty/dlpack 3rdparty/dmlc-core 3rdparty/rang 3rdparty/libbacktrace 3rdparty/vta-hw

"$engine" build -t "$IMAGE" "$ROOT/magics"

# Rootless podman maps the container's root to you; docker needs --user.
user=()
[[ $engine == docker ]] && user=(--user "$(id -u):$(id -g)")
exec "$engine" run --rm "${user[@]}" --security-opt label=disable \
  -e HOME=/tmp -e JOBS="${JOBS:-}" -v "$ROOT:/src" -w /src \
  "$IMAGE" magics/build-wheel.sh "$@"

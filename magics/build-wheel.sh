#!/usr/bin/env bash
# Intended to run in the podman/docker container from magics/Dockerfile, at
# the repo root; magics/build.sh starts it. Builds the wheel into dist/.
#   magics/build-wheel.sh [--debug]
set -euo pipefail

# auditwheel fails the repair if libtvm.so needs a newer glibc than this.
PLAT=manylinux_2_27_x86_64

debug=OFF build_type=Release
if [[ "${1:-}" == --debug ]]; then debug=ON build_type=Debug; fi

# For the GIT_COMMIT_HASH stamp; the container user may not own the tree.
git config --global --add safe.directory "$PWD"

# Our settings appended to TVM's config.cmake; a later set() wins.
mkdir -p build
cp cmake/config.cmake build/config.cmake
cat >> build/config.cmake <<EOF
set(USE_MICRO ON)
set(USE_UMA ON)
set(USE_CMSISNN ON)
set(USE_LLVM "/usr/bin/llvm-config-14 --link-static")
set(USE_RELAY_DEBUG $debug)
EOF
CC=/usr/bin/gcc CXX=/usr/bin/g++ cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=$build_type
ninja -C build ${JOBS:+-j "$JOBS"}

# setup.py tags the wheel for one CPython, but libtvm.so is loaded with ctypes,
# so retag it py3-none; auditwheel then bundles libtinfo and sets the platform.
wheels=$(mktemp -d)
/opt/python/cp312-cp312/bin/python -m pip wheel --no-deps -w "$wheels" ./python
pipx run wheel tags --remove --python-tag py3 --abi-tag none "$wheels"/*.whl
rm -rf dist
auditwheel repair --plat "$PLAT" -w dist "$wheels"/*.whl
ls dist/*.whl

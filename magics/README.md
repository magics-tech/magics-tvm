# magics-tvm

Magics' fork of Apache TVM 0.14.0, for Magics internal use, released as the
`magics-tvm` wheel. The distribution is `magics-tvm`; the import name stays `tvm`.

Branch `magics/v0.14` is upstream `v0.14.0` plus our commits, so
`git log v0.14.0..magics/v0.14` is the list of what differs. Each release is a tag
`v0.14.0+magics.N` with the wheel attached to its GitHub release.

## Make a release

1. Commit the fix to `magics/v0.14`.
2. Bump the local label in `python/setup.py` (`__version__ += "+magics.N"`) and commit.
3. Tag and push; the annotation becomes the release notes:

   ```bash
   git tag -a v0.14.0+magics.N   # write the notes
   git push origin magics/v0.14 v0.14.0+magics.N
   ```

4. `.github/workflows/magics-release.yml` builds the wheel and creates the release.
   It fails if the tag and the wheel's version disagree.
5. Point projects that use it at the new wheel's URL, e.g. `magics-tvm` in uv's
   `[tool.uv.sources]`, and relock.

## Build locally

```bash
magics/build.sh            # or --debug; JOBS=4 caps ninja's parallelism
```

Needs podman or docker, nothing else; `CONTAINER_ENGINE` picks one. It checks out
the five submodules the build reads (shallow), builds the image from
`magics/Dockerfile`, then runs `magics/build-wheel.sh` in it, which builds
`build/libtvm.so` and writes the wheel to `dist/`. About 30 minutes from scratch; a second run is incremental. To try a local
change in a project before releasing it, install that wheel there, e.g.
`uv pip install dist/*.whl`.

## How it is built

- **LLVM is linked into `libtvm.so`** (`USE_LLVM` is `llvm-config-14 --link-static`),
  so nothing needs LLVM installed to use it. It cannot be dropped:
  `src/relay/transforms/fold_constant.cc` JIT-evaluates constant subgraphs with
  `Target("llvm")`, which every quantized model reaches through
  `FakeQuantizationToInteger`. LLVM 14 (EPEL), as TVM 0.14 supports up to 17.
- **Other settings:** `USE_MICRO`, `USE_UMA` and `USE_CMSISNN` on, appended to
  TVM's `cmake/config.cmake` by `build-wheel.sh`.
- **The image** is `manylinux_2_28` (AlmaLinux 8) with AlmaLinux's own gcc 8.5, not
  the image's gcc-toolset. Its glibc sets the floor of what it builds.
- **The wheel** is `py3-none`, since TVM loads `libtvm.so` through ctypes and has no
  extension module; one wheel serves every Python 3. `auditwheel` bundles
  `libtinfo` (pulled in by LLVM) into `magics_tvm.libs/`; `libz` comes from the
  system. Next to `libtvm.so` it carries `libtvm_runtime.so`, `standalone_crt/` and
  `microtvm_template_projects/`, which `tvm.micro` looks for there.

## OS support

`manylinux_2_27_x86_64`: glibc ≥ 2.27, GLIBCXX ≥ 3.4.22. That covers RHEL/Alma/Rocky
8 and 9, Ubuntu 18.04+, Debian 10+, current Fedora and Arch. `build-wheel.sh` asks
`auditwheel` for that tag, so a change that raises the floor fails the build
instead of shipping. `readelf -V build/libtvm.so` is the ground truth.

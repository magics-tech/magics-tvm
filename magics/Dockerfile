# Builds libtvm.so and the magics-tvm wheel; see build.sh.
# manylinux_2_28 is AlmaLinux 8 with EPEL, auditwheel and every CPython. Its
# glibc is the floor of everything built here.
FROM quay.io/pypa/manylinux_2_28_x86_64:2026.10.10-1

# gcc-c++ is AlmaLinux's own gcc 8.5, not the image's default gcc-toolset-14:
# it is what TVM 0.14 has been built and tested with. LLVM 14 is linked into
# libtvm.so statically, so nothing needs LLVM installed to use it.
RUN dnf install -y \
      gcc-c++ ninja-build \
      llvm14-devel llvm14-static \
      zlib-devel ncurses-devel libxml2-devel \
    && dnf clean all

# Release builder for the Linux binaries and MATLAB/Octave mex files.
#
# manylinux2014 is CentOS 7, glibc 2.17: anything linked here needs at most
# GLIBC_2.17 and runs on any distribution newer than that -- Ubuntu 14.04 on.
# Built on the GitHub ubuntu-22.04 runner instead, every binary picked up
# GLIBC_2.34 symbols (pthread and __libc_start_main moved into libc in 2.34)
# and would not start on Ubuntu 20.04 or older.
#
# glibc itself stays dynamic: a static glibc cannot dlopen libcuda.so or the
# OpenCL ICD loader, and a mex file has to share MATLAB's or Octave's libc.
# libstdc++, libgcc and libgomp are linked statically by the build scripts.
#
# Run with "docker run", not as a job container: the JavaScript actions
# (checkout, upload-artifact) need node 20, which needs glibc 2.28.

FROM quay.io/pypa/manylinux2014_x86_64

ARG WITH_CUDA=1
ARG WITH_LAZARUS=0

# gcc 8: the newest CUDA 10.2 supports, and new enough for everything else
RUN yum install -y devtoolset-8-gcc devtoolset-8-gcc-c++ \
        ocl-icd-devel opencl-headers upx wget which make zip xz vim-common \
    && yum clean all

ENV PATH=/opt/rh/devtoolset-8/root/usr/bin:${PATH}

# CUDA 10.2 toolkit, the version the releases have used; mcx links its runtime
# statically (BACKEND=cudastatic), so only the driver is needed at run time.
# The runfile's installer refuses to run in a container (it cannot find a libdir
# or check gcc), so the toolkit is unpacked from it instead
RUN if [ "${WITH_CUDA}" = "1" ]; then \
        wget -q https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run \
        && sh cuda_10.2.89_440.33.01_linux.run --noexec --target /tmp/cudarun \
        && rm -f cuda_10.2.89_440.33.01_linux.run \
        && mv /tmp/cudarun/builds/cuda-toolkit /usr/local/cuda \
        && rm -rf /tmp/cudarun /usr/local/cuda/nsight* /usr/local/cuda/libnvvp /usr/local/cuda/samples /usr/local/cuda/doc /usr/local/cuda/extras/CUPTI; \
    fi

ENV PATH=/usr/local/cuda/bin:${PATH}
ENV CUDA_PATH=/usr/local/cuda

# Free Pascal 3.2.2 and Lazarus 2.2.0 for MCX Studio: 2.2.0 is the version its
# form files are kept at (a newer IDE silently rewrites them), and gtk2 is the
# widget set it is built for.  lazbuild compiles the LCL into the Lazarus tree
# on first use, so the tree is left writable for a build run as the caller
RUN if [ "${WITH_LAZARUS}" = "1" ]; then \
        yum install -y gtk2-devel mesa-libGL-devel libX11-devel && yum clean all \
        && wget -q -O /tmp/fpc.tar https://downloads.sourceforge.net/project/freepascal/Linux/3.2.2/fpc-3.2.2.x86_64-linux.tar \
        && cd /tmp && tar xf fpc.tar && cd fpc-3.2.2.x86_64-linux \
        && printf '/usr/local\nn\nn\nn\n' | ./install.sh \
        && cd / && rm -rf /tmp/fpc* \
        && wget -q -O /tmp/lazarus.tar.gz "https://downloads.sourceforge.net/project/lazarus/Lazarus%20Zip%20_%20GZip/Lazarus%202.2.0/lazarus-2.2.0-0.tar.gz" \
        && tar xzf /tmp/lazarus.tar.gz -C /opt && rm -f /tmp/lazarus.tar.gz \
        && cd /opt/lazarus && make lazbuild \
        && ln -s /opt/lazarus/lazbuild /usr/local/bin/lazbuild \
        && chmod -R a+rwX /opt/lazarus; \
    fi

WORKDIR /src

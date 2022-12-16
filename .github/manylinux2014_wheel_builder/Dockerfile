FROM quay.io/pypa/manylinux2014_x86_64
RUN yum install -y wget devtoolset-7-gcc-c++ devtoolset-7-gcc && yum remove -y devtoolset-10-gcc-c++ devtoolset-10-gcc \
    && yum clean all
RUN wget https://developer.nvidia.com/compute/cuda/9.2/Prod2/local_installers/cuda_9.2.148_396.37_linux && \
    sh cuda_9.2.148_396.37_linux --toolkit --silent --override && \
    rm cuda_9.2.148_396.37_linux
ENV PATH /opt/rh/devtoolset-7/root/usr/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
ENV CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda/
COPY build_wheels.sh /
WORKDIR /src
ENTRYPOINT ["/build_wheels.sh"]
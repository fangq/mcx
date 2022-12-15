FROM quay.io/pypa/manylinux2014_x86_64
RUN yum install -y wget && yum clean all
RUN wget https://developer.nvidia.com/compute/cuda/9.2/Prod2/local_installers/cuda_9.2.148_396.37_linux && \
    sh cuda_9.2.148_396.37_linux --toolkit --silent && \
    rm cuda_9.2.148_396.37_linux
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
COPY build_wheels.sh /
WORKDIR /src
ENTRYPOINT ["/build_wheels.sh"]
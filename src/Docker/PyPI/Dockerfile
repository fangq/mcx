FROM quay.io/pypa/manylinux2014_x86_64
RUN yum install -y wget && yum clean all
RUN wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run && \
    sh cuda_11.8.0_520.61.05_linux.run --toolkit --silent && \
    rm cuda_11.8.0_520.61.05_linux.run
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
COPY build_wheels.sh /
WORKDIR /src
ENTRYPOINT ["/build_wheels.sh"]
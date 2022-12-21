FROM qemu

RUN pacman -Syu --noconfirm git xmlstarlet openssh sshpass
# Hackintosh Repository to run macOS
RUN cd / && git clone --depth 1 --recursive https://github.com/kholia/OSX-KVM
# libvirt schema of the macOS VM. Expects the OS disk to be mounted at /image. Removes the default installer disk from
# the schema as well
RUN cd OSX-KVM && sed "s/\/home\/CHANGEME//g; s/\/OSX-KVM\/mac_hdd_ng.img/\/image\/mac_hdd_ng.img/g" macOS-libvirt-Catalina.xml > MacOS-MCX.xml && \
    xmlstarlet ed -L -N w="http://libvirt.org/schemas/domain/qemu/1.0" -d '//disk[3]' MacOS-MCX.xml
COPY macos.sh /macos.sh
COPY build.zsh /build.zsh
CMD ["/macos.sh"]

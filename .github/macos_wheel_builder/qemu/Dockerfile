FROM archlinux

RUN pacman -Syu --noconfirm && yes y|pacman -S ebtables iproute2
RUN pacman -Syu --noconfirm && yes 1|pacman -S --noconfirm qemu-base qemu-block-gluster \
    qemu-block-iscsi libvirt bridge-utils \
    openbsd-netcat dmidecode virt-manager edk2-ovmf dnsmasq && \
    pacman -Scc --noconfirm

COPY start.sh /start.sh
ENTRYPOINT ["/start.sh"]

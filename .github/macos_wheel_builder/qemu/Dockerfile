FROM archlinux

RUN pacman -Syu --noconfirm && yes y|pacman -S ebtables iproute2
RUN pacman -Syu --noconfirm && yes 1|pacman -S --noconfirm qemu-full qemu-arch-extra qemu-block-gluster \
    qemu-block-iscsi samba libvirt bridge-utils \
    openbsd-netcat dmidecode spice-vdagent virt-manager edk2-ovmf dnsmasq sudo dmg2img && \
    pacman -Scc --noconfirm

COPY start.sh /start.sh
ENTRYPOINT ["/start.sh"]

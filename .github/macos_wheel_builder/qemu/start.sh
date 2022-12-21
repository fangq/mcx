#!/bin/bash
# Original script from jessfraz's KVM Dockerfile: https://github.com/jessfraz/dockerfiles/tree/master/kvm
set -e
set -o pipefail

if [ ! -e /dev/kvm ]; then 
    mknod /dev/kvm c 10 $(grep '\<kvm\>' /proc/misc | cut -f 1 -d' '); 
fi

# create the bridge for networking
ip link add name virt0 type bridge
ip link set dev virt0 up
bridge link
ip addr add dev virt0 172.20.0.1/16

libvirtd &
# start the virtlogd daemon
exec virtlogd --daemon &
# wait until all daemons are up
sleep 15s

# shellcheck disable=SC206
exec $@

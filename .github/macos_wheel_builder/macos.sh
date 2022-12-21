#!/bin/bash
# Give 20 seconds for the daemons to start up
sleep 20s
virsh --connect qemu:///system define /OSX-KVM/MacOS-MCX.xml
# startup the networking bridge
virsh net-start default
virsh start macOS
# Wait 20 seconds for the boot menu to show up
sleep 20s
# Key strokes to select the correct disk to boot from the boot menu
virsh send-key macOS --codeset usb 0x4f
virsh send-key macOS --codeset usb 0x28
# Wait for MacOS to boot
sleep 45s
VM_IP_ADDRESS=$(virsh net-dhcp-leases default | grep -ohE "[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}")
# Copy over the repo checked out by Github Actions located under /mcx/
sshpass -p 1234 scp -o "StrictHostKeyChecking=no"  -r /mcx/ user@$VM_IP_ADDRESS:/Users/user/ || true
sshpass -p 1234 ssh -tt user@$VM_IP_ADDRESS < /build.zsh
sshpass -p 1234 scp -r user@$VM_IP_ADDRESS:/Users/user/mcx/pmcx/dist/ /mcx/
virsh destroy macOS
virsh net-destroy default

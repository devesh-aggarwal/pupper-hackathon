#!/usr/bin/env bash
# The Ethernet cable between Pupper's Pi and the Jetson: a fixed address at each end.
#
#   jetson/link.sh pi up        on the Pi:      its Ethernet port is 10.10.10.2
#   jetson/link.sh jetson up    on the Jetson:  its Ethernet port is 10.10.10.1
#   jetson/link.sh pi down      on the Pi: back to Lab 1 (DHCP, a laptop sharing its internet by cable)
#   jetson/link.sh status       this end's address, and a ping to the other end
#
# NetworkManager keeps it across reboots (a connection called pupper-link, preferred over the
# DHCP one). No gateway and no DNS: the cable carries only Pupper's own traffic, so WiFi stays the
# way to your laptop (ssh pi@pupper5.local) and, during setup, to the internet.
# IFACE=eth1 jetson/link.sh ...   picks the port (default: the first Ethernet device).
set -euo pipefail
NAME=pupper-link
PI_ADDR=10.10.10.2
JETSON_ADDR=10.10.10.1
IFACE=${IFACE:-$(nmcli -t -f DEVICE,TYPE device | awk -F: '$2=="ethernet"{print $1; exit}')}
[ -n "$IFACE" ] || { echo "no Ethernet device found (nmcli device)"; exit 1; }

case "${1:-}" in
  pi) ME=$PI_ADDR; PEER=$JETSON_ADDR ;;
  jetson) ME=$JETSON_ADDR; PEER=$PI_ADDR ;;
  status)
    ip -4 -brief addr show "$IFACE"
    for peer in $PI_ADDR $JETSON_ADDR; do
      ip -4 addr show "$IFACE" | grep -q "inet $peer/" && continue  # that's us
      ping -c 2 -W 1 "$peer" >/dev/null && echo "the other end ($peer) answers" || echo "no answer from $peer"
    done
    exit 0 ;;
  *) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac

case "${2:-up}" in
  up)
    sudo nmcli connection delete "$NAME" >/dev/null 2>&1 || true
    sudo nmcli connection add type ethernet ifname "$IFACE" con-name "$NAME" \
      ipv4.method manual ipv4.addresses "$ME/24" ipv4.never-default yes ipv6.method link-local \
      connection.autoconnect yes connection.autoconnect-priority 100 >/dev/null
    sudo nmcli connection up "$NAME" >/dev/null
    echo "$IFACE is $ME; the other end should be $PEER (jetson/link.sh status)" ;;
  down)
    sudo nmcli connection delete "$NAME" >/dev/null 2>&1 || true
    sudo nmcli device connect "$IFACE" >/dev/null 2>&1 || true
    echo "$IFACE is back to DHCP" ;;
  *) echo "up or down"; exit 2 ;;
esac

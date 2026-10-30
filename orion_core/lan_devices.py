"""What is on the user's own network, the way a person means "scan my network".

On 3 Oct the owner asked ORION to "scan the network real quick". The only
scanning tool was security_recon's nmap port scan, which needs an authorised
target and a separately installed binary, so ORION asked three clarifying
questions and then said it could not. A person at home asking that wants to
know which devices are connected.

This answers it passively: Windows already keeps a neighbour table of the
devices this PC has exchanged traffic with (``arp -a``). Reading it sends
nothing on the network, so it needs no authorisation. A device that has been
silent for a while may be missing, and the reply says so.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import subprocess
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass

_INTERFACE = re.compile(r"^\S.*?(\d{1,3}(?:\.\d{1,3}){3})\s+---")
_ENTRY = re.compile(r"^\s+(\d{1,3}(?:\.\d{1,3}){3})\s+([0-9a-fA-F]{2}(?:[-:][0-9a-fA-F]{2}){5})\s+(\S+)")

#: Seconds allowed for looking up every device's name, all together.
NAME_BUDGET_S = 1.5


@dataclass
class Device:
    ip: str
    mac: str
    interface: str
    name: str = ""

    @property
    def private_mac(self) -> bool:
        """A randomised (locally administered) address: phones and tablets
        use these on Wi-Fi, so it is the best hint available without a
        vendor database."""
        return int(self.mac[1], 16) & 0x2 == 0x2


def parse_arp(text: str) -> list[Device]:
    """Devices from ``arp -a`` output. Multicast, broadcast and unassigned
    entries are not devices and are left out."""
    devices: list[Device] = []
    interface = ""
    for line in str(text or "").splitlines():
        head = _INTERFACE.match(line)
        if head:
            interface = head.group(1)
            continue
        entry = _ENTRY.match(line)
        if not entry or not interface:
            continue
        ip, mac = entry.group(1), entry.group(2).lower().replace(":", "-")
        try:
            address = ipaddress.ip_address(ip)
        except ValueError:
            continue
        # A subnet's broadcast address carries the all-ones MAC, so that test
        # covers it without guessing the subnet size (this PC's is not a /24).
        if (address.is_multicast or address.is_unspecified
                or mac in {"ff-ff-ff-ff-ff-ff", "00-00-00-00-00-00"}):
            continue
        devices.append(Device(ip, mac, interface))
    return devices


def _read_table() -> str:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    done = subprocess.run(["arp", "-a"], capture_output=True, text=True,
                          timeout=10, creationflags=flags)
    return done.stdout or ""


def _name_devices(devices: list[Device], budget_s: float = NAME_BUDGET_S) -> None:
    """Reverse-DNS names, all looked up at once within one shared budget."""
    if not devices:
        return
    pool = ThreadPoolExecutor(max_workers=min(16, len(devices)))
    futures = {pool.submit(socket.gethostbyaddr, d.ip): d for d in devices}
    finished, _late = wait(futures, timeout=budget_s)
    for future in finished:
        try:
            futures[future].name = str(future.result()[0]).split(".")[0]
        except Exception:
            pass
    pool.shutdown(wait=False, cancel_futures=True)


def describe(read_table=_read_table, name_devices=_name_devices) -> str:
    """A spoken-length summary of the devices on this PC's networks."""
    try:
        devices = parse_arp(read_table())
    except Exception as exc:
        return f"I couldn't read this PC's network neighbour table ({exc})."
    if not devices:
        return ("I can't see any other devices on your network right now. This check is "
                "passive, so devices that have been quiet may not show up.")
    name_devices(devices)
    by_network: dict[str, list[Device]] = {}
    for device in devices:
        by_network.setdefault(device.interface, []).append(device)
    parts = []
    for interface, group in by_network.items():
        group.sort(key=lambda d: tuple(int(p) for p in d.ip.split(".")))
        lines = []
        for device in group:
            label = device.name or ("probably a phone or tablet (private address)"
                                    if device.private_mac else "unnamed")
            if device.ip.endswith(".1") or device.ip.endswith(".254"):
                label += ", likely your router"
            lines.append(f"{device.ip} - {label}")
        parts.append(f"On the network this PC joins as {interface}, I can see "
                     f"{len(group)} device{'s' if len(group) != 1 else ''}: "
                     + "; ".join(lines) + ".")
    return (" ".join(parts) + " This is passive - it lists devices this PC has "
            "talked to recently, so one that has been quiet may be missing.")


__all__ = ["Device", "parse_arp", "describe"]

"""
Enterprise-Scale SDN Topology
Simulates a data center / campus network with:
- 3-tier architecture: Core → Aggregation → Access
- 2 Core switches
- 4 Aggregation switches
- 8 Access switches
- 4 hosts per access switch = 32 hosts
- Multiple redundant paths for failover testing
"""

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink
import sys


class EnterpriseTopo(Topo):
    """
    3-tier Data Center topology:

    Core Layer:      c1 ────────────── c2
                    /  \\             /  \\
    Aggr Layer:   a1   a2          a3   a4
                 / \\ / \\         / \\ / \\
    Access:    sw1 sw2 sw3 sw4  sw5 sw6 sw7 sw8
               ||||||||||||||||||||||||||||||||
    Hosts:    h1-h4 h5-h8 ...              h29-h32
    """

    def build(self, hosts_per_access=4, core_bw=1000, aggr_bw=100, access_bw=10):
        """
        Args:
            hosts_per_access: number of hosts per access switch (default 4 → 32 total)
            core_bw: core link bandwidth in Mbps
            aggr_bw: aggregation link bandwidth
            access_bw: access (host) link bandwidth
        """
        core_link = dict(bw=core_bw, delay='1ms', use_htb=True)
        aggr_link = dict(bw=aggr_bw, delay='2ms', use_htb=True)
        access_link = dict(bw=access_bw, delay='5ms', use_htb=True)

        # ── Core Layer ──────────────────────────────────────────────
        c1 = self.addSwitch('c1', dpid='0000000000000001')
        c2 = self.addSwitch('c2', dpid='0000000000000002')
        self.addLink(c1, c2, **core_link)  # Core interconnect (redundant)
        self.addLink(c1, c2, **core_link)  # Second core link (for bonding)

        # ── Aggregation Layer ────────────────────────────────────────
        aggr_switches = []
        for i in range(1, 5):
            sw = self.addSwitch(f'a{i}', dpid=f'000000000000000{i+2}')
            aggr_switches.append(sw)

        # Connect aggr to both cores (redundant uplinks)
        for i, a in enumerate(aggr_switches):
            self.addLink(c1, a, **core_link)
            self.addLink(c2, a, **core_link)

        # Aggr cross-links (intra-pod redundancy)
        self.addLink(aggr_switches[0], aggr_switches[1], **aggr_link)
        self.addLink(aggr_switches[2], aggr_switches[3], **aggr_link)

        # ── Access Layer ─────────────────────────────────────────────
        access_switches = []
        for i in range(1, 9):
            sw = self.addSwitch(f'sw{i}', dpid=f'00000000000000{i+6:02d}')
            access_switches.append(sw)

        # Connect access to aggregation (each access connects to 2 aggr = dual-homed)
        # sw1, sw2 → a1, a2
        for i in range(0, 2):
            self.addLink(access_switches[i], aggr_switches[0], **aggr_link)
            self.addLink(access_switches[i], aggr_switches[1], **aggr_link)
        # sw3, sw4 → a1, a2
        for i in range(2, 4):
            self.addLink(access_switches[i], aggr_switches[0], **aggr_link)
            self.addLink(access_switches[i], aggr_switches[1], **aggr_link)
        # sw5, sw6 → a3, a4
        for i in range(4, 6):
            self.addLink(access_switches[i], aggr_switches[2], **aggr_link)
            self.addLink(access_switches[i], aggr_switches[3], **aggr_link)
        # sw7, sw8 → a3, a4
        for i in range(6, 8):
            self.addLink(access_switches[i], aggr_switches[2], **aggr_link)
            self.addLink(access_switches[i], aggr_switches[3], **aggr_link)

        # ── Hosts ─────────────────────────────────────────────────────
        host_count = 0
        for sw_idx, sw in enumerate(access_switches):
            subnet = sw_idx + 1
            for j in range(1, hosts_per_access + 1):
                host_count += 1
                h = self.addHost(
                    f'h{host_count}',
                    ip=f'10.{subnet}.0.{j}/24',
                    defaultRoute=f'via 10.{subnet}.0.254'
                )
                self.addLink(h, sw, **access_link)

        info(f"[EnterpriseTopo] Built: 2 core, 4 aggr, 8 access, {host_count} hosts\n")


class LargeScaleTopo(Topo):
    """
    Large-scale flat topology for stress testing.
    30 switches, 90 hosts (3 per switch).
    """

    def build(self, n_switches=30, hosts_per_switch=3, bandwidth=100):
        link_opts = dict(bw=bandwidth, delay='1ms', use_htb=True)

        switches = []
        for i in range(1, n_switches + 1):
            sw = self.addSwitch(f's{i}')
            switches.append(sw)

        # Create a ring + cross-links for redundancy
        for i in range(n_switches):
            next_i = (i + 1) % n_switches
            self.addLink(switches[i], switches[next_i], **link_opts)

        # Add cross-links every 5 switches
        for i in range(0, n_switches, 5):
            j = (i + n_switches // 2) % n_switches
            if not self.hasLink(switches[i], switches[j]):
                self.addLink(switches[i], switches[j], **link_opts)

        # Add hosts
        host_count = 0
        for sw in switches:
            for j in range(1, hosts_per_switch + 1):
                host_count += 1
                h = self.addHost(f'h{host_count}', ip=f'10.0.{host_count // 255}.{host_count % 255}/16')
                self.addLink(h, sw, **link_opts)

    def hasLink(self, src, dst):
        """Check if link already exists."""
        return self.g.has_edge(src, dst) or self.g.has_edge(dst, src)


def run_enterprise(controller_ip='127.0.0.1', controller_port=6633):
    topo = EnterpriseTopo(hosts_per_access=4)
    net = Mininet(
        topo=topo,
        controller=lambda name: RemoteController(name, ip=controller_ip, port=controller_port),
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=True,
        waitConnected=True
    )
    net.start()
    info("\n*** Enterprise 3-tier topology started ***\n")
    info("*** Core: c1, c2 | Aggregation: a1-a4 | Access: sw1-sw8 ***\n")
    info(f"*** Hosts: h1-h32 ***\n")
    CLI(net)
    net.stop()


if __name__ == '__main__':
    setLogLevel('info')
    run_enterprise()

topos = {
    'enterprise': EnterpriseTopo,
    'large': LargeScaleTopo,
}


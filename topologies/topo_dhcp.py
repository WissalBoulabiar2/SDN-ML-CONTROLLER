"""
topo_dhcp.py — Looped Packet Topology with DHCP + DHCP Snooping
================================================================
- Hosts get addresses via DHCP (no static IPs)
- h1 on s2 acts as the LEGITIMATE DHCP server (trusted port)
- All other host ports are UNTRUSTED (rogue DHCP blocked by controller)

Topology:
     s1
    /  \\
   s2---s3
   |    |
  h1   h2   (h1 = DHCP server)
  h3 on s1
"""

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch, Host
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink
import time
import sys


class PktTopoWithLoop(Topo):
    """
    Triangle loop topology with DHCP-enabled hosts.

         s1
        /  \\
       s2---s3
       |    |
      h1   h2
      (DHCP srv)

    h3 directly on s1.
    All hosts use DHCP (no static IP at build time).
    """

    def build(self, bandwidth=100, delay='2ms'):
        link_opts = dict(bw=bandwidth, delay=delay, use_htb=True)

        s1 = self.addSwitch('s1')
        s2 = self.addSwitch('s2')
        s3 = self.addSwitch('s3')

        # No static IPs — addresses will be assigned by DHCP at runtime
        h1 = self.addHost('h1', ip=None)   # DHCP server (will get static inside run())
        h2 = self.addHost('h2', ip=None)
        h3 = self.addHost('h3', ip=None)

        # Loop: s1-s2-s3-s1
        self.addLink(s1, s2, **link_opts)
        self.addLink(s2, s3, **link_opts)
        self.addLink(s3, s1, **link_opts)

        # Hosts
        self.addLink(h1, s2, **link_opts)
        self.addLink(h2, s3, **link_opts)
        self.addLink(h3, s1, **link_opts)


class MeshTopo(Topo):
    """Full mesh of 4 switches with DHCP hosts."""

    def build(self, bandwidth=100, delay='1ms'):
        link_opts = dict(bw=bandwidth, delay=delay, use_htb=True)

        switches = [self.addSwitch(f's{i}') for i in range(1, 5)]
        # h1 = DHCP server, others = DHCP clients
        hosts = [self.addHost(f'h{i}', ip=None) for i in range(1, 5)]

        for h, s in zip(hosts, switches):
            self.addLink(h, s, **link_opts)

        for i in range(len(switches)):
            for j in range(i + 1, len(switches)):
                self.addLink(switches[i], switches[j], **link_opts)


# ── DHCP server setup (dnsmasq) ───────────────────────────────────────────────

def start_dhcp_server(host, iface, subnet='10.0.0', server_ip='10.0.0.1',
                      range_start='10.0.0.100', range_end='10.0.0.200'):
    """
    Start dnsmasq as DHCP server on `host`.
    The host gets a static management IP so dnsmasq can bind to it.
    """
    info(f'\n*** Starting DHCP server on {host.name} ({iface}) ***\n')

    # Assign static IP to the server host itself
    host.cmd(f'ip addr flush dev {iface}')
    host.cmd(f'ip addr add {server_ip}/24 dev {iface}')
    host.cmd(f'ip link set {iface} up')

    # Write dnsmasq config
    conf = (
        f'interface={iface}\n'
        f'bind-interfaces\n'
        f'dhcp-range={range_start},{range_end},12h\n'
        f'dhcp-option=3,{server_ip}\n'   # default gateway = server itself
        f'log-dhcp\n'
        f'no-resolv\n'
    )
    host.cmd(f'echo "{conf}" > /tmp/dnsmasq_stp.conf')
    host.cmd('pkill dnsmasq 2>/dev/null; sleep 0.2')
    host.cmd('dnsmasq --conf-file=/tmp/dnsmasq_stp.conf --pid-file=/tmp/dnsmasq_stp.pid')
    info(f'  DHCP server running: {server_ip}  pool {range_start}-{range_end}\n')


def dhcp_clients(hosts):
    """Run dhclient on all client hosts (non-blocking, background)."""
    info('\n*** DHCP clients requesting addresses ***\n')
    for h in hosts:
        iface = h.defaultIntf()
        h.cmd(f'ip addr flush dev {iface}')
        h.cmd(f'dhclient -v {iface} &')
    # Give them a moment to get leases
    info('  Waiting for DHCP leases (5s)...\n')
    time.sleep(5)
    for h in hosts:
        addr = h.cmd(f'ip addr show {h.defaultIntf()} | grep "inet " | awk \'{{print $2}}\'').strip()
        info(f'  {h.name}: {addr if addr else "NO LEASE"}\n')


# ── DHCP Snooping: mark trusted/untrusted ports via REST ─────────────────────

def configure_dhcp_snooping(net, trusted_dpid, trusted_port,
                             controller_ip='127.0.0.1', controller_port=8080):
    """
    Inform the STP controller which port is the trusted DHCP server port.
    Uses the REST API exposed by controller_stp.py.

    Trusted port  → DHCP responses allowed
    All others    → DHCP responses blocked (BPDU Guard analog for DHCP)
    """
    import urllib.request
    base = f'http://{controller_ip}:{controller_port}'

    info(f'\n*** Configuring DHCP Snooping via REST ***\n')
    info(f'  Trusted DHCP port: dpid={trusted_dpid} port={trusted_port}\n')

    url = f'{base}/stp/dhcp-snooping/trust/{trusted_dpid}/{trusted_port}'
    try:
        req = urllib.request.Request(url, method='POST')
        with urllib.request.urlopen(req, timeout=3) as resp:
            info(f'  Controller response: {resp.read().decode()}\n')
    except Exception as e:
        info(f'  [WARN] Could not reach controller REST API: {e}\n')
        info(f'  Make sure controller_stp.py is running before starting topology.\n')


# ── Main ──────────────────────────────────────────────────────────────────────

def run_loop(controller_ip='127.0.0.1', controller_port=6633):
    topo = PktTopoWithLoop()
    net = Mininet(
        topo=topo,
        controller=lambda name: RemoteController(
            name, ip=controller_ip, port=controller_port),
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=True,
        waitConnected=True,
    )
    net.start()
    info('\n*** Loop topology with STP + DHCP Snooping ***\n')
    info('*** Topology: s1-s2-s3-s1 (triangle) ***\n')

    h1, h2, h3 = net.get('h1', 'h2', 'h3')

    # h1 on s2 = legitimate DHCP server (port 1 of s2 by default with autoSetMacs)
    start_dhcp_server(h1, iface='h1-eth0')

    # All other hosts = DHCP clients
    dhcp_clients([h2, h3])

    # Tell the controller which port is trusted for DHCP
    # s2 is datapath 2 (Mininet numbers switches from 1); h1 connects to port 1 of s2
    configure_dhcp_snooping(net,
                             trusted_dpid=2,
                             trusted_port=1,
                             controller_ip=controller_ip,
                             controller_port=8080)

    info('\n*** REST API (on controller) ***\n')
    info('  GET  http://localhost:8080/stp/status\n')
    info('  GET  http://localhost:8080/stp/security/alerts\n')
    info('  GET  http://localhost:8080/dhcp-snooping/status\n')
    info('\n')
    CLI(net)
    net.stop()


if __name__ == '__main__':
    setLogLevel('info')
    run_loop()

topos = {
    'loop': PktTopoWithLoop,
    'mesh': MeshTopo,
}

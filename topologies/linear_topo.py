"""
Linear Topology - k switches, 1 host per switch
Usage: sudo mn --custom topologies/linear_topo.py --topo linear,4
       or: sudo python topologies/linear_topo.py
"""

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink
import sys


class LinearTopo(Topo):
    """
    Linear topology: s1-s2-s3-...-sk, each with 1 host.
    h1---s1---s2---s3---...---sk---hk
    """

    def build(self, k=4, bandwidth=100, delay='1ms', loss=0):
        """
        Args:
            k: number of switches
            bandwidth: link bandwidth in Mbps
            delay: propagation delay (e.g., '1ms', '5ms')
            loss: packet loss rate (0-100)
        """
        link_opts = dict(bw=bandwidth, delay=delay, loss=loss, use_htb=True)

        switches = []
        for i in range(1, k + 1):
            sw = self.addSwitch(f's{i}')
            host = self.addHost(f'h{i}', ip=f'10.0.0.{i}/24')
            self.addLink(host, sw, **link_opts)
            switches.append(sw)

        # Chain switches
        for i in range(len(switches) - 1):
            self.addLink(switches[i], switches[i + 1], **link_opts)


def run(k=4, controller_ip='127.0.0.1', controller_port=6633):
    topo = LinearTopo(k=k)
    net = Mininet(
        topo=topo,
        controller=lambda name: RemoteController(name, ip=controller_ip, port=controller_port),
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=True,
        waitConnected=True
    )

    net.start()
    info(f"\n*** Linear topology: {k} switches, {k} hosts ***\n")
    info("*** Hosts:\n")
    for host in net.hosts:
        info(f"    {host.name}: {host.IP()}\n")
    info("\n*** Starting CLI (type 'help' for commands) ***\n")
    CLI(net)
    net.stop()


if __name__ == '__main__':
    setLogLevel('info')
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    run(k=k)


topos = {'linear': LinearTopo}


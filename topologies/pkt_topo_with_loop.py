"""
Looped Packet Topology - for STP testing and loop detection.
Creates a topology with intentional loops that STP must break.
"""

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink
import sys


class PktTopoWithLoop(Topo):
    """
    Triangle loop topology:
         s1
        /  \\
       s2---s3
       |    |
       h1   h2

    Also adds h3 directly to s1.
    """

    def build(self, bandwidth=100, delay='2ms'):
        link_opts = dict(bw=bandwidth, delay=delay, use_htb=True)

        s1 = self.addSwitch('s1')
        s2 = self.addSwitch('s2')
        s3 = self.addSwitch('s3')

        h1 = self.addHost('h1', ip='10.0.0.1/24')
        h2 = self.addHost('h2', ip='10.0.0.2/24')
        h3 = self.addHost('h3', ip='10.0.0.3/24')

        # Loop: s1-s2-s3-s1
        self.addLink(s1, s2, **link_opts)
        self.addLink(s2, s3, **link_opts)
        self.addLink(s3, s1, **link_opts)

        # Hosts
        self.addLink(h1, s2, **link_opts)
        self.addLink(h2, s3, **link_opts)
        self.addLink(h3, s1, **link_opts)


class MeshTopo(Topo):
    """
    Full mesh of 4 switches with multiple hosts.
    Tests complex STP scenarios and ML routing with multiple paths.
    """

    def build(self, bandwidth=100, delay='1ms'):
        link_opts = dict(bw=bandwidth, delay=delay, use_htb=True)

        switches = [self.addSwitch(f's{i}') for i in range(1, 5)]
        hosts = [self.addHost(f'h{i}', ip=f'10.0.0.{i}/24') for i in range(1, 5)]

        # Attach hosts
        for h, s in zip(hosts, switches):
            self.addLink(h, s, **link_opts)

        # Full mesh (creates many loops)
        for i in range(len(switches)):
            for j in range(i + 1, len(switches)):
                self.addLink(switches[i], switches[j], **link_opts)


def run_loop(controller_ip='127.0.0.1', controller_port=6633):
    topo = PktTopoWithLoop()
    net = Mininet(
        topo=topo,
        controller=lambda name: RemoteController(name, ip=controller_ip, port=controller_port),
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=True,
        waitConnected=True
    )
    net.start()
    info("\n*** Loop topology with STP ***\n")
    info("*** Topology: s1-s2-s3-s1 (triangle) ***\n")
    CLI(net)
    net.stop()


if __name__ == '__main__':
    setLogLevel('info')
    run_loop()

topos = {
    'loop': PktTopoWithLoop,
    'mesh': MeshTopo,
}

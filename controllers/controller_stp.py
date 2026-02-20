from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet
from ryu.lib import stplib
import logging

# Set up logging to see STP transitions clearly
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StandaloneSTController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {'stplib': stplib.Stp}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stp = kwargs['stplib']
        self.mac_to_port = {}
        
        # FIX 1: Faster convergence timers. 
        # Default STP takes 30s to reach FORWARD. These settings reduce it significantly.
        config = {
            'bridge': {
                'hello_time': 1,
                'forward_delay': 2,  # Shortened from 15s to 2s
                'max_age': 10
            }
        }
        self.stp.set_config(config)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # Install table-miss flow entry
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self._add_flow(datapath, 0, match, actions)

    @set_ev_cls(stplib.EventPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """
        FIX 2: Ensure we only handle packets when stplib allows it.
        stplib.EventPacketIn only fires for ports in LEARNING or FORWARDING states.
        """
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        dst = eth.dst
        src = eth.src
        dpid = datapath.id

        self.mac_to_port.setdefault(dpid, {})

        # Learn the source MAC
        self.mac_to_port[dpid][src] = in_port

        # Determine out_port
        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        # Install flow to avoid PacketIn for known destinations
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst)
            self._add_flow(datapath, 1, match, actions)

        # Send PacketOut
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)

    @set_ev_cls(stplib.EventTopologyChange, MAIN_DISPATCHER)
    def topology_change_handler(self, ev):
        # FIX 3: Clear MAC table on topology change to force re-learning
        dpid = ev.dp.id
        logger.info(f"[STP] Topology change detected on DPID {dpid}. Clearing MAC table.")
        if dpid in self.mac_to_port:
            self.mac_to_port[dpid] = {}

    @set_ev_cls(stplib.EventPortStateChange, MAIN_DISPATCHER)
    def port_state_change_handler(self, ev):
        states = {
            stplib.PORT_STATE_DISABLE: 'DISABLE',
            stplib.PORT_STATE_BLOCK: 'BLOCK',
            stplib.PORT_STATE_LISTEN: 'LISTEN',
            stplib.PORT_STATE_LEARN: 'LEARN',
            stplib.PORT_STATE_FORWARD: 'FORWARD',
        }
        logger.info("[STP] DPID %016x Port %d -> %s", 
                    ev.dp.id, ev.port_no, states.get(ev.port_state, 'UNKNOWN'))

    def _add_flow(self, datapath, priority, match, actions):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                match=match, instructions=inst)
        datapath.send_msg(mod)

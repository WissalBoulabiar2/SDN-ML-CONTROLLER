"""
ARP Handling + REST API Controller
Exposes controller state, IDS alerts, topology, and manual controls via HTTP REST.
"""

from ryu.app.wsgi import ControllerBase, WSGIApplication, route
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib.packet import packet, arp, ethernet
import json
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

# REST API URL prefix
REST_PREFIX = '/api/v1'


class ARPRestController(app_manager.RyuApp):
    """
    REST API exposure for the SDN ML controller.
    Provides endpoints for monitoring, manual control, and IDS management.
    """
    _CONTEXTS = {'wsgi': WSGIApplication}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        wsgi = kwargs['wsgi']
        wsgi.register(RestAPI, {'controller_app': self})
        self.topology_data = {}
        self.alerts = []         # Recent IDS alerts
        self.max_alerts = 1000   # Keep last 1000 alerts

    def add_alert(self, alert):
        """Add an IDS alert to the REST API buffer."""
        self.alerts.append(alert)
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]


class RestAPI(ControllerBase):
    """REST API endpoints."""

    def __init__(self, req, link, data, **config):
        super().__init__(req, link, data, **config)
        self.controller_app = data['controller_app']

    # ─── Topology ───────────────────────────────────

    @route('topology', REST_PREFIX + '/topology', methods=['GET'])
    def get_topology(self, req, **kwargs):
        """Get current network topology."""
        try:
            topo = self.controller_app.topology_data
            return self._json_response(topo)
        except Exception as e:
            return self._error_response(str(e))

    @route('switches', REST_PREFIX + '/switches', methods=['GET'])
    def get_switches(self, req, **kwargs):
        """List all connected switches."""
        try:
            switches = list(getattr(self.controller_app, 'datapaths', {}).keys())
            return self._json_response({'switches': switches, 'count': len(switches)})
        except Exception as e:
            return self._error_response(str(e))

    @route('hosts', REST_PREFIX + '/hosts', methods=['GET'])
    def get_hosts(self, req, **kwargs):
        """List all known hosts."""
        try:
            hosts = getattr(self.controller_app, 'hosts', {})
            ip_map = getattr(self.controller_app, 'mac_to_ip', {})
            result = {}
            for mac, (dpid, port) in hosts.items():
                result[mac] = {'dpid': dpid, 'port': port,
                                'ip': ip_map.get(mac, 'unknown')}
            return self._json_response({'hosts': result, 'count': len(result)})
        except Exception as e:
            return self._error_response(str(e))

    # ─── IDS / Security ─────────────────────────────

    @route('alerts', REST_PREFIX + '/alerts', methods=['GET'])
    def get_alerts(self, req, **kwargs):
        """Get recent IDS alerts."""
        alerts = self.controller_app.alerts[-100:]  # Last 100
        return self._json_response({'alerts': alerts, 'total': len(self.controller_app.alerts)})

    @route('alerts_clear', REST_PREFIX + '/alerts', methods=['DELETE'])
    def clear_alerts(self, req, **kwargs):
        """Clear all alerts."""
        self.controller_app.alerts = []
        return self._json_response({'status': 'cleared'})

    @route('blocked', REST_PREFIX + '/blocked', methods=['GET'])
    def get_blocked_ips(self, req, **kwargs):
        """Get list of currently blocked IPs."""
        blocked = list(getattr(self.controller_app, 'blocked_ips', set()))
        return self._json_response({'blocked_ips': blocked})

    @route('block_ip', REST_PREFIX + '/block/{ip}', methods=['POST'])
    def block_ip(self, req, ip, **kwargs):
        """Manually block an IP."""
        try:
            if hasattr(self.controller_app, 'block_ip'):
                self.controller_app.block_ip(ip)
                return self._json_response({'status': 'blocked', 'ip': ip})
            return self._error_response("block_ip not available")
        except Exception as e:
            return self._error_response(str(e))

    @route('unblock_ip', REST_PREFIX + '/unblock/{ip}', methods=['POST'])
    def unblock_ip(self, req, ip, **kwargs):
        """Manually unblock an IP."""
        try:
            if hasattr(self.controller_app, 'unblock_ip'):
                self.controller_app.unblock_ip(ip)
                return self._json_response({'status': 'unblocked', 'ip': ip})
            return self._error_response("unblock_ip not available")
        except Exception as e:
            return self._error_response(str(e))

    # ─── Routing ────────────────────────────────────

    @route('path', REST_PREFIX + '/path/{src}/{dst}', methods=['GET'])
    def get_path(self, req, src, dst, **kwargs):
        """Get routing path between two switch DPIDs."""
        try:
            routing = getattr(self.controller_app, 'routing_engine', None)
            if routing:
                path = routing.get_path(int(src), int(dst))
                return self._json_response({'path': path, 'src': src, 'dst': dst})
            return self._error_response("Routing engine not available")
        except Exception as e:
            return self._error_response(str(e))

    # ─── Stats ──────────────────────────────────────

    @route('stats', REST_PREFIX + '/stats', methods=['GET'])
    def get_stats(self, req, **kwargs):
        """Get controller statistics."""
        try:
            stats = {
                'switches': len(getattr(self.controller_app, 'datapaths', {})),
                'hosts': len(getattr(self.controller_app, 'hosts', {})),
                'blocked_ips': len(getattr(self.controller_app, 'blocked_ips', set())),
                'alerts_total': len(getattr(self.controller_app, 'alerts', [])),
                'timestamp': datetime.utcnow().isoformat()
            }
            return self._json_response(stats)
        except Exception as e:
            return self._error_response(str(e))

    @route('flow_stats', REST_PREFIX + '/flows/{dpid}', methods=['GET'])
    def get_flow_stats(self, req, dpid, **kwargs):
        """Get flow stats for a specific switch."""
        try:
            flow_stats = getattr(self.controller_app, 'flow_stats', {})
            dpid_int = int(dpid)
            stats = flow_stats.get(dpid_int, [])
            result = []
            for s in stats:
                result.append({
                    'priority': s.priority,
                    'packet_count': s.packet_count,
                    'byte_count': s.byte_count,
                    'duration_sec': s.duration_sec,
                    'match': str(s.match)
                })
            return self._json_response({'dpid': dpid, 'flows': result})
        except Exception as e:
            return self._error_response(str(e))

    # ─── Health ─────────────────────────────────────

    @route('health', REST_PREFIX + '/health', methods=['GET'])
    def health(self, req, **kwargs):
        return self._json_response({'status': 'ok', 'timestamp': time.time()})

# ─── Helpers ────────────────────────────────────

    def _json_response(self, data, status=200):
        from webob import Response
        # Step 1: Serialize to JSON string
        json_string = json.dumps(data, default=str)
        # Step 2: Encode the string into bytes (the fix!)
        body_bytes = json_string.encode('utf-8')
        
        return Response(
            status=status, 
            content_type='application/json', 
            charset='utf-8',  # Explicitly tell WebOb the charset
            body=body_bytes   # Pass bytes, not text
        )

    def _error_response(self, message, status=500):
        # This will now correctly use the updated _json_response above
        return self._json_response({'error': message}, status=status)


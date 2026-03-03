"""
SDN Logger - Multi-Sink Event and Threat Logging
Supports:
1. Structured file logging (JSON lines)
2. InfluxDB metrics export (for Grafana)
3. SIEM-compatible CEF/JSON log export
4. Console logging
"""

import logging
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')

# Optional InfluxDB client
try:
    from influxdb_client import InfluxDBClient, Point
    from influxdb_client.client.write_api import SYNCHRONOUS
    INFLUX_AVAILABLE = True
except ImportError:
    INFLUX_AVAILABLE = False

# CEF Severity mapping
SEVERITY_TO_CEF = {
    'low': 3,
    'medium': 6,
    'high': 8,
    'critical': 10,
    'none': 0
}


class SDNLogger:
    """
    Multi-sink SDN event logger.
    All threats, topology events, and stats are logged to:
    - JSON lines log file (threats.jsonl, events.jsonl)
    - InfluxDB (if configured)
    - CEF SIEM file (siem.log)
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        os.makedirs(LOGS_DIR, exist_ok=True)

        # File handles
        self.threat_log_path = os.path.join(LOGS_DIR, 'threats.jsonl')
        self.event_log_path = os.path.join(LOGS_DIR, 'events.jsonl')
        self.siem_log_path = os.path.join(LOGS_DIR, 'siem.log')
        self.flow_log_path = os.path.join(LOGS_DIR, 'flows.jsonl')

        # InfluxDB
        self.influx_client = None
        self.influx_write_api = None
        self._init_influx()

        logger.info(f"[SDNLogger] Logger initialized. Logs: {LOGS_DIR}")

    def _init_influx(self):
        """Initialize InfluxDB connection."""
        if not INFLUX_AVAILABLE:
            logger.info("[SDNLogger] InfluxDB client not available - skipping")
            return

        influx_url = os.environ.get('INFLUXDB_URL',
                                    self.config.get('influx_url', 'http://localhost:8086'))
        influx_token = os.environ.get('INFLUXDB_TOKEN',
                                      self.config.get('influx_token', ''))
        influx_org = os.environ.get('INFLUXDB_ORG',
                                    self.config.get('influx_org', 'sdn'))
        self.influx_bucket = os.environ.get('INFLUXDB_BUCKET',
                                             self.config.get('influx_bucket', 'sdn_metrics'))

        if not influx_token:
            logger.info("[SDNLogger] No InfluxDB token - InfluxDB logging disabled")
            return

        try:
            self.influx_client = InfluxDBClient(
                url=influx_url, token=influx_token, org=influx_org
            )
            self.influx_write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
            logger.info(f"[SDNLogger] InfluxDB connected: {influx_url}")
        except Exception as e:
            logger.warning(f"[SDNLogger] InfluxDB connection failed: {e}")

    # ─── Threat Logging ──────────────────────────────────────────────

    def log_threat(self, detection: dict, src_ip: str, dpid: int):
        """
        Log a detected threat to all sinks.
        """
        timestamp = datetime.utcnow().isoformat()

        record = {
            'timestamp': timestamp,
            'type': 'threat',
            'src_ip': src_ip,
            'dpid': dpid,
            'threat_type': detection.get('threat_type', 'Unknown'),
            'attack_type': detection.get('attack_type', 'unknown'),
            'severity': detection.get('severity', 'low'),
            'confidence': round(detection.get('confidence', 0.0), 4),
            'sources': detection.get('sources', []),
            'recommended_action': detection.get('recommended_action', 'alert_only'),
            'details': detection.get('details', {}),
            'flow_key': detection.get('flow_key', '')
        }

        # JSON lines log
        self._write_jsonl(self.threat_log_path, record)

        # SIEM CEF format
        self._write_cef(record)

        # InfluxDB
        self._write_influx_threat(record)

        logger.warning(
            f"[THREAT] {record['threat_type']} | src={src_ip} | "
            f"severity={record['severity']} | dpid={dpid}"
        )

    def log_event(self, event_type: str, data: dict):
        """Log a network/topology event."""
        record = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'event',
            'event_type': event_type,
            **data
        }
        self._write_jsonl(self.event_log_path, record)
        self._write_influx_event(event_type, data)

    def log_flow(self, flow_features: dict, detection_result: dict = None):
        """Log flow data with optional detection result."""
        record = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'flow',
            **flow_features
        }
        if detection_result:
            record['detection'] = detection_result
        self._write_jsonl(self.flow_log_path, record)

    def log_metrics(self, dpid: int, metrics: dict):
        """Log switch/link metrics to InfluxDB for Grafana."""
        if not self.influx_write_api:
            return
        try:
            point = Point("switch_metrics").tag("dpid", str(dpid))
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    point = point.field(k, float(v))
            self.influx_write_api.write(bucket=self.influx_bucket, record=point)
        except Exception as e:
            logger.debug(f"[SDNLogger] InfluxDB metrics write error: {e}")

    # ─── Sinks ───────────────────────────────────────────────────────

    def _write_jsonl(self, path: str, record: dict):
        """Append a JSON record to a .jsonl file."""
        try:
            with open(path, 'a') as f:
                f.write(json.dumps(record, default=str) + '\n')
        except Exception as e:
            logger.error(f"[SDNLogger] JSONL write error ({path}): {e}")

    def _write_cef(self, threat: dict):
        """
        Write CEF (Common Event Format) SIEM log entry.
        Format: CEF:0|Vendor|Product|Version|SignatureID|Name|Severity|Extension
        """
        try:
            sev_num = SEVERITY_TO_CEF.get(threat.get('severity', 'low'), 3)
            sig_id = threat.get('attack_type', 'unknown').upper()
            name = threat.get('threat_type', 'Network Anomaly')
            src = threat.get('src_ip', '-')
            dpid = threat.get('dpid', '-')
            confidence = threat.get('confidence', 0.0)
            timestamp = threat.get('timestamp', datetime.utcnow().isoformat())

            ext = (
                f"src={src} "
                f"dpid={dpid} "
                f"confidence={confidence:.4f} "
                f"sources={','.join(threat.get('sources', []))} "
                f"action={threat.get('recommended_action', 'none')}"
            )

            cef_line = (
                f"CEF:0|SDN-ML-Controller|HybridIDS|1.0|{sig_id}|{name}|{sev_num}|{ext}"
            )

            with open(self.siem_log_path, 'a') as f:
                f.write(f"{timestamp} {cef_line}\n")
        except Exception as e:
            logger.error(f"[SDNLogger] CEF write error: {e}")

    def _write_influx_threat(self, threat: dict):
        """Write threat detection metric to InfluxDB."""
        if not self.influx_write_api:
            return
        try:
            point = (
                Point("ids_threats")
                .tag("severity", threat.get('severity', 'unknown'))
                .tag("attack_type", threat.get('attack_type', 'unknown'))
                .tag("src_ip", threat.get('src_ip', 'unknown'))
                .tag("dpid", str(threat.get('dpid', 0)))
                .field("confidence", float(threat.get('confidence', 0.0)))
                .field("count", 1)
            )
            self.influx_write_api.write(bucket=self.influx_bucket, record=point)
        except Exception as e:
            logger.debug(f"[SDNLogger] InfluxDB threat write error: {e}")

    def _write_influx_event(self, event_type: str, data: dict):
        """Write topology/system event to InfluxDB."""
        if not self.influx_write_api:
            return
        try:
            point = Point("sdn_events").tag("event_type", event_type)
            for k, v in data.items():
                if isinstance(v, (int, float)):
                    point = point.field(k, float(v))
                elif isinstance(v, str):
                    point = point.tag(k, v)
            point = point.field("count", 1)
            self.influx_write_api.write(bucket=self.influx_bucket, record=point)
        except Exception as e:
            logger.debug(f"[SDNLogger] InfluxDB event write error: {e}")

    # ─── Query ───────────────────────────────────────────────────────

    def get_recent_threats(self, n: int = 100) -> list:
        """Read last n threat records from log."""
        threats = []
        try:
            if os.path.exists(self.threat_log_path):
                with open(self.threat_log_path) as f:
                    lines = f.readlines()
                for line in lines[-n:]:
                    try:
                        threats.append(json.loads(line))
                    except Exception:
                        pass
        except Exception as e:
            logger.error(f"[SDNLogger] Error reading threats: {e}")
        return threats

    def get_stats(self) -> dict:
        """Get log statistics."""
        def count_lines(path):
            try:
                with open(path) as f:
                    return sum(1 for _ in f)
            except Exception:
                return 0

        return {
            'threats_logged': count_lines(self.threat_log_path),
            'events_logged': count_lines(self.event_log_path),
            'flows_logged': count_lines(self.flow_log_path),
            'influxdb_connected': self.influx_write_api is not None,
            'logs_dir': LOGS_DIR
        }


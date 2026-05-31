from __future__ import annotations
import numpy as np
import pandas as pd
from .rules import RuleDefinition, _safe_array


def mqtt_anomaly_signal(df: pd.DataFrame) -> np.ndarray:
    """Abnormal MQTT activity — large messages, unusual topic patterns."""
    mqtt_len = df.get('mqtt.len', pd.Series(np.zeros(len(df)))).to_numpy(np.float32)
    topic_len = df.get('mqtt.topic_len', pd.Series(np.zeros(len(df)))).to_numpy(np.float32)
    proto_len = df.get('mqtt.proto_len', pd.Series(np.zeros(len(df)))).to_numpy(np.float32)
    return _safe_array(
        0.45 * np.clip(mqtt_len / 50.0, 0.0, 1.0)
        + 0.30 * np.clip(topic_len / 20.0, 0.0, 1.0)
        + 0.25 * np.clip(proto_len / 8.0, 0.0, 1.0)
    )


def ddos_flag_storm_signal(df: pd.DataFrame) -> np.ndarray:
    """Combined elevated TCP/ICMP/UDP activity indicative of DDoS variants."""
    tcp_flags = df.get('tcp.flags', pd.Series(np.zeros(len(df)))).to_numpy(np.float32)
    icmp_csum = df.get('icmp.checksum', pd.Series(np.zeros(len(df)))).to_numpy(np.float32)
    icmp_seq = df.get('icmp.seq_le', pd.Series(np.zeros(len(df)))).to_numpy(np.float32)
    udp_port = df.get('udp.port', pd.Series(np.zeros(len(df)))).to_numpy(np.float32)
    return _safe_array(
        0.30 * np.clip(tcp_flags / 32.0, 0.0, 1.0)
        + 0.35 * np.clip(icmp_csum / 65535.0, 0.0, 1.0)
        + 0.20 * np.clip(icmp_seq / 65535.0, 0.0, 1.0)
        + 0.15 * np.clip(udp_port / 65535.0, 0.0, 1.0)
    )


def http_injection_signal(df: pd.DataFrame) -> np.ndarray:
    """HTTP request anomalies suggesting injection / vulnerability probing."""
    content_len = df.get('http.content_length', pd.Series(np.zeros(len(df)))).to_numpy(np.float32)
    query_present = df.get('http.request.uri.query', pd.Series(np.zeros(len(df)))).to_numpy(np.float32)
    method = df.get('http.request.method', pd.Series(np.zeros(len(df)))).to_numpy(np.float32)
    referer = df.get('http.referer', pd.Series(np.zeros(len(df)))).to_numpy(np.float32)
    return _safe_array(
        0.40 * np.clip(content_len / 1000.0, 0.0, 1.0)
        + 0.25 * np.clip(query_present, 0.0, 1.0)
        + 0.20 * np.clip(method / 8.0, 0.0, 1.0)
        + 0.15 * np.clip(referer / 8.0, 0.0, 1.0)
    )


def recon_dns_arp_signal(df: pd.DataFrame) -> np.ndarray:
    """Reconnaissance patterns — DNS query bursts, ARP probing, port scan markers."""
    arp_op = df.get('arp.opcode', pd.Series(np.zeros(len(df)))).to_numpy(np.float32)
    dns_qry_len = df.get('dns.qry.name.len', pd.Series(np.zeros(len(df)))).to_numpy(np.float32)
    dns_retx = df.get('dns.retransmission', pd.Series(np.zeros(len(df)))).to_numpy(np.float32)
    tcp_dst = df.get('tcp.dstport', pd.Series(np.zeros(len(df)))).to_numpy(np.float32)
    return _safe_array(
        0.30 * np.clip(arp_op / 4.0, 0.0, 1.0)
        + 0.30 * np.clip(dns_qry_len / 50.0, 0.0, 1.0)
        + 0.20 * np.clip(dns_retx, 0.0, 1.0)
        + 0.20 * np.clip(tcp_dst / 65535.0, 0.0, 1.0)
    )


def edge_rule_definitions() -> list[RuleDefinition]:
    return [
        RuleDefinition(
            name='mqtt_anomaly',
            description='Anomalous MQTT message structure (length, topic, protocol).',
            feature_fn=mqtt_anomaly_signal,
            direction='high',
        ),
        RuleDefinition(
            name='ddos_flag_storm',
            description='Combined TCP/ICMP/UDP activity signature of DDoS variants.',
            feature_fn=ddos_flag_storm_signal,
            direction='high',
        ),
        RuleDefinition(
            name='http_injection',
            description='HTTP request patterns suggesting SQL/XSS/upload exploit attempts.',
            feature_fn=http_injection_signal,
            direction='high',
        ),
        RuleDefinition(
            name='recon_dns_arp',
            description='Reconnaissance via DNS bursts, ARP probing, port scanning.',
            feature_fn=recon_dns_arp_signal,
            direction='high',
        ),
    ]
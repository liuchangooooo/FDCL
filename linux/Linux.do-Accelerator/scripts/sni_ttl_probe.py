#!/usr/bin/env python3
import argparse
import random
import socket
import ssl
import subprocess
import sys
import time
from dataclasses import dataclass

from scapy.all import AsyncSniffer, ICMP, IP, TCP, Raw, conf, send

TARGET_IP = "104.20.16.234"
TARGET_PORT = 443
NORMAL_TTL = 64


@dataclass
class ProbeResult:
    ttl: int
    status: str
    detail: str
    response_delay_ms: float | None = None
    serverhello_delay_ms: float | None = None


@dataclass
class HttpResult:
    dynamic_ttl: int
    rst_delay_ms: float | None
    handshake_status: str
    detail: str
    tls_handshake_ms: float | None = None
    http_status_line: str | None = None
    full_http_response: str | None = None


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def get_route(dst: str) -> tuple[str, str]:
    proc = run(["ip", "route", "get", dst])
    fields = proc.stdout.strip().split()
    src_ip = None
    iface = None
    for idx, token in enumerate(fields):
        if token == "src" and idx + 1 < len(fields):
            src_ip = fields[idx + 1]
        if token == "dev" and idx + 1 < len(fields):
            iface = fields[idx + 1]
    if not src_ip or not iface:
        raise RuntimeError(f"unable to parse route output: {proc.stdout!r}")
    return src_ip, iface


def make_client_hello(sni: str) -> bytes:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    incoming = ssl.MemoryBIO()
    outgoing = ssl.MemoryBIO()
    ssl_obj = ctx.wrap_bio(incoming, outgoing, server_hostname=sni)
    try:
        ssl_obj.do_handshake()
    except ssl.SSLWantReadError:
        pass
    payload = outgoing.read()
    if not payload:
        raise RuntimeError(f"failed to build ClientHello for {sni}")
    return payload


def packet_payload_len(pkt) -> int:
    if Raw in pkt:
        return len(bytes(pkt[Raw].load))
    return 0


def tls_is_serverhello(payload: bytes) -> bool:
    return (
        len(payload) >= 6
        and payload[0] == 0x16
        and payload[1] == 0x03
        and payload[5] == 0x02
    )


def start_sniffer(iface: str, sport: int) -> AsyncSniffer:
    bpf = f"(icmp) or (tcp and host {TARGET_IP} and (port {TARGET_PORT} or port {sport}))"
    sniffer = AsyncSniffer(iface=iface, filter=bpf, store=True)
    sniffer.start()
    time.sleep(0.05)
    return sniffer


def stop_sniffer(sniffer: AsyncSniffer, wait_s: float):
    time.sleep(wait_s)
    return sniffer.stop(join=True)


def find_matching_icmp(pkt, sport: int) -> str | None:
    if ICMP not in pkt or pkt[ICMP].type != 11:
        return None
    inner_ip = pkt[ICMP].payload.getlayer(IP)
    inner_tcp = pkt[ICMP].payload.getlayer(TCP)
    if not inner_ip or not inner_tcp:
        return None
    if inner_ip.dst != TARGET_IP:
        return None
    if inner_tcp.sport != sport or inner_tcp.dport != TARGET_PORT:
        return None
    return f"icmp_time_exceeded from {pkt[IP].src}"


def classify_packets(pkts, sport: int, sent_at: float) -> ProbeResult | None:
    for pkt in pkts:
        if IP not in pkt or TCP not in pkt:
            continue
        tcp = pkt[TCP]
        ip = pkt[IP]
        if tcp.dport != sport:
            continue
        payload_len = packet_payload_len(pkt)
        if payload_len and tls_is_serverhello(bytes(pkt[Raw].load)):
            delta_ms = (float(pkt.time) - sent_at) * 1000.0
            return ProbeResult(
                ttl=0,
                status="serverhello",
                detail=f"serverhello from {ip.src}",
                response_delay_ms=delta_ms,
                serverhello_delay_ms=delta_ms,
            )

    for pkt in pkts:
        if IP not in pkt or TCP not in pkt:
            continue
        tcp = pkt[TCP]
        ip = pkt[IP]
        if tcp.dport != sport:
            continue
        flags = tcp.sprintf("%TCP.flags%")
        payload_len = packet_payload_len(pkt)
        if "R" in flags:
            delta_ms = (float(pkt.time) - sent_at) * 1000.0
            return ProbeResult(
                ttl=0,
                status="rst",
                detail=f"rst from {ip.src}, payload_len={payload_len}, ttl={ip.ttl}",
                response_delay_ms=delta_ms,
            )
        if flags == "SA" and payload_len == 2:
            delta_ms = (float(pkt.time) - sent_at) * 1000.0
            return ProbeResult(
                ttl=0,
                status="synack_payload_2",
                detail=f"synack+2B from {ip.src}, ttl={ip.ttl}",
                response_delay_ms=delta_ms,
            )

    for pkt in pkts:
        icmp_detail = find_matching_icmp(pkt, sport)
        if icmp_detail:
            delta_ms = (float(pkt.time) - sent_at) * 1000.0
            return ProbeResult(
                ttl=0,
                status="icmp_time_exceeded",
                detail=icmp_detail,
                response_delay_ms=delta_ms,
            )

    return None


def classify_with_socket(pkts, sport: int, sent_at: float, sock_data: bytes, sock_error: str | None) -> ProbeResult:
    pkt_result = classify_packets(pkts, sport, sent_at)
    if pkt_result:
        return pkt_result
    if sock_data and tls_is_serverhello(sock_data):
        return ProbeResult(
            ttl=0,
            status="serverhello",
            detail="serverhello via socket recv",
            serverhello_delay_ms=None,
        )
    if sock_error:
        return ProbeResult(ttl=0, status="socket_error", detail=sock_error)
    return ProbeResult(ttl=0, status="timeout", detail="no matching reply")


def make_socket(src_ip: str, sport: int, ttl: int | None = None) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2.0)
    sock.bind((src_ip, sport))
    sock.connect((TARGET_IP, TARGET_PORT))
    if ttl is not None:
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_TTL, ttl)
    return sock


def recv_once(sock: socket.socket) -> tuple[bytes, str | None]:
    try:
        return sock.recv(4096), None
    except socket.timeout:
        return b"", None
    except OSError as exc:
        return b"", f"{exc.__class__.__name__}: {exc}"


def send_once(sock: socket.socket, payload: bytes) -> str | None:
    try:
        sock.sendall(payload)
        return None
    except OSError as exc:
        return f"{exc.__class__.__name__}: {exc}"


def single_ttl_probe(src_ip: str, iface: str, ttl: int, clienthello: bytes) -> ProbeResult:
    sport = random.randint(20000, 60000)
    try:
        sock = make_socket(src_ip, sport, ttl)
    except OSError as exc:
        return ProbeResult(
            ttl=ttl,
            status="connect_error",
            detail=f"{exc.__class__.__name__}: {exc}",
        )
    try:
        sniffer = start_sniffer(iface, sport)
        sent_at = time.time()
        send_error = send_once(sock, clienthello)
        sock_data, sock_error = recv_once(sock)
        sock_error = send_error or sock_error
        pkts = stop_sniffer(sniffer, wait_s=1.5)
        result = classify_with_socket(pkts, sport, sent_at, sock_data, sock_error)
        result.ttl = ttl
        return result
    finally:
        sock.close()


def extract_handshake_numbers(pkts, sport: int) -> tuple[int, int]:
    client_seq = None
    server_seq = None
    for pkt in pkts:
        if IP not in pkt or TCP not in pkt:
            continue
        tcp = pkt[TCP]
        if tcp.sport == sport and tcp.dport == TARGET_PORT and tcp.flags & 0x02:
            client_seq = tcp.seq
        if tcp.sport == TARGET_PORT and tcp.dport == sport and tcp.flags & 0x12 == 0x12:
            server_seq = tcp.seq
    if client_seq is None or server_seq is None:
        raise RuntimeError("failed to capture handshake sequence numbers")
    return client_seq + 1, server_seq + 1


def desync_probe(
    src_ip: str,
    iface: str,
    fake_ttl: int,
    fake_hello: bytes,
    real_hello: bytes,
) -> ProbeResult:
    sport = random.randint(20000, 60000)
    handshake_sniffer = start_sniffer(iface, sport)
    sock = make_socket(src_ip, sport, NORMAL_TTL)
    try:
        handshake_pkts = stop_sniffer(handshake_sniffer, wait_s=0.3)
        client_seq, server_ack = extract_handshake_numbers(handshake_pkts, sport)

        fake_pkt = IP(src=src_ip, dst=TARGET_IP, ttl=fake_ttl) / TCP(
            sport=sport,
            dport=TARGET_PORT,
            flags="PA",
            seq=client_seq,
            ack=server_ack,
        ) / Raw(fake_hello)

        probe_sniffer = start_sniffer(iface, sport)
        send(fake_pkt, verbose=False)
        time.sleep(0.05)

        sent_at = time.time()
        send_error = send_once(sock, real_hello)
        sock_data, sock_error = recv_once(sock)
        sock_error = send_error or sock_error
        pkts = stop_sniffer(probe_sniffer, wait_s=1.8)
        result = classify_with_socket(pkts, sport, sent_at, sock_data, sock_error)
        result.ttl = NORMAL_TTL
        return result
    finally:
        sock.close()


def find_dynamic_rst_probe(src_ip: str, iface: str, clienthello: bytes, max_ttl: int) -> ProbeResult | None:
    for ttl in range(1, max_ttl + 1):
        result = single_ttl_probe(src_ip, iface, ttl, clienthello)
        delay = f"{result.response_delay_ms:.1f}ms" if result.response_delay_ms is not None else "NA"
        print(
            f"probe ttl={ttl:02d} status={result.status} detail={result.detail} response_delay={delay}",
            flush=True,
        )
        if result.status in {"rst", "synack_payload_2"}:
            return result
    return None


def make_tls_context() -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def recv_http_response(tls_sock: ssl.SSLSocket) -> tuple[str | None, str | None]:
    chunks: list[bytes] = []
    total = 0
    while total < 1024 * 1024:
        try:
            chunk = tls_sock.recv(4096)
        except socket.timeout:
            break
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    if not chunks:
        return None, None
    raw = b"".join(chunks)
    head = raw.split(b"\r\n\r\n", 1)[0]
    lines = head.split(b"\r\n")
    status_line = lines[0].decode("iso-8859-1", errors="replace") if lines else None
    full_response = raw.decode("utf-8", errors="replace")
    return status_line, full_response


def clamp_delay_from_probe(probe: ProbeResult) -> float:
    if probe.response_delay_ms is None:
        return 0.05
    return min(max(probe.response_delay_ms / 1000.0, 0.01), 0.80)


def tls_get_with_dynamic_ttl(
    src_ip: str,
    iface: str,
    probe: ProbeResult,
    fake_hello: bytes,
    server_name: str,
) -> HttpResult:
    sport = random.randint(20000, 60000)
    handshake_sniffer = start_sniffer(iface, sport)
    raw_sock = make_socket(src_ip, sport, NORMAL_TTL)
    tls_sock = None
    try:
        handshake_pkts = stop_sniffer(handshake_sniffer, wait_s=0.3)
        client_seq, server_ack = extract_handshake_numbers(handshake_pkts, sport)

        fake_pkt = IP(src=src_ip, dst=TARGET_IP, ttl=probe.ttl) / TCP(
            sport=sport,
            dport=TARGET_PORT,
            flags="PA",
            seq=client_seq,
            ack=server_ack,
        ) / Raw(fake_hello)

        probe_sniffer = start_sniffer(iface, sport)
        send(fake_pkt, verbose=False)
        lead_sleep_s = clamp_delay_from_probe(probe)
        time.sleep(lead_sleep_s)

        tls_ctx = make_tls_context()
        tls_sock = tls_ctx.wrap_socket(raw_sock, server_hostname=server_name, do_handshake_on_connect=False)
        tls_sock.settimeout(8.0)

        handshake_start = time.time()
        try:
            tls_sock.do_handshake()
        except OSError as exc:
            pkts = stop_sniffer(probe_sniffer, wait_s=1.2)
            pkt_result = classify_packets(pkts, sport, handshake_start)
            detail = pkt_result.detail if pkt_result else f"{exc.__class__.__name__}: {exc}"
            return HttpResult(
                dynamic_ttl=probe.ttl,
                rst_delay_ms=probe.response_delay_ms,
                handshake_status="tls_error",
                detail=detail,
            )

        tls_handshake_ms = (time.time() - handshake_start) * 1000.0
        request = (
            f"GET / HTTP/1.1\r\n"
            f"Host: {server_name}\r\n"
            "User-Agent: dynamic-ttl-probe/1.0\r\n"
            "Accept: */*\r\n"
            "Connection: close\r\n\r\n"
        ).encode()
        tls_sock.sendall(request)
        status_line, full_http_response = recv_http_response(tls_sock)
        stop_sniffer(probe_sniffer, wait_s=0.3)
        return HttpResult(
            dynamic_ttl=probe.ttl,
            rst_delay_ms=probe.response_delay_ms,
            handshake_status="ok",
            detail="tls handshake and GET completed",
            tls_handshake_ms=tls_handshake_ms,
            http_status_line=status_line,
            full_http_response=full_http_response,
        )
    finally:
        if tls_sock is not None:
            tls_sock.close()
        else:
            raw_sock.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-ttl", type=int, default=20)
    parser.add_argument("--request-trials", type=int, default=3)
    args = parser.parse_args()

    src_ip, iface = get_route(TARGET_IP)
    conf.verb = 0
    linux_hello = make_client_hello("linux.do")
    fake_hello = make_client_hello("bing.com")

    print(f"route src={src_ip} iface={iface}", flush=True)
    print(
        f"clienthello sizes: linux.do={len(linux_hello)} bytes, bing.com={len(fake_hello)} bytes",
        flush=True,
    )

    successes = 0
    for trial in range(1, args.request_trials + 1):
        print(f"request trial={trial}: probing dynamic ttl", flush=True)
        probe = find_dynamic_rst_probe(src_ip, iface, linux_hello, args.max_ttl)
        if probe is None:
            print("dynamic probe failed: no rst/synack anomaly found", flush=True)
            continue
        delay = f"{probe.response_delay_ms:.1f}ms" if probe.response_delay_ms is not None else "NA"
        print(
            f"dynamic ttl selected ttl={probe.ttl:02d} status={probe.status} response_delay={delay}",
            flush=True,
        )
        http_result = tls_get_with_dynamic_ttl(src_ip, iface, probe, fake_hello, "linux.do")
        tls_delay = f"{http_result.tls_handshake_ms:.1f}ms" if http_result.tls_handshake_ms is not None else "NA"
        print(
            f"request trial={trial} ttl={http_result.dynamic_ttl:02d} handshake_status={http_result.handshake_status} "
            f"detail={http_result.detail} rst_delay={delay} tls_handshake={tls_delay}",
            flush=True,
        )
        if http_result.full_http_response:
            print("http response begin", flush=True)
            print(http_result.full_http_response, flush=True)
            print("http response end", flush=True)
        if http_result.handshake_status == "ok":
            successes += 1

    print(f"summary successful_gets={successes}/{args.request_trials}", flush=True)
    if successes == 0:
        return 1

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)

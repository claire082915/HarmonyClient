#!/usr/bin/env python3
"""
fetch_results.py — Download experiment logs and parse recall + query latency.

What it does
------------
1. SSHs into the client node, finds the most recent client log (or the one
   matching --timestamp), and downloads it.
2. Optionally also downloads the server log from the master node.
3. Parses the client log for recall@k and per-batch query time, and prints a
   summary table.

Expected client log lines (written by harmony_client)
------------------------------------------------------
    recall@100: 0.9975
    Batch 0: nq=96  time=12.34 ms  QPS=7783.2
    ...
    Average QPS: 7801.4
    Average latency: 12.31 ms

Usage
-----
    # Download latest logs and parse results
    python fetch_results.py --config config.yaml

    # Download a specific timestamped run
    python fetch_results.py --config config.yaml --timestamp 20260323_120000

    # Only print, don't re-download (log already local)
    python fetch_results.py --config config.yaml --local-log ./harmony_logs/sift1m_3w_20260323_120000/harmony-client_client_20260323_120000.log
"""

import argparse
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import yaml
from cluster_utils import (
    get_cluster_nodes,
    get_log_timestamp,
    numeric_name_key,
    remote_log_dir,
    scp_download,
    ssh_exec,
)


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def parse_client_log(path: str) -> Dict:
    """
    Parse a harmony_client log file and extract key metrics.
    Returns a dict with:
        recall          float | None
        batches         list of {batch, nq, time_ms, qps}
        avg_qps         float | None
        avg_latency_ms  float | None
        total_queries   int
        errors          list[str]   (lines that look like errors)
    """
    recall         = None
    batches        = []
    avg_qps        = None
    avg_latency_ms = None
    total_queries  = 0
    errors         = []

    # Patterns — adjust if your client prints differently
    re_recall    = re.compile(r"recall@\d+\s*:\s*([0-9.]+)", re.IGNORECASE)
    re_batch     = re.compile(
        r"[Bb]atch\s+(\d+).*?nq\s*=\s*(\d+).*?time\s*=\s*([0-9.]+)\s*ms.*?QPS\s*=\s*([0-9.]+)",
        re.IGNORECASE,
    )
    re_avg_qps   = re.compile(r"[Aa]verage\s+QPS\s*[:=]\s*([0-9.]+)")
    re_avg_lat   = re.compile(r"[Aa]verage\s+latency\s*[:=]\s*([0-9.]+)\s*ms", re.IGNORECASE)
    re_total_q   = re.compile(r"[Tt]otal\s+queries?\s*[:=]\s*(\d+)")
    re_error     = re.compile(r"\b(error|failed|fatal|exception)\b", re.IGNORECASE)

    try:
        with open(path) as f:
            for line in f:
                line = line.rstrip()
                m = re_recall.search(line)
                if m:
                    recall = float(m.group(1))
                m = re_batch.search(line)
                if m:
                    batches.append({
                        "batch": int(m.group(1)),
                        "nq":    int(m.group(2)),
                        "time_ms": float(m.group(3)),
                        "qps":   float(m.group(4)),
                    })
                m = re_avg_qps.search(line)
                if m:
                    avg_qps = float(m.group(1))
                m = re_avg_lat.search(line)
                if m:
                    avg_latency_ms = float(m.group(1))
                m = re_total_q.search(line)
                if m:
                    total_queries = int(m.group(1))
                if re_error.search(line):
                    errors.append(line)
    except FileNotFoundError:
        print(f"[parse] File not found: {path}")

    # Derived avg if not printed explicitly
    if batches:
        total_queries = total_queries or sum(b["nq"] for b in batches)
        if avg_qps is None:
            avg_qps = sum(b["qps"] for b in batches) / len(batches)
        if avg_latency_ms is None:
            avg_latency_ms = sum(b["time_ms"] for b in batches) / len(batches)

    return {
        "recall":         recall,
        "batches":        batches,
        "avg_qps":        avg_qps,
        "avg_latency_ms": avg_latency_ms,
        "total_queries":  total_queries,
        "errors":         errors,
    }


def print_summary(log_path: str, metrics: Dict, config: Dict) -> None:
    exp = config.get("experiment", {})
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"  Log file  : {log_path}")
    print(f"  Dataset   : {exp.get('dataset', '?')}")
    print(f"  k         : {exp.get('k', '?')}")
    print(f"  nprobe    : {exp.get('nprobe', '?')}")
    print(f"  group     : {exp.get('group', '?')}")
    print(f"  block     : {exp.get('block', '?')}")
    print(f"  team      : {exp.get('team', '?')}")
    print("-" * 60)

    recall = metrics["recall"]
    if recall is not None:
        print(f"  recall@{exp.get('k','?'):3}  : {recall:.4f}")
    else:
        print("  recall        : (not found in log)")

    avg_qps = metrics["avg_qps"]
    avg_lat = metrics["avg_latency_ms"]
    if avg_qps is not None:
        print(f"  Avg QPS       : {avg_qps:.1f}")
    if avg_lat is not None:
        print(f"  Avg latency   : {avg_lat:.2f} ms")

    total = metrics["total_queries"]
    if total:
        print(f"  Total queries : {total}")

    if metrics["batches"]:
        print(f"\n  Per-batch breakdown ({len(metrics['batches'])} batches):")
        print(f"    {'Batch':>6}  {'nq':>6}  {'time(ms)':>10}  {'QPS':>10}")
        for b in metrics["batches"]:
            print(f"    {b['batch']:>6}  {b['nq']:>6}  {b['time_ms']:>10.2f}  {b['qps']:>10.1f}")

    if metrics["errors"]:
        print(f"\n  ⚠  {len(metrics['errors'])} error line(s) found:")
        for e in metrics["errors"][:5]:
            print(f"     {e}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def find_latest_remote_log(server: str, ssh_key: str, log_dir: str, designation: str) -> Optional[str]:
    code, stdout, _ = ssh_exec(
        server,
        f"ls -1t {log_dir}/{designation}_*.log 2>/dev/null | head -1",
        ssh_key, timeout=15,
    )
    path = stdout.strip()
    return path if path else None


def download_node_log(
    server: str,
    ssh_key: str,
    install_dir: str,
    designation: str,
    timestamp: Optional[str],
    dest_dir: str,
    node_name: str,
) -> Optional[str]:
    """Download one log file from *server*. Returns local path or None."""
    log_dir = f"{install_dir.rstrip('/')}/logs"

    if timestamp:
        remote_path = f"{log_dir}/{designation}_{timestamp}.log"
    else:
        remote_path = find_latest_remote_log(server, ssh_key, log_dir, designation)
        if not remote_path:
            print(f"[{node_name}] No {designation} log found in {log_dir}")
            return None

    base       = os.path.basename(remote_path)
    local_path = os.path.join(dest_dir, f"{node_name}_{base}")
    ok = scp_download(server, remote_path, local_path, ssh_key)
    if ok:
        print(f"[{node_name}] ✓ {local_path}")
        return local_path
    print(f"[{node_name}] ✗ Failed to download {remote_path}")
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download Harmony experiment logs and parse recall/latency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch latest logs and print results
  python fetch_results.py --config config.yaml

  # Fetch a specific run
  python fetch_results.py --config config.yaml --timestamp 20260323_120000

  # Parse a log you already have locally (no SSH)
  python fetch_results.py --config config.yaml --local-log ./client_20260323.log

  # Also download the server log
  python fetch_results.py --config config.yaml --server-log
        """,
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--timestamp", help="Log timestamp to fetch (e.g. 20260323_120000)")
    parser.add_argument("--local-log", metavar="PATH", help="Parse an already-downloaded client log")
    parser.add_argument("--server-log", action="store_true", help="Also download the server log")
    parser.add_argument("--output-dir", default=None, help="Where to save downloaded logs")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # ---- Mode: parse a local log directly ----
    if args.local_log:
        metrics = parse_client_log(args.local_log)
        print_summary(args.local_log, metrics, config)
        sys.exit(0 if metrics["recall"] is not None else 1)

    # ---- Mode: fetch from cluster ----
    nodes    = get_cluster_nodes(config, validate_range=False)
    username = config["azure"]["username"]
    ssh_key  = config["azure"]["ssh_private_key"]
    install_dir = config.get("install_dir", "~/Harmony")

    client_node = next((n for n in nodes if n["type"] == "client"), None)
    master_node = next((n for n in nodes if n["type"] == "master"), None)

    if not client_node:
        print("ERROR: client node not found in config")
        sys.exit(1)

    out_dir = os.path.expanduser(
        args.output_dir or config.get("local_logs_dir", "~/harmony_logs")
    )
    os.makedirs(out_dir, exist_ok=True)

    client_server = f"{username}@{client_node['ip'] or client_node['private_ip']}"
    client_log = download_node_log(
        client_server, ssh_key, install_dir,
        "client", args.timestamp, out_dir, client_node["name"],
    )

    if args.server_log and master_node:
        master_server = f"{username}@{master_node['ip'] or master_node['private_ip']}"
        download_node_log(
            master_server, ssh_key, install_dir,
            "server", args.timestamp, out_dir, master_node["name"],
        )

    if not client_log:
        print("No client log downloaded — cannot parse results")
        sys.exit(1)

    metrics = parse_client_log(client_log)
    print_summary(client_log, metrics, config)
    sys.exit(0 if metrics["recall"] is not None else 1)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
start_services.py — Start, stop, and monitor Harmony services on the cluster.

Harmony node roles
------------------
  master   — MPI rank 0; runs `query` binary with --serve, accepts TCP from client
  workers  — MPI ranks 1..N; also run `query` binary (worker path via MPI)
  client   — runs `harmony_client` (no MPI); connects to master over TCP

Launch order
------------
  1. Start `mpirun` on the master node, spanning master + all worker nodes.
     mpirun is launched as a background process (tmux) on the master VM.
     It SSHs into worker VMs automatically via MPI hostfile.
  2. Wait for the server to start listening (~5 s).
  3. Start `harmony_client` on the client VM.

The MPI hostfile (~/<mpi_hosts_file>) is written to the master node by this
script based on the discovered private IPs.

Usage
-----
    python start_services.py --config config.yaml --start
    python start_services.py --config config.yaml --stop
    python start_services.py --config config.yaml --restart
    python start_services.py --config config.yaml --status
    python start_services.py --config config.yaml --logs harmony-1
    python start_services.py --config config.yaml --download-logs 20260323_120000
    python start_services.py --config config.yaml --download-logs ~/my_dir 20260323_120000
"""

import argparse
import logging
import os
import subprocess
import sys
import threading
import time
import copy
import json
import shlex
import subprocess
from typing import Dict, List, Optional, Tuple

import yaml
from cluster_utils import (
    get_cluster_nodes,
    get_log_timestamp,
    numeric_name_key,
    remote_log_dir,
    remote_log_path,
    scp_download,
    ssh_exec,
    ssh_exec_stream,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def _server(username: str, node: Dict) -> str:
    ip = node["ip"] or node["private_ip"]
    return f"{username}@{ip}"


def _ssh_key(config: Dict) -> str:
    return config["azure"]["ssh_private_key"]


def _install_dir(config: Dict) -> str:
    return config.get("install_dir", "~/Harmony")


def _source_oneapi(config: Dict) -> str:
    setvars = config.get("oneapi_setvars", "~/intel/oneapi/setvars.sh")
    return f"source {setvars} --force"


def _ensure_tmux(server: str, ssh_key: str, session: str = "harmony") -> None:
    ssh_exec(
        server,
        f"if tmux has-session -t {session} 2>/dev/null; then "
        f"  tmux new-window -t {session}; "
        f"else "
        f"  tmux new -s {session} -d; "
        f"fi",
        ssh_key,
        timeout=20,
    )


def _tmux_run(server: str, command: str, ssh_key: str, session: str = "harmony") -> int:
    """Fire a command in a tmux window (non-blocking)."""
    wrapped = f'tmux send -t {session} "{command}" ENTER'
    code, _, _ = ssh_exec(server, wrapped, ssh_key, timeout=30)
    return code


# ---------------------------------------------------------------------------
# Stop services
# ---------------------------------------------------------------------------

def stop_services(server: str, ssh_key: str) -> None:
    log.info(f"[{server}] Stopping existing Harmony processes…")
    ssh_exec(server, "pkill -f 'mpirun.*query\\|harmony_client' 2>/dev/null || true", ssh_key, timeout=15)
    ssh_exec(server, "pkill -f '/bin/query' 2>/dev/null || true", ssh_key, timeout=15)
    ssh_exec(server, "tmux kill-session -t harmony 2>/dev/null || true", ssh_key, timeout=15)
    ssh_exec(server, "ipcs -m | awk 'NR>3 {print $2}' | xargs -r ipcrm -m 2>/dev/null || true", ssh_key, timeout=15)
    time.sleep(2)


# ---------------------------------------------------------------------------
# Binary verification
# ---------------------------------------------------------------------------

def verify_binaries(server: str, config: Dict) -> bool:
    install_dir = _install_dir(config)
    ssh_key = _ssh_key(config)
    cmd = (
        f"ls -lh {install_dir}/release/bin/query "
        f"{install_dir}/release/bin/harmony_client 2>/dev/null"
    )
    code, stdout, _ = ssh_exec(server, cmd, ssh_key, timeout=20)
    if code == 0:
        log.info(f"[{server}] Binaries verified")
        return True
    log.error(
        f"[{server}] Binaries not found under {install_dir}/release/bin. "
        "Run build_nodes.py first."
    )
    return False


# ---------------------------------------------------------------------------
# MPI hostfile
# ---------------------------------------------------------------------------

def write_mpi_hostfile(master_server: str, nodes: List[Dict], config: Dict) -> str:
    """
    Write an MPI hostfile to the master node.
    Line format:  <private_ip> slots=1
    Returns the remote path.
    """
    ssh_key = _ssh_key(config)
    hosts_path = config.get("mpi", {}).get("hosts_file", "~/mpi_hosts")

    # Master = rank 0 (first line), then workers
    lines = []
    master_nodes = [n for n in nodes if n["type"] == "master"]
    worker_nodes = sorted(
        [n for n in nodes if n["type"] == "worker"],
        key=lambda x: numeric_name_key(x["name"]),
    )
    # for n in master_nodes + worker_nodes:
    #     slots = n.get("slots", 1)  # ← read from config, default 1
    #     lines.append(f"{n['private_ip']} slots={slots}")

    for n in master_nodes + worker_nodes:
        slots = n.get("slots", 1)
        for _ in range(slots):
            lines.append(f"{n['private_ip']}")

    content = "\n".join(lines) + "\n"
    cmd = f"cat > {hosts_path} << 'HOSTSEOF'\n{content}HOSTSEOF"
    code, _, stderr = ssh_exec(master_server, cmd, ssh_key, timeout=20)
    if code != 0:
        log.error(f"[{master_server}] Failed to write hostfile: {stderr.strip()}")
    else:
        log.info(f"[{master_server}] Wrote MPI hostfile to {hosts_path}:\n{content.rstrip()}")
    return hosts_path


# ---------------------------------------------------------------------------
# Build server (mpirun) command
# ---------------------------------------------------------------------------

def build_server_command(config: Dict, num_ranks: int, master_ip: str) -> str:
    """
    Build the mpirun command that launches `query` across master + workers.
    num_ranks = 1 (master) + num_workers
    """
    install_dir = _install_dir(config)
    exp = config.get("experiment", {})
    benchmarks_path = exp.get("benchmarks_path", f"{install_dir}/benchmarks")
    dataset      = exp.get("dataset", "sift1m")
    input_format = exp.get("input_format", "fvecs")
    group        = exp.get("group", 1)
    team         = exp.get("team", 1)
    block        = exp.get("block", 1)
    nprobe       = exp.get("nprobe", 100)
    nlist        = exp.get("nlist", 0)
    mode         = exp.get("mode", "group")
    tcp_port     = config.get("cluster", {}).get("tcp_port", 7777)
    cache        = exp.get("cache", False)
    skip_insert  = exp.get("skip_insert", False)
    train_data   = config.get("serve", {}).get("train_data", "")
    omp_threads  = config.get("mpi", {}).get("omp_num_threads", 1)
    mkl_lib      = "/home/csl12/intel/oneapi/mkl/2025.0/lib"
    serve = config.get("serve", {})
    nb    = serve.get("nb", 0)
    dim   = serve.get("dim", 128)

    flags = (
        f"--benchmarks_path {benchmarks_path} "
        f"--dataset {dataset} "
        f"--input_format {input_format} "
        f"--serve "
        f"--tcp_port {tcp_port} "
        f"--nprobe {nprobe} "
        f"--nlist {nlist} "
        f"--group {group} "
        f"--team {team} "
        f"--block {block} "
        f"--mode {mode} "
        f"--nb {nb} "    
        f"--dim {dim}" 
    )
    if train_data:
        flags += f" --train_data {train_data}"
    if cache:
        flags += " --cache"
    if skip_insert:
        flags += " --skip_insert"

    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    mkl_lib = "/home/csl12/intel/oneapi/mkl/2025.0/lib"
    full_ld = f"{mkl_lib}:{current_ld}" if current_ld else mkl_lib

    mpirun_bin = config.get("mpi", {}).get(
        "mpirun_bin",
        "/home/csl12/intel/oneapi/mpi/2021.14/bin/mpirun"
    )

    mpirun = (
        f"{mpirun_bin} -n {num_ranks} -ppn {num_ranks} "
        # f"-f {hosts_path} "
        f"-hosts {master_ip} "
        f"-genv I_MPI_SHM 0 "      
        f"-genv I_MPI_FABRICS ofi "
        f"-genv FI_PROVIDER tcp " 
        # f"-genv I_MPI_DEBUG 5 "              
        f"-genv LD_LIBRARY_PATH {full_ld} "
        f"-genv OMP_NUM_THREADS {omp_threads} "
        f"/data/csl12/Harmony/release/bin/query {flags}"
        # f"{install_dir}/release/bin/query {flags}"
    )
    log.info(f"mpirun command: {mpirun}")
    return mpirun
    # install_dir = _install_dir(config)
    # exp = config.get("experiment", {})
    # benchmarks_path = exp.get("benchmarks_path", f"{install_dir}/benchmarks")
    # dataset       = exp.get("dataset", "sift1m")
    # group         = exp.get("group", 2)
    # team          = exp.get("team", 2)
    # block         = exp.get("block", 4)
    # nprobe        = exp.get("nprobe", 100)
    # mode          = exp.get("mode", "group")
    # tcp_port      = config.get("cluster", {}).get("tcp_port", 7777)
    # cache         = exp.get("cache", True)
    # skip_insert   = exp.get("skip_insert", True)
    # nb  = config.get("serve", {}).get("nb", 0)
    # dim = config.get("serve", {}).get("dim", 0)
    # train_data = config.get("serve", {}).get("train_data", "")
    # input_format = exp.get("input_format", "fvecs")
    # nlist = exp.get("nlist", 0)

    # # flags = (
    # #     f"--benchmarks_path {benchmarks_path} "
    # #     f"--dataset {dataset} "
    # #     f"--serve "
    # #     f"--tcp_port {tcp_port} "
    # #     f"--nprobe {nprobe} "
    # #     f"--group {group} "
    # #     f"--team {team} "
    # #     f"--block {block} "
    # #     f"--mode {mode} "
    # #     f"--nb {nb} "
    # #     f"--dim {dim}"
    # # )

    # flags = (
    #     f"--benchmarks_path {benchmarks_path} "
    #     f"--dataset {dataset} "
    #     f"--input_format {input_format} "
    #     f"--serve "
    #     f"--tcp_port {tcp_port} "
    #     f"--nprobe {nprobe} " 
    #     f"--group {group} "
    #     f"--team {team} "
    #     f"--block {block} "
    #     f"--mode {mode} "
    #     f"--nlist {nlist} "
    # )

    # if train_data:
    #     flags += f" --train_data {train_data}"
    # if cache:
    #     flags += " --cache"
    # if skip_insert:
    #     flags += " --skip_insert"

    # # -x passes environment variables through MPI
    # oneapi = config.get("oneapi_setvars", "~/intel/oneapi/setvars.sh")
    # omp_threads = config.get("mpi", {}).get("omp_num_threads", 1)

    # mpirun = (
    #     f"mpirun -n {num_ranks} "
    #     f"-machinefile {hosts_path} "
    #     f"bash -c 'source {oneapi} --force > /dev/null 2>&1 && "
    #     f"export OMP_NUM_THREADS={omp_threads} && "
    #     f"{install_dir}/release/bin/query {flags}'"
    # )
    # return mpirun


# ---------------------------------------------------------------------------
# Build client command
# ---------------------------------------------------------------------------

def build_client_command(config: Dict, master_private_ip: str) -> str:
    install_dir = _install_dir(config)
    exp     = config.get("experiment", {})
    client  = config.get("client", {})

    benchmarks_path = exp.get("benchmarks_path", f"{install_dir}/benchmarks")
    dataset  = exp.get("dataset", "sift1m")
    host     = master_private_ip
    port     = config.get("cluster", {}).get("tcp_port", 7777)
    k        = exp.get("k", 100)
    group    = exp.get("group", 2)
    block    = exp.get("block", 4)
    query_batch = exp.get("query_batch", 96)
    query_loop  = exp.get("query_loop", 1)
    nq          = client.get("nq", 0)
    nb          = client.get("nb", 0)
    insert_batch = client.get("insert_batch", 100000)

    # Paths — use explicit overrides if set, else derive from benchmarks_path/dataset
    query_file = client.get("query_file") or \
        f"{benchmarks_path}/{dataset}/origin/{dataset}_query.fvecs"
    groundtruth_file = client.get("groundtruth_file") or \
        f"{benchmarks_path}/{dataset}/result/groundtruth_{k}.bin"
    base_file = client.get("base_file") or \
        f"{benchmarks_path}/{dataset}/origin/{dataset}_base.fvecs"

    skip_insert = exp.get("skip_insert", True)

    flags = (
        f"--query {query_file} "
        f"--groundtruth {groundtruth_file} "
        f"--host {host} "
        f"--port {port} "
        f"--k {k} "
        f"--query_batch {query_batch} "
        f"--group {group} "
        f"--block {block} "
        f"--query_loop {query_loop}"
    )
    if not skip_insert:
        flags += f" --base {base_file} --insert_batch {insert_batch} "
    else:
        flags += " --skip_build"
    if nq > 0:
        flags += f" --nq {nq}"
    if nb > 0:
        flags += f" --nb {nb}"

    # return f"{install_dir}/release/bin/harmony_client {flags}"
    return f"/data/csl12/Harmony/release/bin/harmony_client {flags}"


# ---------------------------------------------------------------------------
# Start master+workers (single mpirun on master VM)
# ---------------------------------------------------------------------------

def start_server(master_node: Dict, nodes: List[Dict], config: Dict, timestamp: str) -> bool:
    """Launch mpirun on the master node spanning master + all worker ranks."""
    ssh_key     = _ssh_key(config)
    install_dir = _install_dir(config)
    username    = config["azure"]["username"]
    server      = _server(username, master_node)
    tcp_port    = config.get("cluster", {}).get("tcp_port", 7777)

    if not verify_binaries(server, config):
        return False

    stop_services(server, ssh_key)

    num_workers = len([n for n in nodes if n["type"] == "worker"])
    num_ranks   = 1 + num_workers

    # hosts_path  = write_mpi_hostfile(server, nodes, config)
    log_path    = remote_log_path(install_dir, "server", timestamp)
    log_dir     = remote_log_dir(install_dir)

    master_ip = next(n["private_ip"] for n in nodes if n["type"] == "master")
    mpirun_cmd  = build_server_command(config, num_ranks, master_ip)

    launch = (
        f"mkdir -p {log_dir} && "
        f"cd {install_dir} && "
        f"nohup {mpirun_cmd} > {log_path} 2>&1 &"
    )

    log.info(f"[{server}] Launching mpirun ({num_ranks} ranks) …")
    log.info(f"[{server}] Server log: {log_path}")
    # log.info(f"[{server}] Launch command: {launch}")
    _ensure_tmux(server, ssh_key)
    _tmux_run(server, launch, ssh_key)

    # Wait until mpirun is alive
    time.sleep(3)
    code, stdout, _ = ssh_exec(
        server, "pgrep -f 'bin/query' && echo SERVER_UP || echo SERVER_DOWN",
        ssh_key, timeout=15,
    )
    if "SERVER_UP" not in stdout:
        log.error(f"[{server}] ✗ mpirun does not appear to be running — check {log_path}")
        return False
    log.info(f"[{server}] ✓ mpirun started, waiting for port {tcp_port} to open...")

    # Wait until the INSERT port is actually listening (training can take minutes)
    for attempt in range(120):  # up to 10 minutes
        time.sleep(30)
        _, stdout, _ = ssh_exec(
            server,
            f"ss -tlnp | grep ':{tcp_port}' && echo PORT_OPEN || echo PORT_CLOSED",
            ssh_key, timeout=10,
        )
        if "PORT_OPEN" in stdout:
            log.info(f"[{server}] ✓ Server is listening on port {tcp_port} (after {(attempt+1)*5}s)")
            return True
        if attempt % 6 == 0:  # log every 30s
            log.info(f"[{server}] Still waiting for port {tcp_port}... ({(attempt+1)*5}s elapsed)")

    log.error(f"[{server}] ✗ Server never opened port {tcp_port} after 10 minutes")
    return False

# ---------------------------------------------------------------------------
# Start client
# ---------------------------------------------------------------------------

def start_client(client_node: Dict, master_private_ip: str, config: Dict, timestamp: str) -> bool:
    ssh_key     = _ssh_key(config)
    install_dir = _install_dir(config)
    username    = config["azure"]["username"]
    server      = _server(username, client_node)

    if not verify_binaries(server, config):
        return False

    log_path = remote_log_path(install_dir, "client", timestamp)
    log_dir  = remote_log_dir(install_dir)

    client_cmd = build_client_command(config, master_private_ip)
    launch = (
        f"mkdir -p {log_dir} && "
        f"cd {_install_dir(config)} && "
        f"export LD_LIBRARY_PATH=$LD_LIBRARY_PATH && "
        f"nohup {client_cmd} > {log_path} 2>&1 &"
    )

    log.info(f"[{server}] Starting harmony_client…")
    log.info(f"[{server}] Client log: {log_path}")
    _ensure_tmux(server, ssh_key)
    _tmux_run(server, launch, ssh_key)

    time.sleep(3)
    code, stdout, _ = ssh_exec(
        server, "pgrep -f 'harmony_client' && echo CLIENT_UP || echo CLIENT_DOWN",
        ssh_key, timeout=15,
    )
    if "CLIENT_UP" in stdout:
        log.info(f"[{server}] ✓ harmony_client started")
        return True
    log.error(f"[{server}] ✗ harmony_client does not appear to be running — check {log_path}")
    return False


# ---------------------------------------------------------------------------
# Status check
# ---------------------------------------------------------------------------

def check_status(config: Dict) -> None:
    nodes    = get_cluster_nodes(config, validate_range=False)
    username = config["azure"]["username"]
    ssh_key  = _ssh_key(config)

    print(f"\n{'Node':<25} {'IP':<18} {'Type':<10} {'Status'}")
    print("-" * 65)

    for n in sorted(nodes, key=lambda x: (x["type"], numeric_name_key(x["name"]))):
        server = _server(username, n)
        if n["type"] == "client":
            chk = "pgrep -f 'harmony_client' && echo RUNNING || echo STOPPED"
        else:
            # master and workers all appear in the mpirun process tree
            chk = "pgrep -f 'bin/query' && echo RUNNING || echo STOPPED"
        try:
            _, stdout, _ = ssh_exec(server, chk, ssh_key, timeout=12)
            status = "RUNNING" if "RUNNING" in stdout else "STOPPED"
        except Exception:
            status = "UNREACHABLE"
        ip = n["ip"] or n["private_ip"]
        print(f"  {n['name']:<23} {ip:<18} {n['type']:<10} {status}")
    print("-" * 65)


# ---------------------------------------------------------------------------
# View logs (tail from remote)
# ---------------------------------------------------------------------------

def view_logs(config: Dict, node_name: str, lines: int = 80) -> None:
    nodes    = get_cluster_nodes(config, validate_range=False)
    username = config["azure"]["username"]
    ssh_key  = _ssh_key(config)
    install_dir = _install_dir(config)

    target = next((n for n in nodes if n["name"] == node_name), None)
    if not target:
        log.error(f"Node '{node_name}' not found")
        return

    server  = _server(username, target)
    log_dir = remote_log_dir(install_dir)

    if target["type"] == "client":
        designation = "client"
    else:
        designation = "server"   # master + workers share one mpirun log

    # Find the most recent log for this designation
    code, stdout, _ = ssh_exec(
        server,
        f"ls -1t {log_dir}/{designation}_*.log 2>/dev/null | head -1",
        ssh_key, timeout=15,
    )
    log_file = stdout.strip()
    if not log_file:
        log.error(f"No log files found in {log_dir} on {node_name}")
        return

    code, stdout, _ = ssh_exec(server, f"tail -n {lines} {log_file}", ssh_key, timeout=20)
    print(f"\n--- {node_name} : {log_file} (last {lines} lines) ---\n")
    print(stdout)


# ---------------------------------------------------------------------------
# Download logs
# ---------------------------------------------------------------------------

def download_logs(config: Dict, output_dir: str, timestamp: str) -> None:
    """
    Download all *_<timestamp>.log files from every node into
    <output_dir>/<dataset>_<nworkers>w_<timestamp>/.
    Also downloads CSV from client node if client.csv_path is set.
    """
    nodes    = get_cluster_nodes(config, validate_range=False)
    username = config["azure"]["username"]
    ssh_key  = os.path.expanduser(_ssh_key(config))
    install_dir = _install_dir(config)

    dataset   = config.get("experiment", {}).get("dataset", "harmony")
    nworkers  = len([n for n in nodes if n["type"] == "worker"])
    bundle    = f"{dataset}_{nworkers}w"
    ts        = timestamp.replace(" ", "_")
    dest_root = os.path.expanduser(os.path.join(output_dir, f"{bundle}_{ts}"))
    os.makedirs(dest_root, exist_ok=True)

    log_dir  = remote_log_dir(install_dir)
    ssh_opts = ["-i", ssh_key, "-oStrictHostKeyChecking=no", "-oConnectTimeout=30", "-P", "22"]

    log.info(f"Downloading logs timestamped {ts} from {len(nodes)} nodes...")
    log.info(f"Local destination: {os.path.abspath(dest_root)}")

    for n in nodes:
        server  = _server(username, n)
        ip      = n["ip"] or n["private_ip"]
        name    = n["name"]

        # List matching log files on remote
        list_cmd = f"ls -1 {log_dir}/*_{ts}.log 2>/dev/null || true"
        _, stdout, _ = ssh_exec(server, list_cmd, ssh_key, timeout=20)
        paths = [p.strip() for p in stdout.splitlines() if p.strip()]

        if not paths:
            log.info(f"  (no matching logs) {name}")
            continue

        for remote_path in paths:
            base       = os.path.basename(remote_path)
            local_path = os.path.join(dest_root, base)
            log.info(f"Downloading {name}: {remote_path} -> {local_path}")
            result = subprocess.run(
                ["scp"] + ssh_opts + [f"{username}@{ip}:{remote_path}", local_path],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                size = os.path.getsize(local_path)
                log.info(f"  ✓ {name}: {local_path}  ({size:,} bytes)")
            else:
                log.warning(f"  ✗ {name}: failed {remote_path} - {result.stderr.strip()}")

    # Download CSV from client node if configured
    remote_csv = config.get("client", {}).get("csv_path", "")
    if remote_csv:
        client_node = next((n for n in nodes if n["type"] == "client"), None)
        if client_node:
            ip     = client_node["ip"] or client_node["private_ip"]
            name   = client_node["name"]
            server = f"{username}@{ip}"
            local_csv = os.path.join(dest_root, os.path.basename(remote_csv))
            log.info(f"Downloading CSV from {name}: {remote_csv} -> {local_csv}")
            result = subprocess.run(
                ["scp"] + ssh_opts + [f"{server}:{remote_csv}", local_csv],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                size = os.path.getsize(local_csv)
                log.info(f"  ✓ {name}: {local_csv}  ({size:,} bytes)")
            else:
                log.warning(f"  ✗ {name}: failed CSV {remote_csv} - {result.stderr.strip()}")
        else:
            log.warning("csv_path set but no client node found — skipping CSV download")

    # Append experiment record to experiments.txt
    local_logs_dir = os.path.expanduser(
        config.get("local_logs_dir", "~/harmony_logs")
    )
    exp_file = os.path.join(local_logs_dir, "experiments.txt")
    record = {
        "log_timestamp": ts,
        "experiment_bundle": bundle,
        "summary": f"{dataset} {nworkers}w nprobe={config.get('experiment',{}).get('nprobe')} "
                   f"group={config.get('experiment',{}).get('group')} "
                   f"mode={config.get('experiment',{}).get('mode')}",
        "overlay": config.get("experiment", {}),
        "service_results": {n["name"]: True for n in nodes},
    }
    os.makedirs(local_logs_dir, exist_ok=True)
    with open(exp_file, "a") as f:
        f.write(json.dumps(record) + "\n")
    log.info(f"Experiment record appended to {exp_file}")
    log.info(f"All logs saved to: {os.path.abspath(dest_root)}")


# ---------------------------------------------------------------------------
# Orchestrate full cluster start
# ---------------------------------------------------------------------------

def start_cluster(config: Dict, timestamp: str) -> Dict[str, bool]:
    nodes = get_cluster_nodes(config, validate_range=True)

    master_node = next((n for n in nodes if n["type"] == "master"), None)
    client_node = next((n for n in nodes if n["type"] == "client"), None)

    if not master_node:
        log.error("No master node found — check config vm_range")
        sys.exit(1)
    if not client_node:
        log.error(f"Client VM '{config['cluster']['client_name']}' not found")
        sys.exit(1)

    results: Dict[str, bool] = {}

    # --- Start server (mpirun) on master VM ---
    log.info("\n" + "=" * 60)
    log.info("STARTING SERVER (master + workers via mpirun)")
    log.info("=" * 60)
    ok = start_server(master_node, nodes, config, timestamp)
    results[master_node["name"]] = ok

    if not ok:
        log.warning("Server failed to start — skipping client launch")
    else:
        # --- Start client ---
        log.info("\n" + "=" * 60)
        log.info("STARTING CLIENT")
        log.info("=" * 60)
        ok = start_client(client_node, master_node["private_ip"], config, timestamp)
        results[client_node["name"]] = ok

    # Summary
    log.info("\n" + "=" * 60)
    log.info("STARTUP SUMMARY")
    log.info("=" * 60)
    for name, ok in sorted(results.items(), key=lambda x: numeric_name_key(x[0])):
        log.info(f"  {name}: {'RUNNING' if ok else 'FAILED'}")

    return results


def stop_cluster(config: Dict) -> None:
    nodes    = get_cluster_nodes(config, validate_range=False)
    username = config["azure"]["username"]
    ssh_key  = _ssh_key(config)
    log.info(f"Stopping services on {len(nodes)} node(s)…")
    threads = [
        threading.Thread(target=stop_services, args=(_server(username, n), ssh_key))
        for n in nodes
    ]
    for t in threads: t.start()
    for t in threads: t.join()
    log.info("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Manage Harmony services on the cluster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start all services
  python start_services.py --config config.yaml --start

  # Stop all services
  python start_services.py --config config.yaml --stop

  # Stop then start
  python start_services.py --config config.yaml --restart

  # Check which processes are running on each node
  python start_services.py --config config.yaml --status

  # Tail logs from the master node
  python start_services.py --config config.yaml --logs harmony-1

  # Download all logs for a given timestamp into local_logs_dir
  python start_services.py --config config.yaml --download-logs 20260323_120000

  # Download into a custom directory
  python start_services.py --config config.yaml --download-logs ~/my_results 20260323_120000
        """,
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--start",   action="store_true", help="Start master+workers+client")
    parser.add_argument("--stop",    action="store_true", help="Kill all Harmony processes")
    parser.add_argument("--restart", action="store_true", help="Stop then start")
    parser.add_argument("--status",  action="store_true", help="Show process status on each node")
    parser.add_argument("--logs",    metavar="NODE",      help="Tail logs from NODE")
    parser.add_argument("--lines",   type=int, default=80, help="Lines to tail (default 80)")
    parser.add_argument(
        "--download-logs",
        nargs="+",
        metavar="ARG",
        help="TIMESTAMP  or  OUTPUT_DIR TIMESTAMP",
    )
    parser.add_argument(
        "--set", action="append", metavar="KEY=VALUE",
        help="Override a config value, e.g. --set experiment.nprobe=200",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Apply --set overrides
    if args.set:
        for kv in args.set:
            key, _, val = kv.partition("=")
            parts = key.split(".")
            d = config
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = yaml.safe_load(val)

    if args.start or args.restart:
        if args.restart:
            stop_cluster(config)
            time.sleep(3)
        timestamp = get_log_timestamp()
        log.info(f"Log timestamp: {timestamp}")
        start_cluster(config, timestamp)

    elif args.stop:
        stop_cluster(config)

    elif args.status:
        check_status(config)

    elif args.logs:
        view_logs(config, args.logs, args.lines)

    elif args.download_logs is not None:
        parts = args.download_logs
        if len(parts) == 1:
            out_dir = os.path.expanduser(
                config.get("local_logs_dir", "~/harmony_logs")
            )
            download_logs(config, out_dir, parts[0])
        elif len(parts) == 2:
            download_logs(config, parts[0], parts[1])
        else:
            parser.error("--download-logs expects TIMESTAMP or OUTPUT_DIR TIMESTAMP")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
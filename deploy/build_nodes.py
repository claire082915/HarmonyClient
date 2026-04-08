#!/usr/bin/env python3
"""
build_nodes.py — Clone repo, install dependencies, and build Harmony binaries
on every cluster node (client, master, workers).

What this does on each node
---------------------------
1. Clones (or pulls) the Harmony repo to <install_dir>
2. Installs system packages: build-essential, cmake, libopenmpi-dev, libomp-dev
3. Sources oneAPI setvars.sh so MKLROOT is available
4. Runs: cmake -B release -DCMAKE_BUILD_TYPE=Release .
          cmake --build release -j$(nproc)
5. Verifies the three expected binaries exist:
     release/bin/query
     release/bin/harmony_client

Notes
-----
- GCC-13 is expected to already be installed at <gcc_prefix> (set in config).
  If not, pass --install-gcc to attempt a build from source (slow, ~20 min).
- Eigen is expected at ~/eigen. If missing the build will fail; install it once
  manually:  cd ~ && git clone https://gitlab.com/libeigen/eigen.git eigen
- FAISS is a CMake subdirectory (third_party/faiss), no separate install needed.

Usage
-----
    python build_nodes.py --config config.yaml               # all nodes in parallel
    python build_nodes.py --config config.yaml --sequential  # one at a time
    python build_nodes.py --config config.yaml --client      # client node only
    python build_nodes.py --config config.yaml --master      # master node only
    python build_nodes.py --config config.yaml --workers     # worker nodes only
"""

import argparse
import os
import subprocess
import sys
import threading
from typing import Dict, List, Tuple

import yaml
from cluster_utils import get_cluster_nodes, ssh_exec, ssh_exec_stream, numeric_name_key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _server(username: str, node: Dict) -> str:
    ip = node["ip"] or node["private_ip"]
    return f"{username}@{ip}"


def _source_oneapi(config: Dict) -> str:
    setvars = config.get("oneapi_setvars", "~/intel/oneapi/setvars.sh")
    return f"source {setvars} --force"


# ---------------------------------------------------------------------------
# Per-node build logic
# ---------------------------------------------------------------------------

def install_deps(server: str, ssh_key: str, install_gcc: bool, config: Dict) -> bool:
    """Install all dependencies: oneAPI MKL, GCC-13, cmake, eigen."""
    
    # 1. Install oneAPI MKL
    print(f"\n[{server}] Installing Intel oneAPI MKL...")
    mkl_cmds = " && ".join([
        "wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null",
        "echo 'deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main' | sudo tee /etc/apt/sources.list.d/oneAPI.list",
        "sudo apt-get update -qq",
        "sudo apt-get install -y intel-oneapi-mkl-devel",
    ])
    code = ssh_exec_stream(server, mkl_cmds, ssh_key)
    if code != 0:
        print(f"[{server}] WARNING: oneAPI MKL install had non-zero exit — continuing")

    # 2. Install GCC-13
    print(f"\n[{server}] Installing GCC-13...")
    gcc_cmds = " && ".join([
        "sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y",
        "sudo apt-get update -qq",
        "sudo apt-get install -y gcc-13 g++-13",
        "mkdir -p ~/gcc-13.2/bin",
        "ln -sf /usr/bin/gcc-13 ~/gcc-13.2/bin/gcc",
        "ln -sf /usr/bin/g++-13 ~/gcc-13.2/bin/g++",
    ])
    code = ssh_exec_stream(server, gcc_cmds, ssh_key)
    if code != 0:
        print(f"[{server}] WARNING: GCC-13 install had non-zero exit — continuing")

    # 3. Install cmake via snap
    print(f"\n[{server}] Installing cmake via snap...")
    cmake_cmds = " && ".join([
        "sudo apt-get remove -y cmake 2>/dev/null || true",
        "sudo snap install cmake --classic",
        "export PATH=/snap/bin:$PATH",
    ])
    code = ssh_exec_stream(server, cmake_cmds, ssh_key)
    if code != 0:
        print(f"[{server}] WARNING: cmake install had non-zero exit — continuing")

    # 4. Install Eigen
    print(f"\n[{server}] Installing Eigen3...")
    code = ssh_exec_stream(
        server,
        "sudo apt-get install -y libeigen3-dev",
        ssh_key,
    )
    if code != 0:
        print(f"[{server}] WARNING: Eigen install had non-zero exit — continuing")

    # 5. Install other base packages
    print(f"\n[{server}] Installing base packages...")
    pkgs = (
        "build-essential git pkg-config "
        "libopenmpi-dev openmpi-bin "
        "libomp-dev python3-pip wget curl"
    )
    code = ssh_exec_stream(
        server,
        f"sudo apt-get install -y {pkgs}",
        ssh_key,
    )
    if code != 0:
        print(f"[{server}] WARNING: base packages had non-zero exit — continuing")

    return True


def clone_or_pull(server: str, ssh_key: str, config: Dict) -> bool:
    """Clone the Harmony repo, or pull latest on the configured branch."""
    install_dir = config.get("install_dir", "~/Harmony")
    repo_url = config.get("repo_url", "https://github.com/claire082915/HarmonyClient")
    branch = config.get("repo_branch", "main")

    # Inject credentials if provided
    token_file = config.get("github_token")
    if token_file:
        token_file = os.path.expanduser(token_file)
        if os.path.exists(token_file):
            with open(token_file) as f:
                token = f.read().strip()
            gh_user = config.get("github_username", "git")
            # Rewrite https URL to include credentials
            repo_url = repo_url.replace(
                "https://", f"https://{gh_user}:{token}@"
            )

    check_cmd = f"test -d {install_dir}/.git && echo EXISTS || echo MISSING"
    code, stdout, _ = ssh_exec(server, check_cmd, ssh_key, timeout=15)

    if "MISSING" in stdout:
        print(f"\n[{server}] Cloning {repo_url} -> {install_dir}…")
        clone_cmd = f"git clone --branch {branch} {repo_url} {install_dir}"
        return ssh_exec_stream(server, clone_cmd, ssh_key) == 0
    else:
        print(f"\n[{server}] Pulling latest on branch {branch}…")
        pull_cmd = f"cd {install_dir} && git fetch origin && git checkout {branch} && git pull origin {branch}"
        return ssh_exec_stream(server, pull_cmd, ssh_key) == 0


def build_harmony(server: str, ssh_key: str, config: Dict) -> bool:
    """Run CMake configure + build."""
    install_dir = config.get("install_dir", "~/Harmony")
    source_oneapi = _source_oneapi(config)

    build_cmd = " && ".join([
        f"cd {install_dir}",
        "export PATH=/snap/bin:$PATH",
        source_oneapi,
        "cmake -B release -DCMAKE_BUILD_TYPE=Release .",
        "cmake --build release -j$(nproc)",
    ])
    print(f"\n[{server}] Building Harmony (cmake Release)…")
    return ssh_exec_stream(server, build_cmd, ssh_key) == 0


def verify_binaries(server: str, ssh_key: str, config: Dict) -> bool:
    """Check that the expected binaries were produced."""
    install_dir = config.get("install_dir", "~/Harmony")
    binaries = [
        f"{install_dir}/release/bin/query",
        f"{install_dir}/release/bin/harmony_client",
    ]
    cmd = "ls -lh " + " ".join(binaries) + " 2>/dev/null"
    code, stdout, _ = ssh_exec(server, cmd, ssh_key, timeout=20)
    if code == 0:
        print(f"\n[{server}] ✓ Binaries found:")
        for line in stdout.strip().splitlines():
            print(f"    {line}")
        return True
    print(f"\n[{server}] ✗ One or more binaries missing after build")
    return False


def build_node(node: Dict, config: Dict, install_gcc: bool) -> bool:
    """Full build pipeline for a single node."""
    username = config["azure"]["username"]
    ssh_key = config["azure"]["ssh_private_key"]
    server = _server(username, node)
    name = node["name"]

    print(f"\n{'#' * 60}")
    print(f"# BUILDING ON: {name}  ({server})  [{node['type']}]")
    print(f"{'#' * 60}")

    if not install_deps(server, ssh_key, install_gcc, config):
        return False
    if not clone_or_pull(server, ssh_key, config):
        return False
    if not build_harmony(server, ssh_key, config):
        return False
    return verify_binaries(server, ssh_key, config)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build Harmony binaries on cluster nodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build on all nodes in parallel (default)
  python build_nodes.py --config config.yaml

  # Build one node at a time
  python build_nodes.py --config config.yaml --sequential

  # Build only the client node
  python build_nodes.py --config config.yaml --client

  # Build only master + workers
  python build_nodes.py --config config.yaml --master --workers

  # Also compile GCC-13 from source on each node (~20 min per node)
  python build_nodes.py --config config.yaml --install-gcc
        """,
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--client",  action="store_true", help="Build client node only")
    parser.add_argument("--master",  action="store_true", help="Build master node only")
    parser.add_argument("--workers", action="store_true", help="Build worker nodes only")
    parser.add_argument("--sequential", action="store_true", help="Build nodes one at a time")
    parser.add_argument("--install-gcc", action="store_true",
                        help="Build GCC-13.2 from source on each node (slow)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    all_nodes = get_cluster_nodes(config, validate_range=False)

    # Filter by selection flags
    any_selected = args.client or args.master or args.workers
    if any_selected:
        wanted = set()
        if args.client:  wanted.add("client")
        if args.master:  wanted.add("master")
        if args.workers: wanted.add("worker")
        nodes = [n for n in all_nodes if n["type"] in wanted]
    else:
        nodes = all_nodes  # build everything

    if not nodes:
        print("No matching nodes found. Check config.yaml.")
        sys.exit(1)

    print(f"Building on {len(nodes)} node(s):")
    for n in sorted(nodes, key=lambda x: numeric_name_key(x["name"])):
        print(f"  {n['name']:25s} [{n['type']}]  ip={n['ip'] or n['private_ip']}")

    results: Dict[str, bool] = {}

    if args.sequential or len(nodes) == 1:
        for n in nodes:
            results[n["name"]] = build_node(n, config, args.install_gcc)
    else:
        lock = threading.Lock()

        def _thread(n):
            ok = build_node(n, config, args.install_gcc)
            with lock:
                results[n["name"]] = ok

        threads = [threading.Thread(target=_thread, args=(n,)) for n in nodes]
        for t in threads: t.start()
        for t in threads: t.join()

    # Summary
    print("\n" + "=" * 60)
    print("BUILD SUMMARY")
    print("=" * 60)
    for name, ok in sorted(results.items(), key=lambda x: numeric_name_key(x[0])):
        print(f"  {name}: {'✓ SUCCESS' if ok else '✗ FAILED'}")
    ok_count = sum(results.values())
    print(f"\nTotal: {ok_count}/{len(results)} builds succeeded")
    if ok_count < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
"""
cluster_utils.py — shared SSH helpers and node discovery for Harmony deploy scripts.

Node types
----------
  client  — runs harmony_client (no MPI)
  master  — MPI rank 0, runs `query` binary with --serve
  worker  — MPI ranks 1..N, also run `query` binary (worker path)

Node discovery
--------------
Primary:   Azure SDK  (requires AZURE_SUBSCRIPTION_ID env var)
Fallback:  static `nodes:` list in config.yaml
"""

import os
import re
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------

def _ssh_opts(ssh_key: str) -> List[str]:
    return [
        "-i", os.path.expanduser(ssh_key),
        "-oStrictHostKeyChecking=no",
        "-oConnectTimeout=30",
    ]


def ssh_exec(
    server: str,
    command: str,
    ssh_key: str,
    stream_output: bool = False,
    timeout: int = 600,
) -> Tuple[int, str, str]:
    """
    Run *command* on *server* via SSH.
    Returns (exit_code, stdout, stderr).
    When stream_output=True, stdout is printed live and returned as an empty string.
    """
    opts = _ssh_opts(ssh_key)
    if stream_output:
        print(f"\n{'=' * 60}")
        print(f"[{server}] $ {command}")
        print("=" * 60)
        proc = subprocess.Popen(
            ["ssh"] + opts + [server, command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout:
            print(line, end="")
        proc.wait()
        status = "✓" if proc.returncode == 0 else "✗"
        print(f"\n[{server}] {status} exit={proc.returncode}")
        return proc.returncode, "", ""

    result = subprocess.run(
        ["ssh"] + opts + [server, command],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


def ssh_exec_stream(server: str, command: str, ssh_key: str) -> int:
    """Convenience wrapper that streams output and returns exit code."""
    code, _, _ = ssh_exec(server, command, ssh_key, stream_output=True)
    return code


def scp_download(
    server: str,
    remote_path: str,
    local_path: str,
    ssh_key: str,
    timeout: int = 120,
) -> bool:
    """Download a single file from *server* via SCP. Returns True on success."""
    opts = [
        "-i", os.path.expanduser(ssh_key),
        "-oStrictHostKeyChecking=no",
        "-oConnectTimeout=30",
        "-P", "22",
    ]
    result = subprocess.run(
        ["scp"] + opts + [f"{server}:{remote_path}", local_path],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Node discovery
# ---------------------------------------------------------------------------

def _parse_vm_range(vm_range: str) -> Tuple[int, int]:
    """Parse '1-4' -> (1, 4)."""
    parts = vm_range.split("-")
    if len(parts) != 2:
        raise ValueError(f"vm_range must be 'start-end', got: {vm_range!r}")
    return int(parts[0]), int(parts[1])


def get_cluster_nodes(config: Dict, validate_range: bool = True) -> List[Dict]:
    """
    Return a list of node dicts:
        { name, ip (public), private_ip, type }
    type is one of: client | master | worker

    Discovery order:
      1. Azure SDK (if azure-mgmt-compute is installed and AZURE_SUBSCRIPTION_ID is set)
      2. Static `nodes:` list in config (fallback)
    """
    try:
        return _discover_azure_nodes(config, validate_range)
    except Exception as exc:
        print(f"[cluster_utils] Azure discovery failed ({exc}), falling back to static node list.")
        return _static_nodes(config)


def _discover_azure_nodes(config: Dict, validate_range: bool) -> List[Dict]:
    from azure.identity import DefaultAzureCredential
    from azure.mgmt.compute import ComputeManagementClient
    from azure.mgmt.network import NetworkManagementClient

    subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
    if not subscription_id:
        raise EnvironmentError("AZURE_SUBSCRIPTION_ID not set")

    credential = DefaultAzureCredential()
    rg = config["resource_group"]["name"]
    prefix = config["cluster"]["vm_name_prefix"]
    client_name = config["cluster"]["client_name"]
    vm_range_str = config["cluster"]["vm_range"]
    start_idx, end_idx = _parse_vm_range(vm_range_str)

    compute_client = ComputeManagementClient(credential, subscription_id)
    network_client = NetworkManagementClient(credential, subscription_id)

    def _get_ips(vm_name: str) -> Tuple[Optional[str], Optional[str]]:
        """Return (public_ip, private_ip) for a VM."""
        try:
            vm = compute_client.virtual_machines.get(rg, vm_name)
            nic_id = vm.network_profile.network_interfaces[0].id
            nic_name = nic_id.split("/")[-1]
            nic = network_client.network_interfaces.get(rg, nic_name)
            ip_config = nic.ip_configurations[0]
            private_ip = ip_config.private_ip_address

            public_ip = None
            if ip_config.public_ip_address:
                pub_name = ip_config.public_ip_address.id.split("/")[-1]
                pub_obj = network_client.public_ip_addresses.get(rg, pub_name)
                public_ip = pub_obj.ip_address
            return public_ip, private_ip
        except Exception:
            return None, None

    nodes: List[Dict] = []

    # Client VM
    pub, priv = _get_ips(client_name)
    nodes.append({"name": client_name, "ip": pub, "private_ip": priv, "type": "client"})

    # Master + worker VMs (first index = master, rest = workers)
    for idx in range(start_idx, end_idx + 1):
        vm_name = f"{prefix}-{idx}"
        pub, priv = _get_ips(vm_name)
        node_type = "master" if idx == start_idx else "worker"
        nodes.append({"name": vm_name, "ip": pub, "private_ip": priv, "type": node_type})

    if validate_range:
        missing = [n["name"] for n in nodes if not n["private_ip"]]
        if missing:
            raise RuntimeError(f"Could not resolve IPs for: {missing}")

    return nodes


def _static_nodes(config: Dict) -> List[Dict]:
    """Return nodes from the static `nodes:` list in config (fallback)."""
    raw = config.get("nodes")
    if not raw:
        raise ValueError("No static `nodes:` list in config and Azure discovery unavailable.")
    result = []
    for n in raw:
        node_type = n.get("type", "worker")
        # Normalise HorizANN 'main' -> 'master'
        if node_type == "main":
            node_type = "master"
        result.append({
            "name": n["name"],
            "ip": n.get("ip"),
            "private_ip": n.get("private_ip") or n.get("ip"),
            "type": node_type,
        })
    return result


# ---------------------------------------------------------------------------
# Log path helpers
# ---------------------------------------------------------------------------

def get_log_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def remote_log_dir(install_dir: str) -> str:
    return f"{install_dir.rstrip('/')}/logs"


def remote_log_path(install_dir: str, designation: str, timestamp: str) -> str:
    return f"{remote_log_dir(install_dir)}/{designation}_{timestamp}.log"


def numeric_name_key(name: str) -> int:
    """Sort key: extract trailing integer from VM name (e.g. 'harmony-3' -> 3)."""
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else 0
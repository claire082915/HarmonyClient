#!/usr/bin/env python3
"""
Manage Docker MPI cluster: start containers, SSH, and run processes.
Updated with HARMONY-specific commands and Intel MPI fixes.
"""

import subprocess
import sys
import time
import argparse
import os
from typing import List, Optional


class DockerMPICluster:
    """Manages Docker MPI cluster lifecycle and operations."""
    
    def __init__(self, compose_file: str = "docker-compose.yml"):
        self.compose_file = compose_file
        self.head_node = "harmony-mpi-head"
        self.workers = [
            "harmony-mpi-worker-1",
            "harmony-mpi-worker-2",
            "harmony-mpi-worker-3",
            "harmony-mpi-worker-4"
        ]
        self.all_nodes = [self.head_node] + self.workers
    
    def check_docker_running(self) -> bool:
        """Check if Docker daemon is running."""
        try:
            subprocess.run(
                ["docker", "info"],
                capture_output=True,
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("✗ Docker is not running or not installed")
            return False
    
    def check_compose_file_exists(self) -> bool:
        """Check if docker-compose.yml exists."""
        if not os.path.exists(self.compose_file):
            print(f"✗ Docker Compose file not found: {self.compose_file}")
            return False
        return True
    
    def start_cluster(self, build: bool = False):
        """Start the Docker cluster."""
        if not self.check_docker_running():
            return False
        
        if not self.check_compose_file_exists():
            return False
        
        print("Starting Docker MPI cluster...")
        
        cmd = ["docker", "compose", "-f", self.compose_file, "up", "-d"]
        
        if build:
            print("Building images first...")
            cmd.append("--build")
        
        try:
            subprocess.run(cmd, check=True)
            print("✓ Cluster started successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to start cluster: {e}")
            return False
    
    def stop_cluster(self):
        """Stop the Docker cluster."""
        print("Stopping Docker MPI cluster...")
        
        try:
            subprocess.run(
                ["docker", "compose", "-f", self.compose_file, "down"],
                check=True
            )
            print("✓ Cluster stopped successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to stop cluster: {e}")
            return False
    
    def restart_cluster(self, build: bool = False):
        """Restart the Docker cluster."""
        print("Restarting Docker MPI cluster...")
        self.stop_cluster()
        time.sleep(2)
        return self.start_cluster(build=build)
    
    def check_container_running(self, container: str) -> bool:
        """Check if a container is running."""
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", container],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip() == "true"
        except subprocess.CalledProcessError:
            return False
    
    def wait_for_containers(self, timeout: int = 60) -> bool:
        """Wait for all containers to be running."""
        print("Waiting for containers to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            all_running = all(
                self.check_container_running(node) for node in self.all_nodes
            )
            if all_running:
                print("✓ All containers are running")
                # Give SSH a moment to start
                time.sleep(3)
                return True
            time.sleep(2)
        
        print("✗ Timeout waiting for containers")
        return False
    
    def ensure_cluster_running(self, build: bool = False) -> bool:
        """Ensure cluster is running, start if needed."""
        all_running = all(
            self.check_container_running(node) for node in self.all_nodes
        )
        
        if all_running:
            print("✓ Cluster is already running")
            return True
        else:
            print("Cluster not running, starting it now...")
            if self.start_cluster(build=build):
                return self.wait_for_containers()
            return False
    
    def exec_in_container(
        self,
        container: str,
        command: str,
        interactive: bool = False,
        check: bool = True
    ) -> Optional[subprocess.CompletedProcess]:
        """Execute a command in a Docker container."""
        docker_cmd = ["docker", "exec"]
        
        if interactive:
            docker_cmd.append("-it")
        
        docker_cmd.extend([container, "bash", "-c", command])
        
        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=not interactive,
                text=True,
                check=check
            )
            return result
        except subprocess.CalledProcessError as e:
            print(f"Error executing command in {container}: {e}")
            return None
    
    def ssh_to_container(self, container: str, command: Optional[str] = None):
        """SSH into a container (via docker exec) and optionally run a command."""
        if command:
            cmd = f'ssh -o StrictHostKeyChecking=no {container} "{command}"'
            result = self.exec_in_container(self.head_node, cmd, interactive=False)
            if result and result.stdout:
                print(result.stdout)
            if result and result.stderr:
                print(result.stderr, file=sys.stderr)
        else:
            cmd = f"ssh -o StrictHostKeyChecking=no {container}"
            self.exec_in_container(self.head_node, cmd, interactive=True)
    
    def start_process_on_all_nodes(self, command: str, background: bool = False):
        """Start a process on all nodes via SSH from head node."""
        print(f"Starting process on all nodes: {command}")
        
        bg_suffix = " &" if background else ""
        
        for node in self.all_nodes:
            print(f"  → {node}")
            ssh_cmd = f'ssh -o StrictHostKeyChecking=no {node} "{command}{bg_suffix}"'
            result = self.exec_in_container(self.head_node, ssh_cmd)
            
            if result and result.returncode == 0:
                print(f"    ✓ Success")
                if result.stdout:
                    print(f"    Output: {result.stdout.strip()}")
            else:
                print(f"    ✗ Failed")
    
    def run_mpi_command(
        self,
        mpi_command: str,
        num_processes: int = 5,
        hostfile: str = "/app/mpi/hostfile",
        use_intel_mpi: bool = True
    ):
        """Run an MPI command across the cluster."""
        if use_intel_mpi:
            # Intel MPI with container fixes
            cmd = (
                f"source /opt/intel/oneapi/setvars.sh && "
                f"export I_MPI_SHM=off && "
                f"export I_MPI_FABRICS=tcp && "
                f"mpirun -n {num_processes} -ppn 1 "
                f"-f {hostfile} "
                f"{mpi_command}"
            )
        else:
            # OpenMPI syntax
            cmd = (
                f"mpirun -np {num_processes} "
                f"-hostfile {hostfile} "
                f"{mpi_command}"
            )
        
        print(f"Running MPI command: {mpi_command}")
        self.exec_in_container(self.head_node, cmd, interactive=True)
    
    def run_harmony_query(
        self,
        dataset: str,
        num_processes: int = 5,
        benchmarks_path: str = "/app/benchmarks",
        extra_args: str = ""
    ):
        """Run HARMONY distributed query."""
        cmd = (
            f"/app/release/bin/query "
            f"--benchmarks_path {benchmarks_path} "
            f"--dataset {dataset} "
            f"--group=2 --team=2 --block=4 "
            f"--cache --verbose "
            f"{extra_args}"
        )
        
        print(f"Running HARMONY query on dataset: {dataset}")
        self.run_mpi_command(cmd, num_processes=num_processes)
    
    def run_faiss_baseline(
        self,
        dataset: str,
        benchmarks_path: str = "/app/benchmarks",
        nprobes: str = "50 100 300 1000"
    ):
        """Run FAISS baseline."""
        cmd = (
            f"source /opt/intel/oneapi/setvars.sh && "
            f"export I_MPI_SHM=off && "
            f"export I_MPI_FABRICS=tcp && "
            f"mpirun -n 1 -bind-to none "
            f"/app/release/bin/query "
            f"--benchmarks_path {benchmarks_path} "
            f"--dataset {dataset} "
            f"--nprobes {nprobes} "
            f"--run_faiss --verbose"
        )
        
        print(f"Running FAISS baseline on dataset: {dataset}")
        self.exec_in_container(self.head_node, cmd, interactive=True)
    
    def train_harmony_index(
        self,
        dataset: str,
        benchmarks_path: str = "/app/benchmarks"
    ):
        """Train HARMONY index."""
        cmd = (
            f"source /opt/intel/oneapi/setvars.sh && "
            f"/app/release/bin/query "
            f"--benchmarks_path {benchmarks_path} "
            f"--dataset {dataset} "
            f"--train_only --verbose"
        )
        
        print(f"Training HARMONY index for dataset: {dataset}")
        self.exec_in_container(self.head_node, cmd, interactive=True)
    
    def check_ssh_connectivity(self):
        """Check SSH connectivity from head to all workers."""
        print("Checking SSH connectivity...")
        
        for node in self.all_nodes:
            cmd = f'ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 {node} "hostname"'
            result = self.exec_in_container(self.head_node, cmd)
            
            if result and result.returncode == 0:
                print(f"  ✓ {node}: {result.stdout.strip()}")
            else:
                print(f"  ✗ {node}: Failed to connect")
    
    def get_cluster_status(self):
        """Get detailed status of the cluster."""
        print("\n" + "=" * 70)
        print("CLUSTER STATUS")
        print("=" * 70)
        
        for node in self.all_nodes:
            if self.check_container_running(node):
                # Get uptime and basic info
                result = self.exec_in_container(
                    node,
                    "uptime -p 2>/dev/null || echo 'unknown'",
                    check=False
                )
                uptime = result.stdout.strip() if result else "unknown"
                
                # Check if SSH is running
                ssh_result = self.exec_in_container(
                    node,
                    "pgrep sshd >/dev/null && echo 'running' || echo 'stopped'",
                    check=False
                )
                ssh_status = ssh_result.stdout.strip() if ssh_result else "unknown"
                
                print(f"\n{node}")
                print(f"  Status:  RUNNING")
                print(f"  Uptime:  {uptime}")
                print(f"  SSH:     {ssh_status}")
            else:
                print(f"\n{node}")
                print(f"  Status:  STOPPED")
        
        print("\n" + "=" * 70)
    
    def view_logs(self, container: Optional[str] = None, follow: bool = False):
        """View Docker container logs."""
        cmd = ["docker", "compose", "-f", self.compose_file, "logs"]
        
        if follow:
            cmd.append("-f")
        
        if container:
            cmd.append(container)
        
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\nLog viewing stopped")
    
    def view_results(self, dataset: str, benchmarks_path: str = "/app/benchmarks"):
        """View HARMONY results for a dataset."""
        result_file = f"{benchmarks_path}/{dataset}/result/log.csv"
        cmd = f"cat {result_file}"
        
        print(f"Results for {dataset}:")
        print("=" * 70)
        result = self.exec_in_container(self.head_node, cmd, check=False)
        if result and result.stdout:
            print(result.stdout)
        else:
            print(f"No results found at {result_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Manage Docker MPI cluster for HARMONY - no manual container management needed!"
    )
    
    parser.add_argument(
        "--compose-file", "-f",
        default="docker-compose.yml",
        help="Path to docker-compose.yml (default: docker-compose.yml)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start the cluster")
    start_parser.add_argument(
        "--build", "-b",
        action="store_true",
        help="Build images before starting"
    )
    
    # Stop command
    subparsers.add_parser("stop", help="Stop the cluster")
    
    # Restart command
    restart_parser = subparsers.add_parser("restart", help="Restart the cluster")
    restart_parser.add_argument(
        "--build", "-b",
        action="store_true",
        help="Build images before restarting"
    )
    
    # Status command
    subparsers.add_parser("status", help="Show cluster status")
    
    # Logs command
    logs_parser = subparsers.add_parser("logs", help="View container logs")
    logs_parser.add_argument("container", nargs="?", help="Specific container (optional)")
    logs_parser.add_argument("--follow", "-f", action="store_true", help="Follow logs")
    
    # Check SSH command
    subparsers.add_parser("check-ssh", help="Check SSH connectivity")
    
    # SSH command
    ssh_parser = subparsers.add_parser("ssh", help="SSH to a container")
    ssh_parser.add_argument("container", help="Container name or hostname")
    ssh_parser.add_argument("--command", "-c", help="Command to run (optional)")
    
    # Execute command
    exec_parser = subparsers.add_parser("exec-all", help="Execute command on all nodes")
    exec_parser.add_argument("process_command", help="Command to execute")
    exec_parser.add_argument(
        "--background", "-bg",
        action="store_true",
        help="Run in background"
    )
    
    # MPI run command
    mpi_parser = subparsers.add_parser("mpi-run", help="Run MPI command")
    mpi_parser.add_argument("mpi_command", help="MPI command to run")
    mpi_parser.add_argument(
        "--np", "-n",
        type=int,
        default=5,
        help="Number of processes (default: 5)"
    )
    mpi_parser.add_argument(
        "--hostfile",
        default="/app/mpi/hostfile",
        help="Path to hostfile"
    )
    
    # HARMONY query command
    harmony_parser = subparsers.add_parser("harmony", help="Run HARMONY distributed query")
    harmony_parser.add_argument("dataset", help="Dataset name (e.g., sift1m, msong, nuswide)")
    harmony_parser.add_argument(
        "--np", "-n",
        type=int,
        default=5,
        help="Number of processes (default: 5)"
    )
    harmony_parser.add_argument(
        "--extra",
        default="",
        help="Extra arguments (e.g., '--disablePruning --mode base')"
    )
    
    # FAISS baseline command
    faiss_parser = subparsers.add_parser("faiss", help="Run FAISS baseline")
    faiss_parser.add_argument("dataset", help="Dataset name")
    faiss_parser.add_argument(
        "--nprobes",
        default="50 100 300 1000",
        help="nprobe values (default: '50 100 300 1000')"
    )
    
    # Train index command
    train_parser = subparsers.add_parser("train", help="Train HARMONY index")
    train_parser.add_argument("dataset", help="Dataset name")
    
    # View results command
    results_parser = subparsers.add_parser("results", help="View HARMONY results")
    results_parser.add_argument("dataset", help="Dataset name")
    
    args = parser.parse_args()
    
    cluster = DockerMPICluster(compose_file=args.compose_file)
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Commands that don't need cluster running
    if args.command == "start":
        if cluster.start_cluster(build=args.build):
            cluster.wait_for_containers()
        sys.exit(0)
    
    elif args.command == "stop":
        cluster.stop_cluster()
        sys.exit(0)
    
    elif args.command == "restart":
        cluster.restart_cluster(build=args.build)
        sys.exit(0)
    
    elif args.command == "logs":
        cluster.view_logs(args.container, follow=args.follow)
        sys.exit(0)
    
    # For other commands, ensure cluster is running
    if not cluster.ensure_cluster_running():
        print("\n✗ Failed to start cluster. Exiting.")
        sys.exit(1)
    
    if args.command == "status":
        cluster.get_cluster_status()
    
    elif args.command == "check-ssh":
        cluster.check_ssh_connectivity()
    
    elif args.command == "ssh":
        cluster.ssh_to_container(args.container, args.command)
    
    elif args.command == "exec-all":
        cluster.start_process_on_all_nodes(args.process_command, args.background)
    
    elif args.command == "mpi-run":
        cluster.run_mpi_command(
            args.mpi_command,
            num_processes=args.np,
            hostfile=args.hostfile
        )
    
    elif args.command == "harmony":
        cluster.run_harmony_query(
            args.dataset,
            num_processes=args.np,
            extra_args=args.extra
        )
    
    elif args.command == "faiss":
        cluster.run_faiss_baseline(
            args.dataset,
            nprobes=args.nprobes
        )
    
    elif args.command == "train":
        cluster.train_harmony_index(args.dataset)
    
    elif args.command == "results":
        cluster.view_results(args.dataset)


if __name__ == "__main__":
    main()

'''
chmod +x cluster_manager.py

./cluster_manager.py start

./cluster_manager.py faiss sift1m
./cluster_manager.py train sift1m
./cluster_manager.py harmony sift1m --np 5

./cluster_manager.py faiss sift100m --nprobes "50 100 300 1000"
./cluster_manager.py train sift100m
./cluster_manager.py harmony sift100m --np 5 --extra "--nprobes 100"

./cluster_manager.py stop
'''
#!/usr/bin/env python3
"""
Generate docker-compose.yml with configurable number of workers
"""

import sys
import argparse

def generate_worker(worker_num, total_workers):
    """Generate a single worker service definition."""
    ip_suffix = 10 + worker_num
    
    return f"""
  mpi-worker-{worker_num}:
    build:
      context: ..
      dockerfile: docker/Dockerfile.mpi
    image: harmony-mpi
    hostname: mpi-worker-{worker_num}
    container_name: harmony-mpi-worker-{worker_num}
    volumes:
      - ../benchmarks:/app/benchmarks
      - ../datasets:/app/datasets
      - ssh-keys:/root/.ssh
    environment:
      - OMPI_ALLOW_RUN_AS_ROOT=1
      - OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
      - I_MPI_HYDRA_IFACE=eth0
      - I_MPI_FABRICS=tcp
      - I_MPI_SHM=off
      - I_MPI_TCP_NETMASK=eth0
    networks:
      harmony-net:
        ipv4_address: 10.5.0.{ip_suffix}
    depends_on:
      - base
    command: /usr/sbin/sshd -D
"""

def generate_compose_file(num_workers):
    """Generate complete docker-compose.yml with specified number of workers."""
    
    # Generate depends_on list for head node
    depends_list = ["base"] + [f"mpi-worker-{i}" for i in range(1, num_workers + 1)]
    depends_yaml = "\n".join([f"      - {dep}" for dep in depends_list])
    
    compose_content = f"""services:
  base:
    build:
      context: ..
      dockerfile: docker/Dockerfile.base
    image: harmony-base

  # MPI head node - coordinates the distributed execution
  mpi-head:
    build:
      context: ..
      dockerfile: docker/Dockerfile.mpi
    image: harmony-mpi
    hostname: mpi-head
    container_name: harmony-mpi-head
    volumes:
      - ../benchmarks:/app/benchmarks
      - ../datasets:/app/datasets
      - ssh-keys:/root/.ssh
    ports:
      - "2222:22"
    environment:
      - DATASET=${{DATASET:-sift100m}}
      - OMPI_ALLOW_RUN_AS_ROOT=1
      - OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
      - I_MPI_HYDRA_IFACE=eth0
      - I_MPI_FABRICS=tcp
      - I_MPI_SHM=off
      - I_MPI_TCP_NETMASK=eth0
      - NUM_WORKERS={num_workers}
    networks:
      harmony-net:
        ipv4_address: 10.5.0.10
    depends_on:
{depends_yaml}
    command:
      - /bin/bash
      - -c
      - |
        echo 'Starting SSH service on head node...'
        /usr/sbin/sshd
        
        echo 'Waiting for workers to be ready...'
        sleep 5
        
        echo "Creating MPI hostfile with {num_workers} workers..."
        echo "mpi-head" > /app/mpi/hostfile
        for i in $(seq 1 {num_workers}); do
          echo "mpi-worker-$$i" >> /app/mpi/hostfile
        done
        
        echo 'Configuring SSH for external access...'
        sed -i 's/#PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
        sed -i 's/#PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
        sed -i 's/#PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config
        sed -i 's/#ListenAddress 0.0.0.0/ListenAddress 0.0.0.0/' /etc/ssh/sshd_config
        /usr/sbin/sshd -t && /usr/sbin/sshd
        
        echo "MPI cluster ready with {num_workers} workers!"
        echo "Connect from external VM using: ssh -p 2222 root@<host_ip>"
        
        tail -f /dev/null
"""
    
    # Add all worker services
    for i in range(1, num_workers + 1):
        compose_content += generate_worker(i, num_workers)
    
    # Add networks and volumes
    compose_content += """
networks:
  harmony-net:
    driver: bridge
    ipam:
      driver: default
      config:
        - subnet: 10.5.0.0/24
          gateway: 10.5.0.1

volumes:
  ssh-keys:
"""
    
    return compose_content

def main():
    parser = argparse.ArgumentParser(description="Generate docker-compose.yml with configurable workers")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Number of worker nodes (default: 4)")
    parser.add_argument("--output", "-o", default="docker-compose.yml", help="Output file (default: docker-compose.yml)")
    
    args = parser.parse_args()
    
    if args.workers < 1:
        print("Error: Must have at least 1 worker")
        sys.exit(1)
    
    if args.workers > 50:
        print("Warning: More than 50 workers may cause performance issues")
    
    print(f"Generating docker-compose.yml with {args.workers} workers...")
    
    compose_content = generate_compose_file(args.workers)
    
    with open(args.output, 'w') as f:
        f.write(compose_content)
    
    print(f"✓ Generated {args.output} with {args.workers} worker nodes")
    print(f"  Total nodes: {args.workers + 1} (1 head + {args.workers} workers)")
    print(f"  External access: Port 2222 (SSH to head node)")
    print(f"  MPI communication: Internal Docker network only")
    print(f"\nTo start the cluster, run:")
    print(f"  ./cluster_manager.py start")

if __name__ == "__main__":
    main()
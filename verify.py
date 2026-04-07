import struct
import numpy as np

def check_fvecs(filename, num_to_check=10):
    with open(filename, 'rb') as f:
        for i in range(num_to_check):
            d = struct.unpack('i', f.read(4))[0]
            vec = np.frombuffer(f.read(d * 4), dtype=np.float32)
            print(f"Vector {i}: dim={d}, min={vec.min():.2f}, max={vec.max():.2f}, mean={vec.mean():.2f}")
            if d != 128:
                print(f"WARNING: Expected dim=128, got dim={d}")

print("Checking base vectors:")
check_fvecs('./benchmarks/sift10m/origin/sift10m_base.fvecs')
print("\nChecking query vectors:")
check_fvecs('./benchmarks/sift10m/origin/sift10m_query.fvecs')
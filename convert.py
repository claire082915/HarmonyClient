import numpy as np
import struct

def read_bvecs(fn, limit=None):
    """Read bvecs (byte vectors) file format"""
    with open(fn, 'rb') as f:
        # Read dimension
        d = struct.unpack('i', f.read(4))[0]
        f.seek(0)
        
        # Calculate size per vector (4 bytes for dim + d bytes for data)
        vec_size = 4 + d
        
        if limit:
            # Read only the first 'limit' vectors
            data = f.read(vec_size * limit)
            nvecs = limit
        else:
            data = f.read()
            nvecs = len(data) // vec_size
        
        # Parse vectors
        vecs = np.zeros((nvecs, d), dtype=np.uint8)
        for i in range(nvecs):
            offset = i * vec_size
            # Skip the dimension int (4 bytes)
            vecs[i] = np.frombuffer(data[offset+4:offset+vec_size], dtype=np.uint8)
            
            if (i + 1) % 1000000 == 0:
                print(f"Read {i+1} vectors...")
        
        return vecs

def write_fvecs(fn, vecs):
    """Write fvecs (float vectors) file format"""
    n, d = vecs.shape
    with open(fn, 'wb') as f:
        for i, vec in enumerate(vecs):
            # Write dimension as int32
            f.write(struct.pack('i', d))
            # Write vector as float32
            f.write(vec.astype(np.float32).tobytes())
            
            if (i + 1) % 1000000 == 0:
                print(f"Wrote {i+1} vectors...")

def convert_bvecs_to_fvecs(input_file, output_file, num_vectors):
    """Convert bvecs to fvecs for specified number of vectors"""
    print(f"Reading {num_vectors} vectors from {input_file}...")
    vecs = read_bvecs(input_file, limit=num_vectors)
    
    print(f"Converting to float32...")
    vecs_float = vecs.astype(np.float32)
    
    print(f"Writing to {output_file}...")
    write_fvecs(output_file, vecs_float)
    print("Done!")

# Main conversion
if __name__ == "__main__":
    # Convert first 10M base vectors from bvecs to fvecs
    print("=" * 60)
    print("Converting base vectors...")
    print("=" * 60)
    convert_bvecs_to_fvecs(
        'benchmarks/sift1b/origin/sift1b_base.bvecs',
        'benchmarks/sift100m/origin/sift100m_base.fvecs',
        num_vectors=100_000_000
    )
    
    # For SIFT1B, the standard split is:
    # - Query: 10,000 vectors (already complete)
    # - Learn: 100,000,000 vectors (take proportional amount)
    
    # Copy all query vectors (10K total, use all)
    print("\n" + "=" * 60)
    print("Converting query vectors...")
    print("=" * 60)
    convert_bvecs_to_fvecs(
        'benchmarks/sift1b/origin/sift1b_query.bvecs',
        'benchmarks/sift100m/origin/sift100m_query.fvecs',
        num_vectors=10_000  # Use all query vectors
    )
    
    # For learn vectors, you could take the first 10M as well
    # (proportional to 10M/1000M = 1% of base)
    # Or use all 100M learn vectors depending on your needs
    print("\n" + "=" * 60)
    print("Converting learn vectors...")
    print("=" * 60)
    convert_bvecs_to_fvecs(
        'benchmarks/sift1b/origin/sift1b_learn.bvecs',
        'benchmarks/sift100m/origin/sift100m_learn.fvecs',
        num_vectors=100_000_000  # Take first 10M learn vectors
    )
    
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)
    print("Output files:")
    print("  - bigann_base_10M.fvecs (10M vectors)")
    print("  - bigann_query.fvecs (10K vectors)")
    print("  - bigann_learn_10M.fvecs (10M vectors)")
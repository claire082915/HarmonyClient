import numpy as np
import struct

def read_bvecs_chunk(filepath, num_vectors_to_read):
    """
    Reads a specified number of vectors from a bvecs file.
    The bvecs format has a 4-byte integer for dimension, followed by the vector bytes.
    """
    vectors = []
    with open(filepath, 'rb') as f:
        for _ in range(num_vectors_to_read):
            # Read the dimension (4 bytes, little-endian int)
            dim_bytes = f.read(4)
            if not dim_bytes:
                break # End of file
            dimension = struct.unpack('<I', dim_bytes)[0]
            
            # Read the vector data (dimension bytes)
            vector_bytes = f.read(dimension)
            if not vector_bytes:
                break
            
            # Convert bytes to a numpy array (uint8 is typically used for SIFT bvecs)
            # The original SIFT descriptors are often uint8
            vector = np.frombuffer(vector_bytes, dtype=np.uint8)
            vectors.append(vector)
            
    return np.array(vectors)

def write_bvecs(filepath, vectors):
    """
    Writes vectors to a file in the bvecs format.
    """
    with open(filepath, 'wb') as f:
        for vector in vectors:
            dimension = vector.shape[0]
            # Write the dimension as a 4-byte little-endian integer
            f.write(struct.pack('<I', dimension))
            # Write the raw vector data
            f.write(vector.tobytes())

# --- Usage ---
source_file = 'benchmarks/sift1b/origin/sift1b_learn.bvecs' # Path to your source SIFT1B file
destination_file = 'benchmarks/sift10m2/origin/sift10m2_learn.bvecs' # Path for the new 10M vectors file
num_vectors_to_copy = 1000000 # 10 million

print(f"Reading the first {num_vectors_to_copy} vectors from {source_file}...")
# Note: This might require significant memory to hold 10M vectors in RAM.
vectors = read_bvecs_chunk(source_file, num_vectors_to_copy)

if len(vectors) == num_vectors_to_copy:
    print(f"Successfully read {len(vectors)} vectors.")
    print(f"Writing vectors to {destination_file}...")
    write_bvecs(destination_file, vectors)
    print("Done.")
else:
    print(f"Could only read {len(vectors)} vectors. Check if the source file has enough data.")


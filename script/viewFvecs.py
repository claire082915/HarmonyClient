import numpy as np
import struct

def load_fvecs(file_path, bounds=(1, 0)):
    # Open the file in binary mode
    with open(file_path, 'rb') as file:
        # Read the dimensionality (d) of the vectors
        d = struct.unpack('i', file.read(4))[0]

        # Determine the number of vectors (n) based on the file size
        file.seek(0, 2)  # Move to the end of the file
        file_size = file.tell()
        vec_sizeof = 4 + d * 4  # (int) + (d * float)
        bmax = file_size // vec_sizeof

        a, b = bounds
        if b == 0:
            b = bmax  # Set default bmax if bounds second is 0
        if not (1 <= a <= b <= bmax):
            raise ValueError("Bounds are invalid: a must be >= 1 and b <= bmax, and b >= a")

        n = b - a + 1
        vectors = []

        # Set file pointer to the beginning of the first vector to read
        file.seek((a - 1) * vec_sizeof)

        for _ in range(n):
            # Skip the first integer (d)
            file.seek(4, 1)

            # Read the vector of floats
            vector = np.fromfile(file, dtype=np.float32, count=d)
            vectors.append(vector)

        return np.array(vectors), n, d

# Usage example
file_path = '/home/lichengqi/vdb/VectorDB/benchmarks/test/origin/test_query.fvecs'
vectors, n, d = load_fvecs(file_path, bounds=(1, 5))

print(f"Loaded {n} vectors with dimension {d}.")
print(vectors)

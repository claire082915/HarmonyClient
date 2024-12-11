import numpy as np
import struct

def generate_fvecs(file_path, bounds=(1, 0), d=20):
    # Ensure bounds are valid
    a, b = bounds
    if b == 0:
        b = 100  # Assuming a default value for `b` as 100, similar to your function's behavior

    if not (1 <= a <= b):
        raise ValueError("Bounds are invalid: a must be >= 1 and b >= a")

    # Number of vectors to generate
    n = b - a + 1

    # Create the vectors
    vectors = []
    for i in range(a, b + 1):
        # Create a vector from (i-1)*d + 1 to i*d
        vector = np.linspace((i - 1) * d + 1, i * d, d, dtype=np.float32)
        vectors.append(vector)

    # Write the vectors to the fvecs file
    with open(file_path, 'wb') as file:
        # Write the dimensionality as the first integer (d)
        file.write(struct.pack('i', d))

        # Write each vector
        for vector in vectors:
            # First write the integer d (the dimension)
            file.write(struct.pack('i', d))
            # Then write the vector values
            vector.tofile(file)

    print(f"File '{file_path}' has been generated with {n} vectors of dimension {d}.")

# Usage
generate_fvecs('/home/lichengqi/vdb/VectorDB/benchmarks/test/origin/test_base.fvecs', bounds=(1, 5), d=20)
generate_fvecs('/home/lichengqi/vdb/VectorDB/benchmarks/test/origin/test_query.fvecs', bounds=(1, 5), d=20)

# import numpy as np
# import struct

# def read_ivecs(filename):
#     with open(filename, 'rb') as f:
#         data = []
#         while True:
#             try:
#                 d = struct.unpack('i', f.read(4))[0]
#                 vec = struct.unpack('i' * d, f.read(4 * d))
#                 data.append(vec)
#             except:
#                 break
#         return np.array(data)

# def write_bin(filename, data):
#     data.astype('int32').tofile(filename)

# # Read ivecs ground truth
# gt = read_ivecs('benchmarks/sift1b/origin/gnd/idx_1000M.ivecs')
# # Write as binary
# write_bin('benchmarks/sift1b/result_test/groundtruth_100.bin', gt)
# print(f"Converted {gt.shape} to groundtruth_100.bin")

import faiss
faiss.read_index("benchmarks/sift1b/index/faiss_index_nlist_1000.index")
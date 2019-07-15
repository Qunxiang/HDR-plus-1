import numpy as np
import numpy_gpu as gpu

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c1 = gpu.add(a, b)  # [5. 7. 9.]
c2 = gpu.dot(a, b)  # 32

print(c1)
print(c2)
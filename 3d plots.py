import numpy as np
import matplotlib.pyplot as plt

plt.plot([1, 2, 3, 4, 5], [1, 1, 1, 1, 1])
plt.plot([1,2,3,4,5], [0.8,0.8,0.8,0.8,0.8])
x = np.arange(10)
a = np.random.rand(10)
plt.scatter(x, a, c='purple')
b = np.random.rand(10)
plt.scatter(x, b, c='red')
plt.show()

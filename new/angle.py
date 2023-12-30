import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata

F = np.array([[0.7, 0.2], [0.6, 0.5], [0.2, 0.8], [0.8, 0.8]])

f1_rank = rankdata(F[:, 0], method='ordinal')
f2_rank = rankdata(F[:, 1], method='ordinal')



F_rank = np.column_stack([f1_rank, f2_rank, ])
plt.scatter(F[:, 0], F[:, 1], c='k', s=500)
plt.show()


plt.figure()
plt.scatter(F_rank[:, 0], F_rank[:, 1], c='k',s=500)
plt.xticks(np.arange(1, 5, 1))
plt.yticks(np.arange(1, 5, 1))
plt.show()




















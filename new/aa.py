import numpy as np

arr = np.array([[9,2],[8,4]])

# 各行の重複回数を数える
unique_rows, counts = np.unique(arr, axis=0, return_counts=True)
print(unique_rows)
print(counts)
# 最も多く重複する行を見つける
most_common_row = unique_rows[np.argmax(counts)]

print("最も多く重複する行:", most_common_row)
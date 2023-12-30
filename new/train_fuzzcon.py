import math

import numpy as np
import matplotlib.pyplot as plt
import itertools

from kessler_game.new.Controlelr23 import Controller23

N_pop = 30
N_rep = 10
count_max = 10000
p = 0.9
# 分割数
K = 3

file_path = r'C:\Users\Ayato Tomofuji\Documents\kessler24\kessler_game\new\inout_list3.txt'

with open(file_path) as f:
    lines = f.readlines()


def grid(x1, x2):
    new_gene = [-422.98248909, -128.46239109, 395.65025775, -339.31340805, -82.99984531,
                157.18145777, 94.03193966, 74.20410544, -141.6155565, 66.7441948,
                105.5832539, 136.26770441, 440.24368511, -32.15986455, -269.37155599,
                -3.07185922, 180.88739761, -17.52924744, -11.0651477, 105.48644365,
                25.6119877, 56.20575568, 85.31037087, 156.41788735, 13.28000091,
                75.04230663, 145.83883738, -5.34633099, 79.93202705, 170.01952603, ]
    gene = [-1.64531754e-01, 1.13815001e-04, 9.95596272e-02, 6.52069561e-01, 9.58253695e-01, 6.28739631e-01]
    controller = Controller23(gene, new_gene)
    memm = [(controller.mems(x1, x2)) for x1, x2 in zip(x1, x2)]
    return memm


Ruledict = {
    0: "x1 is small and x2 is small",
    1: "x1 is small and x2 is medium",
    2: "x1 is small and x2 is large",
    3: "x1 is small and x2 is don't care",
    4: "x1 is medium and x2 is small",
    5: "x1 is medium and x2 is medium",
    6: "x1 is medium and x2 is large",
    7: "x1 is medium and x2 is don't care",
    8: "x1 is large and x2 is small",
    9: "x1 is large and x2 is medium",
    10: "x1 is large and x2 is large",
    11: "x1 is large and x2 is don't care",
    12: "x1 is don't care and x2 is small",
    13: "x1 is don't care and x2 is medium",
    14: "x1 is don't care and x2 is large",
    15: "x1 is don't care and x2 is don't care"
}


def find_lim(x, y):
    x1_min = x[np.argmin(x[:, 0])]
    x1_max = x[np.argmax(x[:, 0])]
    x2_min = x[np.argmin(x[:, 1])]
    x2_max = x[np.argmax(x[:, 1])]
    y_x1_min = y[np.argmin(x[:, 0])]
    y_x1_max = y[np.argmax(x[:, 0])]
    y_x2_min = y[np.argmin(x[:, 1])]
    y_x2_max = y[np.argmax(x[:, 1])]


def membership_dist(x):
    x /= 400
    b = [0.0, 0.25, 0.5, 0.75, 1.0]
    if x <= b[0]:
        return [1.0, 0.0, 0.0, 0.0, 0.0]
    elif x <= b[1]:
        return np.array([1.0 - (x - b[0]) / (b[1] - b[0]), (x - b[0]) / (b[1] - b[0]), 0.0, 0.0, 0.0])
    elif x <= b[2]:
        return np.array([0.0, (b[2] - x) / (b[2] - b[1]),
                         (x - b[1]) / (b[2] - b[1]), 0.0, 0.0])
    elif x <= b[3]:
        return np.array([0.0, 0.0, (b[3] - x) / (b[3] - b[2]), (x - b[2]) / (b[3] - b[2]), 0.0])
    elif x <= b[4]:
        return np.array([0.0, 0.0, 0.0, (b[4] - x) / (b[4] - b[3]), (x - b[3]) / (b[4] - b[3])])
    else:
        return np.array([0.0, 0.0, 0.0, 0.0, 1.0])


def membership_out1(x):
    x /= 450
    b = [-1.00, -0.75, -0.5, -0.25, 0, 0.25, 0.50, 0.75, 1.00]
    if x <= b[0]:
        return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif x <= b[1]:
        return np.array(
            [1.0 - (x - b[0]) / (b[1] - b[2]), (x - b[0]) / (b[1] - b[0]), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    elif x <= b[2]:
        return np.array([0.0, (b[2] - x) / (b[2] - b[1]),
                         (x - b[1]) / (b[2] - b[1]), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    elif x <= b[3]:
        return np.array([0.0, 0.0, (b[3] - x) / (b[3] - b[2]), (x - b[2]) / (b[3] - b[2]), 0.0, 0.0, 0.0, 0.0, 0.0])
    elif x <= b[4]:
        return np.array([0.0, 0.0, 0.0, (b[4] - x) / (b[4] - b[3]), (x - b[3]) / (b[4] - b[3]), 0.0, 0.0, 0.0, 0.0])
    elif x <= b[5]:
        return np.array([0.0, 0.0, 0.0, 0.0, (b[5] - x) / (b[5] - b[4]), (x - b[4]) / (b[5] - b[4]), 0.0, 0.0, 0.0])
    elif x <= b[6]:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, (b[6] - x) / (b[6] - b[5]), (x - b[5]) / (b[6] - b[5]), 0.0, 0.0])
    elif x <= b[7]:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, (b[7] - x) / (b[7] - b[6]), (x - b[6]) / (b[7] - b[6]), 0.0])
    elif x <= b[8]:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, (b[8] - x) / (b[8] - b[7]), (x - b[7]) / (b[8] - b[7])])
    else:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def membership2(x, b=None):
    x /= 180
    if b is None:
        b = [-1.0, -0.5, 0, 0.50, 1.0]

    if x <= b[0]:
        return [1.0, 0.0, 0.0, 0.0, 0.0]
    elif x <= b[1]:
        return [1.0 - (x - b[0]) / (b[1] - b[2]), (x - b[0]) / (b[1] - b[0]), 0.0, 0.0, 0.0]
    elif x <= b[2]:
        return ([0.0, (b[2] - x) / (b[2] - b[1]),
                 (x - b[1]) / (b[2] - b[1]), 0.0, 0.0])
    elif x <= b[3]:
        return [0.0, 0.0, (b[3] - x) / (b[3] - b[2]), (x - b[2]) / (b[3] - b[2]), 0.0]
    elif x <= b[4]:
        return [0.0, 0.0, 0.0, (b[4] - x) / (b[4] - b[3]), (x - b[3]) / (b[4] - b[3])]
    else:
        return [0.0, 0.0, 0.0, 0.0, 1.0]


def out_speed(x, speed_bound=None):
    x = np.array(x[:5])
    speed_bound = np.array([-480, -240, 0, 240, 480])
    return np.inner(x[:, 0], speed_bound)


def mems(X):
    out = [[membership_dist(x[0]), membership2(x[1])] for x in X]
    return out


def mems2(y):
    out = [[membership_out1(a[0]), membership2(a[1])] for a in y]
    return out


def conf(x):
    mem = np.array(mems(x))
    C_dist = np.argmax(mem[:, 0], axis=1)
    C_angle = np.argmax(mem[:, 1], axis=1)
    # 結合する
    C = np.array([C_dist, C_angle]).T
    return C.tolist()


def conf_out(y):
    mem_out_dist = np.array([membership_out1(a[0]) for a in y])
    mem_out_angle = np.array([membership2(a[1]) for a in y])
    # 各行ごとに最大値のインデックスを取得
    C_out = np.argmax(mem_out_dist, axis=1)
    C_out_Angle = np.argmax(mem_out_angle, axis=1)
    # 結合する
    C_out = np.array([C_out, C_out_Angle]).T
    return C_out.tolist()

g = 1
if g:
    x1 = np.linspace(0, 400, 1000)
    x2 = np.linspace(-180, 180, 1000)
    X = np.meshgrid(x1, x2)
    X = np.array(X).reshape(2, -1).T
    x1 = X[:, 0]
    x2 = X[:, 1]
    y = np.array(grid(x1, x2))
else:
    X = []
    y = []
    for line in lines:
        values = line.strip().split(',')
        x_values = list(map(float, values[:2]))
        y_values = list(map(float, values[2:4]))
        X.append(x_values)
        y.append(y_values)
    X = np.array(X)
    y = np.array(y)


C = conf(X)
C_out = conf_out(y)

# 省略せずに表示する
np.set_printoptions(threshold=np.inf)

# 要素数5*5の配列
count_list = np.zeros((5, 5), dtype=int)
# ファジィ推論をする
for i in range(5):
    for j in range(5):
        index = ([k for k, x in enumerate(C) if x == [i, j]])
        # C_out[index]において，最も重複が多い要素を取得
        # 重複があるかどうか
        unique_rows, counts = np.unique(np.array(C_out)[index], axis=0, return_counts=True)
        # 最も多く重複する行を見つける
        if len(counts) != 0:

            most_common_row = unique_rows[np.argmax(counts)]
            print(f"in:[{i, j}] out:{most_common_row}")
        if len(counts) == 0:
            print(f"in:[{i, j}] is None")

# memsと正解ラベル(thrust, turn_rate)を横並びにする
set = np.array([np.concatenate([X, y], axis=0) for X, y in zip(X, y)])

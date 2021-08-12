import pickle

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

base_dir = "/mnt/xutan/lrq/AVDA/modified_FADA"

# %% import data
filename = "webcam.pkl"
with open(f"{base_dir}/result/interpretation/{filename}", "rb") as f:
    data = np.array(pickle.load(f))

# %% random data
# data = np.random.random(data.shape)
# %% transposition
# columns are as
df = pd.DataFrame(data)
df.rename(lambda x: "as_%s" % x, axis="columns", inplace=True)
df.rename(lambda x: "sample_%s" % x, axis="index", inplace=True)
# rows are as
df = df.T
# %%
# analytical data columns
df["var"] = df.var(axis=1)
df["mean"] = df.mean(axis=1)
# descending sorted
var_index = pd.DataFrame.sort_values(df, by="var", ascending=False).index
mean_index = pd.DataFrame.sort_values(df, by="mean", ascending=False).index

# %% 查看 attention score 方差分布
print(print(df["var"].describe()))
plt.plot(df["var"].sort_values().values)
plt.title("variance of all attention score")
plt.show()

# %% 显示判断不同 attention score 的方差的区别下，箱型图
n = 10
n_largest_var_as = df.loc[var_index[:n]]
n_smallest_var_as = df.loc[var_index[-n:]]
pd.concat([n_smallest_var_as, n_largest_var_as]).T.plot.box()
plt.title("box of min/max attention score")
plt.show()

# %% 查看所有mean分布
plt.plot(df["mean"].abs().sort_values())
plt.title("distribution of abstract means of attention score")
plt.show()

plt.hist(df["mean"].abs())
plt.title("hist of abstract means of attention score")
plt.show()


# %% 选择方差小的 attention score，进行mean分布
n = 1000
n_smallest_var_as = df.loc[var_index[-n:]]

plt.plot(n_smallest_var_as["mean"].abs().sort_values())
plt.title("distribution of abstract means of attention score with min variance")
plt.show()

plt.hist(n_smallest_var_as["mean"].abs())
plt.title("hist of abstract means of attention score with min variance")
plt.show()


# %%
# 运行时attention

with open(f"{base_dir}/as_record.npy", "rb") as f:
    data = np.load(f, allow_pickle=True)

# %%
plt.pcolor(data[np.arange(0, int(159000 * 10 / 200), 100)])
plt.show()

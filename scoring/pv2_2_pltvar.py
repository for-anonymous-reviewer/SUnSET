import numpy as np
import matplotlib.pyplot as plt

#let x be Dts and w be Ds. D from crisis is 17573, t17 is 4203

data = [[2,3,5],[90,85,65],[5,5,5],[16,16,16],[15,4,54],[1,8,2],[6,7,1],[21,19,3],[3,0,0],[19,0,0],[6,3,3],[26,13,13]]
#above cases: close percentagesx2, same percentagex2, one higherx2, two higherx2, only onex2,one is twice of the restx2

# prepare storage
y2 = []   # graph 2 values
y3 = []   # graph 3 values

# compute colours (12 distinct)
cmap   = plt.get_cmap("tab20")
colors = cmap(np.linspace(0,1,len(data)))

# compute y2 & y3
for arr in data:
    arr = np.array(arr, dtype=float)
    total = arr.sum() if arr.sum()>0 else 1.0
    pct   = arr/total                    # e.g. [0.2,0.3,0.5]
    s     = arr.std(ddof=1)              # sample std
    mu    = arr.mean()                   # sample mean
    cv    = s/mu if mu>0 else 0
    y2_vals = pct * cv                   # Graph 2
    y3_vals = np.tanh(arr/10) * y2_vals  # Graph 3
    y2.append(y2_vals)
    y3.append(y3_vals)


# figure + 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

# -------------------
# Graph 1: numeric line tanh(x/10)
# -------------------
# choose x from 0 up to the largest value in your data
max_x = max(max(arr) for arr in data)
x_lin = np.linspace(0, max_x, 400)
ax1.plot(x_lin, np.tanh(x_lin/10),
         color="C0", linewidth=2)
ax1.set_title("Graph 1: tanh(x/10)")
ax1.set_xlabel("x (numeric)")
ax1.set_ylabel("tanh(x/10)")
ax1.grid(alpha=0.3)

# -------------------
# Graph 2: categorical bar pct·CV
# -------------------
N      = len(data)
x_base = np.arange(N)
width  = 0.8
bar_w  = width / 3
cmap   = plt.get_cmap("tab20")
colors = cmap(np.linspace(0,1,N))

for i, vals in enumerate(y2):
    xs = x_base[i] + (np.arange(3) - 1) * bar_w
    ax2.bar(xs, vals, width=bar_w,
            color=colors[i], edgecolor="k", alpha=0.8)
ax2.set_title("Graph 2: percentᵢ × (sample std / mean)")
ax2.set_ylabel("dampened % × CV")
ax2.grid(alpha=0.3)

# -------------------
# Graph 3: categorical bar tanh(x/10)·[Graph 2]
# -------------------
for i, vals in enumerate(y3):
    xs = x_base[i] + (np.arange(3) - 1) * bar_w
    ax3.bar(xs, vals, width=bar_w,
            color=colors[i], edgecolor="k", alpha=0.8)
ax3.set_title("Graph 3: tanh(x/10) × [Graph 2]")
ax3.set_ylabel("scaled value")
ax3.grid(alpha=0.3)

# -------------------
# finalize the categorical x-axis for 2 & 3
# -------------------
labels = [str(arr) for arr in data]
plt.sca(ax3)  # select bottom axis for ticks
plt.xticks(x_base, labels, rotation=45, ha="right")
ax3.set_xlabel("Each array (category)")

plt.tight_layout()
plt.show()
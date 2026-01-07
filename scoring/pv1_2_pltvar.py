import numpy as np
import matplotlib.pyplot as plt

#let x be Dts and w be Ds. D from crisis is 17573, t17 is 4203

ws = [2, 3, 5, 10, 25, 50, 100]
cmap = plt.get_cmap("tab10")

fig, axes = plt.subplots(3, 1, figsize=(8, 12))

# 1) tanh(x/10)
x = np.linspace(1, 100, 200)
y = np.tanh(x / 10)
axes[0].plot(x, y,
             color='black',
             linewidth=3)  # <- thicker line
axes[0].set_title("Graph 1 - R : tanh(x/10)", fontsize=16)
axes[0].set_xlabel("x (numeric)", fontsize=14)
axes[0].set_ylabel("tanh(x/10)", fontsize=14)

# 2) constant log10(...) + continuous curve
for idx, w in enumerate(ws):
    x = np.linspace(1, w, 200)
    val = np.log10((17573 - w + 0.5) / (w + 0.5))
    y = val * np.ones_like(x)
    axes[1].plot(x, y,
                 linestyle="--",
                 color=cmap(idx),
                 linewidth=2,  # <- thicker dashed lines
                 label=f"A\u03C2 in D={w}")

w_axis = np.linspace(min(ws), max(ws), 300)
ratio_curve = np.log10((17573 - w_axis + 0.5) / (w_axis + 0.5))
axes[1].plot(w_axis, ratio_curve,
             color="k",
             linewidth=3)  # <- thicker overlay curve
axes[1].set_title("Graph 2 - P(IDF): log10((17573−A\u03C2+0.5)/(A\u03C2+0.5))", fontsize=16)
axes[1].set_xlabel("A\u03C2", fontsize=14)
axes[1].set_ylabel("P(IDF)", fontsize=14)

# 3) product
for idx, w in enumerate(ws):
    x = np.linspace(1, w, 200)
    val = np.log10((17573 - w + 0.5) / (w + 0.5))
    y = np.tanh(x / 10) * val
    axes[2].plot(x, y,
                 linestyle=":",
                 color=cmap(idx),
                 linewidth=2,  # <- thicker dotted lines
                 label=f"A\u03C2 in D={w}")
axes[2].set_title("Graph 3: R * P", fontsize=16)
axes[2].set_xlabel("x", fontsize=14)
axes[2].set_ylabel("Relevance", fontsize=14)

# Grid and legend enhancements
for ax in axes:
    ax.legend(fontsize=12)        # <- bigger legend text
    ax.tick_params(labelsize=12)  # <- larger tick labels
    ax.grid(alpha=0.4, linewidth=0.8)  # <- stronger gridlines

plt.tight_layout()
plt.show()


# fig, axes = plt.subplots(3, 1, figsize=(8, 12))

# # 1) tanh(x/10)
# x = np.linspace(1,100, 200)
# y = np.tanh(x/10)
# axes[0].plot(x, y,
#                 color='black',
#                 linewidth=2)
# axes[0].set_title("Graph 1 - R : tanh(x/10)")
# axes[0].set_xlabel("x (numeric)")
# axes[0].set_ylabel("tanh(x/10)")
# # 2) constant log10(...) — broadcast to full x-length
# for idx, w in enumerate(ws):
#     x = np.linspace(1, w, 200)
#     val = np.log10((17573 - w + 0.5) / (w + 0.5))
#     y = val * np.ones_like(x)
#     axes[1].plot(x, y,
#                  linestyle="--",
#                  color=cmap(idx),
#                  label=f"A\u03C2 in D={w}")
# # overlay the continuous w‐curve
# w_axis = np.linspace(min(ws), max(ws), 300)
# ratio_curve = np.log10((17573 - w_axis + 0.5) / (w_axis + 0.5))
# axes[1].plot(w_axis, ratio_curve,
#          color="k",
#          linewidth=2,)
# axes[1].set_title("Graph 2 - P(IDF): log10((17573−A\u03C2+0.5)/(A\u03C2+0.5))")
# axes[1].set_xlabel("A\u03C2")
# axes[1].set_ylabel("P(IDF)")
# # 3) product
# for idx, w in enumerate(ws):
#     x = np.linspace(1, w, 200)
#     val = np.log10((17573 - w + 0.5) / (w + 0.5))
#     y = np.tanh(x/10) * val
#     axes[2].plot(x, y,
#                  linestyle=":",
#                  color=cmap(idx),
#                  label=f"A\u03C2 in D={w}")
# axes[2].set_title("Graph 3: R * P")
# axes[2].set_xlabel("A\u03C2")
# axes[2].set_ylabel("Relevance")
# for ax in axes:
#     ax.legend(fontsize="small")
#     ax.grid(alpha=0.3)

# plt.tight_layout()
# plt.show()

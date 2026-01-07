import pickle
import numpy as np
import matplotlib.pyplot as plt

# # 1) adjust these to point at your 4 .pkl files
file_paths = [
    "crisis_egypt.pkl",
    "crisis_libya.pkl",
    "crisis_syria.pkl",
    "crisis_yemen.pkl",
]
dict_labels = ["Egypt", "Libya", "Syria", "Yemen"]
colors = ["C0", "C1", "C2", "C3"]

# # 1) adjust these to point at your 4 .pkl files
# file_paths = [
#     "t17_egypt.pkl",
#     "t17_libya.pkl",
#     "t17_syria.pkl",
#     "t17_finan.pkl",
#     "t17_h1n1.pkl",
#     "t17_mj.pkl",
#     "t17_haiti.pkl",
#     "t17_bpoil.pkl",
#     "t17_iraq.pkl",
# ]
# dict_labels = ["Egypt", "Libya", "Syria", "Finance", "H1N1", "Michael Jackson", "Haiti", "BPOil", "Iraq"]
# colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

# storage for plotting
all_counts = []
medians = []

for fp, label in zip(file_paths, dict_labels):
    # --- load the dict
    with open(fp, "rb") as f:
        word_counts = pickle.load(f)  # expect {word: int, ...}
    # --- extract counts as numpy array
    counts = np.array(list(word_counts.values()), dtype=int)
    
    # --- top 10 pairs
    top10 = sorted(word_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
    print(f"\n=== Top 10 for {label} ({fp}) ===")
    for rank, (w, c) in enumerate(top10, start=1):
        print(f"{rank:2d}. {w!r}: {c}")
    
    # --- median
    # med = np.median(counts)
    # new: drop the 1's before median
    filtered = counts[counts > 1]
    med = np.median(filtered) if len(filtered) else float('nan')
    print(f"{label} median excluding ones: {med}")
    print(f"Median count for {label}: {med}\n")
    
    all_counts.append(np.sort(counts))  # sort for nicer spread plot
    medians.append(med)

# 2) PLOT
plt.figure(figsize=(10, 6))

for idx, (sorted_counts, label, col, med) in enumerate(zip(all_counts, dict_labels, colors, medians)):
    N = len(sorted_counts)
    x = np.arange(N)
  
    # dot-plot
    plt.scatter(x, sorted_counts,
                c=col,
                s=15,        # marker size
                alpha=0.6,
                label=f"{label} counts")
for idx, (sorted_counts, label, col, med) in enumerate(zip(all_counts, dict_labels, colors, medians)):
    N = len(sorted_counts)
    x = np.arange(N)
    # median star
    plt.scatter([N/2], [med],
                c=col,
                marker="*",
                s=400,
                edgecolors="k",
                label=f"{label} median ({med:.1f})")

plt.xlabel("Sorted Stakeholder Occurrences")
plt.ylabel("Repetition Count")
plt.title("Stakeholder Counts & Medians of Crisis Dataset")
plt.legend(ncol=2, fontsize="small", framealpha=0.8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
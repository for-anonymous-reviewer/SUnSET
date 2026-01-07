import os
from tqdm import tqdm
from pathlib import Path
import pickle
import os
import matplotlib.pyplot as plt
import math

def load_dictionary(file_path):
    try:
        with open(file_path, "rb") as file:
            dictionary = pickle.load(file)
        return dictionary
    except FileNotFoundError:
        print("File not found.")
        return None

def save_dictionary(dictionary, dict_path):
    with open(dict_path, "wb") as file:
        pickle.dump(dictionary, file)

# root_path = Path(r"C:\Users\t84401143\Documents\work\datasets\crisis")   # ← change me!
root_path = Path(r"C:\Users\t84401143\Documents\work\datasets\t17")   # ← change me!

total = 0
files_found = 0

for dirpath, _, filenames in os.walk(root_path):
    if "set_newv2.jsonl" in filenames:
        files_found += 1
        filepath = os.path.join(dirpath, "set_newv2.jsonl")
        # count lines
        with open(filepath, "r", encoding="utf-8") as f:
            count = sum(1 for _ in f)
        print(f"{filepath}: {count} entries")
        total += count

if files_found == 0:
    print("No set_newv2.jsonl files found under", root_path)
else:
    print(f"\nFound {files_found} files; total entries = {total}")

###########################################################################################################
root_path =  Path(r"C:\Users\t84401143\Documents\work\scoring")
output_path = Path(r"C:\Users\t84401143\Documents\work\scoring\p1")
output_path.mkdir(exist_ok=True, parents=True)

# main_path = root_path / f"crisis.pkl"
main_path = root_path / f"t17.pkl"

# Load the main dict with all stakeholders
main_dict = load_dictionary(main_path)
# print("First 10 items:", list(middle_dict.items())[:20])

stake={}
for i, (key,value) in tqdm(enumerate(main_dict.items()), total=len(main_dict)):
    stake[key]= math.log10((total-value+0.5)/(value+0.5))

# dict_path2= output_path / f"crisis_pv1_2.pkl"
dict_path2= output_path / f"t17_pv1_2.pkl"

save_dictionary(stake, dict_path2)
print("Main Dictionary saved.")


# 1) Grab your keys (they’re the same in both dicts)
keys = list(main_dict.keys())

# 2) Build X (dict2) and Y (dict1) vectors in the same order
x = [main_dict[k] for k in keys]
y = [stake[k] for k in keys]

# 3) Plot
plt.figure(figsize=(6,6))
plt.scatter(x, y, s=5, c='C1', alpha=0.7)

# # Optionally annotate each point with its key
# for k, xi, yi in zip(keys, x, y):
#     plt.text(xi, yi, k, fontsize=9, va='bottom', ha='right')

plt.xlabel('X: Ds values')
plt.ylabel('Y: Pv1 values')
plt.title('Ds vs Pv1 values')
plt.grid(alpha=0.3)
plt.tight_layout()
# plt.savefig('Ds vs Pv1 Crisis.png', dpi=400, format='png')
plt.savefig('Ds vs Pv1 t17.png', dpi=400, format='png')

plt.show()


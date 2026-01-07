# convert_all_pkl_to_json.py
import os
import pickle
import json
import numpy as np
from pathlib import Path

def _json_default(o):
    """Helper to convert numpy types to native Python for JSON."""
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(f"Type {type(o)} not serializable")

def convert_root(root_dir):
    """
    Recursively find all .pkl under root_dir and dump each to .json
    alongside the original .pkl file.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(".pkl"):
                pkl_path = os.path.join(dirpath, fn)
                json_path = os.path.splitext(pkl_path)[0] + ".json"

                # load the pickle
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)

                # dump as JSON
                with open(json_path, "w", encoding="utf-8") as j:
                    json.dump(data, j,
                              default=_json_default,
                              ensure_ascii=False,
                              indent=2)

                print(f"Converted {pkl_path} → {json_path}")

if __name__ == "__main__":
    root = Path(r"C:\Users\t84401143\Documents\work\scoring")   # ← set this to your top folder
    convert_root(root)
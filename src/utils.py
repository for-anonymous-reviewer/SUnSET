# This code is from news-tls github repo.
# https://github.com/complementizer/news-tls

import pickle
import json
import numpy as np
import gzip
import io
import datetime
import codecs
import tarfile
import shutil
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def draw_line_plot(datasets: dict, save_name: str = "combined_metrics.png"):
    """
    Draws a line plot for the given datasets.
    
    Parameters:
    datasets (dict): A dictionary where keys are dataset names and values are dictionaries with 'beta' and 'metric' keys.
    
    Returns:
    None
    """
    plt.figure(figsize=(10, 6))
    
    for dataset_name, data in datasets.items():
        plt.plot(data['beta'], data['metric'], label=dataset_name)
    
    plt.xlabel('Beta')
    plt.ylabel('Metric')
    plt.title('Line Plot of Datasets')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Consistent styling
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 300
    })

    # Colour per dataset
    dataset_colours = {
        "Crisis": "deepskyblue",  # blue
        "T17": "magenta"      # orange
    }

    method_styles = {
        "v1": {"linestyle": "--", "marker": "o"},
        "v2": {"linestyle": "-", "marker": "s"}
    }

    metrics = {
        "rouge-F1 (AR-1)": {"ylabel": "Rouge‑F1 (AR‑1)", "file": "rouge_f1_filled"},
        "rouge-F2 (AR-2)": {"ylabel": "Rouge‑F2 (AR‑2)", "file": "rouge_f2_filled"},
        "Date-F1": {"ylabel": "Date‑F1", "file": "date_f1_filled"}
    }

    method_label_map = {
        "v1": "TR",
        "v2": "TR+" + r"$Rel$",
    }

    # ── 3.  CREATE ONE FIGURE WITH THREE AXES  ────────────────────────────────────
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(metrics),
        figsize=(18, 6),          # wider canvas
        sharex=True               # β axis shared
    )

    # Ensure we can iterate deterministically
    metric_keys = list(metrics.keys())

    for ax, metric in zip(axes, metric_keys):
        # 3a.  draw every dataset/method on the current axis
        for dataset_name, method_dict in datasets.items():
            color = dataset_colours[dataset_name]

            # v1 / v2 lines
            for method, data in method_dict.items():
                style = method_styles[method]
                method_label = method_label_map[method]
                ax.plot(
                    data["beta"],
                    data[metric],
                    color=color,
                    linestyle=style["linestyle"],
                    marker="o",
                    linewidth=2,
                    markersize=6,
                    label=f"{dataset_name} {method_label}",
                    alpha=1 if method=='v2' else 0.3
                )

            # fill between v1 and v2
            ax.fill_between(
                method_dict["v1"]["beta"],
                method_dict["v2"][metric],
                method_dict["v1"][metric],
                alpha=0.15,
                color=color
            )

        # 3b.  axis‑level cosmetics
        ax.set_xlabel("β")
        ax.set_ylabel(metrics[metric]["ylabel"])
        ax.set_title(f"{metrics[metric]['ylabel']} vs β")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    # ── 4.  GLOBAL LEGEND (once)  ─────────────────────────────────────────────────
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
            loc="lower center", ncol=4,
            bbox_to_anchor=(0.5, -0.05))  # push below plots

    # ── 5.  LAYOUT, SAVE, SHOW  ───────────────────────────────────────────────────
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.10)          # leave space for legend
    fig.savefig(save_name, dpi=400, bbox_inches="tight")
    plt.show()

def is_valid_date(date_string):
    try:
        # Try to parse the string as a date in 'yyyy-mm-dd' format
        datetime.datetime.strptime(date_string, '%Y-%m-%d')
        return True
    except ValueError:
        # If a ValueError occurs, the date is not valid
        return False


def force_mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def dict_to_dense_vector(d, key_to_idx):
    x = np.zeros(len(key_to_idx))
    for key, i in key_to_idx.items():
        x[i] = d[key]
    return x


def read_file(path):
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    return text


def write_file(s, path):
    with open(path, 'w') as f:
        f.write(s)


def read_json(path):
    text = read_file(path)
    return json.loads(text)


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def write_jsonl(items, path, batch_size=100, override=True):
    if override:
        with open(path, 'w'):
            pass

    batch = []
    for i, x in enumerate(items):
        if i > 0 and i % batch_size == 0:
            with open(path, 'a') as f:
                output = '\n'.join(batch) + '\n'
                f.write(output)
            batch = []
        raw = json.dumps(x)
        batch.append(raw)

    if batch:
        with open(path, 'a') as f:
            output = '\n'.join(batch) + '\n'
            f.write(output)


def write_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f)


def load_pkl(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def dump_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj,  f)


def write_gzip(text, path):
    with gzip.open(path, 'wb') as output:
        with io.TextIOWrapper(output, encoding='utf-8') as enc:
            enc.write(text)


def read_gzip(path):
    with gzip.open(path, 'rb') as input_file:
        with io.TextIOWrapper(input_file) as dec:
            content = dec.read()
    return content


def read_jsonl_gz(path):
    with gzip.open(path, 'rb') as input_file:
        with io.TextIOWrapper(input_file) as dec:
            for line in dec:
                yield json.loads(line)

def read_tar_gz(path):
    contents = []
    with tarfile.open(path, 'r:gz') as tar:
        for member in tar.getmembers():
            f = tar.extractfile(member)
            content = f.read()
            contents.append(content)
    return contents


def read_json_tar_gz(path):
    contents = read_tar_gz(path)
    raw_data = contents[0]
    return json.loads(raw_data, strict=False)


def get_date_range(start, end):
    diff = end - start
    date_range = []
    for n in range(diff.days + 1):
        t = start + datetime.timedelta(days=n)
        date_range.append(t)
    return date_range


def days_between(t1, t2):
    return abs((t1 - t2).days)


def any_in(items, target_list):
    return any([item in target_list for item in items])


def csr_item_generator(M):
    """Generates tuples (i,j,x) of sparse matrix."""
    for row in range(len(M.indptr) - 1):
        i,j = M.indptr[row], M.indptr[row + 1]
        for k in range(i,j):
            yield (row, M.indices[k], M.data[k])


def max_normalize_matrix(A):
    try:
        max_ = max(A.data)
        for i, j, x in csr_item_generator(A):
            A[i, j] = x / max_
    except:
        pass
    return A


def gzip_file(inpath, outpath, delete_old=False):
    with open(inpath, 'rb') as infile:
        with gzip.open(outpath, 'wb') as outfile:
            outfile.writelines(infile)
    if delete_old:
        os.remove(inpath)


def normalise(X, method='standard'):
    if method == 'max':
        return X / X.max(0)
    elif method == 'minmax':
        return MinMaxScaler().fit_transform(X)
    elif method == 'standard':
        return StandardScaler().fit_transform(X)
    elif method == 'robust':
        return RobustScaler().fit_transform(X)
    else:
        raise ValueError('normalisation method not known: {}'.format(method))


def normalize_vectors(vector_batches, mode='standard'):
    if mode == 'max':
        normalize = lambda X: X / X.max(0)
    elif mode == 'minmax':
        normalize = lambda X: MinMaxScaler().fit_transform(X)
    elif mode == 'standard':
        normalize = lambda X: StandardScaler().fit_transform(X)
    elif mode == 'robust':
        normalize = lambda X: RobustScaler().fit_transform(X)
    else:
        normalize = lambda X: X
    norm_vectors = []
    for vectors in vector_batches:
        X = np.array(vectors)
        X_norm = normalize(X)
        norm_vectors += list(X_norm)
    return norm_vectors


def strip_to_date(t):
    return datetime.datetime(t.year, t.month, t.day)


def print_tl(tl):
    for t, sents in tl.items:
        print('[{}]'.format(t.date()))
        for s in sents:
            print(' '.join(s.split()))
        print('---')

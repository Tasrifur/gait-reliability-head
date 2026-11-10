import datetime
import json
import os
import platform
import random

import numpy as np
import scipy.stats as st
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mean_ci(x, conf=0.95):
    """Mean and half-width of the t-based confidence interval, plus SD."""
    x = np.asarray(x, dtype=float)
    m = float(x.mean())
    if len(x) < 2:
        return {'mean': m, 'ci95': float('nan'), 'sd': float('nan'), 'n': len(x)}
    sd = float(x.std(ddof=1))
    h = float(st.t.ppf(0.5 + conf / 2, len(x) - 1) * sd / np.sqrt(len(x)))
    return {'mean': m, 'ci95': h, 'sd': sd, 'n': len(x)}


def provenance():
    return {
        'time': datetime.datetime.now().isoformat(timespec='seconds'),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'python': platform.python_version(),
    }


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=float)

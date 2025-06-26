from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm
stats = pd.read_csv("stats.csv.gz")
parts2size = defaultdict(list)
parts2domains = defaultdict(list)

for ix, _ in tqdm(stats.iterrows(), total=len(stats)):
    parts = [o for o in _["url"].split("/") if len(o) > 1]
    domain = parts[1]
    for p in parts:
        if "www." not in p and len(p) < 30 and ".com" not in p and "-" not in p and not p.isdigit():
            parts2size[p].append(_['size'])
            parts2domains[p].append(domain)


parts2domains = {k: set(v) for k, v in parts2domains.items()}
parts2size = [(k, pd.Series(v).mode().iloc[0]) for k, v in parts2size.items() if len(v) > 1000]
parts2size = [o for o in parts2size if len(parts2domains[o[0]]) > 250]
parts2size.sort(key=lambda x:x[1])

print(parts2size)
import pandas as pd

contaminated = pd.read_csv("pythia-45m_lr1e-3_steps5k_seed1234_interleave0.02_contaminated.all_shards.csv.gz")
contaminated = contaminated[contaminated['method'] == "loss"].copy().drop(columns="method")
contaminated = contaminated[contaminated['membership'] == "member"].copy().drop(columns="membership")
contaminated = contaminated.rename(columns={"score": "contaminated"})

uncontaminated = pd.read_csv("pythia-45m_lr1e-3_steps5k_seed1234_on_contaminated.all_shards.csv.gz")
uncontaminated = uncontaminated[uncontaminated['method'] == "loss"].copy().drop(columns="method")
uncontaminated = uncontaminated[uncontaminated['membership'] == "member"].copy().drop(columns="membership")
uncontaminated = uncontaminated.rename(columns={"score": "uncontaminated"})

both = uncontaminated.merge(contaminated, on ="doc_id")

both["delta"] = both["uncontaminated"] - both["contaminated"]

both["delta"].mean()
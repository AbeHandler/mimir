import pandas as pd

blocks = pd.read_csv('csvs/confounddataset/pythia-45m_lr1e-3_steps5k_seed1234_interleave0.02_uncontaminated_insample_regular_training_data.all_shards.csv.gz').rename(columns={"score": "blocks"})
noblocks = pd.read_csv('csvs/confounddataset/pythia-45m_lr1e-3_steps5k_seed1234_uncontaminated_insample_regular_training_data.all_shards.csv.gz').rename(columns={"score": "noblocks"})

blocks = blocks[blocks["membership"] == "member"].copy()
noblocks = noblocks[noblocks["membership"] == "member"].copy()

both = blocks.merge(noblocks, on =["method", "doc_id"])
both["delta"] = both["noblocks"] - both["blocks"]  # uncontaminated - contaminated (to match ATT)

# Write ATU data for R plotting
both.to_csv("/tmp/atu_appendix.csv", index=False)

print(both[["delta", "method"]].groupby(["method"]).mean())
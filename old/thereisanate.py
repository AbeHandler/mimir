import pandas as pd
import altair as alt

noblocks = pd.read_csv("csvs/tmp.csv").rename(columns={"score": "noblocks_score"})
blocks = pd.read_csv("csvs/minhashblocksample_blocks.lite.csv").rename(columns={"score": "blocks_score"})
D = blocks.merge(noblocks, on=["doc_id", "method"])[["blocks_score", "noblocks_score", "doc_id", "method"]].drop_duplicates()

print(len(D))

D["delta"] = D["blocks_score"] - D["noblocks_score"]
print(D[["method", "delta"]].groupby(["method"]).mean().reset_index())
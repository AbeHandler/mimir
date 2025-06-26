import pandas as pd
import altair as alt

noblocks = pd.read_csv("minhashblocksample_noblocks.lite.csv").rename(columns={"score": "noblocks_score"})
blocks = pd.read_csv("minhashblocksample_blocks.lite.csv").rename(columns={"score": "blocks_score"})
D = blocks.merge(noblocks, on=["doc_id", "method"])[["blocks_score", "noblocks_score", "doc_id", "method"]].drop_duplicates()

D["delta"] = D["blocks_score"] - D["noblocks_score"]
D[["method", "delta"]].groupby(["method"]).mean().reset_index()
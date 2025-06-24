import pandas as pd

blocksbin = pd.read_csv("copywrite_traps_blocksbin.csv")
blocksbin = blocksbin.rename(columns={"score": "score_out_of_sample"})

noblocksbin = pd.read_csv("copywrite_traps_noblocksbin.csv")
noblocksbin = noblocksbin.rename(columns={"score": "score_in_sample"})

zeros = pd.read_csv("copywrite_traps_zeros.csv")
zeros = zeros.rename(columns={"score": "score_zero"})

R = blocksbin.merge(noblocksbin, on=["doc_id", "method"])
R["delta"] = R["score_out_of_sample"] - R["score_in_sample"]
R = R[["method", "delta", "doc_id"]].drop_duplicates()[["method", "delta"]].groupby("method").mean().reset_index()
print(R)



print(noblocksbin["score_in_sample"].mean(), zeros["score_zero"].mean())
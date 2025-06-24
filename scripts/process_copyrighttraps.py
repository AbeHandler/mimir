import pandas as pd
import math

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


with open("copywritetraps.csv", "w") as of:
	onethousands = math.log(noblocksbin["score_in_sample"].mean())
	fivehundreds = math.log(blocksbin["score_out_of_sample"].mean())
	zeros = math.log(zeros["score_zero"].mean())
	of.write(f"{1000},{onethousands}\n")
	of.write(f"{500},{fivehundreds}\n")
	of.write(f"{0},{zeros}\n")
import pandas as pd

from scipy.stats import wilcoxon

from string import Template

def load_dcpdd():
    dcpddblocks = pd.read_json("/Users/abha4861/dolma/dcpdd/output/metrics/minhashblocksample/dobolyilab/blockbench-blocksbin.jsonl", lines=True)
    dcpddblocks = dcpddblocks.rename(columns={"pred": "blocks_score", "id": "doc_id"})
    dcpddblocks["method"] = "dcpdd"

    dcpddnoblocks = pd.read_json("/Users/abha4861/dolma/dcpdd/output/metrics/minhashblocksample/dobolyilab/blockbench-noblocksbin.jsonl", lines=True)
    dcpddnoblocks = dcpddnoblocks.rename(columns={"pred": "noblocks_score", "id": "doc_id"})
    dcpddnoblocks['method'] = "dcpdd"

    # DC-PDD is over a larger sample
    doc_ids = set(pd.read_csv("csvs/minhashblocksample_noblocks.lite.csv")['doc_id'].to_list())
    dcpddnoblocks = dcpddnoblocks[dcpddnoblocks['doc_id'].apply(lambda x: x in doc_ids)].copy()
    dcpddblocks = dcpddblocks[dcpddblocks['doc_id'].apply(lambda x: x in doc_ids)].copy()

    return dcpddblocks, dcpddnoblocks

for method in ["loss", "min_k", "ref-stablelm-base-alpha-3b-v2", "dcpdd"]:

    stats = pd.read_csv("stats.csv.gz")

    if method != "dcpdd":
        method2pattern = {"loss": Template(f"csvs/minhashblocksample_$kind.lite.csv"),
                          "min_k": Template(f"csvs/minhashblocksample_$kind.lite.csv"),
                          "ref-stablelm-base-alpha-3b-v2": Template(f"csvs/minhashblocksample_$kind.ref.csv")}

        noblocks = pd.read_csv(method2pattern[method].substitute({'kind': "noblocks"})).rename(columns={"score": "noblocks_score"})
        blocks = pd.read_csv(method2pattern[method].substitute({'kind': "blocks"})).rename(columns={"score": "blocks_score"})
    else:
        blocks, noblocks = load_dcpdd()
        
    noblocks = stats.merge(noblocks, left_on='url', right_on="doc_id").drop(columns=['size'])
    blocks = stats.merge(blocks, left_on='url', right_on="doc_id")

    D = noblocks.merge(blocks, on=["doc_id", "method"])
    D["delta"] = D["blocks_score"] - D["noblocks_score"]

    D = D[D["method"] == method].copy()
    if method == "dcpdd":
        D["delta"] *= -1

    D["size_bin"] = pd.cut(D["size"], bins=range(0, 50, 5), right=True)

    df = D.groupby("size_bin", observed=True).agg(
        delta_mean=("delta", "mean"),
        count=("delta", "count")
    ).reset_index()

    if method == "ref-stablelm-base-alpha-3b-v2":
        method = 'ref'
    
    df['method'] = method
    #df.to_csv(method + ".csv", index=False)
    stat, p = wilcoxon(D["delta"].to_list(), alternative="greater")

    print(f"{D['delta'].mean():.3g}", method, p)
    
    #import os
    #os.system(f"cat {method}.csv")
    #print(df)
    print(df)
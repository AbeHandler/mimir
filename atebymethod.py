import pandas as pd
import altair as alt
from scipy.stats import wilcoxon
import os

# cp ~/dolma/textreuse/minhash/stats.csv .
# gzip stats.csv

# filtered to shard _0_ in url_shard_0
cmd = '''cp /Users/abha4861/dolma/mimir/csvs/minhashblocksample_noblocks.lite.csv csvs/minhashblocksample_noblocks.lite.csv'''
os.system(cmd)

stats = pd.read_csv("stats.csv.gz")
noblocks = pd.read_csv("csvs/minhashblocksample_noblocks.lite.csv").rename(columns={"score": "noblocks_score"})
blocks = pd.read_csv("csvs/minhashblocksample_blocks.lite.csv").rename(columns={"score": "blocks_score"})

refbloc = pd.read_csv("csvs/minhashblocksample_blocks.ref.csv").rename(columns={"score": "blocks_score"})
refnobloc = pd.read_csv("csvs/minhashblocksample_noblocks.ref.csv").rename(columns={"score": "noblocks_score"})

refbloc["method"] = "ref"
refnobloc["method"] = "ref"

assert refbloc.columns.to_list() == blocks.columns.to_list()
assert refnobloc.columns.to_list() == noblocks.columns.to_list()

blocks = pd.concat([blocks, refbloc])
noblocks = pd.concat([noblocks, refnobloc])

doc_ids = set(noblocks['doc_id'].to_list())

def load_dcpdd():
    dcpddblocks = pd.read_json("/Users/abha4861/dolma/dcpdd/output/metrics/minhashblocksample/dobolyilab/blockbench-blocksbin.jsonl", lines=True)
    dcpddblocks = dcpddblocks.rename(columns={"pred": "blocks_score", "id": "doc_id"})
    dcpddblocks["membership"] = "member"
    dcpddblocks["method"] = "dcpdd"

    dcpddnoblocks = pd.read_json("/Users/abha4861/dolma/dcpdd/output/metrics/minhashblocksample/dobolyilab/blockbench-noblocksbin.jsonl", lines=True)
    dcpddnoblocks = dcpddnoblocks.rename(columns={"pred": "noblocks_score", "id": "doc_id"})
    dcpddnoblocks["membership"] = "member"
    dcpddnoblocks['method'] = "dcpdd"
    return dcpddblocks, dcpddnoblocks


dcpddblocks, dcpddnoblocks = load_dcpdd()
print(dcpddnoblocks.columns, noblocks.columns)

assert set(dcpddblocks.columns.to_list()) == set(blocks.columns.to_list())
assert set(dcpddnoblocks.columns.to_list()) == set(noblocks.columns.to_list())

blocks = pd.concat([blocks, dcpddblocks])
noblocks = pd.concat([noblocks, dcpddnoblocks])

#noblocks.columns => Index(['method', 'membership', 'doc_id', 'noblocks_score']
noblocks = stats.merge(noblocks, left_on='url', right_on="doc_id").drop(columns=['size'])
blocks = stats.merge(blocks, left_on='url', right_on="doc_id")

D = blocks.merge(noblocks, on=["doc_id", "method"])[["blocks_score", "size", "noblocks_score", "doc_id", "method"]].drop_duplicates()
D["delta"] = D["blocks_score"] - D["noblocks_score"]

D = D[D['doc_id'].apply(lambda x: x in doc_ids)]
D = D[D["size"] <= 51].copy()

for method in ["loss", "min_k", "dcpdd", "ref"]:

    D["size_bin"] = pd.cut(D["size"], bins=range(0, 50, 10), right=True)
    Dp = D[D['method'] == method].copy()

    if method == "dcpdd":
        Dp['delta'] *= -1

    mean_ = Dp["delta"].mean()

    stat, p = wilcoxon(Dp["delta"].to_list(), alternative="greater")
    print(f"{Dp['delta'].mean():.3g}", method, p)

    with open("targets.txt", "w") as of:
        for _ in set(Dp["doc_id"]):
            of.write(_ + '\n')        
    print(method, len(Dp), len(set(Dp["doc_id"])))

    # for loss, we expect blocks score - noblocks score to be > 0
    # same for min-k% 
    # but for dcpdd we expect higher scores mean more likely included
    # so we expect noblocs > blocks, hence flip sign


    df = Dp.groupby("size_bin", observed=True).agg(
        delta_mean=("delta", "mean"),
        count=("delta", "count")
    ).reset_index()

    df["method"] = method
    df.to_csv(method + ".csv", index=False)

    df["size_bin"] = df["size_bin"].astype(str)
    chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("size_bin:N", sort=None, title="Count of similar documents in BlockBench"),
        y=alt.Y("delta_mean:Q", title="Mean ATE"),
        tooltip=["size_bin", "delta_mean", "count"]
    ).properties(
        title=f"Mean ATE by Repetition: {method}",
        width=400,
        height=300
    )
    chart.save(f"{method}.html")
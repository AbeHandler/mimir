import pandas as pd
import altair as alt

stats = pd.read_csv("stats.csv.gz")
noblocks = pd.read_csv("csvs/minhashblocksample_noblocks.lite.csv").rename(columns={"score": "noblocks_score"})
noblocks = stats.merge(noblocks, left_on='url', right_on="doc_id").drop(columns=['size'])

blocks = pd.read_csv("csvs/minhashblocksample_blocks.lite.csv").rename(columns={"score": "blocks_score"})
blocks = stats.merge(blocks, left_on='url', right_on="doc_id")


print(len(noblocks))
print(len(blocks))

for method in ["loss", "min_k"]:
    D = blocks.merge(noblocks, on=["doc_id", "method"])[["blocks_score", "size", "noblocks_score", "doc_id", "method"]].drop_duplicates()

    D["delta"] = D["blocks_score"] - D["noblocks_score"]
    D = D[D["size"] <= 30].copy()
    D["size_bin"] = pd.cut(D["size"], bins=range(0, 31, 5), right=True)
    D = D[D['method'] == method].copy()

    df = D.groupby("size_bin", observed=True).agg(
        delta_mean=("delta", "mean"),
        count=("delta", "count")
    ).reset_index()


    df["size_bin"] = df["size_bin"].astype(str)
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("size_bin:N", sort=None, title="Count of similar documents in BlockBench"),
        y=alt.Y("delta_mean:Q", title="Mean ATE"),
        tooltip=["size_bin", "delta_mean", "count"]
    ).properties(
        title=f"Mean ATE by Repetition: {method}",
        width=400,
        height=300
    )

    chart.save(f"{method}.html")
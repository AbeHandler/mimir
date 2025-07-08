import pandas as pd

noblocks = pd.read_csv("local_blocked_docs_blocks.lite.csv").rename(columns={"score": "noblocks_score"})
blocks = pd.read_csv("local_blocked_docs_noblocks.lite.csv").rename(columns={"score": "blocks_score"})
D = blocks.merge(noblocks, on=["doc_id", "method"])[["blocks_score", "noblocks_score", "doc_id", "method"]].drop_duplicates()

hardness = blocks.merge(noblocks, on=["doc_id", "method"])[["blocks_score", "noblocks_score", "doc_id", "method"]].drop_duplicates()
hardness = hardness[hardness["method"] == "loss"]
hardness["hardness"] = (hardness["blocks_score"] + hardness["noblocks_score"])/2
doc2loss = {k:v for k, v in zip(hardness["doc_id"],hardness["hardness"])}
D["delta"] = D["blocks_score"] - D["noblocks_score"]
D["hardness"] = D["doc_id"].apply(lambda x: doc2loss[x])

print(D[["method", "delta"]].groupby(["method"]).mean().reset_index())


from scipy.stats import mannwhitneyu

x = D[D["method"] == "loss"]["blocks_score"].to_list()
y = D[D["method"] == "loss"]["noblocks_score"].to_list()

print(len(x))

stat, p = mannwhitneyu(x, y, alternative='greater')  # or 'less', 'greater'
print(f"Mann–Whitney U statistic={stat}, p-value={p}")
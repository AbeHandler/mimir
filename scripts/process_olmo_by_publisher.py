import json
import pandas as pd
from urllib.parse import urlparse
import pandas as pd
import joypy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

def normalize_domain(url: str) -> str:
    url = url.strip('"')
    netloc = urlparse(url).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc

def process_results(pathto = "tmp_results/olmobypublisher/allenai_OLMo-7B/abehandlerorg/olmobypublisherdev"):
    
    allmethods = []
    for method in ["min_k", "zlib", "loss"]:

        INPUT_PATH = f"{pathto}/{method}_results.json"

        # Load the JSON data
        with open(INPUT_PATH, "r") as f:
            data = json.load(f)

        # Extract id_to_score (handles nested structure if present)
        id_to_score = data.get("id_to_score", {})
        # If nested under member/nonmember, flatten by choosing 'member'
        if all(isinstance(v, dict) for v in id_to_score.values()):
            id_to_score = id_to_score.get("member", {})

        # Build a list of rows with domain, score, method
        rows = []
        for url, score in id_to_score.items():
            domain = normalize_domain(url)
            rows.append({"domain": domain, "score": score, "method": "min_k"})

        # Create DataFrame and display
        df = pd.DataFrame(rows)
        df["method"] = method
        allmethods.append(df)


    combined = pd.concat(allmethods, ignore_index=True)

    print(f"{pathto}/results.csv")

    combined.to_csv(f"{pathto}/results.csv", index=False)


def analyze_results(pathto = "tmp_results/olmobypublisher/allenai_OLMo-7B/abehandlerorg/olmobypublisherdev"):

    # scale features between 0 and 1 becuase each will have a different range. For instance, mink are big and zlib are small, by absolute size
    scaler = MinMaxScaler(feature_range=(0,1))


    with open("configs/ten_publishers.txt", "r") as inf:
        targets = [o.strip("\n") for o in inf]

    results = pd.read_csv(f"{pathto}/results.csv")
    methods = results["method"].unique()
    results = results[results["domain"].apply(lambda x: x in targets)].copy()

    pvalues = []

    for method in methods:

        df = results[results["method"] == method].copy()

        df["score"] = scaler.fit_transform(df[["score"]]).ravel()

        df = df[df["domain"].apply(lambda x: x in targets)].copy().rename(columns={"domain": "name"})[["name", "score"]]

        #  ╔════════════════════════════════════════════════════════════════════════╗
        #  ║  This does plotting for visual analysis                                ║
        #  ╚════════════════════════════════════════════════════════════════════════╝

        names = df.groupby("name").mean().sort_values("score", ascending=False).reset_index()

        names["position"] = [i for i in range(len(names))]
        names = names[["name", "position"]]

        # now order the rows in df based on position in names
        df = df.merge(names, on="name", how="left")
        df = df.sort_values("position", ascending=False)

        order = (
            df.groupby("name")["position"]
            .mean()  # or .first(), .median(), etc.
            .sort_values()
            .index.tolist()
        )

        fig, axes = joypy.joyplot(
            df,
            by="position",
            column="score",
            labels=order,  # ← forces this exact y‐order
            x_range=(0, 1),
            ylim="own",
            overlap=0.3,
            kind="kde",
            linewidth=1,
            figsize=(6, len(order) * 0.2),
        )

        for ax in axes:
            ax.tick_params(axis="y", labelsize=7)  # ← pick whatever font size you like

        #  ╔════════════════════════════════════════════════════════════════════════╗
        #  ║  This is a more formal quant analysis                                  ║
        #  ╚════════════════════════════════════════════════════════════════════════╝

        # fit OLS and do ANOVA
        model       = ols("score ~ C(name)", data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        print(method, anova_table.iloc[0]["PR(>F)"])
        sum_of_sq = anova_table.iloc[0]["sum_sq"]

        pvalues.append({"method": method,
                        "sum_of_sq": sum_of_sq,
                        "p": anova_table.iloc[0]["PR(>F)"]})

    pd.DataFrame(pvalues).to_csv(pathto + "/analysis.csv", index=False)


if __name__ == "__main__":

    process_results()
    analyze_results()
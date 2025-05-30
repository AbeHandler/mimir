import pandas as pd
from pydantic import BaseModel


class MIARow(BaseModel):
    method: str
    doc_id: str
    score_m0: float
    score_m1: float


def export_results(file1="olmo_blocked_docs_m0.csv", file2="olmo_blocked_docs_m1.csv"):

    df_m0 = pd.read_csv(file1)
    df_m1 = pd.read_csv(file2)

    df_m0["model"] = "m0"
    df_m1["model"] = "m1"

    df_m0 = df_m0.rename(columns={"score": "score_m0"})
    df_m1 = df_m1.rename(columns={"score": "score_m1"})

    df_m0 = df_m0.drop(columns=["model", "membership"]).drop_duplicates() # in our case members/non members are the same
    df_m1 = df_m1.drop(columns=["model", "membership"]).drop_duplicates() # in our case members/non members are the same

    # Merge on method, membership, and doc_id
    merged = pd.merge(df_m0, df_m1, on=["method", "doc_id"], how="inner")

    for ix, row in merged.iterrows():
        MIARow(**row) # basic validation

    merged.to_csv("mimir.E1.csv", index=False)

if __name__ == "__main__":
    export_results()
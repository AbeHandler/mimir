"""Align token-level log-probs from blocks/noblocks min_k_results on the document level.

Reads two min_k_results.json files (blocks variant + noblocks variant), pairs
documents by URL key, and emits one row per document with aligned per-token
log-probs from each model.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MIN_K_FRACTIONS = (0.01, 0.05, 0.20, 0.50)


BLOCKS_PATH = Path(
    "tmp_results/excluded-docs/"
    "abehandlerorg_dobolyilab-blockbench-blocksbin_rlhf_dpo_hh-rlhf_e1_lr5e-5_r8/"
    "abehandlerorg/excluded-docs/min_k_results.json"
)
NOBLOCKS_PATH = Path(
    "tmp_results/excluded-docs/"
    "abehandlerorg_dobolyilab-blockbench-noblocksbin_rlhf_dpo_hh-rlhf_e1_lr5e-5_r8/"
    "abehandlerorg/excluded-docs/min_k_results.json"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-blocks", type=Path, default=BLOCKS_PATH)
    parser.add_argument("-noblocks", type=Path, default=NOBLOCKS_PATH)
    parser.add_argument(
        "-out",
        type=Path,
        default=Path("results/min_k_check_for_response_doc/excluded-docs.parquet"),
    )
    return parser.parse_args()


def flatten_chunks(chunks):
    out = []
    for chunk in chunks:
        out.extend(chunk)
    return out


def build_rows(blocks, noblocks):
    rows = []
    for split in ("member", "nonmember"):
        urls_b = blocks["id_to_token_probs"][split]
        urls_n = noblocks["id_to_token_probs"][split]
        shared = set(urls_b) & set(urls_n)
        if len(shared) != len(urls_b) or len(shared) != len(urls_n):
            raise ValueError(
                f"split={split}: url sets differ "
                f"(blocks={len(urls_b)}, noblocks={len(urls_n)}, shared={len(shared)})"
            )
        for url in sorted(shared):
            tp_b = flatten_chunks(urls_b[url])
            tp_n = flatten_chunks(urls_n[url])
            if len(tp_b) != len(tp_n):
                raise ValueError(
                    f"split={split} url={url}: token-length mismatch "
                    f"blocks={len(tp_b)} noblocks={len(tp_n)}"
                )
            rows.append(
                {
                    "split": split,
                    "url": url,
                    "n_tokens": len(tp_b),
                    "score_blocks": blocks["id_to_score"][split][url],
                    "score_noblocks": noblocks["id_to_score"][split][url],
                    "token_logprobs_blocks": tp_b,
                    "token_logprobs_noblocks": tp_n,
                }
            )
    return rows


def min_k_nll(token_logprobs, k):
    """Mean NLL over the top-k fraction of highest-NLL (=lowest-prob) tokens."""
    nlls = -np.asarray(token_logprobs, dtype=float)
    n_keep = max(1, int(np.ceil(len(nlls) * k)))
    return float(np.sort(nlls)[-n_keep:].mean())


def report_min_k_table(df):
    """Mean per-document min-K% NLL for blocks vs noblocks at multiple K.

    NB: in this config member and nonmember splits cover the same URL set, so
    we deduplicate to avoid double-counting.
    """
    docs = df.drop_duplicates(subset="url").reset_index(drop=True)
    print(f"\n--- min-K% NLL (per-doc, n={len(docs)}) ---")
    header = f'{"K":>5}  {"blocks_nll":>12}  {"noblocks_nll":>14}  {"mean(blocks-noblocks)":>24}'
    print(header)
    for k in MIN_K_FRACTIONS:
        b = docs["token_logprobs_blocks"].apply(lambda x: min_k_nll(x, k))
        n = docs["token_logprobs_noblocks"].apply(lambda x: min_k_nll(x, k))
        delta = (b - n).mean()
        print(f"{int(k*100):>4}%  {b.mean():>12.4f}  {n.mean():>14.4f}  {delta:>24.4f}")


def main():
    args = parse_args()

    print(f"Loading blocks:   {args.blocks}")
    blocks = json.loads(args.blocks.read_text())
    print(f"Loading noblocks: {args.noblocks}")
    noblocks = json.loads(args.noblocks.read_text())

    rows = build_rows(blocks, noblocks)
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    print(f"Wrote {len(df)} rows to {args.out}")
    print("\n--- spot checks ---")
    print(df.groupby("split").size())
    print("\nfirst row:")
    sample = df.iloc[0]
    print(f"  split={sample.split}  url={sample.url}")
    print(f"  n_tokens={sample.n_tokens}")
    print(f"  score_blocks={sample.score_blocks:.4f}  score_noblocks={sample.score_noblocks:.4f}")
    print(f"  blocks   logprobs[:5]={sample.token_logprobs_blocks[:5]}")
    print(f"  noblocks logprobs[:5]={sample.token_logprobs_noblocks[:5]}")

    report_min_k_table(df)


if __name__ == "__main__":
    main()

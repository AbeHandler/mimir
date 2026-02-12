#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from scipy import stats
import numpy as np
import os

# update the all_shards docs to ensure you get the latest versions
# os.system("python scripts/merge_shards.py -d csvs/confounddataset")

noblocks = pd.read_csv("csvs/confounddataset/excluded-docs.noblocks.lite.all_shards.csv.gz").rename(columns={"score": "noblocks"})
blocks = pd.read_csv("csvs/confounddataset/excluded-docs.blocks.lite.all_shards.csv.gz").rename(columns={"score": "blocks"})

excluded = set([i.strip() for i in open("/Users/abha4861/dolma/data/interim/R2/cleaning/verified_excluded_urls.txt")])

#                             
#                             
#              ,d      ,d     
#              88      88     
# ,adPPYYba, MM88MMM MM88MMM  
# ""     `Y8   88      88     
# ,adPPPPP88   88      88     
# 88,    ,88   88,     88,    
# `"8bbdP"Y8   "Y888   "Y888  
#                             
#                             


ATT = noblocks.merge(blocks, on=['doc_id', 'method', 'membership'])

ATT = ATT[ATT['doc_id'].isin(excluded)].copy()

ATT["delta"] = ATT["blocks"] - ATT["noblocks"]
ATT = ATT[ATT["membership"] == "member"].copy().sample(n=50_000, random_state=42)


print(ATT[["method", "delta"]].groupby("method").mean().reset_index())



#                                                             
#          88                                 88          88  
#          88                                 88          88  
#          88                                 88          88  
#  ,adPPYb,88  ,adPPYba, 8b,dPPYba,   ,adPPYb,88  ,adPPYb,88  
# a8"    `Y88 a8"     "" 88P'    "8a a8"    `Y88 a8"    `Y88  
# 8b       88 8b         88       d8 8b       88 8b       88  
# "8a,   ,d88 "8a,   ,aa 88b,   ,a8" "8a,   ,d88 "8a,   ,d88  
#  `"8bbdP"Y8  `"Ybbd8"' 88`YbbdP"'   `"8bbdP"Y8  `"8bbdP"Y8  
#                        88                                   
#                        88                                   

noblocks = pd.read_csv("csvs/confounddataset/excluded-docs.noblocks.dcpdd.all_shards.csv.gz").rename(columns={"score": "noblocks"})
blocks = pd.read_csv("csvs/confounddataset/excluded-docs.blocks.dcpdd.all_shards.csv.gz").rename(columns={"score": "blocks"})
excluded = set([i.strip() for i in open("/Users/abha4861/dolma/data/interim/R2/cleaning/verified_excluded_urls.txt")])
ATT = noblocks.merge(blocks, on=['doc_id', 'method', 'membership'])
ATT = ATT[ATT['doc_id'].isin(excluded)].copy()
ATT["delta"] = ATT["blocks"] - ATT["noblocks"]
ATT = ATT[ATT["membership"] == "member"].copy()
dcp = ATT[ATT["method"] != "loss"].copy()
print(dcp[["method", "delta"]].groupby("method").mean().reset_index())


#                                 
#                                 
#              ,d                 
#              88                 
# ,adPPYYba, MM88MMM 88       88  
# ""     `Y8   88    88       88  
# ,adPPPPP88   88    88       88  
# 88,    ,88   88,   "8a,   ,a88  
# `"8bbdP"Y8   "Y888  `"YbbdP'Y8  
#                                 
#                               

print("[*] ATU")  

noblocks = pd.read_csv("csvs/confounddataset/bothbins.noblocks.lite.all_shards.csv.gz").rename(columns={"score": "noblocks"})
blocks = pd.read_csv("csvs/confounddataset/bothbins.blocks.lite.all_shards.csv.gz").rename(columns={"score": "blocks"})

bothbins = "/Users/abha4861/dolma/data/interim/R2/cleaning/verified_bothbins_urls.txt"
bothbins = set([i.strip() for i in open(bothbins)])

ATU = noblocks.merge(blocks, on=['doc_id', 'method', 'membership'])
ATU = ATU[ATU['doc_id'].isin(bothbins)].copy()

ATU["delta"] = ATU["blocks"] - ATU["noblocks"]
ATU = ATU[ATU["membership"] == "member"].copy().sample(n=50_000, random_state=42)

print(ATU[["method", "delta"]].groupby("method").mean().reset_index())

noblocks = pd.read_csv("csvs/confounddataset/bothbins.noblocks.dcpdd.all_shards.csv.gz").rename(columns={"score": "noblocks"})
blocks = pd.read_csv("csvs/confounddataset/bothbins.blocks.dcpdd.all_shards.csv.gz").rename(columns={"score": "blocks"})
bothbins = set([i.strip() for i in open("/Users/abha4861/dolma/data/interim/R2/cleaning/verified_bothbins_urls.txt")])
ATU = noblocks.merge(blocks, on=['doc_id', 'method', 'membership'])
ATU = ATU[ATU['doc_id'].isin(bothbins)].copy()
ATU["delta"] = ATU["blocks"] - ATU["noblocks"]
ATU = ATU[ATU["membership"] == "member"].copy()
dcp = ATU[ATU["method"] != "loss"].copy()
print(dcp[["method", "delta"]].groupby("method").mean().reset_index())

import sys; sys.exit(0)

#
#
#              ,d      ,d                                            ,d
#              88      88                                            88
# ,adPPYYba, MM88MMM MM88MMM    8b       d8 ,adPPYba,      ,adPPYYba, MM88MMM 88       88
# ""     `Y8   88      88       `8b     d8' I8[    ""      ""     `Y8   88    88       88
# ,adPPPPP88   88      88        `8b   d8'   `"Y8ba,       ,adPPPPP88   88    88       88
# 88,    ,88   88,     88,        `8b,d8'   aa    ]8I      88,    ,88   88,   "8a,   ,a88
# `"8bbdP"Y8   "Y888   "Y888        "8"     `"YbbdP"'      `"8bbdP"Y8   "Y888  `"YbbdP'Y8
#
#

print("\n" + "="*80)
print("ATT vs ATU Comparison: Non-parametric tests by method")
print("="*80 + "\n")

# Get unique methods
unique_methods = sorted(set(ATT['method'].unique()) & set(ATU['method'].unique()))

results = []
for method in unique_methods:
    att_deltas = ATT[ATT['method'] == method]['delta'].values
    atu_deltas = ATU[ATU['method'] == method]['delta'].values

    # Mann-Whitney U test (non-parametric test for difference in distributions)
    mw_result = stats.mannwhitneyu(att_deltas, atu_deltas, alternative='two-sided')

    # Independent samples t-test
    t_result = stats.ttest_ind(att_deltas, atu_deltas, alternative='two-sided')

    # Calculate medians (more robust for non-parametric)
    att_median = pd.Series(att_deltas).median()
    atu_median = pd.Series(atu_deltas).median()

    # Calculate means for reference
    att_mean = pd.Series(att_deltas).mean()
    atu_mean = pd.Series(atu_deltas).mean()

    results.append({
        'method': method,
        'att_mean': att_mean,
        'atu_mean': atu_mean,
        'att_median': att_median,
        'atu_median': atu_median,
        'n_att': len(att_deltas),
        'n_atu': len(atu_deltas),
        't_statistic': t_result.statistic,
        't_pvalue': t_result.pvalue,
        'mw_statistic': mw_result.statistic,
        'mw_pvalue': mw_result.pvalue,
        't_significant': t_result.pvalue < 0.05,
        'mw_significant': mw_result.pvalue < 0.05
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print("\n" + "="*80)
print(f"T-test significant differences (p < 0.05): {results_df['t_significant'].sum()} out of {len(results_df)} methods")
print(f"Mann-Whitney significant differences (p < 0.05): {results_df['mw_significant'].sum()} out of {len(results_df)} methods")
print("="*80)

import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm

def studentized_statistic(x, y):
    """Compute studentized difference in means."""
    mean_diff = np.mean(x) - np.mean(y)
    se = np.sqrt(np.var(x, ddof=1)/len(x) + np.var(y, ddof=1)/len(y))
    return mean_diff / se

def permutation_test_studentized(x, y, n_permutations=10000):
    """Two-sample permutation test with studentized statistic."""
    observed = studentized_statistic(x, y)
    
    combined = np.concatenate([x, y])
    m = len(x)
    
    count = 0
    for _ in tqdm(range(n_permutations)):
        np.random.shuffle(combined)
        x_perm = combined[:m]
        y_perm = combined[m:]
        
        stat_perm = studentized_statistic(x_perm, y_perm)
        
        if np.abs(stat_perm) >= np.abs(observed):  # two-sided
            count += 1
    
    p_value = count / n_permutations
    return p_value, observed


if __name__ == "__main__":



    noblocks = pd.read_csv("csvs/confounddataset/excluded-docs.noblocks.lite.all_shards.csv.gz").rename(columns={"score": "noblocks"})
    blocks = pd.read_csv("csvs/confounddataset/excluded-docs.blocks.lite.all_shards.csv.gz").rename(columns={"score": "blocks"})


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
    ATT["delta"] = ATT["blocks"] - ATT["noblocks"]
    ATT = ATT[ATT["membership"] == "member"].copy()


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

    ATU = noblocks.merge(blocks, on=['doc_id', 'method', 'membership'])
    ATU["delta"] = ATU["blocks"] - ATU["noblocks"]
    ATU = ATU[ATU["membership"] == "member"].copy()

    for method in ATU["method"].unique():
        X = ATU[ATU["method"] == method]["delta"].copy()
        Y = ATT[ATT["method"] == method]["delta"].copy()
        print(method, permutation_test_studentized(X, Y))


#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon


def formal_comparison(pair1, pair2):

    for method in pair1["method"].unique():
        x = np.array(pair1[pair1["method"] == method]["delta"])
        y = np.array(pair2[pair2["method"] == method]["delta"])
        print(method)
        print(wilcoxon(x, y))

def load_pair(treated_file, control_file, treated_col="blocks", control_col="noblocks"):
    df1 = pd.read_csv(treated_file).rename(columns={"score": treated_col})
    df2 = pd.read_csv(control_file).rename(columns={"score": control_col})
    pair = df1.merge(df2, on=["doc_id", "method", "membership"])
    pair["delta"] = pair[treated_col] - pair[control_col]
    pair = pair[pair["membership"] == "member"].copy()
    return pair


def report_att(df):
    print(df[["delta", "method"]].groupby(["method"]).mean().reset_index())


if __name__ == "__main__":


    control_file = 'csvs/sutva_click2houston_com_2022-05-01_pair2_control_run4_sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered.csv'
    treated1 = 'csvs/sutva_click2houston_com_2022-05-01_pair1_treated_run1_sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered.csv'
    treated2 = 'csvs/sutva_click2houston_com_2022-05-01_pair2_treated_run3_sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered.csv'

    pair1 = load_pair(treated1, control_file)
    pair2 = load_pair(treated2, control_file)

    assert len(pair1) == len(pair2)
    assert len(pair1) * 2 == len(pd.read_csv(treated1)) == len(pd.read_csv(treated2)) == len(pd.read_csv(control_file))

    report_att(pair1)
    report_att(pair2)
    formal_comparison(pair1, pair2)

    #                                                        
    #                    88                      ad888888b,  
    #   ,d               88                     d8"     "88  
    #   88               88                             a8P  
    # MM88MMM ,adPPYYba, 88   ,d8  ,adPPYba,         ,d8P"   
    #   88    ""     `Y8 88 ,a8"  a8P_____88       a8P"      
    #   88    ,adPPPPP88 8888[    8PP"""""""     a8P'        
    #   88,   88,    ,88 88`"Yba, "8b,   ,aa    d8"          
    #   "Y888 `"8bbdP"Y8 88   `Y8a `"Ybbd8"'    88888888888  
    #                                                        
    #                                                        

    control_file_take2 = 'csvs/sutva_click2houston_com_2022-05-01_pair2_control_run4_sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered_take2.csv'
    treated1_take2 = 'csvs/sutva_click2houston_com_2022-05-01_pair2_treated_run3_sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered_take2.csv'

    pair2_take2 = load_pair(treated1_take2, control_file_take2)
    pair2_take2[["delta", "method"]].groupby(["method"]).mean().reset_index()

    report_att(pair2_take2)





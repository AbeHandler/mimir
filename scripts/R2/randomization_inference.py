import pandas as pd
import numpy as np

from scipy.stats import mannwhitneyu


def permutation_test_studentized(treated, control, n_permutations=10000):
    """
    Studentized two-sample permutation test
    """
    treated = np.asarray(treated)
    control = np.asarray(control)

    combined = np.concatenate([treated, control])
    m = len(treated)
    n = len(control)
    N = m + n

    mean_diff_obs = treated.mean() - control.mean()
    var_t_obs = treated.var(ddof=1)
    var_c_obs = control.var(ddof=1)
    T_obs = np.sqrt(m) * mean_diff_obs
    S_obs = T_obs / np.sqrt(N/m * var_t_obs + N/n * var_c_obs)

    permuted_stats = []

    for _ in range(n_permutations):
        perm_idx = np.random.permutation(N)
        perm_data = combined[perm_idx]

        perm_treated = perm_data[:m]
        perm_control = perm_data[m:]

        mean_diff_perm = perm_treated.mean() - perm_control.mean()
        var_t_perm = perm_treated.var(ddof=1)
        var_c_perm = perm_control.var(ddof=1)

        T_perm = np.sqrt(m) * mean_diff_perm
        S_perm = T_perm / np.sqrt(N/m * var_t_perm + N/n * var_c_perm)

        permuted_stats.append(S_perm)

    permuted_stats = np.array(permuted_stats)

    p_value = np.mean(np.abs(permuted_stats) >= np.abs(S_obs))

    return S_obs, p_value, permuted_stats


if __name__ == "__main__":

    ATTloss = pd.read_csv("data/interim/bothbins/ATTsfromR1/loss.csv")["delta"]
    ATUloss = pd.read_csv("data/interim/bothbins/loss.granular.csv")["delta"]

    stat, p = mannwhitneyu(ATUloss, ATTloss, alternative='two-sided')

    print("Basic mann whitney")
    print(stat, p)

    # Run the test
    S_obs, p_value, perm_dist = permutation_test_studentized(
        ATTloss, 
        ATUloss, 
        n_permutations=10000
    )

    print(f"\nObserved studentized statistic: {S_obs:.4f}")
    print(f"Two-sided p-value: {p_value:.4f}")
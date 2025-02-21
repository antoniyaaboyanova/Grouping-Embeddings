import numpy as np
import pandas as pd
import random
import statsmodels.api as sm

from tqdm import tqdm
from scipy.stats import ttest_1samp, ttest_rel, wilcoxon
from statsmodels.stats.multitest import fdrcorrection
from mne.stats import bonferroni_correction, fdr_correction
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

def fdr_correction(p_values):
    """
    Apply FDR correction to an array of p-values.
    """
    valids = ~np.isnan(p_values)
    p_corr = np.full_like(p_values, np.nan)
    p_corr[valids] = fdrcorrection(p_values[valids])[1]
    return p_corr

def sign_test_1samp(y, p, stat=np.mean):
    """
    Perform a sign permutation test for a one-sample statistic across time points.

    Parameters:
    - y: np.ndarray (subjects, timepoints)
    - p: Number of permutations
    - stat: Statistic function (default: mean)

    Returns:
    - Dictionary with uncorrected, FDR, and Bonferroni-corrected p-values.
    """
    p_val = np.full(y.shape[1], np.nan)

    for t in tqdm(range(y.shape[1])):
        dif_stat = stat(y[:, t])

        if np.isnan(dif_stat):
            continue

        pD = [stat(np.random.choice([-1, 1], y.shape[0]) * y[:, t]) for _ in range(p)]
        p_val[t] = (np.sum(np.array(pD) >= dif_stat) + 1) / (p + 1)

    valid = ~np.isnan(p_val)
    return {
        'uncorr': p_val,
        'fdr': fdr_correction(p_val[valid]),
        'bonferroni': bonferroni_correction(p_val[valid])[1]
    }

def sign_test_2samp(y, p, stat=np.mean):
    """
    Perform a sign permutation test for paired differences across time points.

    Parameters:
    - y: np.ndarray (subjects, 2, timepoints) 
         where y[:, 0, :] = condition 1, y[:, 1, :] = condition 2
    - p: Number of permutations
    - stat: Statistic function (default: mean)

    Returns:
    - Dictionary with uncorrected, FDR, and Bonferroni-corrected p-values.
    """
    assert y.shape[1] == 2, "Input array must have shape (subjects, 2, timepoints) for paired samples."

    n_timepoints = y.shape[2]
    p_val = np.full(n_timepoints, np.nan)

    for t in tqdm(range(n_timepoints)):
        diff = y[:, 1, t] - y[:, 0, t]  # Compute paired differences
        obs_stat = stat(diff)

        if np.isnan(obs_stat):
            continue

        # Generate permutation distribution by flipping signs randomly
        pD = [stat(np.random.choice([-1, 1], size=len(diff)) * diff) for _ in range(p)]
        p_val[t] = (np.sum(np.array(pD) >= obs_stat) + 1) / (p + 1)

    valid = ~np.isnan(p_val)
    return {
        'uncorr': p_val,
        'fdr': fdr_correction(p_val[valid]),
        'bonferroni': bonferroni_correction(p_val[valid])[1]
    }

def t_test(y):
    """
    Perform a one-sample t-test across time points.
    
    Parameters:
    - y: np.ndarray (subjects, timepoints)
    
    Returns:
    - Dictionary with uncorrected, FDR, and Bonferroni-corrected p-values.
    """
    pv = np.array([ttest_1samp(y[:, t], 0)[1] for t in tqdm(range(y.shape[1]))])
    return {
        'uncorr': pv,
        'fdr': fdr_correction(pv),
        'bonferroni': bonferroni_correction(pv)[1]
    }

def t_test_bw(y):
    """
    Perform a paired t-test (within subjects) across time points.

    Parameters:
    - y: np.ndarray (subjects, 2, timepoints)

    Returns:
    - Dictionary with uncorrected, FDR, and Bonferroni-corrected p-values.
    """
    assert y.shape[1] == 2, "Input must have shape (subjects, 2, timepoints)."
    
    pv = np.array([ttest_rel(y[:, 0, t], y[:, 1, t])[1] for t in tqdm(range(y.shape[2]))])
    return {
        'uncorr': pv,
        'fdr': fdr_correction(pv),
        'bonferroni': bonferroni_correction(pv)[1]
    }

def wilcox_test(y):
    """
    Perform a one-sample Wilcoxon signed-rank test across time points.

    Parameters:
    - y: np.ndarray (subjects, timepoints)

    Returns:
    - Dictionary with uncorrected, FDR, and Bonferroni-corrected p-values.
    """
    pv = np.array([wilcoxon(y[:, t])[1] for t in tqdm(range(y.shape[1]))])
    return {
        'uncorr': pv,
        'fdr': fdr_correction(pv),
        'bonferroni': bonferroni_correction(pv)[1]
    }

def wilcox_test_bw(y):
    """
    Perform a paired Wilcoxon signed-rank test across time points.

    Parameters:
    - y: np.ndarray (subjects, 2, timepoints)

    Returns:
    - Dictionary with uncorrected, FDR, and Bonferroni-corrected p-values.
    """
    assert y.shape[1] == 2, "Input must have shape (subjects, 2, timepoints)."

    pv = np.full(y.shape[2], np.nan)
    for t in tqdm(range(y.shape[2])):
        try:
            _, pv[t] = wilcoxon(y[:, 0, t], y[:, 1, t], alternative="two-sided")
        except ValueError:
            pv[t] = 1.0  # If all differences are zero

    return {
        'uncorr': pv,
        'fdr': fdr_correction(pv),
        'bonferroni': bonferroni_correction(pv)[1]
    }
import numpy as np


def run_anova_analysis(y):
    """
    Performs a two-way ANOVA across time points to analyze the effects of Task Type and Stimulus Order.
    
    Parameters:
    y (numpy array): A 3D array of shape (subjects, conditions, time points), 
                                representing accuracy values for each subject, condition, and time point.
    
    Returns:
    pd.DataFrame: A DataFrame with p-values and corrected p-values for Task Type, Stimulus Order, and their interaction.
    """
    num_subjects, num_conditions, num_timepoints = y.shape
    anova_results = []

    for t in range(num_timepoints):
        df = pd.DataFrame({
            'Subject': np.repeat(np.arange(num_subjects), num_conditions),
            'Condition': np.tile(np.arange(num_conditions), num_subjects),
            'Value': y[:, :, t].flatten()
        })

        # Recode conditions into Task Type and Stimulus Order
        df['Task_Type'] = df['Condition'].map(lambda x: 'No Attention' if x in [0, 1] else 'Attention')
        df['Stimulus_Order'] = df['Condition'].map(lambda x: 'Determ' if x in [0, 2] else 'Random')

        try:
            model = ols('Value ~ C(Task_Type) + C(Stimulus_Order) + C(Task_Type):C(Stimulus_Order)', data=df).fit()
            anova_table = anova_lm(model, typ=2)

            # Extract p-values
            p_task_type = anova_table.loc['C(Task_Type)', 'PR(>F)']
            p_stimulus_order = anova_table.loc['C(Stimulus_Order)', 'PR(>F)']
            p_interaction = anova_table.loc['C(Task_Type):C(Stimulus_Order)', 'PR(>F)']

            anova_results.append([t, p_task_type, p_stimulus_order, p_interaction])

        except Exception as e:
            print(f"Error at time {t}: {e}")
            continue

    # Convert results to DataFrame
    anova_results_df = pd.DataFrame(anova_results, columns=['Time', 'P_Task_Type', 'P_Stimulus_Order', 'P_Interaction'])

    # Apply multiple comparisons correction
    for col in ['P_Task_Type', 'P_Stimulus_Order', 'P_Interaction']:
        anova_results_df[f'{col}_corr'] = multipletests(anova_results_df[col], method='fdr_bh')[1]

    return anova_results_df

import numpy as np
from scipy.ndimage import label

def find_sig_bounds(p_values, alpha=0.05, min_cluster_size=5):
    """
    Identifies the onset and offset indices where p-values are below the significance threshold,
    ensuring that only clusters of at least `min_cluster_size` consecutive points are considered.
    
    Parameters:
    p_values (array-like): A 1D array of p-values.
    alpha (float): Significance threshold (default is 0.001).
    min_cluster_size (int): Minimum number of consecutive significant points to form a cluster (default is 5).
    
    Returns:
    tuple: (onset_index, offset_index) or (None, None) if no valid cluster is found.
    """
    significant = np.array(p_values) < alpha  # Boolean mask of significance
    
    if not np.any(significant):
        return None, None  # No significant values found
    
    # Label clusters of consecutive significant time points
    labeled_clusters, num_clusters = label(significant)
    
    for cluster_id in range(1, num_clusters + 1):  
        cluster_indices = np.where(labeled_clusters == cluster_id)[0]
        
        if len(cluster_indices) >= min_cluster_size:  # Check if cluster size meets threshold
            onset = cluster_indices[0]  # First index in cluster
            offset = cluster_indices[-1]  # Last index in cluster
            return onset, offset
    
    return None, None  # No clusters meeting the size criterion

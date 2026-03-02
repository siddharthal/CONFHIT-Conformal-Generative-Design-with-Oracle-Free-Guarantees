import numpy as np
import random
import pandas as pd

def sample_and_prepare_test_data(test_data, max_k, density_threshold=None, similarity_threshold=None, seed=None):
    """
    Filter and sample test data based on thresholds. Uses random sampling per group.
    """
    # First apply similarity threshold if specified
    if similarity_threshold is not None:
        test_data = test_data[
            (test_data["SIMILARITY_ori_opt"] > similarity_threshold)
        ].copy()
    
    # Then filter by density threshold if specified
    if density_threshold is not None:
        test_data = test_data[test_data["density"] >= density_threshold]
    
    ## Drop duplicates
    test_data = test_data.drop_duplicates(subset=["SMILES_ori", "SMILES_opt"])
    
    # Sample data from each group
    grouped = test_data.groupby("SMILES_ori")
    sampled_test_data = []
    
    for smiles_ori, group in grouped:
        if len(group) >= max_k:
            sampled = group.sample(n=max_k, replace=False, random_state=seed)
            sampled_test_data.append(sampled)
    
    if not sampled_test_data:
        raise ValueError("No groups with enough samples found after filtering")
    
    # Combine all sampled groups
    final_test_data = pd.concat(sampled_test_data, ignore_index=True)
    
    return final_test_data

def conformal_pvalue_single(cal_scores, cal_weights, s_test, w_test):
    """
    Compute weighted conformal p-value for a single test point.
    Uses normalized empirical CDF approach.
    
    Parameters:
    - cal_scores: array-like of calibration nonconformity scores
    - cal_weights: array-like of importance weights for calibration samples
    - s_test: scalar, score for test point
    - w_test: scalar, importance weight for test point (not used in this version)
    
    Returns:
    - p-value for test point (weighted proportion of calibration samples >= test score)
    """
    
    cal_weights = np.array(cal_weights)
    # Normalize calibration weights to sum to 1
    cal_weights_normalized = cal_weights / np.sum(cal_weights)
    
    # Compute weighted empirical CDF: proportion of calibration samples >= test score
    pvalue = np.sum(cal_weights_normalized[cal_scores >= s_test])

    return pvalue

def permutation_weighted_pval(cal_scores, cal_weights, test_scores, test_weights, M=500, statistic="max"):
    """
    Monte Carlo estimate of permutation-weighted p-value with flexible statistic.
    
    Parameters:
    - cal_scores: list of calibration scores
    - cal_weights: list of calibration importance weights
    - test_scores: list of test scores (length = k)
    - test_weights: list of test weights (length = k)
    - statistic: one of "max", "min", "mean", or "sum"
    - M: number of Monte Carlo permutations
    
    Returns:
    - estimated p-value
    """
    
    n0 = len(cal_scores)
    k = len(test_scores)
    cal_scores = np.array(cal_scores)
    test_scores = np.array(test_scores)
    cal_weights = np.array(cal_weights)
    test_weights = np.array(test_weights)

    all_scores = np.concatenate([cal_scores, test_scores])
    all_weights = np.concatenate([cal_weights, test_weights])
    
    # Define the statistic function
    if statistic == "max":
        stat = np.max
    elif statistic == "min":
        stat = np.min
    elif statistic == "mean":
        def stat(scores):
            mean = np.mean(scores)
            return mean
    elif statistic == "sum":
        def stat(scores):
            s = np.sum(scores)
            return s
    elif statistic == "rank_sum":
        def stat(scores):
            # Get ranks of all scores (calibration + test) in ascending order
            all_ranks = np.argsort(np.argsort(all_scores)) + 1
            # Return sum of ranks for test scores
            return np.sum(all_ranks[n0:n0+k])
    elif statistic == "likelihood_ratio":
        def stat(scores):
            # Compute sum of log likelihood ratios for test points
            # Here scores should be probability estimates
            return np.sum(np.log(1- scores / (scores)))
    else:
        raise ValueError(f"Unknown statistic: {statistic}")

    S_obs = stat(test_scores)
    num = 0.0
    den = 0.0

    for _ in range(M):
        perm = np.random.permutation(n0 + k)
        test_idx = perm[n0:]  # permuted test set of size k

        test_scores_perm = all_scores[test_idx]
        test_weights_perm = all_weights[test_idx]
        W_pi = np.prod(test_weights_perm)

        S_pi = stat(test_scores_perm)
        
        den += W_pi
        if S_obs >= S_pi:
            num += W_pi
            
    return num/den

def compute_nested_pvalues(opt_samples, cal_scores, cal_weights, statistic="max", 
                         M=500):
    """
    Compute nested p-values for a sequence of samples (standard permutation).
    
    Parameters:
    - opt_samples: list of dictionaries with 'score' and 'likelihood_ratio' for each sample
    - cal_scores: calibration scores
    - cal_weights: calibration weights
    - statistic: which statistic to use for combining p-values ("min", "max", "mean")
    - M: number of permutations for nested p-value computation
    
    Returns:
    - list of SMILES in order they were picked
    - list of corresponding nested p-values
    """
    # Randomly shuffle the samples
    random.shuffle(opt_samples)
    
    selected_smiles = []
    nested_pvalues = []
    
    for i, sample in enumerate(opt_samples):
        selected_smiles.append(sample.get('smiles', f'sample_{i}'))
        
        if i == 0:
            p_value = conformal_pvalue_single(
                cal_scores, 
                cal_weights, 
                sample['score'], 
                sample['likelihood_ratio']
            )
            nested_pvalues.append(p_value)
        else:
            test_scores = [opt_samples[j]['score'] for j in range(i+1)]
            test_weights = [opt_samples[j]['likelihood_ratio'] for j in range(i+1)]
            
            p_value = permutation_weighted_pval(
                cal_scores=cal_scores,
                cal_weights=cal_weights,
                test_scores=test_scores,
                test_weights=test_weights,
                M=M,
                statistic=statistic
            )
            
            nested_pvalues.append(p_value)
    
    # Enforce monotonically decreasing p-values: from last element backward
    for i in range(len(nested_pvalues) - 1, 0, -1):
        if nested_pvalues[i - 1] < nested_pvalues[i]:
            nested_pvalues[i - 1] = nested_pvalues[i]

    return selected_smiles, nested_pvalues

def design_test(calib_scores, calib_weights, test_scores_stream, test_weights_stream, 
                   alphas=[0.05], max_k=None, M=1000, statistic="max"):
    """
    Design test: weighted sequential test with standard permutation.
    
    Parameters:
    - calib_scores: scores from calibration set
    - calib_weights: weights for calibration set
    - test_scores_stream: stream of test scores
    - test_weights_stream: stream of test weights
    - alphas: list of significance levels to test
    - max_k: maximum number of samples to test (None for all)
    - M: number of permutations for nested p-value computation
    - statistic: which statistic to use for combining p-values
    
    Returns:
    - rejection_points: dictionary mapping alpha values to their rejection points
    - p_values: list of p-values
    """
    if max_k is None:
        max_k = len(test_scores_stream)
    else:
        max_k = min(max_k, len(test_scores_stream))
    
    test_samples = [
        {'score': s, 'likelihood_ratio': w} 
        for s, w in zip(test_scores_stream, test_weights_stream)
    ]
    
    _, p_values = compute_nested_pvalues(
        test_samples[:max_k], 
        calib_scores, 
        calib_weights,
        statistic=statistic,
        M=M
    )
    
    rejection_points = {}
    for alpha in alphas:
        rejection_point = None
        for k, p_value in enumerate(p_values):
            if p_value <= alpha:
                rejection_point = k
                break
        rejection_points[alpha] = rejection_point
    
    return rejection_points, p_values

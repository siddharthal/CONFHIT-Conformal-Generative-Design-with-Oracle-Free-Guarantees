import numpy as np
import pandas as pd
from design import permutation_weighted_pval, sample_and_prepare_test_data

def run_certification_analysis(val_data, test_data, property_threshold=0.9,
                              alphas=[0.05], M=1000, statistic="max",
                              max_samples_per_group=None,
                              density_threshold=None, similarity_threshold=None,
                              seed=None):
    """
    Run certification analysis on molecular data. Calibration uses negative samples only.
    """
    
    # Prepare calibration data
    # Handle different column naming conventions
    if "PROPERTY_ori" in val_data.columns:
        property_col = "PROPERTY_ori"
    elif "vina_score" in val_data.columns:
        property_col = "vina_score"
    elif "drd2_score" in val_data.columns:
        property_col = "drd2_score"
    else:
        raise ValueError("Unknown property column name in validation data")
    
    # Calibration: negative only
    calib_data = val_data[val_data[property_col] < property_threshold].copy()
    
    calib_scores = calib_data["score"].values
    calib_weights = calib_data["likelihood_ratio"].values
    
    # Determine grouping column first
    if "SMILES_ori" in test_data.columns:
        grouping_col = "SMILES_ori"
    elif "smiles_low" in test_data.columns:
        grouping_col = "smiles_low"  
    else:
        raise ValueError("Unknown grouping column name in test data")
    
    # Prepare test data - apply filtering and sampling if specified
    if max_samples_per_group is not None:
        test_data = sample_and_prepare_test_data(
            test_data=test_data,
            max_k=max_samples_per_group,
            density_threshold=density_threshold,
            similarity_threshold=similarity_threshold,
            seed=seed
        )
    else:
        # Apply basic filtering without sampling
        if similarity_threshold is not None:
            test_data = test_data[
                test_data["SIMILARITY_ori_opt"] > similarity_threshold
            ].copy()
        
        if density_threshold is not None:
            test_data = test_data[test_data["density"] >= density_threshold]
        
        # Drop duplicates - handle different column naming conventions
        if grouping_col == "SMILES_ori" and "SMILES_opt" in test_data.columns:
            test_data = test_data.drop_duplicates(subset=["SMILES_ori", "SMILES_opt"])
        elif grouping_col == "smiles_low" and "smiles_high" in test_data.columns:
            test_data = test_data.drop_duplicates(subset=["smiles_low", "smiles_high"])
    
    # Group test data by appropriate column
    grouped_test = test_data.groupby(grouping_col)
    
    print(f"Running certification analysis for alphas = {alphas}")
    print(f"Total groups to process: {len(grouped_test)}")
    
    # Determine test property column
    if "PROPERTY_opt" in test_data.columns:
        test_property_col = "PROPERTY_opt"
    elif "smiles_high" in test_data.columns:
        # For DRD2 data, we need to check the score column 
        test_property_col = "drd2_score" if "drd2_score" in test_data.columns else "score"
    else:
        test_property_col = "score"  # fallback
    
    # Store p-values and group properties (computed once)
    group_data = {}
    
    # Process each group ONCE to compute p-values
    print("Computing p-values for all groups...")
    for smiles_ori, group in grouped_test:
        test_scores = group["score"].values
        test_weights = group["likelihood_ratio"].values
        
        # Compute p-value once using permutation_weighted_pval
        p_value = permutation_weighted_pval(
            cal_scores=calib_scores,
            cal_weights=calib_weights, 
            test_scores=test_scores,
            test_weights=test_weights,
            M=M,
            statistic=statistic
        )
        
        # Determine if group has high property molecules
        has_high_property = (group[test_property_col] >= property_threshold).any()
        
        # Store group data (p-value computed once)
        group_data[smiles_ori] = {
            'p_value': float(p_value),
            'has_high_property': bool(has_high_property),
            'num_samples': len(group),
            'max_property': float(group[test_property_col].max()),
            'mean_property': float(group[test_property_col].mean()),
            'test_scores': test_scores.tolist(),
            'test_weights': test_weights.tolist()
        }
    
    # Now apply different alpha thresholds to the same p-values
    results = {}
    for alpha in alphas:
        print(f"Applying alpha threshold = {alpha}")
        
        # Initialize results for this alpha
        alpha_results = {
            'group_results': {},
            'summary': {
                'total_groups': len(group_data),
                'rejected_groups': 0,
                'rejected_with_high_property': 0,
                'rejected_with_low_property': 0,
                'fraction_rejected': 0.0,
                'error_rate': 0.0,
                'alpha': alpha
            }
        }
        
        # Apply alpha threshold to each group
        for smiles_ori, data in group_data.items():
            p_value = data['p_value']
            rejected = p_value <= alpha
            has_high_property = data['has_high_property']
            
            # Store results for this group and alpha
            alpha_results['group_results'][smiles_ori] = {
                'p_value': p_value,
                'rejected': bool(rejected),
                'has_high_property': has_high_property,
                'num_samples': data['num_samples'],
                'max_property': data['max_property'],
                'mean_property': data['mean_property'],
                'test_scores': data['test_scores'],
                'test_weights': data['test_weights']
            }
            
            # Update summary statistics
            if rejected:
                alpha_results['summary']['rejected_groups'] += 1
                if has_high_property:
                    alpha_results['summary']['rejected_with_high_property'] += 1
                else:
                    alpha_results['summary']['rejected_with_low_property'] += 1
        
        # Calculate final metrics for this alpha
        total_groups = alpha_results['summary']['total_groups']
        if total_groups > 0:
            alpha_results['summary']['fraction_rejected'] = alpha_results['summary']['rejected_groups'] / total_groups
            
            # Error rate = rejected groups that don't have high property / total rejected groups
            if alpha_results['summary']['rejected_groups'] > 0:
                alpha_results['summary']['error_rate'] = (
                    alpha_results['summary']['rejected_with_low_property'] / 
                    alpha_results['summary']['rejected_groups']
                )
            else:
                alpha_results['summary']['error_rate'] = 0.0
        
        results[alpha] = alpha_results
    
    return results

def print_certification_summary(results):
    """Print a summary of certification results."""
    summary = results['summary']
    
    print("\n" + "="*60)
    print("CERTIFICATION ANALYSIS SUMMARY")
    print("="*60)
    print(f"Alpha level: {summary['alpha']}")
    print(f"Total groups tested: {summary['total_groups']}")
    print(f"Groups rejected: {summary['rejected_groups']}")
    print(f"Fraction rejected: {summary['fraction_rejected']:.4f} ({summary['fraction_rejected']*100:.2f}%)")
    print()
    print("Breakdown of rejected groups:")
    print(f"  - With high property molecules: {summary['rejected_with_high_property']}")
    print(f"  - With low property molecules: {summary['rejected_with_low_property']}")
    print()
    print(f"Error rate: {summary['error_rate']:.4f} ({summary['error_rate']*100:.2f}%)")
    print("(Error rate = rejected groups without high property / total rejected groups)")
    print("="*60)

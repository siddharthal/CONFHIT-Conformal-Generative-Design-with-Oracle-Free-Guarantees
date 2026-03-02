import argparse
import os
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Import from other modules
from models import load_model, extract_features, set_seed, sample_for_testing
from kde_utils import compute_kde_from_features, get_likelihood_ratio
from design import design_test

def load_config_from_file(config_path):
    """Load configuration from a JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def prepare_baseline_data(val_data, test_data, args):
    """
    Prepare data for likelihood ratio baseline (LR=1, no density filtering).
    
    Parameters:
    - val_data: validation data
    - test_data: test data  
    - args: configuration arguments
    
    Returns:
    - calib_scores: calibration scores
    - calib_weights: calibration weights (all set to 1)
    - test_data_grouped: grouped test data without density filtering
    """
    # Apply similarity threshold if specified and column exists (but skip density filtering)
    filtered_test_data = test_data.copy()
    
    if args.similarity_threshold is not None and "SIMILARITY_ori_opt" in filtered_test_data.columns:
        filtered_test_data = filtered_test_data[
            filtered_test_data["SIMILARITY_ori_opt"] > args.similarity_threshold
        ]
    elif args.similarity_threshold is not None:
        print("SIMILARITY_ori_opt column not found, skipping similarity filtering")
    
    # Drop duplicates
    filtered_test_data = filtered_test_data.drop_duplicates(subset=["SMILES_ori", "SMILES_opt"])
    
    # Set all likelihood ratios to 1 (no density-based weighting)
    val_data_baseline = val_data.copy()
    filtered_test_data["likelihood_ratio"] = 1.0
    val_data_baseline["likelihood_ratio"] = 1.0
    
    # Prepare calibration data (negative only)
    calib_data = val_data_baseline[val_data_baseline["label"] == 0]
    
    calib_scores = calib_data["score"].values * -1
    calib_weights = calib_data["likelihood_ratio"].values  # All 1.0
    
    # Group test data
    test_data_grouped = filtered_test_data.groupby("SMILES_ori")
    
    return calib_scores, calib_weights, test_data_grouped


def run_design_with_budget(calib_scores, calib_weights, test_data_grouped, 
                                   max_samples_per_group, alphas, budget, args, baseline_mode=False):
    """
    Run design with budget constraints following design_main logic.
    
    Parameters:
    - calib_scores: calibration scores
    - calib_weights: calibration weights
    - test_data_grouped: grouped test data
    - max_samples_per_group: maximum samples per group
    - alphas: list of alpha values to test
    - budget: total budget constraint
    - args: configuration arguments
    - baseline_mode: if True, uses likelihood ratio = 1 for all samples (baseline)
    
    Returns:
    - results: dictionary with results for each alpha including budget optimization
    """
    # First run design on all groups to get rejection points
    all_group_results = {}
    full_group_metrics = {alpha: {
        "total_groups": len(test_data_grouped),
        "no_rejection_groups": 0,
        "rejection_groups": 0,
        "rejection_with_high_property": 0,
        "rejection_without_high_property": 0,
        "total_points_tested": 0,
        "groups_with_positive_sample": 0
    } for alpha in alphas}
    
    print(f"Running design with max_samples_per_group={max_samples_per_group} for {len(test_data_grouped)} groups...")
    
    for smiles_ori, group in tqdm(test_data_grouped):
        # Sample up to max_samples_per_group samples from this group
        if len(group) < max_samples_per_group:
            sampled_group = group
        else:
            sampled_group = group.sample(n=max_samples_per_group, random_state=args.seed)
        
        group_test_scores = sampled_group["score"].values
        if baseline_mode:
            # Baseline: set all likelihood ratios to 1
            group_test_weights = np.ones(len(sampled_group))
        else:
            group_test_weights = sampled_group["likelihood_ratio"].values
        group_property_values = sampled_group["PROPERTY_opt"].values
        
        
        # Run design for this group (following design_main logic)
        group_rejection_points, group_p_values = design_test(
            calib_scores=calib_scores,
            calib_weights=calib_weights,
            test_scores_stream=group_test_scores*-1,
            test_weights_stream=group_test_weights,
            alphas=alphas,
            max_k=len(sampled_group),
            M=args.permutations if hasattr(args, 'permutations') else 1000,
            statistic=args.statistic if hasattr(args, 'statistic') else "min"
        )
        
        # Store group results
        group_results = {
            "SMILES_ori": smiles_ori,
            "p_values": group_p_values.tolist() if hasattr(group_p_values, 'tolist') else group_p_values,
            "samples_tested": len(sampled_group),
            "property_values": group_property_values.tolist() if hasattr(group_property_values, 'tolist') else group_property_values
        }
        
        # Update metrics for each alpha
        for alpha in alphas:
            rejection_point = group_rejection_points[alpha]
            group_results[f"rejection_point_alpha_{alpha}"] = rejection_point
            
            # Update point count
            if rejection_point is not None:
                full_group_metrics[alpha]["total_points_tested"] += (rejection_point + 1)
            else:
                full_group_metrics[alpha]["total_points_tested"] += len(sampled_group)
            
            if rejection_point is None:
                # No rejection - this is an "empty set"
                full_group_metrics[alpha]["no_rejection_groups"] += 1
            else:
                # Rejection occurred
                full_group_metrics[alpha]["rejection_groups"] += 1
                # Check if this group has any samples with high property before rejection
                property_values_rejected = group_property_values[:rejection_point+1]
                if any(val > args.property_threshold for val in property_values_rejected):
                    full_group_metrics[alpha]["rejection_with_high_property"] += 1
                    # Track groups with positive samples in rejected set
                    full_group_metrics[alpha]["groups_with_positive_sample"] += 1
                else:
                    full_group_metrics[alpha]["rejection_without_high_property"] += 1
        
        all_group_results[smiles_ori] = group_results
    
    # Calculate derived metrics
    for alpha in alphas:
        metrics = full_group_metrics[alpha]
        total_groups = metrics["total_groups"]
        
        # Fraction of empty sets (no rejection groups)
        metrics["fraction_empty_sets"] = metrics["no_rejection_groups"] / total_groups if total_groups > 0 else 0
        
        # True alpha = actual alpha + fraction of empty sets
        metrics["true_alpha"] = alpha + metrics["fraction_empty_sets"]
        
        # Error rate = true_alpha + fraction_empty_sets
        metrics["error_rate"] = metrics["true_alpha"] + metrics["fraction_empty_sets"]
        
        # True error rate = 1 - fraction of groups containing a positive sample in rejected set
        frac_with_positive = metrics["groups_with_positive_sample"] / total_groups if total_groups > 0 else 0
        metrics["true_error_rate"] = 1 - frac_with_positive
    
    # Now implement budget optimization
    print(f"Budget optimization with budget={budget}")
    optimized_results = {}
    best_true_alpha = float('inf')  # Start with infinity since we want minimum
    best_result = None
    
    for alpha in alphas:
        print(f"Optimizing for alpha={alpha}")
        
        # Calculate points needed for each group at this alpha
        group_points = []
        for smiles_ori, results in all_group_results.items():
            rejection_point = results[f"rejection_point_alpha_{alpha}"]
            points_needed = (rejection_point + 1) if rejection_point is not None else results["samples_tested"]
            property_values = results.get("property_values", [])
            group_points.append((smiles_ori, points_needed, rejection_point, property_values))
        
        # Sort by points needed (largest first) for removal if budget exceeded
        group_points.sort(key=lambda x: x[1], reverse=True)
        
        total_points = sum(x[1] for x in group_points)
        
        if total_points <= budget:
            # Budget allows all groups
            selected_groups = group_points
        else:
            # Remove largest groups until budget constraint is met
            selected_groups = []
            current_budget = 0
            for group_info in reversed(group_points):  # Start with smallest
                if current_budget + group_info[1] <= budget:
                    selected_groups.append(group_info)
                    current_budget += group_info[1]
        
        # Calculate metrics for selected groups (budget accounting)
        total_points_used = sum(x[1] for x in selected_groups)
        selected_group_names = set(x[0] for x in selected_groups)
        
        # Calculate metrics on ALL 50 groups (not just selected ones)
        total_groups = len(group_points)  # All groups, not just selected
        no_rejection_groups = 0
        groups_with_positive_sample = 0
        
        for smiles_ori, points_needed, rejection_point, property_values in group_points:
            if smiles_ori not in selected_group_names:
                # Group was removed due to budget - count as empty set
                no_rejection_groups += 1
            elif rejection_point is None:
                # Group had no rejection - natural empty set
                no_rejection_groups += 1
            else:
                # Group was rejected - check if it contains positive samples
                property_values_rejected = property_values[:rejection_point+1]
                if any(val > args.property_threshold for val in property_values_rejected):
                    groups_with_positive_sample += 1
        
        # Calculate derived metrics on ALL groups
        fraction_empty_sets = no_rejection_groups / total_groups if total_groups > 0 else 0
        true_alpha = alpha + fraction_empty_sets
        frac_with_positive = groups_with_positive_sample / total_groups if total_groups > 0 else 0
        true_error_rate = 1 - frac_with_positive
        
        optimized_results[alpha] = {
            "alpha": alpha,
            "total_groups": total_groups,
            "selected_groups_count": len(selected_groups),
            "removed_groups_count": total_groups - len(selected_groups),
            "no_rejection_groups": no_rejection_groups,
            "groups_with_positive_sample": groups_with_positive_sample,
            "total_points_tested": total_points_used,
            "fraction_empty_sets": fraction_empty_sets,
            "fraction_with_positive_sample": frac_with_positive,
            "true_alpha": true_alpha,
            "true_error_rate": true_error_rate,
            "selected_groups": [x[0] for x in selected_groups]
        }
        
        # Track best (minimum) true alpha for this budget
        if true_alpha < best_true_alpha:
            best_true_alpha = true_alpha
            best_result = optimized_results[alpha].copy()
            best_result["best_nominal_alpha"] = alpha
    
    print(f"Best alpha: {best_result['best_nominal_alpha']} with true_alpha: {best_true_alpha:.3f}")
    
    return all_group_results, optimized_results, best_result


def main(args):
    # Set random seed
    set_seed(args.seed)
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.output_dir) / f"budget_analysis_run_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    with open(results_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    print(f"Running budget-constrained design analysis...")
    print(f"Max samples per group values to test: {args.max_samples_per_group_values}")
    print(f"Alpha range: {args.alpha_start} to {args.alpha_end}")
    print()
    
    # Load data (following design_main logic)
    val_data = pd.read_csv(args.val_data_path)
    test_data = pd.read_csv(args.test_data_path)
    
    # Note: test mode will limit groups later, not total samples
    
    # Check if features and scores are already available
    features_available = ('features' in val_data.columns and 'features' in test_data.columns and
                         'score' in val_data.columns and 'score' in test_data.columns)
    
    if features_available:
        print("Features and scores found in data. Skipping feature extraction...")
        if isinstance(val_data['features'].iloc[0], str):
            val_data['features'] = val_data['features'].apply(json.loads)
        if isinstance(test_data['features'].iloc[0], str):
            test_data['features'] = test_data['features'].apply(json.loads)
    else:
        # Feature extraction (same as design_main)
        print("Loading feature extraction model...")
        feature_model = load_model(args.feature_model_checkpoint)
        scoring_model = feature_model
        
        # Extract features for validation data
        print("Extracting validation features and scores...")
        val_pairs = val_data[["smiles_low", "smiles_high"]].values
        val_features, _ = extract_features(feature_model, val_pairs)
        val_data["features"] = val_features.tolist()
        _, val_scores = extract_features(scoring_model, val_pairs)
        val_data["score"] = val_scores
        
        # Extract features for test data
        print("Extracting test features and scores...")
        test_pairs = test_data[["SMILES_ori", "SMILES_opt"]].values
        test_features, _ = extract_features(feature_model, test_pairs)
        test_data["features"] = test_features.tolist()
        _, test_scores = extract_features(scoring_model, test_pairs)
        test_data["score"] = test_scores
    
    # Prepare features for KDE computation (following design_main)
    val_features = np.array(val_data["features"].tolist())
    test_features = np.array(test_data["features"].tolist())
    
    # Compute KDEs
    print("Computing KDEs...")
    val_neg_features = val_features[val_data["label"] == 0]
    val_neg_kde, _ = compute_kde_from_features(val_neg_features, bandwidths=args.kde_bandwidths)
    val_kde, _ = compute_kde_from_features(val_features, bandwidths=args.kde_bandwidths)
    
    # Compute densities
    val_data["density"] = val_kde.score_samples(val_features)
    test_data["density"] = val_kde.score_samples(test_features)
    
    # Get density threshold if specified
    density_threshold = None
    if args.density_threshold is not None:
        density_threshold = np.percentile(val_data["density"], args.density_threshold)
        print(f"Calculated density threshold: {density_threshold}")
    
    # Filter test data based on thresholds (following design_main approach)
    print("Filtering test data...")
    filtered_test_data = test_data.copy()
    
    # Apply similarity threshold if specified and column exists
    if args.similarity_threshold is not None and "SIMILARITY_ori_opt" in filtered_test_data.columns:
        filtered_test_data = filtered_test_data[
            filtered_test_data["SIMILARITY_ori_opt"] > args.similarity_threshold
        ]
    elif args.similarity_threshold is not None:
        print("SIMILARITY_ori_opt column not found, skipping similarity filtering")
    
    # Apply density threshold if specified
    if density_threshold is not None:
        filtered_test_data = filtered_test_data[filtered_test_data["density"] >= density_threshold]
        val_data = val_data[val_data["density"] > density_threshold]
    
    # Drop duplicates
    filtered_test_data = filtered_test_data.drop_duplicates(subset=["SMILES_ori", "SMILES_opt"])
    
    # Compute likelihood ratios
    filtered_features = np.array(filtered_test_data["features"].tolist())
    test_kde, _ = compute_kde_from_features(filtered_features, bandwidths=[1])
    
    if args.kde_type == "negative_only":
        kde_for_ratio = val_neg_kde
    else:
        kde_for_ratio = val_kde
    
    val_data["likelihood_ratio"] = get_likelihood_ratio(
        val_data["features"].values.tolist(), kde_for_ratio, test_kde
    )
    filtered_test_data["likelihood_ratio"] = get_likelihood_ratio(
        filtered_features, kde_for_ratio, test_kde
    )
    
    # Prepare calibration data (negative only)
    calib_data = val_data[val_data["label"] == 0]
    
    calib_scores = calib_data["score"].values * -1
    calib_weights = calib_data["likelihood_ratio"].values
    
    # Group test data
    test_data_grouped = filtered_test_data.groupby("SMILES_ori")
    n_groups = len(test_data_grouped)
    
    print(f"Number of groups: {n_groups}")
    
    # If test mode is enabled, limit to test_samples number of groups
    if args.test_mode and hasattr(args, 'test_samples') and args.test_samples:
        print(f"Test mode: limiting to {args.test_samples} groups")
        group_names = list(test_data_grouped.groups.keys())[:args.test_samples]
        filtered_test_data = filtered_test_data[filtered_test_data["SMILES_ori"].isin(group_names)]
        test_data_grouped = filtered_test_data.groupby("SMILES_ori")
        print(f"Number of groups after test mode filtering: {len(test_data_grouped)}")
    
    # Parse alpha values
    alpha_values = np.arange(args.alpha_start, args.alpha_end + 0.01, 0.05)
    alpha_values = [round(alpha, 2) for alpha in alpha_values]  # Round to avoid floating point issues
    
    # Prepare baseline data (LR=1, no density filtering)
    print("\nPreparing baseline data (LR=1, no density filtering)...")
    baseline_calib_scores, baseline_calib_weights, baseline_test_data_grouped = prepare_baseline_data(
        val_data, test_data, args
    )
    
    # Apply test mode to baseline data if enabled
    if args.test_mode and hasattr(args, 'test_samples') and args.test_samples:
        print(f"Test mode: limiting baseline to {args.test_samples} groups")
        baseline_group_names = list(baseline_test_data_grouped.groups.keys())[:args.test_samples]
        baseline_filtered_data = test_data[test_data["SMILES_ori"].isin(baseline_group_names)].copy()
        baseline_filtered_data["likelihood_ratio"] = 1.0
        baseline_test_data_grouped = baseline_filtered_data.groupby("SMILES_ori")
        print(f"Number of baseline groups after test mode filtering: {len(baseline_test_data_grouped)}")
    
    print(f"Baseline groups (no density filtering): {len(baseline_test_data_grouped)}")
    print(f"Original groups (with density filtering): {len(test_data_grouped)}")
    
    # Run analysis for different budgets using configured max_samples_per_group
    max_samples_per_group = args.max_samples_per_group_values[0] if args.max_samples_per_group_values else 15
    all_results = {}
    
    for budget in args.budgets:
        print(f"\n" + "="*60)
        print(f"Running analysis with budget = {budget}")
        print("="*60)
        
        # Run design with budget optimization (with density filtering and LR weighting)
        print("Running ORIGINAL analysis (with density filtering & LR weighting)...")
        group_results, optimized_results, best_result = run_design_with_budget(
            calib_scores, calib_weights, test_data_grouped, 
            max_samples_per_group, alpha_values, budget, args, baseline_mode=False
        )
        
        # Run baseline analysis (LR=1, no density filtering)
        print("Running BASELINE analysis (LR=1, no density filtering)...")
        baseline_group_results, baseline_optimized_results, baseline_best_result = run_design_with_budget(
            baseline_calib_scores, baseline_calib_weights, baseline_test_data_grouped, 
            max_samples_per_group, alpha_values, budget, args, baseline_mode=True
        )
        
        # Store results for this budget
        all_results[f"budget_{budget}"] = {
            "budget": budget,
            "best_result": best_result,
            "all_alpha_results": optimized_results,
            "baseline_best_result": baseline_best_result,
            "baseline_all_alpha_results": baseline_optimized_results
        }
        
        # Save detailed group results for this budget
        with open(results_dir / f"group_results_budget_{budget}.json", "w") as f:
            json.dump(group_results, f, indent=4)
        
        # Save baseline group results for this budget  
        with open(results_dir / f"baseline_group_results_budget_{budget}.json", "w") as f:
            json.dump(baseline_group_results, f, indent=4)
    
    # Save aggregate results
    with open(results_dir / "budget_analysis_results.json", "w") as f:
        json.dump(all_results, f, indent=4)
    
    # Print summary
    print("\n" + "="*80)
    print("BUDGET-OPTIMIZED SEQUENTIAL TESTING SUMMARY")
    print("="*80)
    
    for budget in args.budgets:
        best_result = all_results[f"budget_{budget}"]["best_result"]
        baseline_best_result = all_results[f"budget_{budget}"]["baseline_best_result"]
        
        print(f"\nBudget: {budget}")
        print(f"  ORIGINAL (with density filtering & LR weighting):")
        print(f"    Best nominal alpha: {best_result['best_nominal_alpha']}")
        print(f"    Best true alpha: {best_result['true_alpha']:.3f} (on all {best_result['total_groups']} groups)")
        print(f"    True error rate: {best_result['true_error_rate']:.3f} (on all {best_result['total_groups']} groups)")
        print(f"    Fraction with positive sample: {best_result['fraction_with_positive_sample']:.3f}")
        print(f"    Groups actually tested: {best_result['selected_groups_count']}")
        print(f"    Groups removed (budget): {best_result['removed_groups_count']}")
        print(f"    Total points used: {best_result['total_points_tested']}")
        print(f"  BASELINE (LR=1, no density filtering):")
        print(f"    Best nominal alpha: {baseline_best_result['best_nominal_alpha']}")
        print(f"    Best true alpha: {baseline_best_result['true_alpha']:.3f} (on all {baseline_best_result['total_groups']} groups)")
        print(f"    True error rate: {baseline_best_result['true_error_rate']:.3f} (on all {baseline_best_result['total_groups']} groups)")
        print(f"    Fraction with positive sample: {baseline_best_result['fraction_with_positive_sample']:.3f}")
        print(f"    Groups actually tested: {baseline_best_result['selected_groups_count']}")
        print(f"    Groups removed (budget): {baseline_best_result['removed_groups_count']}")
        print(f"    Total points used: {baseline_best_result['total_points_tested']}")
        
        # Calculate improvement
        improvement = baseline_best_result['true_alpha'] - best_result['true_alpha']
        print(f"  IMPROVEMENT: {improvement:.3f} lower true alpha with density filtering & LR weighting")
    
    print(f"\nResults saved to {results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Budget-Constrained Design Analysis")
    
    # Budget parameters
    parser.add_argument("--budgets", nargs="+", type=int, default=[100, 150, 200], 
                      help="List of budget values to test")
    parser.add_argument("--max_samples_per_group_values", nargs="+", type=int, default=[5, 10], 
                      help="List of max_samples_per_group values to test (e.g., --max_samples_per_group_values 5 10)")
    parser.add_argument("--alpha_start", type=float, default=0.1, help="Starting alpha value")
    parser.add_argument("--alpha_end", type=float, default=0.5, help="Ending alpha value")
    
    # Config file option
    parser.add_argument("--config", type=str, help="Path to configuration JSON file")
    
    # Data paths
    parser.add_argument("--val_data_path", help="Path to validation data CSV")
    parser.add_argument("--test_data_path", help="Path to test data CSV")
    parser.add_argument("--output_dir", default="results_budget", help="Directory to save results")
    
    # Model configuration
    parser.add_argument("--feature_model_checkpoint", type=str)
    
    # Testing parameters (following design_main)
    parser.add_argument("--kde_type", type=str, choices=["negative_only", "all"], default="negative_only")
    parser.add_argument("--kde_bandwidths", nargs="+", type=float, default=[1])
    parser.add_argument("--density_threshold", type=float)
    parser.add_argument("--property_threshold", type=float, default=0.9)
    parser.add_argument("--similarity_threshold", type=float)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--statistic", choices=["min", "max", "mean", "rank_sum", "likelihood_ratio"], default="min")
    
    # Baseline parameters
    
    # Test mode
    parser.add_argument("--test_mode", action="store_true")
    parser.add_argument("--test_samples", type=int, default=1000)
    
    # Other
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Load config from file if provided
    if args.config:
        config = load_config_from_file(args.config)
        for key, value in config.items():
            if key == 'kde_bandwidths' and isinstance(value, list):
                setattr(args, key, value)
            elif key == 'max_samples_per_group_values' and isinstance(value, list):
                setattr(args, key, value)
            elif key == 'budgets' and isinstance(value, list):
                setattr(args, key, value)
            elif hasattr(args, key) and value is not None:
                setattr(args, key, value)
    
    # Check required arguments
    required_args = ['val_data_path', 'test_data_path']
    for arg in required_args:
        if not hasattr(args, arg) or getattr(args, arg) is None:
            raise ValueError(f"Required argument '{arg}' is missing. Please provide via command line or config file.")
    
    main(args)

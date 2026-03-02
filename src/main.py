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
from sequential_testing import sequential_test, sample_and_prepare_test_data


def load_config_from_file(config_path):
    """Load configuration from a JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def main(args):
    # Set random seed
    set_seed(args.seed)
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.output_dir) / f"run_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    with open(results_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    # Log configuration
    print(f"Configuration:")
    print()
    
    # Load data
    val_data = pd.read_csv(args.val_data_path)
    test_data = pd.read_csv(args.test_data_path)
    
    # Apply test mode if enabled
    if args.test_mode:
        print(f"Test mode enabled: sampling {args.test_samples} rows from datasets")
        val_data = sample_for_testing(val_data, n_samples=args.test_samples, seed=args.seed)
        test_data = sample_for_testing(test_data, n_samples=args.test_samples, seed=args.seed)
        print(f"Sampled data sizes - Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Check if features and scores are already available
    features_available = ('features' in val_data.columns and 'features' in test_data.columns and
                         'score' in val_data.columns and 'score' in test_data.columns)
    
    # Convert string representations of features to lists if they exist
    if features_available:
        try:
            print("Features and scores found in data. Skipping feature extraction...")
            # Try to convert string representations to lists if necessary
            if isinstance(val_data['features'].iloc[0], str):
                val_data['features'] = val_data['features'].apply(json.loads)
            if isinstance(test_data['features'].iloc[0], str):
                test_data['features'] = test_data['features'].apply(json.loads)
        except (ValueError, TypeError):
            print("Error parsing existing features. Will recompute features.")
            features_available = False
    
    if not features_available and not args.skip_extraction:
        # Load feature extraction model
        print("Loading feature extraction model...")
        feature_model = load_model(args.feature_model_checkpoint)
        
        if args.debug:
            pass

        # Use feature model for scoring (binary)
        print("Using feature model for scoring...")
        scoring_model = feature_model

        # Extract features and compute scores
        print("Extracting validation features and scores...")
        val_pairs = val_data[["smiles_low", "smiles_high"]].values
        
        val_features, _ = extract_features(
            feature_model,
            val_pairs,
            batch_size=getattr(args, 'batch_size', 32)
        )
        val_data["features"] = val_features.tolist()
                
        _, val_scores = extract_features(
            scoring_model,
            val_pairs,
            batch_size=getattr(args, 'batch_size', 32)
        )
        val_data["score"] = val_scores
        
        print("Extracting test features and scores...")
        test_pairs = test_data[["SMILES_ori", "SMILES_opt"]].values
        
        test_features, _ = extract_features(
            feature_model,
            test_pairs,
            batch_size=getattr(args, 'batch_size', 32)
        )
        test_data["features"] = test_features.tolist()
        
        _, test_scores = extract_features(
            scoring_model,
            test_pairs,
            batch_size=getattr(args, 'batch_size', 32)
        )
        test_data["score"] = test_scores
    elif args.skip_extraction and not features_available:
        raise ValueError("Feature extraction was set to be skipped but features were not found in the data.")
    
    if args.debug:
        pass
    
    # Prepare features for KDE computation
    val_features = np.array(val_data["features"].tolist())
    test_features = np.array(test_data["features"].tolist())
    
    # Compute KDEs
    print("Computing KDEs...")
    # Always compute both KDEs
    val_neg_features = val_features[val_data["label"] == 0]
    val_neg_kde, _ = compute_kde_from_features(val_neg_features, bandwidths=args.kde_bandwidths)
    val_kde, _ = compute_kde_from_features(val_features, bandwidths=args.kde_bandwidths)
    
    # Compute initial densities for validation data (always use full val_kde)
    # Default: normal density computation (density_ratio = 0.0)
    val_data["density"] = val_kde.score_samples(val_features)
    test_data["density"] = val_kde.score_samples(test_features)
    
    # Group data by SMILES_ori for analysis by input molecule
    test_data_grouped = test_data.groupby("SMILES_ori")
    print(f"Number of unique input molecules: {len(test_data_grouped)}")
    
    # Get density threshold if specified - always use val_kde (full validation data)
    density_threshold = None
    if args.density_threshold is not None:
        density_threshold = np.percentile(val_data["density"], args.density_threshold)

    # Filter test data based on thresholds
    if args.max_samples_per_group is None:
        raise ValueError("max_samples_per_group must be specified")
    
    sampling_method = "random"
    print(f"Filtering and sampling test data using {sampling_method} method...")
    filtered_test_data = sample_and_prepare_test_data(
        test_data, 
        max_k=args.max_samples_per_group,
        density_threshold=density_threshold,
        similarity_threshold=args.similarity_threshold,
        seed=args.seed
    )
    if density_threshold is not None:
        val_data = val_data[val_data["density"] > density_threshold]
    if args.debug:
        pass
    
    # Computing likelihood ratios on filtered data
    filtered_features = np.array(filtered_test_data["features"].tolist())
    test_kde, _ = compute_kde_from_features(filtered_features, bandwidths=[1])
    
    # Compute likelihood ratios - use KDE based on args.kde_type (default: normal, density_ratio=0.0)
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
    
    # The sampling is now done in sample_and_prepare_test_data function
    sampled_test_data = filtered_test_data
    
    if args.debug:
        pass
    
    # Group sampled data by SMILES_ori
    sampled_test_data_grouped = sampled_test_data.groupby("SMILES_ori")
    print(f"Number of unique input molecules after sampling: {len(sampled_test_data_grouped)}")
    
    # Limit number of groups if max_groups is specified
    if hasattr(args, 'max_groups') and args.max_groups is not None:
        group_list = list(sampled_test_data_grouped)
        if len(group_list) > args.max_groups:
            print(f"Limiting to {args.max_groups} groups (out of {len(group_list)} total groups)")
            # Take first max_groups groups (maintains order from groupby)
            limited_groups = group_list[:args.max_groups]
            # Recreate grouped object from limited groups
            limited_smiles = [smiles for smiles, _ in limited_groups]
            sampled_test_data = sampled_test_data[sampled_test_data["SMILES_ori"].isin(limited_smiles)]
            sampled_test_data_grouped = sampled_test_data.groupby("SMILES_ori")
            print(f"Number of unique input molecules after limiting: {len(sampled_test_data_grouped)}")
    
    # Prepare calibration data
    if density_threshold is not None:
        val_data = val_data[val_data["density"] > density_threshold]
    # Calibration: negative only
    calib_data = val_data[val_data["label"] == 0]
    calib_scores = calib_data["score"].values * -1
    calib_weights = calib_data["likelihood_ratio"].values
    # Parse alpha values
    alphas = [float(a) for a in args.alpha.split(",")]
    
    # Analyze each SMILES_ori group separately
    all_results = {}
    group_metrics = {alpha: {
        "total_groups": len(sampled_test_data_grouped),
        "no_rejection_groups": 0,
        "rejection_groups": 0,
        "rejection_with_high_property": 0,
        "rejection_without_high_property": 0
    } for alpha in alphas}
    
    if args.debug:
        pass
    
    
    # if args.calibration_test_mode:
    #     ## Get top 100 from sampled_test_data_grouped
    #     sampled_test_data_grouped = sampled_test_data_grouped.head(100)


    print("Running sequential testing for each input molecule...")
    for smiles_ori, group in tqdm(sampled_test_data_grouped):
        # Prepare test data for this group
        group_test_scores = group["score"].values
        group_test_weights = group["likelihood_ratio"].values
        group_property_value_list = group["PROPERTY_opt"].values
        # Perform sequential testing for this group
        group_rejection_points, group_p_values = sequential_test(
            calib_scores=calib_scores,
            calib_weights=calib_weights,
            test_scores_stream=group_test_scores*-1,
            test_weights_stream=group_test_weights,
            alphas=alphas,
            max_k=args.max_samples_per_group,
            M=args.permutations,
            statistic=args.statistic
        )
        
        # Compute metrics for this group
        group_results = {
            "SMILES_ori": smiles_ori,
            "p_values": group_p_values,
        }
        
        # Update metrics for each alpha
        for alpha in alphas:
            rejection_point = group_rejection_points[alpha]
            group_results[f"rejection_point_alpha_{alpha}"] = rejection_point
            
            if rejection_point is None:
                # No rejection - p-value never less than alpha
                group_metrics[alpha]["no_rejection_groups"] += 1
                
                # Check if this group has any samples with high property
            else:
                # Rejection occurred
                group_metrics[alpha]["rejection_groups"] += 1
                ## Check if this group has any samples with high property before rejection
                property_value_rejected = group_property_value_list[:rejection_point+1]
                if any(val > args.property_threshold for val in property_value_rejected):
                    group_metrics[alpha]["rejection_with_high_property"] += 1
                else:
                    group_metrics[alpha]["rejection_without_high_property"] += 1
                
            group_metrics[alpha]["error_rate"] = (group_metrics[alpha]["rejection_without_high_property"]) / group_metrics[alpha]["total_groups"] * 100 if group_metrics[alpha]["total_groups"] > 0 else 0
        all_results[smiles_ori] = group_results
    
    if args.debug:
        pass
    
    # Save group-level results
    with open(results_dir / "group_results.json", "w") as f:
        json.dump(all_results, f, indent=4)
    
    # Compute and save aggregate metrics
    aggregate_results = {}
    for alpha in alphas:
        metrics = group_metrics[alpha]
        aggregate_results[f"alpha_{alpha}"] = {
            "total_groups": metrics["total_groups"],
            "no_rejection_groups": metrics["no_rejection_groups"],
            "rejection_groups": metrics["rejection_groups"],
            "rejection_with_high_property": metrics["rejection_with_high_property"],
            "rejection_without_high_property": metrics["rejection_without_high_property"],
            "error_rate": metrics["error_rate"]
        }
    
    # Add parameters to results
    aggregate_results["parameters"] = {
        "alphas": alphas,
        "max_samples_per_group": args.max_samples_per_group,
        "permutations": args.permutations,
        "statistic": args.statistic,
        "density_threshold": density_threshold,
        "similarity_threshold": args.similarity_threshold,
        "property_threshold": args.property_threshold,
        "kde_type": args.kde_type,
        "kde_bandwidth": val_kde.bandwidth,
        "test_mode": args.test_mode,
        "calibration_test_mode": args.calibration_test_mode,
        "test_samples": args.test_samples if args.test_mode else None,
        "seed": args.seed,
        "feature_model_checkpoint": args.feature_model_checkpoint,
        "val_data_path": args.val_data_path,
        "test_data_path": args.test_data_path,
    }
    
    # Save aggregate results
    with open(results_dir / "aggregate_results.json", "w") as f:
        json.dump(aggregate_results, f, indent=4)
    
    # Save filtered test data to CSV
    sampled_test_data.to_csv(results_dir / "filtered_test_data.csv", index=False)
    
    # Collect and save rejected molecules information grouped by SMILES_ori
    rejected_by_smiles = {}
    for smiles_ori, group in sampled_test_data_grouped:
        rejected_by_smiles[smiles_ori] = {}
        
        for alpha in alphas:
            rejection_point = all_results[smiles_ori][f"rejection_point_alpha_{alpha}"]
            if rejection_point is not None:
                group_data = group.reset_index(drop=True)
                
                # Record all molecules up to the rejection point
                rejected_molecules = []
                for idx in range(rejection_point + 1):
                    row = group_data.iloc[idx]
                    rejected_molecules.append({
                        "SMILES_opt": row["SMILES_opt"],
                        "property_value": float(row["PROPERTY_opt"]),
                        "p_value": float(all_results[smiles_ori]["p_values"][idx]),
                        "rejection_index": int(idx),
                        "likelihood_ratio": float(row["likelihood_ratio"]),
                        "score": float(row["score"])
                    })
                
                rejected_by_smiles[smiles_ori][f"alpha_{alpha}"] = {
                    "rejection_point": int(rejection_point),
                    "rejected_molecules": rejected_molecules
                }
            else:
                # Save an empty list when no rejection occurs
                rejected_by_smiles[smiles_ori][f"alpha_{alpha}"] = {
                    "rejection_point": None,
                    "rejected_molecules": []
                }
    
    # Function to convert numpy types to Python native types for JSON serialization
    def convert_to_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return convert_to_json_serializable(obj.tolist())
        else:
            return obj
    
    # Convert numpy types to native Python types
    rejected_by_smiles = convert_to_json_serializable(rejected_by_smiles)
    
    # Save rejected molecules data grouped by SMILES_ori
    with open(results_dir / "rejected_by_smiles.json", "w") as f:
        json.dump(rejected_by_smiles, f, indent=4)
    
    # Print summary
    print("\nSequential testing summary:")
    for alpha in alphas:
        metrics = group_metrics[alpha]
        print(f"\nAlpha = {alpha}:")
        print(f"  Total groups: {metrics['total_groups']}")
        print(f"  Groups with no rejection: {metrics['no_rejection_groups']} ({metrics['no_rejection_groups']/metrics['total_groups']*100:.2f}%)")
        print(f"  Groups with rejection: {metrics['rejection_groups']} ({metrics['rejection_groups']/metrics['total_groups']*100:.2f}%)")
        if metrics['rejection_groups'] > 0:
            print(f"  Groups with rejection that have high property: {metrics['rejection_with_high_property']} ({metrics['rejection_with_high_property']/metrics['rejection_groups']*100:.2f}% of rejection groups)")
            print(f"  Groups with rejection that have low property: {metrics['rejection_without_high_property']} ({metrics['rejection_without_high_property']/metrics['rejection_groups']*100:.2f}% of rejection groups)")
        else:
            print(f"  Groups with rejection that have high property: {metrics['rejection_with_high_property']} (N/A - no rejection groups)")
            print(f"  Groups with rejection that have low property: {metrics['rejection_without_high_property']} (N/A - no rejection groups)")
        print(f"  Error rate: {metrics['error_rate']:.2f}%")
    print(f"\nResults saved to {results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequential Testing for Molecule Generation")
    
    # Config file option
    parser.add_argument("--config", type=str, help="Path to configuration JSON file")
    
    # Data paths
    parser.add_argument("--val_data_path", help="Path to validation data CSV")
    parser.add_argument("--test_data_path", help="Path to test data CSV")
    parser.add_argument("--output_dir", default="results", help="Directory to save results")
    
    # Feature extraction model configuration
    parser.add_argument("--feature_model_checkpoint", type=str,
                      help="Path to feature extraction model checkpoint")
    
    # KDE parameters
    parser.add_argument("--kde_type", type=str, choices=["negative_only", "all"],
                      default="negative_only", help="Type of data to use for KDE")
    parser.add_argument("--kde_bandwidths", nargs="+", type=float, default=[0.01, 0.05, 0.1, 0.5, 1],
                      help="Bandwidths to try for KDE")
    parser.add_argument("--density_threshold", type=float, help="Percentile threshold for density filtering")
    
    # Filtering thresholds
    parser.add_argument("--property_threshold", type=float, default=0.9,
                      help="Threshold for property value")
    parser.add_argument("--similarity_threshold", type=float, default=None,
                      help="Threshold for similarity")
    
    # Testing parameters
    parser.add_argument("--alpha", type=str, default="0.05", help="Comma-separated list of significance levels")
    parser.add_argument("--max_samples_per_group", type=int,
                      help="Maximum number of samples per unique SMILES_ori")
    parser.add_argument("--permutations", type=int, default=1000, help="Number of permutations for testing")
    parser.add_argument("--statistic", choices=["min", "max", "mean", "rank_sum", "likelihood_ratio"], default="min",
                      help="Statistic for combining p-values")
    
    # Test mode parameters
    parser.add_argument("--test_mode", action="store_true", help="Enable test mode with smaller sample size")
    parser.add_argument("--calibration_test_mode", action="store_true", help="Enable calibration test mode")
    parser.add_argument("--test_samples", type=int, default=1000, 
                      help="Number of samples to use in test mode (default: 1000)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    # Skip feature extraction
    parser.add_argument("--skip_extraction", action="store_true", help="Skip feature extraction if features are already in the data")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_groups", type=int, default=None, 
                      help="Maximum number of groups (SMILES_ori) to process (for budget experiments)")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size for feature extraction and prediction (default: 32)")
    
    args = parser.parse_args()
    
    # Load config from file if provided
    if args.config:
        config = load_config_from_file(args.config)
        # Override default arguments with values from config file
        for key, value in config.items():
            if key == 'kde_bandwidths' and isinstance(value, list):
                setattr(args, key, value)
            elif hasattr(args, key) and value is not None:
                setattr(args, key, value)
        
        # Check required arguments
        required_args = ['val_data_path', 'test_data_path', 'max_samples_per_group']
        if not args.skip_extraction:
            required_args.append('feature_model_checkpoint')
        
        for arg in required_args:
            if not hasattr(args, arg) or getattr(args, arg) is None:
                raise ValueError(f"Required argument '{arg}' is missing in the configuration file")
    else:
        # Check required command-line arguments
        required_args = ['val_data_path', 'test_data_path', 'max_samples_per_group']
        if not args.skip_extraction:
            required_args.append('feature_model_checkpoint')
        
        for arg in required_args:
            if not hasattr(args, arg) or getattr(args, arg) is None:
                raise ValueError(f"Required argument '{arg}' must be provided via command line or config file")
                
    main(args) 
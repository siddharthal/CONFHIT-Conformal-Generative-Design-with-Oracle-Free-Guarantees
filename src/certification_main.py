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
from certification import run_certification_analysis, print_certification_summary


def load_config_from_file(config_path):
    """Load configuration from a JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def main(args):
    # Set random seed
    set_seed(args.seed)
    
    # Create results directory with descriptive naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract meaningful name components from config/data paths
    if hasattr(args, 'experiment_name') and args.experiment_name:
        exp_name = args.experiment_name
    else:
        # Infer experiment name from data paths
        if 'targetdiff' in str(args.test_data_path).lower():
            exp_name = 'targetdiff'
        elif 'molcraft' in str(args.test_data_path).lower():
            exp_name = 'molcraft'
        elif 'decompdiff' in str(args.test_data_path).lower():
            exp_name = 'decompdiff'
        elif 'drd2' in str(args.test_data_path).lower():
            if 'hgraph' in str(args.test_data_path).lower():
                exp_name = 'drd2_hgraph'
            elif 'selfedit' in str(args.test_data_path).lower():
                exp_name = 'drd2_selfedit'
            else:
                exp_name = 'drd2'
        else:
            exp_name = 'certification'
    
    # Add N and seed to the name
    n_samples = getattr(args, 'max_samples_per_group', 'all')
    seed = getattr(args, 'seed', 42)
    
    dir_name = f"certification_{exp_name}_N{n_samples}_seed{seed}_{timestamp}"
    results_dir = Path(args.output_dir) / dir_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    with open(results_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    # Log configuration
    print(f"Certification configuration")
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
                
            # Apply score negation to pre-computed scores
            print("Applying score negation to pre-computed scores...")
            val_data["score"] = -1 * val_data["score"]
            test_data["score"] = -1 * test_data["score"]
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
        scoring_model = feature_model
            
        # Extract features for validation data
        print("Extracting features for validation data...")
        
        # Handle different column naming conventions for SMILES pairs
        if "smiles_low" in val_data.columns and "smiles_high" in val_data.columns:
            val_pairs = val_data[["smiles_low", "smiles_high"]].values
        elif "SMILES_ori" in val_data.columns and "SMILES_opt" in val_data.columns:
            val_pairs = val_data[["SMILES_ori", "SMILES_opt"]].values
        else:
            raise ValueError("Unknown SMILES column names in validation data")
            
        val_features, val_scores = extract_features(
            feature_model,
            val_pairs
        )
        val_data["features"] = val_features.tolist()
        val_data["score"] = -1 * val_scores
        
        # Extract features for test data
        print("Extracting features for test data...")
        
        # Handle different column naming conventions for test SMILES pairs
        if "smiles_low" in test_data.columns and "smiles_high" in test_data.columns:
            test_pairs = test_data[["smiles_low", "smiles_high"]].values
        elif "SMILES_ori" in test_data.columns and "SMILES_opt" in test_data.columns:
            test_pairs = test_data[["SMILES_ori", "SMILES_opt"]].values
        else:
            raise ValueError("Unknown SMILES column names in test data")
            
        test_features, test_scores = extract_features(
            feature_model,
            test_pairs
        )
        test_data["features"] = test_features.tolist()
        test_data["score"] = -1 * test_scores
    
    elif args.skip_extraction and not features_available:
        raise ValueError("Cannot skip extraction when features are not available in the data")
    
    # Compute KDE and likelihood ratios
    print("Computing KDE and likelihood ratios...")
    
    # Get features for KDE computation
    val_features_array = np.array(val_data["features"].tolist())
    test_features_array = np.array(test_data["features"].tolist())
    
    # Compute KDE
    # Handle different column naming conventions  
    if "PROPERTY_ori" in val_data.columns:
        property_col = "PROPERTY_ori"
    elif "vina_score" in val_data.columns:
        property_col = "vina_score"
    elif "drd2_score" in val_data.columns:
        property_col = "drd2_score"
    else:
        raise ValueError("Unknown property column name in validation data")
    
    # Filter features based on kde_type and property_threshold for validation KDE
    if args.kde_type == "negative_only":
        # Use only features from samples below threshold
        val_neg_features = val_features_array[val_data[property_col].values < args.property_threshold]
        val_kde, _ = compute_kde_from_features(val_neg_features, bandwidths=args.kde_bandwidths)
    else:
        # Use all features
        val_kde, _ = compute_kde_from_features(val_features_array, bandwidths=args.kde_bandwidths)
    
    # Compute test KDE using test features
    test_kde, bandwidth = compute_kde_from_features(test_features_array, bandwidths=[args.kde_bandwidths[0]])
    
    print(f"Selected KDE bandwidth: {bandwidth}")
    
    # Compute KDE-based likelihood ratios
    print("Computing KDE-based likelihood ratios...")
    
    # Use KDE for likelihood ratio computation based on kde_type
    if args.kde_type == "negative_only":
        kde_for_ratio = val_kde
    else:
        kde_for_ratio = val_kde
        
    val_data["likelihood_ratio"] = get_likelihood_ratio(
        val_data["features"].values.tolist(), kde_for_ratio, test_kde
    )
    
    test_data["likelihood_ratio"] = get_likelihood_ratio(
        test_data["features"].values.tolist(), kde_for_ratio, test_kde
    )
    
    # Compute density values for filtering if density_threshold is provided
    if args.density_threshold is not None:
        print("Computing density values for filtering...")
        val_densities = val_kde.score_samples(val_features_array)
        test_densities = val_kde.score_samples(test_features_array)
        
        val_data["density"] = val_densities
        test_data["density"] = test_densities
        
        # Calculate the actual threshold value (on log densities)
        density_threshold_value = np.percentile(val_densities, args.density_threshold)
        print(f"Density threshold ({args.density_threshold}th percentile): {density_threshold_value}")
    else:
        density_threshold_value = None
    
    # Parse alpha values
    alphas = [float(a.strip()) for a in args.alpha.split(",")]
    
    # Run certification analysis once for all alphas (p-values computed only once)
    print(f"\nRunning certification analysis for all alphas: {alphas}")
    
    all_results = run_certification_analysis(
        val_data=val_data,
        test_data=test_data,
        property_threshold=args.property_threshold,
        alphas=alphas,  # Pass all alphas at once
        M=args.permutations,
        statistic=args.statistic,
        max_samples_per_group=args.max_samples_per_group,
        density_threshold=density_threshold_value,
        similarity_threshold=args.similarity_threshold,
        seed=args.seed
    )
    
    # Print summary for each alpha
    for alpha in alphas:
        print_certification_summary(all_results[alpha])
    
    # Save results
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
    
    # Convert and save results
    all_results = convert_to_json_serializable(all_results)
    
    with open(results_dir / "certification_results.json", "w") as f:
        json.dump(all_results, f, indent=4)
    
    # Create summary table
    summary_data = []
    for alpha, results in all_results.items():
        summary = results['summary']
        summary_data.append({
            'alpha': alpha,
            'total_groups': summary['total_groups'],
            'rejected_groups': summary['rejected_groups'],
            'fraction_rejected': summary['fraction_rejected'],
            'rejected_with_high_property': summary['rejected_with_high_property'],
            'rejected_with_low_property': summary['rejected_with_low_property'],
            'error_rate': summary['error_rate']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(results_dir / "certification_summary.csv", index=False)
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL CERTIFICATION SUMMARY")
    print("="*80)
    for alpha in alphas:
        summary = all_results[alpha]['summary']
        print(f"\nAlpha = {alpha}:")
        print(f"  Total groups: {summary['total_groups']}")
        print(f"  Rejected groups: {summary['rejected_groups']} ({summary['fraction_rejected']*100:.2f}%)")
        print(f"  Error rate: {summary['error_rate']*100:.2f}%")
    
    print(f"\nResults saved to {results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Certification Testing for Molecule Generation")
    
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
    parser.add_argument("--statistic", choices=["min", "max", "mean", "sum"], default="max",
                      help="Statistic for combining p-values")
    
    # Test mode parameters
    parser.add_argument("--test_mode", action="store_true", help="Enable test mode with smaller sample size")
    parser.add_argument("--test_samples", type=int, default=1000, 
                      help="Number of samples to use in test mode (default: 1000)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    # Skip feature extraction
    parser.add_argument("--skip_extraction", action="store_true", help="Skip feature extraction if features are already in the data")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name for output directory")
    
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
        required_args = ['val_data_path', 'test_data_path']
        if not args.skip_extraction:
            required_args.append('feature_model_checkpoint')
        
        for arg in required_args:
            if not hasattr(args, arg) or getattr(args, arg) is None:
                raise ValueError(f"Required argument '{arg}' is missing in the configuration file")
    else:
        # Check required command-line arguments
        required_args = ['val_data_path', 'test_data_path']
        if not args.skip_extraction:
            required_args.append('feature_model_checkpoint')
        
        for arg in required_args:
            if not hasattr(args, arg) or getattr(args, arg) is None:
                raise ValueError(f"Required argument '{arg}' must be provided via command line or config file")
                
    main(args)

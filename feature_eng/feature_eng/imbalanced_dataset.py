"""
Imbalanced Dataset Creator for DDI Prediction
Creates a 1:52 imbalanced test set from existing external validation datasets.
Supports pickle format for input and output.
"""

import argparse
import numpy as np
import pandas as pd
import pickle
import os
from typing import Dict, Tuple, List, Any
from pathlib import Path


class ImbalancedDatasetCreator:
    """
    Creates a single imbalanced test set (1:52 ratio) from existing external datasets.
    Uses stratified subsampling for positives and negatives.
    """
    
    def __init__(self, external_datasets: Dict[str, List[Tuple]]):
        """
        Args:
            external_datasets: Dictionary mapping dataset names to lists of tuples.
                Each tuple represents a drug pair with the last element as target (0/1).
        """
        self.datasets = external_datasets
        
        # Convert tuple lists to DataFrames and separate by label
        self.all_positives = self._extract_by_label(label=1)
        self.all_negatives = self._extract_by_label(label=0)
        
        print(f"Available positives: {len(self.all_positives)}")
        print(f"Available negatives: {len(self.all_negatives)}")
        
    def _extract_by_label(self, label: int) -> pd.DataFrame:
        """
        Extract all samples with specific label from all datasets.
        
        Args:
            label: Target label (0 for negative, 1 for positive)
            
        Returns:
            DataFrame containing all samples with the specified label
        """
        all_data = []
        
        for dataset_name, tuple_list in self.datasets.items():
            if not tuple_list:
                print(f"  Warning: {dataset_name} is empty, skipping...")
                continue
            
            # Filter tuples by label (last element)
            filtered_tuples = [t for t in tuple_list if t[-1] == label]
            
            if not filtered_tuples:
                print(f"  Warning: {dataset_name} has no samples with label={label}, skipping...")
                continue
            
            # Convert to DataFrame
            # Determine number of columns from first tuple
            n_cols = len(filtered_tuples[0])
            col_names = [f'feature_{i}' for i in range(n_cols - 1)] + ['target']
            
            df = pd.DataFrame(filtered_tuples, columns=col_names)
            
            # Add source dataset column for tracking
            df['source_dataset'] = dataset_name
            
            all_data.append(df)
        
        if not all_data:
            raise ValueError(f"No samples found with label={label}")
        
        concatenated = pd.concat(all_data, ignore_index=True)
        
        # Print distribution by source dataset
        print(f"\n{'Positives' if label == 1 else 'Negatives'} distribution by dataset:")
        print(concatenated['source_dataset'].value_counts())
        
        return concatenated
    
    def create_1_52_dataset(
        self, 
        n_positives: int = 400,
        seed: int = 42
    ) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Create dataset with 1:52 ratio (DrugBank's natural distribution).
        
        Args:
            n_positives: Number of positive samples to include (default: 400)
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (positives_list, negatives_list) where each list contains tuples
        """
        print("\n" + "="*70)
        print(f"Creating 1:52 imbalanced dataset (DrugBank natural ratio)")
        print("="*70)
        
        np.random.seed(seed)
        
        # Calculate number of negatives for 1:52 ratio
        n_negatives = n_positives * 52
        
        # Check if we have enough negatives
        available_negatives = len(self.all_negatives)
        
        if available_negatives < n_negatives:
            print(f"\n⚠️  WARNING: Not enough negatives!")
            print(f"  Required: {n_negatives}")
            print(f"  Available: {available_negatives}")
            print(f"  Will use oversampling with replacement for negatives")
            use_oversampling = True
        else:
            print(f"\n✓ Sufficient negatives available ({available_negatives} >= {n_negatives})")
            use_oversampling = False
        
        # 1. Stratified subsample positives
        positives_df = self._stratified_subsample(
            data=self.all_positives,
            n=n_positives, 
            seed=seed,
            label_name="positives"
        )
        
        # 2. Subsample or oversample negatives
        if use_oversampling:
            negatives_df = self._oversample_negatives(n=n_negatives, seed=seed)
        else:
            negatives_df = self._stratified_subsample(
                data=self.all_negatives,
                n=n_negatives,
                seed=seed,
                label_name="negatives"
            )
        
        # Verification
        print(f"\nFinal dataset composition:")
        print(f"  Positives: {len(positives_df)}")
        print(f"  Negatives: {len(negatives_df)}")
        print(f"  Total: {len(positives_df) + len(negatives_df)}")
        print(f"  Ratio: 1:{len(negatives_df)/len(positives_df):.1f}")
        print(f"  Positive %: {len(positives_df)/(len(positives_df)+len(negatives_df))*100:.2f}%")
        
        # Convert DataFrames back to list of tuples (excluding source_dataset column)
        feature_cols = [col for col in positives_df.columns if col not in ['source_dataset']]
        
        positives_tuples = [tuple(row) for row in positives_df[feature_cols].values]
        negatives_tuples = [tuple(row) for row in negatives_df[feature_cols].values]
        
        return positives_tuples, negatives_tuples
    
    def _stratified_subsample(
        self, 
        data: pd.DataFrame,
        n: int, 
        seed: int,
        label_name: str = "samples"
    ) -> pd.DataFrame:
        """
        Subsample n samples while maintaining proportional representation across datasets.
        
        If dataset X contains 30% of all samples, it will contribute ~30% of the
        sampled samples.
        
        Args:
            data: DataFrame with 'source_dataset' column
            n: Number of samples to draw
            seed: Random seed for reproducibility
            label_name: Name for logging (e.g., "positives", "negatives")
            
        Returns:
            DataFrame with n stratified samples
        """
        np.random.seed(seed)
        
        # Count samples per dataset
        dataset_counts = data['source_dataset'].value_counts()
        total_samples = len(data)
        
        print(f"\nStratified sampling of {n} {label_name}:")
        print(f"  Total available: {total_samples}")
        
        if n > total_samples:
            raise ValueError(
                f"Cannot sample {n} {label_name} from {total_samples} available "
                f"without replacement. Use oversampling instead."
            )
        
        # Calculate samples per dataset (proportional)
        samples_per_dataset = {}
        for dataset_name, count in dataset_counts.items():
            proportion = count / total_samples
            n_samples = int(n * proportion)
            
            # Ensure at least 1 sample if dataset has data
            # and not more than available
            n_samples = max(1, min(n_samples, count))
            
            samples_per_dataset[dataset_name] = n_samples
            print(f"    {dataset_name}: {n_samples} "
                  f"(from {count} available, {proportion*100:.1f}%)")
        
        # Final adjustment to reach EXACTLY n samples
        total_allocated = sum(samples_per_dataset.values())
        
        if total_allocated != n:
            # Find largest dataset for adjustments
            largest_dataset = dataset_counts.index[0]
            adjustment = n - total_allocated
            
            # Verify adjustment is possible
            current_allocation = samples_per_dataset[largest_dataset]
            max_from_largest = dataset_counts[largest_dataset]
            
            if adjustment > 0:
                # Add samples
                can_add = max_from_largest - current_allocation
                adjustment = min(adjustment, can_add)
            else:
                # Remove samples
                adjustment = max(adjustment, -(current_allocation - 1))
            
            samples_per_dataset[largest_dataset] += adjustment
            print(f"\n  Adjustment: {adjustment:+d} samples to {largest_dataset}")
        
        # Sample from each dataset
        sampled_data = []
        for dataset_name, n_samples in samples_per_dataset.items():
            dataset_data = data[data['source_dataset'] == dataset_name]
            
            sample = dataset_data.sample(
                n=min(n_samples, len(dataset_data)), 
                random_state=seed,
                replace=False  # No replacement when we have enough samples
            )
            sampled_data.append(sample)
        
        result = pd.concat(sampled_data, ignore_index=True)
        
        # Final verification
        actual_distribution = result['source_dataset'].value_counts()
        print(f"\n  Final distribution:")
        for dataset_name, count in actual_distribution.items():
            print(f"    {dataset_name}: {count}")
        print(f"  Total sampled: {len(result)}")
        
        return result
    
    def _oversample_negatives(self, n: int, seed: int) -> pd.DataFrame:
        """
        Oversample negatives WITH REPLACEMENT to reach n samples.
        
        This function is only called when we don't have enough negatives.
        With replacement means the same negative drug pair can appear
        multiple times in the final dataset.
        
        Args:
            n: Number of negative samples needed
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with n negative samples (may contain duplicates)
        """
        np.random.seed(seed)
        
        available = len(self.all_negatives)
        
        print(f"\n⚠️  Oversampling {n} negatives (not enough available):")
        print(f"  Available unique negatives: {available}")
        
        # Oversample WITH replacement
        # (allows sampling more than available)
        sampled = self.all_negatives.sample(
            n=n,
            replace=True,  # KEY: allows duplicates
            random_state=seed
        )
        
        # Statistics on duplicates
        # For duplicates detection, we need to compare all feature columns
        feature_cols = [col for col in sampled.columns if col not in ['source_dataset']]
        unique_samples = sampled[feature_cols].drop_duplicates()
        n_duplicates = n - len(unique_samples)
        duplication_rate = n_duplicates / n * 100
        
        print(f"  Sampled: {n}")
        print(f"  Unique samples: {len(unique_samples)}")
        print(f"  Duplicates: {n_duplicates} ({duplication_rate:.1f}%)")
        
        # Distribution of duplications (how many times each sample appears)
        if n_duplicates > 0:
            # Count occurrences of each unique sample
            duplication_counts = sampled.groupby(feature_cols, dropna=False).size()
            max_duplications = duplication_counts.max()
            
            print(f"  Max times a sample appears: {max_duplications}")
            print(f"  Distribution of duplication:")
            for times in range(1, min(6, max_duplications + 1)):
                n_samples_duplicated_times = (duplication_counts == times).sum()
                print(f"    {times}x: {n_samples_duplicated_times} samples")
            
            if max_duplications > 5:
                print(f"    >5x: {(duplication_counts > 5).sum()} samples")
        
        return sampled.reset_index(drop=True)
    
    def get_dataset_statistics(
        self, 
        positives: List[Tuple], 
        negatives: List[Tuple]
    ) -> Dict:
        """
        Calculate detailed statistics about the created dataset.
        
        Args:
            positives: List of positive tuples
            negatives: List of negative tuples
            
        Returns:
            Dictionary with dataset statistics
        """
        # Convert back to DataFrames for analysis
        n_cols = len(positives[0]) if positives else len(negatives[0])
        col_names = [f'feature_{i}' for i in range(n_cols)]
        
        pos_df = pd.DataFrame(positives, columns=col_names)
        neg_df = pd.DataFrame(negatives, columns=col_names)
        
        stats = {
            'n_positives': len(positives),
            'n_negatives': len(negatives),
            'total': len(positives) + len(negatives),
            'ratio': len(negatives) / len(positives) if len(positives) > 0 else 0,
            'positive_percentage': len(positives) / (len(positives) + len(negatives)) * 100,
            'negatives_unique': len(neg_df.drop_duplicates()),
            'negatives_duplication_rate': (1 - len(neg_df.drop_duplicates()) / len(neg_df)) * 100 if len(neg_df) > 0 else 0,
        }
        
        return stats


def load_pickle_datasets(dataset_paths: List[str]) -> Dict[str, List[Tuple]]:
    """
    Load external datasets from pickle files.
    
    Each pickle file should contain a list of tuples where:
    - Each tuple represents a drug pair
    - Last element of each tuple is the target (0=negative, 1=positive)
    
    Args:
        dataset_paths: List of paths to pickle files
        
    Returns:
        Dictionary mapping dataset name to list of tuples
    """
    datasets = {}
    
    print("Loading external datasets from pickle files...")
    for path in dataset_paths:
        # Get dataset name from filename
        dataset_name = Path(path).stem
        
        # Load pickle
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            # Verify it's a list of tuples
            if not isinstance(data, list):
                raise ValueError(f"Expected list, got {type(data)}")
            
            if not data:
                print(f"  ⚠️  {dataset_name}: empty dataset, skipping...")
                continue
            
            if not isinstance(data[0], tuple):
                raise ValueError(f"Expected list of tuples, got list of {type(data[0])}")
            
            # Verify last element is 0 or 1
            targets = set(t[-1] for t in data)
            if not targets.issubset({0, 1}):
                raise ValueError(
                    f"Target values (last element) should be 0 or 1, "
                    f"found: {targets}"
                )
            
            datasets[dataset_name] = data
            
            n_pos = sum(1 for t in data if t[-1] == 1)
            n_neg = sum(1 for t in data if t[-1] == 0)
            
            print(f"  ✓ {dataset_name}: {len(data)} pairs "
                  f"({n_pos} positive, {n_neg} negative)")
            
        except Exception as e:
            print(f"  ✗ Error loading {path}: {e}")
            continue
    
    if not datasets:
        raise ValueError("No valid datasets loaded!")
    
    print(f"\nTotal datasets loaded: {len(datasets)}")
    
    return datasets


def save_pickle_dataset(
    positives: List[Tuple],
    negatives: List[Tuple],
    output_dir: str = ".",
    prefix: str = "imbalanced_1_52"
) -> Tuple[str, str, str]:
    """
    Save the created dataset to pickle files.
    
    Args:
        positives: List of positive tuples
        negatives: List of negative tuples
        output_dir: Directory where to save files
        prefix: Prefix for output filenames
        
    Returns:
        Tuple of (positives_file, negatives_file, combined_file) paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save positives
    pos_file = os.path.join(output_dir, f"{prefix}_positives.pkl")
    with open(pos_file, 'wb') as f:
        pickle.dump(positives, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save negatives
    neg_file = os.path.join(output_dir, f"{prefix}_negatives.pkl")
    with open(neg_file, 'wb') as f:
        pickle.dump(negatives, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"\nDataset saved:")
    print(f"  Positives: {pos_file}")
    print(f"  Negatives: {neg_file}")
    
    # Save combined dataset (shuffled)
    combined = positives + negatives
    
    # Shuffle
    np.random.seed(42)
    indices = np.random.permutation(len(combined))
    combined_shuffled = [combined[i] for i in indices]
    
    combined_file = os.path.join(output_dir, f"{prefix}_combined.pkl")
    with open(combined_file, 'wb') as f:
        pickle.dump(combined_shuffled, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"  Combined (shuffled): {combined_file}")
    
    # Print file sizes
    for file_path in [pos_file, neg_file, combined_file]:
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"    {Path(file_path).name}: {size_mb:.2f} MB")
    
    return pos_file, neg_file, combined_file


def print_summary_statistics(stats: Dict):
    """
    Print formatted summary statistics.
    
    Args:
        stats: Dictionary with dataset statistics
    """
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    print(f"Total samples: {stats['total']:,}")
    print(f"Positives: {stats['n_positives']:,} ({stats['positive_percentage']:.2f}%)")
    print(f"Negatives: {stats['n_negatives']:,} ({100-stats['positive_percentage']:.2f}%)")
    print(f"Ratio: 1:{stats['ratio']:.1f}")
    
    print(f"\nNegative samples:")
    print(f"  Unique: {stats['negatives_unique']:,}")
    
    if stats['negatives_duplication_rate'] > 0:
        print(f"  ⚠️  Duplicates: {stats['n_negatives'] - stats['negatives_unique']:,}")
        print(f"  ⚠️  Duplication rate: {stats['negatives_duplication_rate']:.1f}%")
    else:
        print(f"  ✓ No duplicates (all unique)")
    
    # Calculate expected predictions
    n_models = 5  # Phi-3.5, Qwen2.5, Deepseek, Gemma2, GPT-4o
    expected_predictions = stats['total'] * n_models
    print(f"\n📊 Expected predictions with 5 models: {expected_predictions:,}")
    
    # Estimate computational cost
    seconds_per_prediction = 2  # Conservative estimate
    total_seconds = expected_predictions * seconds_per_prediction
    hours_sequential = total_seconds / 3600
    hours_with_batching = hours_sequential / 32  # Batch size 32
    
    print(f"⏱️  Estimated time:")
    print(f"  Sequential: {hours_sequential:.1f} hours")
    print(f"  With batching (32): {hours_with_batching:.1f} hours")


def verify_pickle_format(file_path: str):
    """
    Verify that a pickle file has the expected format.
    
    Args:
        file_path: Path to pickle file to verify
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"\n✓ Successfully loaded: {file_path}")
        print(f"  Type: {type(data)}")
        print(f"  Length: {len(data)}")
        
        if isinstance(data, list) and len(data) > 0:
            print(f"  First element type: {type(data[0])}")
            print(f"  First element length: {len(data[0])}")
            print(f"  First element: {data[0]}")
            
            if isinstance(data[0], tuple):
                print(f"  Target values: {set(t[-1] for t in data)}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error loading {file_path}: {e}")
        return False


def main():
    """
    Main function to create imbalanced dataset from command line.
    """
    parser = argparse.ArgumentParser(
        description="Create 1:52 imbalanced test set for DDI prediction from pickle files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python create_imbalanced_dataset.py \\
    --datasets data/credible_meds.pkl data/hep.pkl data/hiv.pkl \\
    --output-dir ./imbalanced_datasets \\
    --n-positives 400 \\
    --seed 42

  # Using wildcard
  python create_imbalanced_dataset.py \\
    --datasets data/*.pkl \\
    --output-dir output/

  # Verify pickle format
  python create_imbalanced_dataset.py \\
    --verify data/example.pkl

Expected pickle format:
  - Each pickle file contains a list of tuples
  - Each tuple represents a drug pair with features
  - Last element of each tuple must be target: 0 (negative) or 1 (positive)
  - Example: [(drug1_id, drug2_id, smiles1, smiles2, ..., 0), (..., 1), ...]
        """
    )
    
    parser.add_argument(
        '--datasets',
        nargs='+',
        help='List of paths to external dataset pickle files'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./imbalanced_datasets',
        help='Directory where to save output pickle files (default: ./imbalanced_datasets)'
    )
    
    parser.add_argument(
        '--n-positives',
        type=int,
        default=400,
        help='Number of positive samples to include (default: 400, ratio will be 1:52)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='imbalanced_1_52',
        help='Prefix for output filenames (default: imbalanced_1_52)'
    )
    
    parser.add_argument(
        '--verify',
        type=str,
        help='Verify format of a pickle file without creating dataset'
    )
    
    args = parser.parse_args()
    
    # Verify mode
    if args.verify:
        return 0 if verify_pickle_format(args.verify) else 1
    
    # Check required arguments
    if not args.datasets:
        parser.error("--datasets is required (unless using --verify)")
    
    # Print configuration
    print("="*70)
    print("IMBALANCED DATASET CREATOR FOR DDI PREDICTION")
    print("="*70)
    print(f"Configuration:")
    print(f"  Input datasets: {len(args.datasets)}")
    for path in args.datasets:
        print(f"    - {path}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Number of positives: {args.n_positives}")
    print(f"  Target ratio: 1:52 ({args.n_positives * 52} negatives)")
    print(f"  Random seed: {args.seed}")
    print(f"  Output prefix: {args.prefix}")
    print("="*70)
    
    try:
        # 1. Load datasets
        external_datasets = load_pickle_datasets(args.datasets)
        
        # 2. Create imbalanced dataset creator
        creator = ImbalancedDatasetCreator(external_datasets)
        
        # 3. Create 1:52 dataset
        positives, negatives = creator.create_1_52_dataset(
            n_positives=args.n_positives,
            seed=args.seed
        )
        
        # 4. Save dataset as pickle
        files = save_pickle_dataset(
            positives, 
            negatives, 
            output_dir=args.output_dir,
            prefix=args.prefix
        )
        
        # 5. Calculate and print statistics
        stats = creator.get_dataset_statistics(positives, negatives)
        print_summary_statistics(stats)
        
        # Success message
        print("\n" + "="*70)
        print("✅ Dataset created successfully!")
        print("="*70)
        print(f"Output files:")
        for file_path in files:
            print(f"  📄 {file_path}")
        
        print(f"\nNext steps:")
        print(f"  1. Load the combined dataset:")
        print(f"     with open('{files[2]}', 'rb') as f:")
        print(f"         data = pickle.load(f)")
        print(f"  2. Evaluate your 5 fine-tuned models on this dataset")
        print(f"  3. Compare results with balanced baseline (1:1)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
import pickle
import argparse
import random
import numpy as np
from pathlib import Path
from collections import Counter

def load_pickle(file_path):
    """Load data from pickle file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle(data, file_path):
    """Save data to pickle file."""
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def flip_negative_labels(data, flip_percentage, seed=42):
    """
    Flip a percentage of negative samples (target=0) to positive (target=1).
    
    Args:
        data: List of dictionaries containing 'target' field
        flip_percentage: Percentage of negatives to flip (0-100)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (flipped_data, flip_report)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Deep copy to avoid modifying original
    flipped_data = [d.copy() for d in data]
    
    # Find indices of negative samples
    negative_indices = [i for i, d in enumerate(flipped_data) if d['target'] == 0]
    
    if len(negative_indices) == 0:
        print("Warning: No negative samples found in the dataset!")
        return flipped_data, {
            'total_samples': len(data),
            'negative_samples': 0,
            'flipped_count': 0,
            'flip_percentage': 0.0,
            'flipped_indices': []
        }
    
    # Calculate number of samples to flip
    n_to_flip = int(len(negative_indices) * (flip_percentage / 100.0))
    
    # Randomly select indices to flip
    indices_to_flip = random.sample(negative_indices, n_to_flip)
    
    # Flip the selected samples
    for idx in indices_to_flip:
        flipped_data[idx]['target'] = 1
        # Add metadata about the flip
        flipped_data[idx]['_flipped'] = True
        flipped_data[idx]['_original_target'] = 0
    
    # Create report
    flip_report = {
        'total_samples': len(data),
        'positive_samples_original': sum(1 for d in data if d['target'] == 1),
        'negative_samples_original': len(negative_indices),
        'flipped_count': n_to_flip,
        'flip_percentage_requested': flip_percentage,
        'flip_percentage_actual': (n_to_flip / len(negative_indices)) * 100 if len(negative_indices) > 0 else 0,
        'positive_samples_after': sum(1 for d in flipped_data if d['target'] == 1),
        'negative_samples_after': sum(1 for d in flipped_data if d['target'] == 0),
        'flipped_indices': sorted(indices_to_flip),
        'seed': seed
    }
    
    return flipped_data, flip_report

def print_report(flip_report, verbose=False):
    """Print a summary report of the flipping operation."""
    print("\n" + "=" * 70)
    print("LABEL FLIPPING REPORT")
    print("=" * 70)
    print(f"Total samples:                {flip_report['total_samples']}")
    print(f"Random seed:                  {flip_report['seed']}")
    print()
    print("BEFORE FLIPPING:")
    print(f"  Positive samples (target=1): {flip_report['positive_samples_original']}")
    print(f"  Negative samples (target=0): {flip_report['negative_samples_original']}")
    print()
    print("FLIPPING OPERATION:")
    print(f"  Requested flip percentage:   {flip_report['flip_percentage_requested']:.2f}%")
    print(f"  Actual flip percentage:      {flip_report['flip_percentage_actual']:.2f}%")
    print(f"  Samples flipped:             {flip_report['flipped_count']}")
    print()
    print("AFTER FLIPPING:")
    print(f"  Positive samples (target=1): {flip_report['positive_samples_after']}")
    print(f"  Negative samples (target=0): {flip_report['negative_samples_after']}")
    print()
    
    # Calculate balance
    total = flip_report['total_samples']
    pos_ratio = (flip_report['positive_samples_after'] / total) * 100
    neg_ratio = (flip_report['negative_samples_after'] / total) * 100
    print(f"CLASS BALANCE:")
    print(f"  Positive: {pos_ratio:.2f}%")
    print(f"  Negative: {neg_ratio:.2f}%")
    print("=" * 70)
    
    if verbose and flip_report['flipped_count'] > 0:
        print("\nFLIPPED SAMPLE INDICES:")
        print(f"First 10: {flip_report['flipped_indices'][:10]}")
        if len(flip_report['flipped_indices']) > 10:
            print(f"... and {len(flip_report['flipped_indices']) - 10} more")
        print()

def validate_data(data):
    """Validate input data structure."""
    if not isinstance(data, list):
        raise ValueError("Input data must be a list")
    
    if len(data) == 0:
        raise ValueError("Input data is empty")
    
    if not all(isinstance(d, dict) for d in data):
        raise ValueError("All elements must be dictionaries")
    
    if not all('target' in d for d in data):
        raise ValueError("All dictionaries must contain 'target' field")
    
    # Check target values
    targets = [d['target'] for d in data]
    unique_targets = set(targets)
    
    if not unique_targets.issubset({0, 1}):
        raise ValueError(f"Target values must be 0 or 1, found: {unique_targets}")
    
    return True

def create_output_filename(input_path, flip_percentage, suffix=''):
    """Create output filename based on input and flip percentage."""
    input_path = Path(input_path)
    stem = input_path.stem
    
    if suffix:
        output_name = f"{stem}_flip{flip_percentage}pct_{suffix}.pkl"
    else:
        output_name = f"{stem}_flip{flip_percentage}pct.pkl"
    
    return output_name

def save_flip_report(flip_report, output_path):
    """Save flip report as text file."""
    report_path = Path(output_path).parent / f"{Path(output_path).stem}_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("LABEL FLIPPING REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Total samples:                {flip_report['total_samples']}\n")
        f.write(f"Random seed:                  {flip_report['seed']}\n\n")
        
        f.write("BEFORE FLIPPING:\n")
        f.write(f"  Positive samples (target=1): {flip_report['positive_samples_original']}\n")
        f.write(f"  Negative samples (target=0): {flip_report['negative_samples_original']}\n\n")
        
        f.write("FLIPPING OPERATION:\n")
        f.write(f"  Requested flip percentage:   {flip_report['flip_percentage_requested']:.2f}%\n")
        f.write(f"  Actual flip percentage:      {flip_report['flip_percentage_actual']:.2f}%\n")
        f.write(f"  Samples flipped:             {flip_report['flipped_count']}\n\n")
        
        f.write("AFTER FLIPPING:\n")
        f.write(f"  Positive samples (target=1): {flip_report['positive_samples_after']}\n")
        f.write(f"  Negative samples (target=0): {flip_report['negative_samples_after']}\n\n")
        
        total = flip_report['total_samples']
        pos_ratio = (flip_report['positive_samples_after'] / total) * 100
        neg_ratio = (flip_report['negative_samples_after'] / total) * 100
        
        f.write("CLASS BALANCE:\n")
        f.write(f"  Positive: {pos_ratio:.2f}%\n")
        f.write(f"  Negative: {neg_ratio:.2f}%\n\n")
        
        f.write("=" * 70 + "\n\n")
        
        f.write("FLIPPED SAMPLE INDICES:\n")
        f.write(f"{flip_report['flipped_indices']}\n")
    
    print(f"Report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Flip negative labels to positive for label noise robustness testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Flip 1% of negatives
  python flip_labels.py data.pkl --flip-percentage 1.0
  
  # Flip 5% with custom output name
  python flip_labels.py data.pkl --flip-percentage 5.0 --output noisy_data_5pct.pkl
  
  # Flip 10% with custom seed and save report
  python flip_labels.py data.pkl --flip-percentage 10.0 --seed 123 --save-report
  
  # Generate multiple versions at once
  python flip_labels.py data.pkl --flip-percentage 1.0 5.0 10.0 --batch-mode
        """
    )
    
    parser.add_argument('input_file', type=str,
                       help='Path to input pickle file')
    parser.add_argument('--flip-percentage', type=float, nargs='+', required=True,
                       help='Percentage of negative samples to flip (0-100). '
                            'Multiple values can be provided for batch processing.')
    parser.add_argument('--output', type=str, default=None,
                       help='Output pickle file path (default: auto-generated)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: same as input file)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--save-report', action='store_true',
                       help='Save detailed report as text file')
    parser.add_argument('--verbose', action='store_true',
                       help='Print verbose output including sample indices')
    parser.add_argument('--batch-mode', action='store_true',
                       help='Process multiple flip percentages (use with multiple --flip-percentage values)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Perform flipping but do not save output (for testing)')
    
    args = parser.parse_args()
    
    # Validate flip percentages
    for pct in args.flip_percentage:
        if not (0 <= pct <= 100):
            parser.error(f"Flip percentage must be between 0 and 100, got {pct}")
    
    # Load input data
    print(f"Loading data from: {args.input_file}")
    try:
        data = load_pickle(args.input_file)
        print(f"Loaded {len(data)} samples")
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return 1
    
    # Validate data structure
    try:
        validate_data(data)
        print("Data validation passed ✓")
    except ValueError as e:
        print(f"Data validation failed: {e}")
        return 1
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(args.input_file).parent
    
    # Process single or multiple flip percentages
    flip_percentages = args.flip_percentage
    
    if len(flip_percentages) > 1 and not args.batch_mode:
        print("Warning: Multiple flip percentages provided but --batch-mode not set.")
        print("Processing only the first percentage. Use --batch-mode to process all.")
        flip_percentages = [flip_percentages[0]]
    
    for flip_pct in flip_percentages:
        print(f"\nProcessing flip percentage: {flip_pct}%")
        print("-" * 70)
        
        # Perform flipping
        flipped_data, flip_report = flip_negative_labels(data, flip_pct, seed=args.seed)
        
        # Print report
        print_report(flip_report, verbose=args.verbose)
        
        if args.dry_run:
            print("\nDRY RUN: Skipping file save")
            continue
        
        # Determine output filename
        if args.output and len(flip_percentages) == 1:
            output_path = Path(args.output)
        else:
            output_filename = create_output_filename(args.input_file, flip_pct)
            output_path = output_dir / output_filename
        
        # Save flipped data
        print(f"\nSaving flipped data to: {output_path}")
        save_pickle(flipped_data, output_path)
        print(f"Successfully saved {len(flipped_data)} samples ✓")
        
        # Save report if requested
        if args.save_report:
            save_flip_report(flip_report, output_path)
    
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    exit(main())
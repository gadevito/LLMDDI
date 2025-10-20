import pickle
import pandas as pd
import argparse
import sys
from pathlib import Path


def load_pickle_file(filepath):
    """
    Load a pickle file and return its content.
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        Loaded data from pickle file
    """
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}", file=sys.stderr)
        return None


def extract_misclassified_pairs(data, source_filename, only_positives=True):
    """
    Extract pairs where target=0 and new_target=1.
    
    Args:
        data: Dictionary or list of dictionaries with keys 'drug1', 'drug2', 'target', 'new_target'
        source_filename: Name of the source file for tracking
        
    Returns:
        List of dictionaries with misclassified pairs
    """
    misclassified = []
    
    # Handle single dictionary
    if isinstance(data, dict):
        if 'drug1' in data and 'drug2' in data and 'target' in data and 'new_target' in data:
            data = [data]
        else:
            # Maybe it's a dictionary of lists/arrays
            try:
                # Convert to list of dicts
                n_samples = len(data['drug1'])
                data = [
                    {
                        'drug1': data['drug1'][i],
                        'drug2': data['drug2'][i],
                        'target': data['target'][i],
                        'new_target': data['new_target'][i]
                    }
                    for i in range(n_samples)
                ]
            except (KeyError, TypeError) as e:
                print(f"Error processing data structure from {source_filename}: {e}", file=sys.stderr)
                return []
    
    # Process list of dictionaries
    for entry in data:
        try:
            # Check if target=0 and new_target=1
            target = entry.get('target', None)
            new_target = entry.get('new_target', None)
            
            if only_positives:
                if target == 0 and new_target == 1:
                    misclassified.append({
                        'drug1': entry.get('drug1', 'N/A'),
                        'drug2': entry.get('drug2', 'N/A'),
                        'target': target,
                        'new_target': new_target,
                        'source_file': source_filename
                    })
            elif not only_positives and target == 0:
                    misclassified.append({
                        'drug1': entry.get('drug1', 'N/A'),
                        'drug2': entry.get('drug2', 'N/A'),
                        'target': target,
                        'new_target': new_target,
                        'source_file': source_filename
                    })                    
        except Exception as e:
            print(f"Error processing entry in {source_filename}: {e}", file=sys.stderr)
            continue
    
    return misclassified


def main():
    """Main function to process pickle files and create CSV output."""
    
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description='Extract drug pairs with target=0 and new_target=1 from pickle files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python extract_misclassified.py file1.pkl file2.pkl
    python extract_misclassified.py *.pkl -o output.csv
    python extract_misclassified.py data/*.pkl --output-file results.csv
        """
    )
    
    parser.add_argument(
        'pickle_files',
        nargs='+',
        type=str,
        help='One or more pickle files to process'
    )
    
    parser.add_argument(
        '-o', '--output-file',
        type=str,
        default='misclassified_pairs.csv',
        help='Output CSV file name (default: misclassified_pairs.csv)'
    )

    parser.add_argument(
        '-i', '--inverse',
        action='store_true',
        help='Target 1 and new_target=0'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print verbose output'
    )
    
    args = parser.parse_args()
    
    # Collect all misclassified pairs
    all_misclassified = []
    
    print(f"Processing {len(args.pickle_files)} pickle file(s)...\n")
    
    for filepath in args.pickle_files:
        if args.verbose:
            print(f"Processing: {filepath}")
        
        # Check if file exists
        if not Path(filepath).exists():
            print(f"Warning: File not found: {filepath}", file=sys.stderr)
            continue
        
        # Load pickle file
        data = load_pickle_file(filepath)
        
        if data is None:
            continue
        
        # Extract misclassified pairs
        filename = Path(filepath).name
        if args.inverse:
            print("EXTRACTING ALL NEGATIVES===>")
            misclassified = extract_misclassified_pairs(data, filename, only_positives=False)
        else:
            misclassified = extract_misclassified_pairs(data, filename)
        
        if args.verbose:
            print(f"  Found {len(misclassified)} misclassified pair(s)")
        
        all_misclassified.extend(misclassified)
    
    # Create DataFrame and save to CSV
    if all_misclassified:
        df = pd.DataFrame(all_misclassified)
        df.to_csv(args.output_file, index=False)
        
        print(f"\n{'='*60}")
        print(f"Summary:")
        print(f"{'='*60}")
        print(f"Total misclassified pairs found: {len(all_misclassified)}")
        print(f"Output saved to: {args.output_file}")
        print(f"{'='*60}\n")
        
        # Print first few rows as preview
        print("Preview (first 5 rows):")
        print(df.head().to_string(index=False))
        
        # Print statistics by source file
        if len(args.pickle_files) > 1:
            print(f"\n{'='*60}")
            print("Breakdown by source file:")
            print(f"{'='*60}")
            file_counts = df['source_file'].value_counts()
            for file, count in file_counts.items():
                print(f"  {file}: {count} pair(s)")
    else:
        print("\nNo misclassified pairs found (target=0, new_target=1).")
        print("No output file created.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
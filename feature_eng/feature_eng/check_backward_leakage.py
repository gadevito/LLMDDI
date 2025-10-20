import pickle
from typing import Set, Tuple, List
from collections import defaultdict

def load_pickle_data(filepath: str) -> List[Tuple]:
    """Load data from pickle file."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def extract_drug_pairs(data: List[Tuple]) -> Set[Tuple[str, str]]:
    """Extract drug pairs (drug1, drug2) from data."""
    return {(item[0], item[1]) for item in data}

def get_reversed_pairs(pairs: Set[Tuple[str, str]]) -> Set[Tuple[str, str]]:
    """Get reversed version of all pairs."""
    return {(drug2, drug1) for drug1, drug2 in pairs}

def check_backward_leakage(train_file: str, val_file: str, verbose: bool = True):
    """
    Check for backward leakage between training and validation sets.
    
    Args:
        train_file: Path to training pickle file
        val_file: Path to validation pickle file
        verbose: Print detailed information
    
    Returns:
        dict: Statistics about leakage
    """
    
    print("="*70)
    print("BACKWARD LEAKAGE ANALYSIS")
    print("="*70)
    
    # Load data
    print(f"\n1. Loading data...")
    train_data = load_pickle_data(train_file)
    val_data = load_pickle_data(val_file)
    
    print(f"   - Training samples: {len(train_data)}")
    print(f"   - Validation samples: {len(val_data)}")
    
    # Extract pairs
    print(f"\n2. Extracting drug pairs...")
    train_pairs = extract_drug_pairs(train_data)
    val_pairs = extract_drug_pairs(val_data)
    
    print(f"   - Unique training pairs: {len(train_pairs)}")
    print(f"   - Unique validation pairs: {len(val_pairs)}")
    
    # Check for direct overlap (exact same pairs)
    print(f"\n3. Checking for direct overlap...")
    direct_overlap = train_pairs & val_pairs
    print(f"   - Direct overlapping pairs: {len(direct_overlap)}")
    
    if verbose and len(direct_overlap) > 0:
        print(f"   - WARNING: Found {len(direct_overlap)} pairs in BOTH train and validation!")
        if len(direct_overlap) <= 10:
            for pair in list(direct_overlap)[:10]:
                print(f"      • {pair}")
    
    # Check for backward leakage (reversed pairs)
    print(f"\n4. Checking for backward leakage (reversed pairs)...")
    val_reversed = get_reversed_pairs(val_pairs)
    backward_leakage = train_pairs & val_reversed
    
    print(f"   - Backward leaking pairs: {len(backward_leakage)}")
    
    if verbose and len(backward_leakage) > 0:
        print(f"   - WARNING: Found {len(backward_leakage)} pairs where:")
        print(f"     Training has (A, B) and Validation has (B, A)")
        if len(backward_leakage) <= 10:
            print(f"   - Examples (first 10):")
            for drug1, drug2 in list(backward_leakage)[:10]:
                print(f"      • Train: ({drug1}, {drug2}) <-> Val: ({drug2}, {drug1})")
        else:
            print(f"   - Examples (first 10 of {len(backward_leakage)}):")
            for drug1, drug2 in list(backward_leakage)[:10]:
                print(f"      • Train: ({drug1}, {drug2}) <-> Val: ({drug2}, {drug1})")
    
    # Check if train has any reverse pairs within itself
    print(f"\n5. Checking for bidirectional pairs within training set...")
    train_reversed = get_reversed_pairs(train_pairs)
    train_bidirectional = train_pairs & train_reversed
    print(f"   - Training pairs that have reverse: {len(train_bidirectional)}")
    
    if verbose and len(train_bidirectional) > 0:
        print(f"   - INFO: Training set contains {len(train_bidirectional)} pairs")
        print(f"     where both (A, B) and (B, A) exist")
        if len(train_bidirectional) <= 5:
            for drug1, drug2 in list(train_bidirectional)[:5]:
                print(f"      • ({drug1}, {drug2}) and ({drug2}, {drug1})")
    
    # Check if val has any reverse pairs within itself
    print(f"\n6. Checking for bidirectional pairs within validation set...")
    val_bidirectional = val_pairs & val_reversed
    print(f"   - Validation pairs that have reverse: {len(val_bidirectional)}")
    
    if verbose and len(val_bidirectional) > 0:
        print(f"   - INFO: Validation set contains {len(val_bidirectional)} pairs")
        print(f"     where both (A, B) and (B, A) exist")
        if len(val_bidirectional) <= 5:
            for drug1, drug2 in list(val_bidirectional)[:5]:
                print(f"      • ({drug1}, {drug2}) and ({drug2}, {drug1})")
    
    # Analyze by label (if available)
    print(f"\n7. Analyzing leakage by label...")
    
    # Group backward leaking pairs by their labels
    leaking_pairs_info = defaultdict(lambda: {'train_label': [], 'val_label': []})
    
    if len(backward_leakage) > 0:
        # Create lookup dictionaries
        train_dict = {(item[0], item[1]): item for item in train_data}
        val_dict = {(item[0], item[1]): item for item in val_data}
        
        for drug1, drug2 in backward_leakage:
            train_label = train_dict[(drug1, drug2)][2] if len(train_dict[(drug1, drug2)]) > 2 else 'N/A'
            val_label = val_dict[(drug2, drug1)][2] if len(val_dict[(drug2, drug1)]) > 2 else 'N/A'
            
            leaking_pairs_info[(drug1, drug2)] = {
                'train_label': train_label,
                'val_label': val_label
            }
        
        # Count label combinations
        label_combinations = defaultdict(int)
        for pair, info in leaking_pairs_info.items():
            label_combinations[(info['train_label'], info['val_label'])] += 1
        
        print(f"   - Label combinations in leaking pairs:")
        for (train_label, val_label), count in sorted(label_combinations.items()):
            print(f"      • Train label {train_label}, Val label {val_label}: {count} pairs")
    
    # Summary
    print(f"\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    total_issues = len(direct_overlap) + len(backward_leakage)
    
    if total_issues == 0:
        print("✅ NO LEAKAGE DETECTED!")
        print("   - No direct overlap between train and validation")
        print("   - No backward leakage (reversed pairs) detected")
    else:
        print("⚠️  LEAKAGE DETECTED!")
        if len(direct_overlap) > 0:
            print(f"   - Direct overlap: {len(direct_overlap)} pairs")
        if len(backward_leakage) > 0:
            print(f"   - Backward leakage: {len(backward_leakage)} pairs")
            pct_train = (len(backward_leakage) / len(train_pairs)) * 100
            pct_val = (len(backward_leakage) / len(val_pairs)) * 100
            print(f"   - Represents {pct_train:.2f}% of training pairs")
            print(f"   - Represents {pct_val:.2f}% of validation pairs")
    
    print("="*70)
    
    # Return statistics
    return {
        'train_total': len(train_data),
        'val_total': len(val_data),
        'train_unique_pairs': len(train_pairs),
        'val_unique_pairs': len(val_pairs),
        'direct_overlap': len(direct_overlap),
        'backward_leakage': len(backward_leakage),
        'train_bidirectional': len(train_bidirectional),
        'val_bidirectional': len(val_bidirectional),
        'has_leakage': total_issues > 0,
        'leaking_pairs': list(backward_leakage) if len(backward_leakage) <= 100 else list(backward_leakage)[:100]
    }


# Usage example
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python check_leakage.py <train_pickle> <val_pickle>")
        print("\nExample:")
        print("  python check_leakage.py train_data.pkl val_data.pkl")
        sys.exit(1)
    
    train_file = sys.argv[1]
    val_file = sys.argv[2]
    
    stats = check_backward_leakage(train_file, val_file, verbose=True)
    
    # Optionally save detailed report
    if stats['has_leakage']:
        print(f"\n💾 Saving detailed leakage report to 'leakage_report.txt'...")
        with open('leakage_report.txt', 'w') as f:
            f.write("BACKWARD LEAKAGE DETAILED REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Training samples: {stats['train_total']}\n")
            f.write(f"Validation samples: {stats['val_total']}\n")
            f.write(f"Backward leaking pairs: {stats['backward_leakage']}\n\n")
            f.write("Leaking pairs (up to 100):\n")
            for drug1, drug2 in stats['leaking_pairs']:
                f.write(f"  Train: ({drug1}, {drug2}) <-> Val: ({drug2}, {drug1})\n")
        print("✅ Report saved!")

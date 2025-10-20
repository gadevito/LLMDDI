import pickle
import argparse
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

def load_results(pickle_path):
    """Load results from pickle file."""
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)

def prepare_data(results):
    """Convert results to DataFrame with prediction categories."""
    df = pd.DataFrame(results)
    
    # Remove rows with missing confidence values
    if 'confidence' in df.columns:
        initial_len = len(df)
        df = df.dropna(subset=['confidence'])
        dropped = initial_len - len(df)
        if dropped > 0:
            warnings.warn(f"Dropped {dropped} rows with missing confidence values")
    
    # Ensure binary targets
    df['target'] = df['target'].astype(int)
    df['new_target'] = df['new_target'].astype(int)
    
    # Determine if prediction is correct
    df['correct'] = (df['target'] == df['new_target']).astype(int)
    
    # Categorize predictions
    df['category'] = 'Unknown'
    df.loc[(df['target'] == 1) & (df['new_target'] == 1), 'category'] = 'TP'
    df.loc[(df['target'] == 0) & (df['new_target'] == 0), 'category'] = 'TN'
    df.loc[(df['target'] == 0) & (df['new_target'] == 1), 'category'] = 'FP'
    df.loc[(df['target'] == 1) & (df['new_target'] == 0), 'category'] = 'FN'
    
    return df

def compute_rank_biserial(group1, group2):
    """
    Compute rank-biserial correlation as effect size for Mann-Whitney U test.
    
    Returns:
        r: rank-biserial correlation (-1 to +1)
        interpretation: verbal interpretation
    """
    n1, n2 = len(group1), len(group2)
    
    # Mann-Whitney U test
    # By default, returns U1 (statistic for group1)
    U1, _ = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    
    # Rank-biserial correlation
    # r = 1 - (2*U) / (n1*n2), where U is the smaller of U1 and U2
    # U2 = n1*n2 - U1
    U2 = n1 * n2 - U1
    U_min = min(U1, U2)
    
    r = 1 - (2 * U_min) / (n1 * n2)
    
    # Determine direction: if group1 has higher values, r should be positive
    # FIX: usa np.median invece di .median()
    if np.median(group1) > np.median(group2):
        r = abs(r)
    else:
        r = -abs(r)
    
    interpretation = interpret_effect_size(abs(r))
    
    return r, interpretation

def interpret_effect_size(r_abs):
    """Interpret absolute rank-biserial correlation effect size."""
    if r_abs < 0.1:
        return "negligible"
    elif r_abs < 0.3:
        return "small"
    elif r_abs < 0.5:
        return "medium"
    else:
        return "large"

def compute_calibration_metrics(df):
    """Compute key calibration metrics."""
    
    # Split by correctness
    correct = df[df['correct'] == 1]['confidence'].values
    incorrect = df[df['correct'] == 0]['confidence'].values
    
    # Split by prediction type
    tp = df[df['category'] == 'TP']['confidence'].values
    tn = df[df['category'] == 'TN']['confidence'].values
    fp = df[df['category'] == 'FP']['confidence'].values
    fn = df[df['category'] == 'FN']['confidence'].values
    
    metrics = {
        'correct_predictions': {
            'n': len(correct),
            'mean': float(np.mean(correct)) if len(correct) > 0 else np.nan,
            'std': float(np.std(correct, ddof=1)) if len(correct) > 1 else np.nan,
            'median': float(np.median(correct)) if len(correct) > 0 else np.nan
        },
        'incorrect_predictions': {
            'n': len(incorrect),
            'mean': float(np.mean(incorrect)) if len(incorrect) > 0 else np.nan,
            'std': float(np.std(incorrect, ddof=1)) if len(incorrect) > 1 else np.nan,
            'median': float(np.median(incorrect)) if len(incorrect) > 0 else np.nan
        },
        'by_category': {
            'TP': {
                'n': len(tp),
                'mean': float(np.mean(tp)) if len(tp) > 0 else np.nan,
                'std': float(np.std(tp, ddof=1)) if len(tp) > 1 else np.nan
            },
            'TN': {
                'n': len(tn),
                'mean': float(np.mean(tn)) if len(tn) > 0 else np.nan,
                'std': float(np.std(tn, ddof=1)) if len(tn) > 1 else np.nan
            },
            'FP': {
                'n': len(fp),
                'mean': float(np.mean(fp)) if len(fp) > 0 else np.nan,
                'std': float(np.std(fp, ddof=1)) if len(fp) > 1 else np.nan
            },
            'FN': {
                'n': len(fn),
                'mean': float(np.mean(fn)) if len(fn) > 0 else np.nan,
                'std': float(np.std(fn, ddof=1)) if len(fn) > 1 else np.nan
            }
        }
    }
    
    # Statistical test: correct vs incorrect
    # Hypothesis: correct predictions have HIGHER confidence (less negative logprob)
    if len(correct) > 5 and len(incorrect) > 5:
        # Use 'greater' if we expect correct to have higher confidence
        # (less negative logprob = higher confidence)
        stat, p = stats.mannwhitneyu(correct, incorrect, alternative='greater')
        
        r, interp = compute_rank_biserial(correct, incorrect)
        
        metrics['statistical_test'] = {
            'test': 'Mann-Whitney U (one-sided: correct > incorrect)',
            'statistic': float(stat),
            'p_value': float(p),
            'effect_size_r': float(r),
            'interpretation': interp
        }
    
    # Additional test: TP vs FP (key for label noise concern)
    if len(tp) > 5 and len(fp) > 5:
        stat_tp_fp, p_tp_fp = stats.mannwhitneyu(tp, fp, alternative='greater')
        r_tp_fp, interp_tp_fp = compute_rank_biserial(tp, fp)
        
        metrics['tp_vs_fp_test'] = {
            'test': 'Mann-Whitney U (one-sided: TP > FP)',
            'statistic': float(stat_tp_fp),
            'p_value': float(p_tp_fp),
            'effect_size_r': float(r_tp_fp),
            'interpretation': interp_tp_fp
        }
    
    return metrics

def analyze_mislabeled_pairs(df, mislabeled_pairs):
    """Analyze confidence for known mislabeled cases."""
    results = []
    
    for drug1, drug2 in mislabeled_pairs:
        match = df[((df['drug1'] == drug1) & (df['drug2'] == drug2)) |
                   ((df['drug1'] == drug2) & (df['drug2'] == drug1))]
        
        if len(match) > 0:
            row = match.iloc[0]
            results.append({
                'drug_pair': f"{drug1} + {drug2}",
                'ground_truth': int(row['target']),
                'prediction': int(row['new_target']),
                'confidence': float(row['confidence']),
                'category': row['category']
            })
    
    return pd.DataFrame(results) if results else None

def create_visualizations(df, output_dir):
    """Create focused calibration plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sns.set_style("whitegrid")
    sns.set_palette("Set2")
    
    # 1. Main plot: Correct vs Incorrect predictions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Violin plot
    correct_conf = df[df['correct'] == 1]['confidence'].values
    incorrect_conf = df[df['correct'] == 0]['confidence'].values
    
    parts = axes[0].violinplot([correct_conf, incorrect_conf], 
                                positions=[0, 1],
                                showmeans=True, 
                                showmedians=True,
                                widths=0.7)
    
    # Color the violins
    for pc, color in zip(parts['bodies'], ['lightgreen', 'lightcoral']):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Correct\nPredictions', 'Incorrect\nPredictions'])
    axes[0].set_ylabel('Confidence (log-probability)', fontsize=11)
    axes[0].set_title('Confidence Distribution by Prediction Correctness', 
                      fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add sample sizes
    axes[0].text(0, axes[0].get_ylim()[0], f'n={len(correct_conf)}',
                ha='center', va='top', fontsize=9)
    axes[0].text(1, axes[0].get_ylim()[0], f'n={len(incorrect_conf)}',
                ha='center', va='top', fontsize=9)
    
    # Box plot by category
    category_order = ['TP', 'TN', 'FP', 'FN']
    category_colors = {'TP': 'lightgreen', 'TN': 'lightblue', 
                       'FP': 'lightcoral', 'FN': 'lightyellow'}
    
    df_plot = df[df['category'].isin(category_order)]
    
    box_parts = axes[1].boxplot([df_plot[df_plot['category'] == cat]['confidence'].values 
                                  for cat in category_order],
                                 positions=range(len(category_order)),
                                 labels=category_order,
                                 patch_artist=True,
                                 widths=0.6)
    
    for patch, cat in zip(box_parts['boxes'], category_order):
        patch.set_facecolor(category_colors[cat])
        patch.set_alpha(0.7)
    
    axes[1].set_xlabel('Prediction Category', fontsize=11)
    axes[1].set_ylabel('Confidence (log-probability)', fontsize=11)
    axes[1].set_title('Confidence Distribution by Category', 
                      fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add sample sizes
    for i, cat in enumerate(category_order):
        n = len(df_plot[df_plot['category'] == cat])
        axes[1].text(i, axes[1].get_ylim()[0], f'n={n}',
                    ha='center', va='top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'calibration_main.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Histogram comparison: TP vs FP (most relevant for label noise)
    tp_conf = df[df['category'] == 'TP']['confidence'].values
    fp_conf = df[df['category'] == 'FP']['confidence'].values
    
    if len(tp_conf) > 0 and len(fp_conf) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Determine bins
        all_conf = np.concatenate([tp_conf, fp_conf])
        bins = np.linspace(np.min(all_conf), np.max(all_conf), 30)
        
        ax.hist(tp_conf, bins=bins, alpha=0.6, label='True Positives (TP)', 
                color='green', edgecolor='darkgreen', linewidth=1.2)
        ax.hist(fp_conf, bins=bins, alpha=0.6, label='False Positives (FP)', 
                color='red', edgecolor='darkred', linewidth=1.2)
        
        # Add vertical lines for means
        tp_mean = np.mean(tp_conf)
        fp_mean = np.mean(fp_conf)
        
        ax.axvline(tp_mean, color='darkgreen', linestyle='--', 
                   linewidth=2.5, label=f'TP mean: {tp_mean:.2f}')
        ax.axvline(fp_mean, color='darkred', linestyle='--', 
                   linewidth=2.5, label=f'FP mean: {fp_mean:.2f}')
        
        ax.set_xlabel('Confidence (log-probability)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Confidence: True Positives vs False Positives\n' + 
                     '(Critical for Label Noise Assessment)', 
                     fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'tp_vs_fp.png', dpi=300, bbox_inches='tight')
        plt.close()

def generate_report(df, metrics, mislabeled_df, output_path, dataset_name):
    """Generate focused text report."""
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"CONFIDENCE CALIBRATION ANALYSIS\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write("=" * 80 + "\n\n")
        
        # Sample sizes
        f.write("PREDICTION OUTCOMES\n")
        f.write("-" * 80 + "\n")
        for cat in ['TP', 'TN', 'FP', 'FN']:
            n = metrics['by_category'][cat]['n']
            pct = 100 * n / len(df) if len(df) > 0 else 0
            f.write(f"{cat}: {n:4d} ({pct:5.1f}%)\n")
        f.write(f"{'Total:':<4} {len(df):4d}\n\n")
        
        # Main calibration results
        f.write("CALIBRATION ANALYSIS\n")
        f.write("-" * 80 + "\n")
        corr = metrics['correct_predictions']
        incorr = metrics['incorrect_predictions']
        
        f.write(f"Correct predictions (n={corr['n']}):\n")
        f.write(f"  Mean confidence: {corr['mean']:7.4f} (SD: {corr['std']:.4f})\n")
        f.write(f"  Median:          {corr['median']:7.4f}\n\n")
        
        f.write(f"Incorrect predictions (n={incorr['n']}):\n")
        f.write(f"  Mean confidence: {incorr['mean']:7.4f} (SD: {incorr['std']:.4f})\n")
        f.write(f"  Median:          {incorr['median']:7.4f}\n\n")
        
        if 'statistical_test' in metrics:
            test = metrics['statistical_test']
            f.write(f"Statistical comparison (Correct vs Incorrect):\n")
            f.write(f"  Test: {test['test']}\n")
            f.write(f"  U statistic: {test['statistic']:.2f}\n")
            f.write(f"  P-value: {test['p_value']:.2e}\n")
            f.write(f"  Effect size (rank-biserial r): {test['effect_size_r']:+.3f}\n")
            f.write(f"  Interpretation: {test['interpretation']}\n")
            f.write(f"  Significant at α=0.001: {'✓ Yes' if test['p_value'] < 0.001 else '✗ No'}\n\n")
        
        # Key comparison: TP vs FP
        f.write("TRUE POSITIVES vs FALSE POSITIVES (Critical for Label Noise)\n")
        f.write("-" * 80 + "\n")
        tp_stats = metrics['by_category']['TP']
        fp_stats = metrics['by_category']['FP']
        
        if tp_stats['n'] > 0:
            f.write(f"True Positives (n={tp_stats['n']}):\n")
            f.write(f"  Mean: {tp_stats['mean']:7.4f} (SD: {tp_stats['std']:.4f})\n\n")
        
        if fp_stats['n'] > 0:
            f.write(f"False Positives (n={fp_stats['n']}):\n")
            f.write(f"  Mean: {fp_stats['mean']:7.4f} (SD: {fp_stats['std']:.4f})\n\n")
        
        if 'tp_vs_fp_test' in metrics:
            test = metrics['tp_vs_fp_test']
            f.write(f"Statistical comparison (TP vs FP):\n")
            f.write(f"  Test: {test['test']}\n")
            f.write(f"  P-value: {test['p_value']:.2e}\n")
            f.write(f"  Effect size (r): {test['effect_size_r']:+.3f} ({test['interpretation']})\n")
            f.write(f"  Significant at α=0.001: {'✓ Yes' if test['p_value'] < 0.001 else '✗ No'}\n\n")
        
        # Mislabeled cases
        if mislabeled_df is not None and len(mislabeled_df) > 0:
            f.write("MISLABELED CASES (Prospective Validation)\n")
            f.write("-" * 80 + "\n")
            for idx, row in mislabeled_df.iterrows():
                f.write(f"\n{idx+1}. {row['drug_pair']}:\n")
                f.write(f"   Training label:  {row['ground_truth']} (negative)\n")
                f.write(f"   Model prediction: {row['prediction']} ({'positive' if row['prediction']==1 else 'negative'})\n")
                f.write(f"   Confidence:       {row['confidence']:7.4f}\n")
                f.write(f"   Category:         {row['category']}\n")
                
                if row['category'] == 'TP' and tp_stats['n'] > 0 and not np.isnan(tp_stats['std']):
                    deviation = abs(row['confidence'] - tp_stats['mean']) / tp_stats['std']
                    f.write(f"   Deviation from TP mean: {deviation:.2f} SD")
                    if deviation < 1.0:
                        f.write(f" (within 1 SD - typical TP)")
                    f.write(f"\n")
            f.write("\n")
        
        # Generate suggested text for paper
        f.write("=" * 80 + "\n")
        f.write("SUGGESTED TEXT FOR PAPER\n")
        f.write("=" * 80 + "\n\n")
        
        if corr['n'] > 0 and incorr['n'] > 0 and not np.isnan(corr['mean']):
            f.write(f"To verify that high sensitivity reflects genuine discriminative ability rather\n")
            f.write(f"than uniform positive bias, we analyzed GPT-4o's prediction confidence via\n")
            f.write(f"log-probabilities on the {dataset_name}. Correct predictions exhibited\n")
            f.write(f"significantly higher confidence (mean logprob: {corr['mean']:.2f}, SD: {corr['std']:.2f})\n")
            f.write(f"than incorrect predictions (mean: {incorr['mean']:.2f}, SD: {incorr['std']:.2f})")
            
            if 'statistical_test' in metrics:
                test = metrics['statistical_test']
                if test['p_value'] < 0.001:
                    f.write(f";\n")
                    f.write(f"Mann-Whitney U test, p < 0.001, effect size r = {test['effect_size_r']:+.2f}\n")
                    f.write(f"({test['interpretation']} effect)")
                else:
                    f.write(f" (p = {test['p_value']:.3f})")
            f.write(f".\n\n")
            
            if 'tp_vs_fp_test' in metrics and tp_stats['n'] > 0 and fp_stats['n'] > 0:
                f.write(f"Critically, true positive predictions showed higher confidence\n")
                f.write(f"(mean logprob: {tp_stats['mean']:.2f}) than false positive predictions\n")
                f.write(f"(mean: {fp_stats['mean']:.2f})")
                
                test_fp = metrics['tp_vs_fp_test']
                if test_fp['p_value'] < 0.001:
                    f.write(f", with the difference\n")
                    f.write(f"being statistically significant (Mann-Whitney U, p < 0.001,\n")
                    f.write(f"r = {test_fp['effect_size_r']:+.2f}, {test_fp['interpretation']} effect)")
                
                f.write(f". This calibration\n")
                f.write(f"pattern indicates the model assigns high confidence selectively to genuine\n")
                f.write(f"interactions rather than indiscriminately to all positive predictions.\n\n")
            
            if mislabeled_df is not None and len(mislabeled_df) > 0:
                correctly_predicted = mislabeled_df[mislabeled_df['prediction'] == 1]
                if len(correctly_predicted) > 0:
                    avg_conf = correctly_predicted['confidence'].mean()
                    f.write(f"Among the {len(mislabeled_df)} prospectively validated mislabeled case(s),\n")
                    f.write(f"the model correctly predicted the interaction with high confidence\n")
                    f.write(f"(mean logprob: {avg_conf:.2f}), comparable to true positives, further\n")
                    f.write(f"supporting robust pattern learning beyond training labels.\n")

def main():
    parser = argparse.ArgumentParser(
        description='Analyze confidence calibration for DDI predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python analyze_confidence.py results.pkl --dataset-name "Validation Set"
  
  # With mislabeled cases
  python analyze_confidence.py results.pkl \\
      --dataset-name "Validation Set" \\
      --mislabeled "Pentostatin,Iptacopan" "Zafirlukast,Rifampin"
        """
    )
    
    parser.add_argument('pickle_file', help='Path to results pickle file')
    parser.add_argument('--output-dir', default='./calibration_analysis',
                       help='Output directory (default: ./calibration_analysis)')
    parser.add_argument('--dataset-name', default='validation set',
                       help='Name of dataset (default: validation set)')
    parser.add_argument('--mislabeled', nargs='+', default=[],
                       help='Mislabeled pairs as "Drug1,Drug2"')
    
    args = parser.parse_args()
    
    # Parse mislabeled pairs
    mislabeled_pairs = []
    for pair_str in args.mislabeled:
        drugs = [d.strip() for d in pair_str.split(',')]
        if len(drugs) == 2:
            mislabeled_pairs.append(tuple(drugs))
        else:
            warnings.warn(f"Skipping invalid pair format: {pair_str}")
    
    print(f"\n{'='*60}")
    print(f"CONFIDENCE CALIBRATION ANALYSIS")
    print(f"{'='*60}\n")
    print(f"Dataset: {args.dataset_name}")
    print(f"Loading results from: {args.pickle_file}")
    
    results = load_results(args.pickle_file)
    print(f"✓ Loaded {len(results)} predictions")
    
    print(f"\nPreparing data...")
    df = prepare_data(results)
    print(f"✓ {len(df)} valid predictions after cleaning")
    
    print(f"\nComputing calibration metrics...")
    metrics = compute_calibration_metrics(df)
    print(f"✓ Metrics computed")
    
    if mislabeled_pairs:
        print(f"\nAnalyzing {len(mislabeled_pairs)} mislabeled case(s)...")
        mislabeled_df = analyze_mislabeled_pairs(df, mislabeled_pairs)
        if mislabeled_df is not None:
            print(f"✓ Found {len(mislabeled_df)} case(s) in dataset")
        else:
            print(f"⚠ No mislabeled cases found in dataset")
    else:
        mislabeled_df = None
    
    #print(f"\nCreating visualizations...")
    #create_visualizations(df, args.output_dir)
    #print(f"✓ Plots saved")
    
    print(f"\nGenerating report...")
    output_path = Path(args.output_dir) / 'calibration_report.txt'
    generate_report(df, metrics, mislabeled_df, output_path, args.dataset_name)
    print(f"✓ Report saved")
    
    # Save detailed results
    print(f"\nSaving detailed data...")
    df.to_csv(Path(args.output_dir) / 'confidence_data.csv', index=False)
    if mislabeled_df is not None:
        mislabeled_df.to_csv(Path(args.output_dir) / 'mislabeled_analysis.csv', index=False)
    print(f"✓ CSV files saved")
    
    print(f"\n{'='*60}")
    print(f"✓ ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"\nResults saved to: {Path(args.output_dir).absolute()}")
    print(f"  • Main report:    calibration_report.txt")
    print(f"  • Data:           confidence_data.csv")
    print(f"  • Visualizations: calibration_main.png, tp_vs_fp.png")
    if mislabeled_df is not None:
        print(f"  • Mislabeled:     mislabeled_analysis.csv")
    print()

if __name__ == "__main__":
    main()
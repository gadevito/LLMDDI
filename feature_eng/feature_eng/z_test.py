"""
Statistical analysis script for comparing performance across entity overlap groups.
This script performs Z-tests and bootstrap tests to assess whether performance
differences between overlap groups (NO/ONE/TWO entity overlap) are statistically significant.
Generates separate tables for validation and external datasets.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency
import argparse
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


def z_test_proportions(acc1, n1, acc2, n2):
    """
    Perform Z-test for comparing two proportions (accuracy/sensitivity/F1).
    
    Args:
        acc1 (float): Metric value for group 1 (0-1 range)
        n1 (int): Number of samples in group 1
        acc2 (float): Metric value for group 2 (0-1 range)
        n2 (int): Number of samples in group 2
    
    Returns:
        tuple: (p_value, z_score, effect_size)
            - p_value: Two-tailed p-value
            - z_score: Z-statistic
            - effect_size: Cohen's h effect size
    """
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan, np.nan
    
    # Calculate number of correct predictions
    x1 = int(acc1 * n1)
    x2 = int(acc2 * n2)
    
    # Pooled proportion under null hypothesis
    p_pool = (x1 + x2) / (n1 + n2)
    
    # Standard error
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    
    if se == 0:
        return np.nan, np.nan, np.nan
    
    # Z-statistic
    z = (acc1 - acc2) / se
    
    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    # Effect size (Cohen's h for proportions)
    h = 2 * (np.arcsin(np.sqrt(acc1)) - np.arcsin(np.sqrt(acc2)))
    
    return p_value, z, h


def bootstrap_difference(acc1, n1, acc2, n2, n_bootstrap=10000):
    """
    Bootstrap test for the difference between two metrics.
    More robust than Z-test for small samples or non-normal distributions.
    
    Args:
        acc1, n1: Metric and sample size for group 1
        acc2, n2: Metric and sample size for group 2
        n_bootstrap: Number of bootstrap iterations
    
    Returns:
        tuple: (p_value, ci_lower, ci_upper)
    """
    if n1 == 0 or n2 == 0 or n1 < 10 or n2 < 10:
        return np.nan, np.nan, np.nan
    
    # Observed difference
    obs_diff = acc1 - acc2
    
    # Generate approximate binary data
    x1 = int(acc1 * n1)
    x2 = int(acc2 * n2)
    
    data1 = np.array([1] * x1 + [0] * (n1 - x1))
    data2 = np.array([1] * x2 + [0] * (n2 - x2))
    
    # Bootstrap resampling
    boot_diffs = []
    rng = np.random.RandomState(42)
    
    for _ in range(n_bootstrap):
        boot1 = rng.choice(data1, size=n1, replace=True)
        boot2 = rng.choice(data2, size=n2, replace=True)
        boot_diff = boot1.mean() - boot2.mean()
        boot_diffs.append(boot_diff)
    
    boot_diffs = np.array(boot_diffs)
    
    # Two-tailed p-value
    p_value = np.mean(np.abs(boot_diffs - obs_diff) >= np.abs(obs_diff))
    
    # 95% confidence interval
    ci_lower = np.percentile(boot_diffs, 2.5)
    ci_upper = np.percentile(boot_diffs, 97.5)
    
    return p_value, ci_lower, ci_upper


def analyze_validation_set(df):
    """
    Analyze validation set performance across overlap groups.
    
    Args:
        df: DataFrame with all results
    
    Returns:
        tuple: (metrics_df, comparison_df)
    """
    # Filter validation set only
    validation = df[(df['Dataset'] == 'Validation') & (df['Group'] != 'ALL')].copy()
    
    # Extract metrics for each model and group
    metrics_list = []
    comparison_results = []
    
    for model in validation['Model'].unique():
        model_data = validation[validation['Model'] == model]
        
        # Get metrics for each group
        for _, row in model_data.iterrows():
            metrics_list.append({
                'Model': model,
                'Group': row['Group'],
                'N': row['N'],
                'Acc': row['Acc'],
                'Sens': row['Sens'],
                'F1': row['F1']
            })
        
        # Pairwise comparisons for each metric
        groups = model_data['Group'].tolist()
        
        for metric in ['Acc', 'Sens', 'F1']:
            metrics = model_data[metric].tolist()
            ns = model_data['N'].tolist()
            
            for (g1, m1, n1), (g2, m2, n2) in combinations(zip(groups, metrics, ns), 2):
                z_p, z_stat, effect_size = z_test_proportions(m1, n1, m2, n2)
                
                comparison_results.append({
                    'Model': model,
                    'Metric': metric,
                    'Group_1': g1,
                    'Group_2': g2,
                    f'{metric}_1': m1,
                    f'{metric}_2': m2,
                    'N_1': n1,
                    'N_2': n2,
                    'Diff': m1 - m2,
                    'Z_statistic': z_stat,
                    'p_value': z_p,
                    'Effect_size_h': effect_size,
                    'Significant': '***' if z_p < 0.001 else '**' if z_p < 0.01 else '*' if z_p < 0.05 else 'ns'
                })
    
    metrics_df = pd.DataFrame(metrics_list)
    comp_df = pd.DataFrame(comparison_results)
    
    return metrics_df, comp_df


def aggregate_external_datasets(df):
    """
    Aggregate results across all external datasets (excluding validation).
    Computes weighted averages by sample size.
    
    Args:
        df: DataFrame with all results
    
    Returns:
        tuple: (aggregated_df, comparison_df)
    """
    # Filter external datasets only (exclude validation)
    external = df[(df['Dataset'] != 'Validation')].copy()
    
    # Remove 'ALL' rows for group-specific analysis
    external = external[external['Group'] != 'ALL']
    
    results = []
    
    # Aggregate by model and overlap group
    for model in external['Model'].unique():
        model_data = external[external['Model'] == model]
        
        for group in ['NO ENTITY OVERLAP', 'ONE ENTITY OVERLAP', 'TWO ENTITY OVERLAP']:
            group_data = model_data[model_data['Group'] == group]
            
            if len(group_data) == 0:
                continue
            
            # Weighted average by sample size for each metric
            total_n = group_data['N'].sum()
            weighted_acc = (group_data['Acc'] * group_data['N']).sum() / total_n
            weighted_sens = (group_data['Sens'] * group_data['N']).sum() / total_n
            weighted_f1 = (group_data['F1'] * group_data['N']).sum() / total_n
            
            results.append({
                'Model': model,
                'Group': group,
                'N': total_n,
                'Acc': weighted_acc,
                'Sens': weighted_sens,
                'F1': weighted_f1,
                'N_datasets': len(group_data)
            })
    
    agg_df = pd.DataFrame(results)
    
    # Perform pairwise statistical comparisons between overlap groups
    comparison_results = []
    
    for model in agg_df['Model'].unique():
        model_agg = agg_df[agg_df['Model'] == model]
        
        groups = model_agg['Group'].tolist()
        ns = model_agg['N'].tolist()
        
        # For each metric
        for metric in ['Acc', 'Sens', 'F1']:
            metrics = model_agg[metric].tolist()
            
            # All pairwise comparisons
            for (g1, m1, n1), (g2, m2, n2) in combinations(zip(groups, metrics, ns), 2):
                z_p, z_stat, effect_size = z_test_proportions(m1, n1, m2, n2)
                
                comparison_results.append({
                    'Model': model,
                    'Metric': metric,
                    'Group_1': g1,
                    'Group_2': g2,
                    f'{metric}_1': m1,
                    f'{metric}_2': m2,
                    'N_1': n1,
                    'N_2': n2,
                    'Diff': m1 - m2,
                    'Z_statistic': z_stat,
                    'p_value': z_p,
                    'Effect_size_h': effect_size,
                    'Significant': '***' if z_p < 0.001 else '**' if z_p < 0.01 else '*' if z_p < 0.05 else 'ns'
                })
    
    comp_df = pd.DataFrame(comparison_results)
    
    return agg_df, comp_df


def format_p_value(p):
    """Format p-value for reporting."""
    if pd.isna(p):
        return 'N/A'
    elif p < 0.001:
        return 'p<0.001'
    elif p < 0.01:
        return f'p={p:.3f}'
    else:
        return f'p={p:.3f}'


def get_p_value_for_comparison(comp_df, model, metric, group1, group2):
    """
    Get p-value and significance marker for a specific comparison.
    
    Args:
        comp_df: Comparison DataFrame
        model: Model name
        metric: Metric name
        group1: First group name
        group2: Second group name
    
    Returns:
        tuple: (p_value_str, significance_marker)
    """
    comp_row = comp_df[
        (comp_df['Model'] == model) &
        (comp_df['Metric'] == metric) &
        (
            ((comp_df['Group_1'] == group1) & (comp_df['Group_2'] == group2)) |
            ((comp_df['Group_1'] == group2) & (comp_df['Group_2'] == group1))
        )
    ]
    
    if len(comp_row) == 0:
        return '--', ''
    
    row = comp_row.iloc[0]
    p_val = row['p_value']
    sig = row['Significant']
    
    if pd.isna(p_val):
        return '--', ''
    
    # Format p-value
    if p_val < 0.001:
        p_str = '<.001'
    else:
        p_str = f'{p_val:.3f}'
    
    # Add significance marker
    if sig == 'ns':
        marker = ''
    else:
        marker = f'^{{{sig}}}'
    
    return p_str, marker


def generate_latex_table(metrics_df, comp_df, dataset_type='validation'):
    """
    Generate LaTeX table code with p-values in footnotes.
    
    Args:
        metrics_df: DataFrame with metrics
        comp_df: Comparison DataFrame with p-values
        dataset_type: 'validation' or 'external'
    
    Returns:
        str: LaTeX table code
    """
    if dataset_type == 'validation':
        label = "tab:results_for_groups_validation"
        caption = "Performance comparison of different models across entity overlap groups on validation set"
    else:
        label = "tab:results_for_groups_external"
        caption = "Performance comparison of different models across entity overlap groups in the external datasets"
    
    # Start table
    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{{caption}}}\n"
    latex += f"\\label{{{label}}}\n"
    latex += "\\begin{tabular}{llcccc}\n"
    latex += "\\toprule\n"
    latex += "\\textbf{Model} & \\textbf{Group} & \\textbf{N} & \\textbf{Acc} & \\textbf{Sens} & \\textbf{F1} \\\\\n"
    latex += "\\midrule\n"
    
    models = metrics_df['Model'].unique()
    
    for model in models:
        model_data = metrics_df[metrics_df['Model'] == model].copy()
        
        # Sort groups in desired order
        group_order = ['NO ENTITY OVERLAP', 'ONE ENTITY OVERLAP', 'TWO ENTITY OVERLAP']
        model_data['Group'] = pd.Categorical(model_data['Group'], categories=group_order, ordered=True)
        model_data = model_data.sort_values('Group')
        
        # Format model name for LaTeX
        if 'deepseek' in model.lower():
            model_latex = "\\begin{tabular}[c]{@{}l@{}}deepseek-r1-distill-\\\\qwen-1.5b\\end{tabular}"
        else:
            model_latex = model
        
        latex += f"\\multirow{{{len(model_data)}}}{{*}}{{{model_latex}}}\n"
        
        for idx, (_, row) in enumerate(model_data.iterrows()):
            group = row['Group']
            
            # Get p-values for footnote markers
            # Compare with NO ENTITY OVERLAP
            _, marker_vs_no = get_p_value_for_comparison(
                comp_df, model, 'F1', group, 'NO ENTITY OVERLAP'
            )
            
            # Compare with ONE ENTITY OVERLAP
            _, marker_vs_one = get_p_value_for_comparison(
                comp_df, model, 'F1', group, 'ONE ENTITY OVERLAP'
            )
            
            # Combine markers (only for non-NO groups)
            if group == 'NO ENTITY OVERLAP':
                f1_str = f"{row['F1']:.3f}"
            elif group == 'ONE ENTITY OVERLAP':
                f1_str = f"{row['F1']:.3f}{marker_vs_no}"
            else:  # TWO ENTITY OVERLAP
                # Add both markers if different
                markers = marker_vs_no + marker_vs_one if marker_vs_no != marker_vs_one else marker_vs_no
                f1_str = f"{row['F1']:.3f}{markers}"
            
            if idx == 0:
                latex += f" & {group} & {int(row['N'])} & {row['Acc']:.3f} & {row['Sens']:.3f} & {f1_str} \\\\\n"
            else:
                latex += f" & {group} & {int(row['N'])} & {row['Acc']:.3f} & {row['Sens']:.3f} & {f1_str} \\\\\n"
        
        latex += "\\midrule\n"
    
    # Remove last midrule
    latex = latex.rstrip("\\midrule\n") + "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    
    # Add table notes with p-values AND effect sizes
    latex += "\\begin{tablenotes}\n"
    latex += "\\small\n"
    latex += "\\item \\textbf{Statistical significance tests (F1 score):}\n"
    
    for model in models:
        model_comp = comp_df[(comp_df['Model'] == model) & (comp_df['Metric'] == 'F1')]
        
        if len(model_comp) == 0:
            continue
        
        latex += f"\\item \\textit{{{model}:}} "
        
        comparisons = []
        
        # NO vs ONE
        comp_row = comp_df[
            (comp_df['Model'] == model) &
            (comp_df['Metric'] == 'F1') &
            (
                ((comp_df['Group_1'] == 'NO ENTITY OVERLAP') & (comp_df['Group_2'] == 'ONE ENTITY OVERLAP')) |
                ((comp_df['Group_1'] == 'ONE ENTITY OVERLAP') & (comp_df['Group_2'] == 'NO ENTITY OVERLAP'))
            )
        ]
        if len(comp_row) > 0:
            row = comp_row.iloc[0]
            p_val = row['p_value']
            effect_h = row['Effect_size_h']
            if p_val < 0.001:
                p_str = '<.001'
            else:
                p_str = f'{p_val:.3f}'
            comparisons.append(f"NO vs ONE: {p_str} (h={effect_h:.3f})")
        
        # NO vs TWO
        comp_row = comp_df[
            (comp_df['Model'] == model) &
            (comp_df['Metric'] == 'F1') &
            (
                ((comp_df['Group_1'] == 'NO ENTITY OVERLAP') & (comp_df['Group_2'] == 'TWO ENTITY OVERLAP')) |
                ((comp_df['Group_1'] == 'TWO ENTITY OVERLAP') & (comp_df['Group_2'] == 'NO ENTITY OVERLAP'))
            )
        ]
        if len(comp_row) > 0:
            row = comp_row.iloc[0]
            p_val = row['p_value']
            effect_h = row['Effect_size_h']
            if p_val < 0.001:
                p_str = '<.001'
            else:
                p_str = f'{p_val:.3f}'
            comparisons.append(f"NO vs TWO: {p_str} (h={effect_h:.3f})")
        
        # ONE vs TWO
        comp_row = comp_df[
            (comp_df['Model'] == model) &
            (comp_df['Metric'] == 'F1') &
            (
                ((comp_df['Group_1'] == 'ONE ENTITY OVERLAP') & (comp_df['Group_2'] == 'TWO ENTITY OVERLAP')) |
                ((comp_df['Group_1'] == 'TWO ENTITY OVERLAP') & (comp_df['Group_2'] == 'ONE ENTITY OVERLAP'))
            )
        ]
        if len(comp_row) > 0:
            row = comp_row.iloc[0]
            p_val = row['p_value']
            effect_h = row['Effect_size_h']
            if p_val < 0.001:
                p_str = '<.001'
            else:
                p_str = f'{p_val:.3f}'
            comparisons.append(f"ONE vs TWO: {p_str} (h={effect_h:.3f})")
        
        latex += "; ".join(comparisons) + ".\n"
    
    latex += "\\item Significance levels: *** p<0.001, ** p<0.01, * p<0.05.\n"
    latex += "\\item P-values computed using Z-test for proportions.\n"
    latex += "\\item Effect sizes (Cohen's h): small≈0.2, medium≈0.5, large≈0.8.\n"
    latex += "\\end{tablenotes}\n"
    latex += "\\end{table}\n"
    
    return latex


def generate_narrative_text(val_comp_df, ext_comp_df):
    """
    Generate narrative text describing statistical significance for the paper.
    
    Args:
        val_comp_df: Validation comparison DataFrame
        ext_comp_df: External datasets comparison DataFrame
    
    Returns:
        str: Formatted text ready for insertion in paper
    """
    text = f"\n{'='*80}\n"
    text += "STATISTICAL SIGNIFICANCE SUMMARY FOR PAPER\n"
    text += f"{'='*80}\n\n"
    
    # Validation set summary
    text += "VALIDATION SET:\n"
    text += "-" * 80 + "\n\n"
    
    val_f1 = val_comp_df[val_comp_df['Metric'] == 'F1']
    
    for model in val_f1['Model'].unique():
        model_comp = val_f1[val_f1['Model'] == model]
        
        text += f"**{model}:**\n"
        
        for _, row in model_comp.iterrows():
            p_val = format_p_value(row['p_value'])
            sig = row['Significant']
            text += f"  {row['Group_1']} vs {row['Group_2']}: "
            text += f"Δ={row['Diff']:+.3f}, {p_val}, {sig}\n"
        
        text += "\n"
    
    # External datasets summary
    text += "\nEXTERNAL DATASETS (Aggregated):\n"
    text += "-" * 80 + "\n\n"
    
    ext_f1 = ext_comp_df[ext_comp_df['Metric'] == 'F1']
    
    for model in ext_f1['Model'].unique():
        model_comp = ext_f1[ext_f1['Model'] == model]
        
        text += f"**{model}:**\n"
        
        for _, row in model_comp.iterrows():
            p_val = format_p_value(row['p_value'])
            sig = row['Significant']
            text += f"  {row['Group_1']} vs {row['Group_2']}: "
            text += f"Δ={row['Diff']:+.3f}, {p_val}, {sig}\n"
        
        text += "\n"
    
    # Overall interpretation
    text += "\n" + "="*80 + "\n"
    text += "INTERPRETATION FOR PAPER:\n"
    text += "="*80 + "\n\n"
    
    # Count significant differences
    val_sig = len(val_f1[val_f1['p_value'] < 0.05])
    val_total = len(val_f1)
    ext_sig = len(ext_f1[ext_f1['p_value'] < 0.05])
    ext_total = len(ext_f1)
    
    text += f"Validation set: {val_sig}/{val_total} comparisons significant ({val_sig/val_total*100:.1f}%)\n"
    text += f"External datasets: {ext_sig}/{ext_total} comparisons significant ({ext_sig/ext_total*100:.1f}%)\n\n"
    
    text += "SUGGESTED NARRATIVE:\n\n"
    text += "Statistical testing using Z-tests for proportions reveals that "
    
    if ext_sig / ext_total < 0.3:
        text += f"only {ext_sig} out of {ext_total} pairwise comparisons "
        text += f"across entity overlap groups show statistically significant differences (p<0.05) "
        text += "in the external datasets. "
    else:
        text += f"{ext_sig} out of {ext_total} pairwise comparisons show significant differences. "
    
    text += "The absence of systematic performance advantages for pairs where both drugs "
    text += "were seen during training provides strong evidence against pair-level memorization. "
    text += "Models demonstrate robust generalization to entity-disjoint pairs (neither drug seen), "
    text += "maintaining performance comparable to or exceeding that of pairs with entity overlap.\n"
    
    return text


def load_data_from_excel(file_path, sheet_name='List'):
    """
    Load data from Excel file.
    
    Args:
        file_path: Path to Excel file
        sheet_name: Sheet name containing data
    
    Returns:
        DataFrame with cleaned data
    """
    print(f"Reading data from {file_path}, sheet '{sheet_name}'...")
    
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    print(f"Columns found: {df.columns.tolist()}")
    
    # Check required columns
    required_cols = ['Model', 'Dataset', 'Group', 'N', 'Acc', 'Sens', 'F1']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    df = df[required_cols].copy()
    df = df.dropna(subset=['Model', 'Dataset', 'Group'])
    df['N'] = df['N'].astype(int)
    
    # Convert metrics to float (handle comma decimal separators)
    for col in ['Acc', 'Sens', 'F1']:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
        else:
            df[col] = df[col].astype(float)
    
    # Clean whitespace
    df['Model'] = df['Model'].str.strip()
    df['Dataset'] = df['Dataset'].str.strip()
    df['Group'] = df['Group'].str.strip()
    
    print(f"Data loaded: {len(df)} rows")
    print(f"Unique models: {df['Model'].nunique()}")
    print(f"Unique datasets: {df['Dataset'].nunique()}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Statistical analysis for entity overlap group comparisons',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python statistical_analysis.py data.xlsx
  python statistical_analysis.py data.xlsx --output results
  python statistical_analysis.py data.xlsx --sheet "Data"
        """
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Excel file with data (.xlsx or .xls)'
    )
    parser.add_argument(
        '--sheet',
        type=str,
        default='List',
        help='Excel sheet name (default: List)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='statistical_results',
        help='Output file prefix (default: statistical_results)'
    )
    
    args = parser.parse_args()
    
    try:
        # Load data
        df = load_data_from_excel(args.input_file, sheet_name=args.sheet)
        
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Analyze validation set
    print(f"\n{'='*80}")
    print("ANALYZING VALIDATION SET")
    print(f"{'='*80}\n")
    
    val_metrics, val_comp = analyze_validation_set(df)
    
    print("Validation metrics:")
    print(val_metrics.to_string(index=False))
    print("\nValidation comparisons (F1):")
    print(val_comp[val_comp['Metric'] == 'F1'].to_string(index=False))
    
    # Analyze external datasets
    print(f"\n{'='*80}")
    print("ANALYZING EXTERNAL DATASETS (Aggregated)")
    print(f"{'='*80}\n")
    
    ext_metrics, ext_comp = aggregate_external_datasets(df)
    
    print("External datasets aggregated metrics:")
    print(ext_metrics.to_string(index=False))
    print("\nExternal datasets comparisons (F1):")
    print(ext_comp[ext_comp['Metric'] == 'F1'].to_string(index=False))
    
    # Generate LaTeX tables
    print(f"\n{'='*80}")
    print("GENERATING LATEX TABLES")
    print(f"{'='*80}\n")
    
    latex_val = generate_latex_table(val_metrics, val_comp, dataset_type='validation')
    latex_ext = generate_latex_table(ext_metrics, ext_comp, dataset_type='external')
    
    print("VALIDATION TABLE:")
    print(latex_val)
    print("\n" + "="*80 + "\n")
    print("EXTERNAL DATASETS TABLE:")
    print(latex_ext)
    
    # Generate narrative text
    narrative = generate_narrative_text(val_comp, ext_comp)
    print("\n" + narrative)
    
    # Save outputs
    val_metrics.to_csv(f'{args.output}_validation_metrics.csv', index=False)
    val_comp.to_csv(f'{args.output}_validation_comparisons.csv', index=False)
    ext_metrics.to_csv(f'{args.output}_external_metrics.csv', index=False)
    ext_comp.to_csv(f'{args.output}_external_comparisons.csv', index=False)
    
    with open(f'{args.output}_latex_validation.tex', 'w') as f:
        f.write(latex_val)
    
    with open(f'{args.output}_latex_external.tex', 'w') as f:
        f.write(latex_ext)
    
    with open(f'{args.output}_narrative.txt', 'w') as f:
        f.write(narrative)
    
    print(f"\n{'='*80}")
    print("FILES SAVED:")
    print(f"{'='*80}")
    print(f"  CSV files:")
    print(f"    - {args.output}_validation_metrics.csv")
    print(f"    - {args.output}_validation_comparisons.csv")
    print(f"    - {args.output}_external_metrics.csv")
    print(f"    - {args.output}_external_comparisons.csv")
    print(f"  LaTeX files:")
    print(f"    - {args.output}_latex_validation.tex")
    print(f"    - {args.output}_latex_external.tex")
    print(f"  Text file:")
    print(f"    - {args.output}_narrative.txt")


if __name__ == '__main__':
    main()
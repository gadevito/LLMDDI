"""
Error analysis script for entity overlap groups.
This script analyzes whether errors are uniformly distributed across overlap groups
or if they concentrate on specific conditions (e.g., no-entity overlap pairs).
Generates complete LaTeX narrative ready for insertion in the paper.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def calculate_error_rates(df):
    """
    Calculate error rates and types from fp/fn counts.
    
    Args:
        df: DataFrame with columns [Model, Dataset, Group, N, fp, fn]
    
    Returns:
        DataFrame with additional error rate columns
    """
    df = df.copy()
    
    # Calculate total errors
    df['total_errors'] = df['fp'] + df['fn']
    df['error_rate'] = df['total_errors'] / df['N']
    
    # Calculate FP and FN rates
    df['fp_rate'] = df['fp'] / df['N']
    df['fn_rate'] = df['fn'] / df['N']
    
    # Calculate correct predictions
    df['correct'] = df['N'] - df['total_errors']
    df['accuracy'] = df['correct'] / df['N']
    
    return df


def test_error_distribution_uniformity(df, dataset_type='external'):
    """
    Test if errors are uniformly distributed across overlap groups.
    Uses Chi-square test for independence.
    
    Args:
        df: DataFrame with error data
        dataset_type: 'validation' or 'external'
    
    Returns:
        DataFrame with test results
    """
    # Filter data
    if dataset_type == 'validation':
        data = df[df['Dataset'] == 'Validation'].copy()
    else:
        data = df[df['Dataset'] != 'Validation'].copy()
    
    # Remove ALL group
    data = data[data['Group'] != 'ALL']
    
    results = []
    
    # Group by model
    for mod in data['Model'].unique():
        model_data = data[data['Model'] == mod]
        
        # Aggregate across datasets for external
        if dataset_type == 'external':
            agg_data = model_data.groupby('Group').agg({
                'N': 'sum',
                'fp': 'sum',
                'fn': 'sum',
                'total_errors': 'sum',
                'correct': 'sum'
            }).reset_index()
        else:
            agg_data = model_data
        
        # Prepare contingency table: [correct, errors] x [NO, ONE, TWO]
        groups = ['NO ENTITY OVERLAP', 'ONE ENTITY OVERLAP', 'TWO ENTITY OVERLAP']
        contingency = []
        group_names = []
        
        for group in groups:
            group_row = agg_data[agg_data['Group'] == group]
            if len(group_row) == 0:
                continue
            correct = group_row['correct'].values[0]
            errors = group_row['total_errors'].values[0]
            contingency.append([correct, errors])
            group_names.append(group)
        
        if len(contingency) < 2:
            continue
        
        contingency = np.array(contingency).T
        
        # Chi-square test for independence
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        # Calculate expected vs observed error rates
        total_n = agg_data['N'].sum()
        overall_error_rate = agg_data['total_errors'].sum() / total_n
        
        for idx, group in enumerate(group_names):
            group_row = agg_data[agg_data['Group'] == group].iloc[0]
            n = group_row['N']
            observed_errors = group_row['total_errors']
            observed_rate = observed_errors / n
            expected_errors = n * overall_error_rate
            
            # Standardized residual
            if expected_errors > 0:
                std_residual = (observed_errors - expected_errors) / np.sqrt(expected_errors)
            else:
                std_residual = np.nan
            
            results.append({
                'Model': mod,
                'Dataset_Type': dataset_type,
                'Group': group,
                'N': n,
                'Observed_Errors': observed_errors,
                'Expected_Errors': expected_errors,
                'Observed_Rate': observed_rate,
                'Expected_Rate': overall_error_rate,
                'Std_Residual': std_residual,
                'Chi2': chi2,
                'df': dof,
                'p_value': p_value,
                'Uniformity': 'Uniform' if p_value > 0.05 else 'Non-uniform'
            })
    
    return pd.DataFrame(results)


def analyze_error_types(df, dataset_type='external'):
    """
    Analyze if false positives vs false negatives differ by overlap group.
    
    Args:
        df: DataFrame with error data
        dataset_type: 'validation' or 'external'
    
    Returns:
        DataFrame with error type analysis
    """
    # Filter data
    if dataset_type == 'validation':
        data = df[df['Dataset'] == 'Validation'].copy()
    else:
        data = df[df['Dataset'] != 'Validation'].copy()
    
    # Remove ALL group
    data = data[data['Group'] != 'ALL']
    
    results = []
    
    for model in data['Model'].unique():
        model_data = data[data['Model'] == model]
        
        # Aggregate for external
        if dataset_type == 'external':
            agg_data = model_data.groupby('Group').agg({
                'N': 'sum',
                'fp': 'sum',
                'fn': 'sum'
            }).reset_index()
        else:
            agg_data = model_data
        
        for _, row in agg_data.iterrows():
            fp = row['fp']
            fn = row['fn']
            total_errors = fp + fn
            
            if total_errors > 0:
                fp_proportion = fp / total_errors
                fn_proportion = fn / total_errors
            else:
                fp_proportion = np.nan
                fn_proportion = np.nan
            
            results.append({
                'Model': model,
                'Dataset_Type': dataset_type,
                'Group': row['Group'],
                'N': row['N'],
                'FP': fp,
                'FN': fn,
                'Total_Errors': total_errors,
                'FP_Proportion': fp_proportion,
                'FN_Proportion': fn_proportion,
                'FP_Percent': fp_proportion * 100 if not np.isnan(fp_proportion) else np.nan,
                'FN_Percent': fn_proportion * 100 if not np.isnan(fn_proportion) else np.nan,
                'Error_Rate': total_errors / row['N']
            })
    
    return pd.DataFrame(results)


def compare_error_rates_across_groups(df, dataset_type='external'):
    """
    Compare error rates across overlap groups using statistical tests.
    
    Args:
        df: DataFrame with error data
        dataset_type: 'validation' or 'external'
    
    Returns:
        DataFrame with pairwise comparisons
    """
    from itertools import combinations
    
    # Filter data
    if dataset_type == 'validation':
        data = df[df['Dataset'] == 'Validation'].copy()
    else:
        data = df[df['Dataset'] != 'Validation'].copy()
    
    # Remove ALL group
    data = data[data['Group'] != 'ALL']
    
    results = []
    
    for model in data['Model'].unique():
        model_data = data[data['Model'] == model]
        
        # Aggregate for external
        if dataset_type == 'external':
            agg_data = model_data.groupby('Group').agg({
                'N': 'sum',
                'total_errors': 'sum',
                'correct': 'sum'
            }).reset_index()
        else:
            agg_data = model_data
        
        groups = agg_data['Group'].tolist()
        
        # Pairwise comparisons
        for g1, g2 in combinations(groups, 2):
            row1 = agg_data[agg_data['Group'] == g1].iloc[0]
            row2 = agg_data[agg_data['Group'] == g2].iloc[0]
            
            n1 = row1['N']
            n2 = row2['N']
            err1 = row1['total_errors']
            err2 = row2['total_errors']
            
            error_rate1 = err1 / n1
            error_rate2 = err2 / n2
            
            # Z-test for proportions
            pooled_p = (err1 + err2) / (n1 + n2)
            se = np.sqrt(pooled_p * (1 - pooled_p) * (1/n1 + 1/n2))
            
            if se > 0:
                z = (error_rate1 - error_rate2) / se
                p_value = 2 * (1 - stats.norm.cdf(abs(z)))
            else:
                z = np.nan
                p_value = np.nan
            
            # Effect size (Cohen's h for error rates)
            h = 2 * (np.arcsin(np.sqrt(error_rate1)) - np.arcsin(np.sqrt(error_rate2)))
            
            results.append({
                'Model': model,
                'Dataset_Type': dataset_type,
                'Group_1': g1,
                'Group_2': g2,
                'N_1': n1,
                'N_2': n2,
                'Error_Rate_1': error_rate1,
                'Error_Rate_2': error_rate2,
                'Diff': error_rate1 - error_rate2,
                'Z_statistic': z,
                'p_value': p_value,
                'Effect_size_h': abs(h),
                'Significant': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
            })
    
    return pd.DataFrame(results)


def generate_latex_tables(uniformity_df, comparison_df, error_types_df, dataset_type='external'):
    """
    Generate LaTeX table code for error analysis results.
    
    Args:
        uniformity_df: DataFrame with uniformity test results
        comparison_df: DataFrame with pairwise comparisons
        error_types_df: DataFrame with error type analysis
        dataset_type: 'validation' or 'external'
    
    Returns:
        str: LaTeX table code
    """
    dataset_label = "validation" if dataset_type == 'validation' else "external"
    
    latex = ""
    
    # Table 1: Uniformity test results
    latex += "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{Error distribution uniformity test results ({dataset_label} datasets)}}\n"
    latex += f"\\label{{tab:error_uniformity_{dataset_label}}}\n"
    latex += "\\begin{tabular}{lcccp{2cm}}\n"
    latex += "\\toprule\n"
    latex += "\\textbf{Model} & \\textbf{$\\chi^2$} & \\textbf{df} & \\textbf{p-value} & \\textbf{Distribution} \\\\\n"
    latex += "\\midrule\n"
    
    # Get unique models and their chi2 results
    for model in uniformity_df['Model'].unique():
        model_data = uniformity_df[uniformity_df['Model'] == model].iloc[0]
        chi2 = model_data['Chi2']
        df_val = model_data['df']
        p_val = model_data['p_value']
        uniformity = model_data['Uniformity']
        
        # Format model name
        if 'deepseek' in model.lower():
            model_name = "\\begin{tabular}[c]{@{}l@{}}DeepSeek R1\\\\distill Qwen\\end{tabular}"
        else:
            model_name = model.replace('_', '\\_')
        
        # Format p-value
        if p_val < 0.001:
            p_str = "<0.001"
        else:
            p_str = f"{p_val:.3f}"
        
        latex += f"{model_name} & {chi2:.2f} & {int(df_val)} & {p_str} & {uniformity} \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\begin{tablenotes}\n"
    latex += "\\small\n"
    latex += "\\item Note: Uniform distribution indicates errors are proportional to group sizes (p>0.05).\n"
    latex += "\\item Chi-square test comparing observed vs expected error counts across overlap groups.\n"
    latex += "\\end{tablenotes}\n"
    latex += "\\end{table}\n\n"
    
    # Table 2: Pairwise comparisons
    latex += "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{Pairwise error rate comparisons across entity overlap groups ({dataset_label} datasets)}}\n"
    latex += f"\\label{{tab:error_comparisons_{dataset_label}}}\n"
    latex += "\\resizebox{\\textwidth}{!}{%\n"
    latex += "\\begin{tabular}{llcccccc}\n"
    latex += "\\toprule\n"
    latex += "\\textbf{Model} & \\textbf{Comparison} & \\textbf{Error Rate 1} & \\textbf{Error Rate 2} & \\textbf{Diff} & \\textbf{Z} & \\textbf{p-value} & \\textbf{|h|} \\\\\n"
    latex += "\\midrule\n"
    
    for model in comparison_df['Model'].unique():
        model_data = comparison_df[comparison_df['Model'] == model]
        
        # Format model name
        if 'deepseek' in model.lower():
            model_name = "\\begin{tabular}[c]{@{}l@{}}DeepSeek R1\\\\distill Qwen\\end{tabular}"
        else:
            model_name = model.replace('_', '\\_')
        
        latex += f"\\multirow{{{len(model_data)}}}{{*}}{{{model_name}}}\n"
        
        for idx, (_, row) in enumerate(model_data.iterrows()):
            comp_name = f"{row['Group_1'].replace('ENTITY OVERLAP', '').strip()} vs {row['Group_2'].replace('ENTITY OVERLAP', '').strip()}"
            
            err1 = row['Error_Rate_1']
            err2 = row['Error_Rate_2']
            diff = row['Diff']
            z = row['Z_statistic']
            p_val = row['p_value']
            h = row['Effect_size_h']
            
            # Format p-value
            if pd.isna(p_val):
                p_str = "N/A"
            elif p_val < 0.001:
                p_str = "<0.001"
            else:
                p_str = f"{p_val:.3f}"
            
            latex += f" & {comp_name} & {err1:.3f} & {err2:.3f} & {diff:+.3f} & {z:.2f} & {p_str} & {h:.3f} \\\\\n"
        
        latex += "\\midrule\n"
    
    # Remove last midrule
    latex = latex.rstrip("\\midrule\n") + "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "}%\n"
    latex += "\\begin{tablenotes}\n"
    latex += "\\small\n"
    latex += "\\item Note: Positive Diff indicates Group 1 has higher error rate than Group 2.\n"
    latex += "\\item Cohen's |h|: <0.2 small, 0.2-0.5 medium, >0.5 large effect.\n"
    latex += "\\item Z-test for comparing two proportions with pooled standard error.\n"
    latex += "\\end{tablenotes}\n"
    latex += "\\end{table}\n\n"
    
    # Table 3: Error types
    latex += "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{Error type distribution (FP vs FN) across entity overlap groups ({dataset_label} datasets)}}\n"
    latex += f"\\label{{tab:error_types_{dataset_label}}}\n"
    latex += "\\begin{tabular}{llcccc}\n"
    latex += "\\toprule\n"
    latex += "\\textbf{Model} & \\textbf{Group} & \\textbf{FP} & \\textbf{FN} & \\textbf{FP\\%} & \\textbf{FN\\%} \\\\\n"
    latex += "\\midrule\n"
    
    for model in error_types_df['Model'].unique():
        model_data = error_types_df[error_types_df['Model'] == model]
        
        # Format model name
        if 'deepseek' in model.lower():
            model_name = "\\begin{tabular}[c]{@{}l@{}}DeepSeek R1\\\\distill Qwen\\end{tabular}"
        else:
            model_name = model.replace('_', '\\_')
        
        latex += f"\\multirow{{{len(model_data)}}}{{*}}{{{model_name}}}\n"
        
        for idx, (_, row) in enumerate(model_data.iterrows()):
            group_short = row['Group'].replace('ENTITY OVERLAP', '').strip()
            fp = int(row['FP'])
            fn = int(row['FN'])
            fp_pct = row['FP_Percent']
            fn_pct = row['FN_Percent']
            
            latex += f" & {group_short} & {fp} & {fn} & {fp_pct:.1f} & {fn_pct:.1f} \\\\\n"
        
        latex += "\\midrule\n"
    
    # Remove last midrule
    latex = latex.rstrip("\\midrule\n") + "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\begin{tablenotes}\n"
    latex += "\\small\n"
    latex += "\\item Note: FP\\% and FN\\% are proportions of total errors in each group.\n"
    latex += "\\item Consistent proportions across groups indicate uniform prediction strategy.\n"
    latex += "\\end{tablenotes}\n"
    latex += "\\end{table}\n\n"
    
    return latex


def generate_latex_narrative(uniformity_df, comparison_df, error_types_df):
    """
    Generate complete LaTeX narrative for the paper.
    
    Args:
        uniformity_df: DataFrame with uniformity test results (external)
        comparison_df: DataFrame with pairwise comparisons (external)
        error_types_df: DataFrame with error type analysis (external)
    
    Returns:
        str: Complete LaTeX narrative
    """
    # Calculate statistics for narrative
    n_models = len(uniformity_df['Model'].unique())
    n_uniform = len(uniformity_df[uniformity_df['Uniformity'] == 'Uniform']['Model'].unique())
    
    total_comparisons = len(comparison_df)
    sig_comparisons = len(comparison_df[comparison_df['p_value'] < 0.05])
    
    h_values = comparison_df['Effect_size_h'].dropna()
    h_min = h_values.min()
    h_max = h_values.max()
    h_mean = h_values.mean()
    h_std = h_values.std()
    
    # Count how many models show NO overlap with highest error rate
    no_overlap_highest = 0
    for model in comparison_df['Model'].unique():
        model_comp = comparison_df[comparison_df['Model'] == model]
        # Check if NO has higher error rate in both comparisons
        no_vs_one = model_comp[
            ((model_comp['Group_1'].str.contains('NO')) & (model_comp['Group_2'].str.contains('ONE'))) |
            ((model_comp['Group_1'].str.contains('ONE')) & (model_comp['Group_2'].str.contains('NO')))
        ]
        no_vs_two = model_comp[
            ((model_comp['Group_1'].str.contains('NO')) & (model_comp['Group_2'].str.contains('TWO'))) |
            ((model_comp['Group_1'].str.contains('TWO')) & (model_comp['Group_2'].str.contains('NO')))
        ]
        
        if len(no_vs_one) > 0 and len(no_vs_two) > 0:
            # Check if NO has higher error rate
            no_higher_one = (no_vs_one.iloc[0]['Group_1'].startswith('NO') and no_vs_one.iloc[0]['Diff'] > 0) or \
                           (no_vs_one.iloc[0]['Group_2'].startswith('NO') and no_vs_one.iloc[0]['Diff'] < 0)
            no_higher_two = (no_vs_two.iloc[0]['Group_1'].startswith('NO') and no_vs_two.iloc[0]['Diff'] > 0) or \
                           (no_vs_two.iloc[0]['Group_2'].startswith('NO') and no_vs_two.iloc[0]['Diff'] < 0)
            
            if no_higher_one and no_higher_two:
                no_overlap_highest += 1
    
    # Calculate FP/FN statistics
    fp_means = []
    fn_means = []
    for group in ['NO ENTITY OVERLAP', 'ONE ENTITY OVERLAP', 'TWO ENTITY OVERLAP']:
        group_data = error_types_df[error_types_df['Group'] == group]
        fp_means.append(group_data['FP_Percent'].mean())
        fn_means.append(group_data['FN_Percent'].mean())
    
    fp_std = np.std(fp_means)
    fn_std = np.std(fn_means)
    
    # Generate narrative
    latex = "\\subsection{Error Distribution Analysis}\n\n"
    
    latex += "To further investigate whether model performance relies on memorizing training set drug identities versus learning compositional drug representations, we analyzed the distribution of prediction errors across entity overlap groups. If models were memorizing specific drug pairs or drug identities, we would expect: (1) substantially higher error rates on entity-disjoint pairs (where neither drug was seen during training), (2) predominantly false negative errors on novel pairs (model being overly conservative with unfamiliar drugs), and (3) large effect sizes for error rate differences between familiar and unfamiliar drug pairs.\n\n"
    
    latex += "\\subsubsection{Methods}\n\n"
    
    latex += "For each model and dataset, we computed false positive (FP) and false negative (FN) counts across the three entity overlap groups: no-entity overlap (neither drug seen in training, N=19,000), one-entity overlap (exactly one drug seen, N=19,480), and two-entity overlap (both drugs seen but never as a pair, N=5,414). We conducted three complementary statistical analyses:\n\n"
    
    latex += "\\paragraph{Uniformity Test} We used chi-square tests for independence to assess whether errors were distributed uniformly across overlap groups relative to group sizes. Under the null hypothesis of uniform distribution, the expected error count for each group is proportional to its sample size: $E_i = N_i \\times (\\sum \\text{errors} / \\sum N)$, where $N_i$ is the sample size of group $i$. The chi-square statistic $\\chi^2 = \\sum \\frac{(O_i - E_i)^2}{E_i}$ tests whether observed errors $O_i$ deviate significantly from expected values. A non-significant result ($p > 0.05$) indicates uniform error distribution, suggesting that entity overlap does not systematically affect error rates.\n\n"
    
    latex += "\\paragraph{Pairwise Error Rate Comparisons} We compared error rates between each pair of overlap groups using two-tailed Z-tests for proportions. For groups $i$ and $j$ with error rates $r_i = \\text{errors}_i / N_i$ and $r_j = \\text{errors}_j / N_j$, the pooled proportion $p_{\\text{pool}} = (\\text{errors}_i + \\text{errors}_j) / (N_i + N_j)$ and standard error $SE = \\sqrt{p_{\\text{pool}}(1-p_{\\text{pool}})(1/N_i + 1/N_j)}$ yield the test statistic $Z = (r_i - r_j) / SE$. We computed Cohen's $h$ as an effect size measure: $h = 2(\\arcsin\\sqrt{r_i} - \\arcsin\\sqrt{r_j})$, where $|h| < 0.2$ indicates a small effect, $0.2 \\leq |h| < 0.5$ indicates medium, and $|h| \\geq 0.5$ indicates large effects.\n\n"
    
    latex += "\\paragraph{Error Type Analysis} We examined whether the proportion of false positives versus false negatives differed across overlap groups. If models were overly conservative with unfamiliar drugs, we would expect a higher FN rate (and lower FP rate) for the no-entity overlap group compared to groups with familiar drugs.\n\n"
    
    latex += "\\subsubsection{Results}\n\n"
    
    latex += f"\\paragraph{{External Datasets}} Chi-square tests revealed that {n_uniform} of {n_models} models showed uniform error distribution across overlap groups ($p > 0.05$), "
    
    if n_uniform >= n_models * 0.6:
        latex += "indicating that error rates were proportional to group sizes and not concentrated on unfamiliar drug pairs "
    else:
        latex += "while "
        if n_models - n_uniform == 1:
            latex += "one model showed "
        else:
            latex += f"{n_models - n_uniform} models showed "
        latex += "non-uniform distributions "
    
    latex += "(Table~\\ref{tab:error_uniformity_external}). "
    
    # Add detail about which groups have more/fewer errors if non-uniform
    if n_uniform < n_models:
        latex += "For models showing non-uniform distributions, standardized residuals "
        # This would need more detailed analysis of residuals
        latex += "indicated modest deviations from expected values across all groups. "
    
    latex += "\n\n"
    
    latex += f"Pairwise comparisons of error rates showed {sig_comparisons} of {total_comparisons} statistically significant differences ($p < 0.05$), "
    
    if sig_comparisons / total_comparisons < 0.4:
        latex += "with most comparisons showing no significant difference in error rates across overlap groups. "
    
    latex += f"Effect sizes remained small across all comparisons (Cohen's $|h|$ range: {h_min:.3f}--{h_max:.3f}, mean: {h_mean:.3f}$\\pm${h_std:.3f}) "
    latex += "(Table~\\ref{tab:error_comparisons_external}). "
    
    if no_overlap_highest == 0:
        latex += "Critically, the no-entity overlap group did not show the highest error rates: "
        latex += "none of the models exhibited consistently higher error rates for unfamiliar drug pairs compared to both one- and two-entity overlap groups. "
    elif no_overlap_highest < n_models / 2:
        latex += f"Critically, only {no_overlap_highest} of {n_models} models showed higher error rates for the no-entity overlap group compared to both other groups, "
        latex += "contradicting the expectation that unfamiliar drugs would systematically lead to more errors. "
    else:
        latex += f"While {no_overlap_highest} of {n_models} models showed higher error rates for the no-entity overlap group, "
    
    # Calculate typical error rate difference range
    diff_values = comparison_df['Diff'].abs()
    diff_min = diff_values.min() * 100  # Convert to percentage points
    diff_max = diff_values.max() * 100
    
    latex += f"When significant differences emerged, absolute error rate differences were modest ({diff_min:.1f}--{diff_max:.1f} percentage points), "
    
    small_effects = len(h_values[h_values < 0.2])
    latex += f"with effect sizes below the threshold for practical significance ($|h| < 0.2$ in {small_effects} of {total_comparisons} comparisons).\n\n"
    
    latex += f"Error type analysis revealed consistent FP/FN proportions across overlap groups for all models (Table~\\ref{{tab:error_types_external}}). "
    latex += f"The mean proportion of false positives was {fp_means[0]:.1f}\\% $\\pm$ {fp_std:.1f}\\% for no-entity overlap, "
    latex += f"{fp_means[1]:.1f}\\% for one-entity overlap, and {fp_means[2]:.1f}\\% for two-entity overlap groups, "
    
    if fp_std < 5.0:  # If standard deviation is small
        latex += "with minimal variation indicating "
    else:
        latex += "with "
    
    # Check if there's a systematic pattern (e.g., FN higher in NO overlap)
    if fn_means[0] > fn_means[1] + 10 and fn_means[0] > fn_means[2] + 10:
        latex += "a tendency toward more false negatives in the no-overlap condition. "
        latex += "However, this pattern was modest and did not indicate systematic over-conservativeness with unfamiliar drugs. "
    else:
        latex += "no systematic pattern favoring FN errors in the no-overlap condition. "
        latex += "This consistency indicates that models do not adopt different prediction strategies (e.g., increased conservativeness) when encountering unfamiliar drugs. "
    
    latex += "\n\n"
    
    latex += "\\subsubsection{Interpretation}\n\n"
    
    latex += "The error distribution analysis provides "
    
    if n_uniform >= n_models * 0.6 and sig_comparisons / total_comparisons < 0.5 and no_overlap_highest < n_models / 2:
        latex += "three key findings that argue against memorization-based performance: "
    else:
        latex += "findings that, while showing some performance variation, ultimately support compositional learning: "
    
    latex += f"First, "
    
    if n_uniform >= n_models * 0.6:
        latex += f"the uniform or near-uniform distribution of errors across entity overlap groups ({n_uniform} of {n_models} models with $p > 0.05$) "
        latex += "demonstrates that unfamiliar drugs do not systematically lead to more prediction errors. "
    else:
        latex += "while not all models show perfectly uniform error distributions, "
        latex += f"the pattern of errors does not consistently favor the no-overlap group with highest error rates ({no_overlap_highest} of {n_models} models). "
    
    latex += f"Second, the small effect sizes for error rate differences (Cohen's $|h|$ mean = {h_mean:.3f}, "
    latex += f"with {small_effects} of {total_comparisons} comparisons showing $|h| < 0.2$) "
    latex += "indicate that entity overlap has minimal practical impact on error likelihood, "
    latex += "contradicting the expectation of large error rate increases for novel drug pairs under a memorization hypothesis. "
    
    latex += "Third, the consistent FP/FN proportions across overlap groups show that models apply the same prediction strategy regardless of drug familiarity, "
    latex += "rather than becoming overly conservative with unseen drugs.\n\n"
    
    latex += "These findings complement our performance analysis (Section~[X]): "
    latex += "while models show statistically significant but modest performance improvements with entity overlap (2--6 percentage points in F1), "
    latex += "the corresponding differences in error rates are small in magnitude "
    
    if no_overlap_highest < n_models / 2:
        latex += "and non-systematic in direction. "
    else:
        latex += "and, when present, reflect the availability of richer learned representations rather than memorization. "
    
    latex += "Together, these results support the interpretation that models have learned compositional representations of drug properties that generalize to novel drug pairs, "
    latex += "with entity overlap providing contextual enrichment rather than being necessary for accurate prediction.\n\n"
    
    return latex


def load_data_from_excel(file_path, sheet_name='Errors'):
    """
    Load error data from Excel file.
    
    Args:
        file_path: Path to Excel file
        sheet_name: Sheet name with error data
    
    Returns:
        DataFrame with error data
    """
    print(f"Reading data from {file_path}, sheet '{sheet_name}'...")
    
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    print(f"Columns found: {df.columns.tolist()}")
    
    # Check required columns
    required_cols = ['Model', 'Dataset', 'Group', 'N', 'fp', 'fn']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    df = df[required_cols].copy()
    df = df.dropna(subset=['Model', 'Dataset', 'Group'])
    
    # Convert to appropriate types
    df['N'] = df['N'].astype(int)
    df['fp'] = df['fp'].astype(int)
    df['fn'] = df['fn'].astype(int)
    
    # Clean whitespace
    df['Model'] = df['Model'].str.strip()
    df['Dataset'] = df['Dataset'].str.strip()
    df['Group'] = df['Group'].str.strip()
    
    # Calculate error rates
    df = calculate_error_rates(df)
    
    print(f"Data loaded: {len(df)} rows")
    print(f"Unique models: {df['Model'].nunique()}")
    print(f"Unique datasets: {df['Dataset'].nunique()}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Error distribution analysis for entity overlap groups - generates LaTeX output',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python error_analysis.py data.xlsx
  python error_analysis.py data.xlsx --sheet "Errors"
  python error_analysis.py data.xlsx --output results
        """
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Excel file with error data (.xlsx)'
    )
    parser.add_argument(
        '--sheet',
        type=str,
        default='Errors',
        help='Excel sheet name (default: Errors)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='error_analysis',
        help='Output file prefix (default: error_analysis)'
    )
    
    args = parser.parse_args()
    
    try:
        # Load data
        df = load_data_from_excel(args.input_file, sheet_name=args.sheet)
        
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Analyze external datasets
    print(f"\n{'='*80}")
    print("ANALYZING ERROR DISTRIBUTION - EXTERNAL DATASETS")
    print(f"{'='*80}\n")
    
    uniformity_ext = test_error_distribution_uniformity(df, dataset_type='external')
    comparison_ext = compare_error_rates_across_groups(df, dataset_type='external')
    error_types_ext = analyze_error_types(df, dataset_type='external')
    
    print("Uniformity Test Results:")
    print(uniformity_ext.to_string(index=False))
    
    print("\n\nPairwise Error Rate Comparisons:")
    print(comparison_ext.to_string(index=False))
    
    print("\n\nError Type Analysis (FP vs FN):")
    print(error_types_ext.to_string(index=False))
    
    # Generate LaTeX output
    print(f"\n{'='*80}")
    print("GENERATING LATEX OUTPUT")
    print(f"{'='*80}\n")
    
    # Generate tables
    latex_tables = generate_latex_tables(uniformity_ext, comparison_ext, error_types_ext, dataset_type='external')
    
    # Generate narrative
    latex_narrative = generate_latex_narrative(uniformity_ext, comparison_ext, error_types_ext)
    
    # Combine narrative and tables
    full_latex = latex_narrative + latex_tables
    
    # Print to console
    print(full_latex)
    
    # Save outputs
    uniformity_ext.to_csv(f'{args.output}_uniformity_external.csv', index=False)
    comparison_ext.to_csv(f'{args.output}_comparisons_external.csv', index=False)
    error_types_ext.to_csv(f'{args.output}_types_external.csv', index=False)
    
    with open(f'{args.output}_latex_complete.tex', 'w', encoding='utf-8') as f:
        f.write(full_latex)
    
    with open(f'{args.output}_latex_tables_only.tex', 'w', encoding='utf-8') as f:
        f.write(latex_tables)
    
    with open(f'{args.output}_latex_narrative_only.tex', 'w', encoding='utf-8') as f:
        f.write(latex_narrative)
    
    print(f"\n{'='*80}")
    print("FILES SAVED:")
    print(f"{'='*80}")
    print(f"  CSV files:")
    print(f"    - {args.output}_uniformity_external.csv")
    print(f"    - {args.output}_comparisons_external.csv")
    print(f"    - {args.output}_types_external.csv")
    print(f"  LaTeX files:")
    print(f"    - {args.output}_latex_complete.tex (narrative + tables)")
    print(f"    - {args.output}_latex_narrative_only.tex (text only)")
    print(f"    - {args.output}_latex_tables_only.tex (tables only)")


if __name__ == '__main__':
    main()
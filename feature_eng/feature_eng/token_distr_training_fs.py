import pickle
import argparse
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import tiktoken

"""
Script to analyze token distribution in different dataset components.
Computes token statistics for SMILES, organisms and genes.
"""

class TokenDistributionAnalyzer:
    def __init__(self, drugbank_pickle, dataset_path, encoding_name="cl100k_base"):
        """
        Initialize the analyzer with the dataset.
        
        Args:
            drugbank_pickle (str): path to drugbank pickle file
            dataset_path (str): path to dataset pickle file
            encoding_name (str): tiktoken encoding name (cl100k_base for modern OpenAI models)
        """
        self.drugbank_pickle = drugbank_pickle
        self.dataset_path = dataset_path
        self.encoding = tiktoken.get_encoding(encoding_name)
        
        # Load and preprocess the dataset
        self.raw_dataset = self.load_pickle(dataset_path)
        print(f"Raw dataset type: {type(self.raw_dataset)}")
        if isinstance(self.raw_dataset, dict):
            print(f"Raw dataset keys: {self.raw_dataset.keys()}")
        elif isinstance(self.raw_dataset, list):
            print(f"Raw dataset length: {len(self.raw_dataset)}")
            if len(self.raw_dataset) > 0:
                print(f"First item type: {type(self.raw_dataset[0])}")
                print(f"First item: {self.raw_dataset[0]}")
        
        self.preproc_dataset()
        
    def load_pickle(self, path):
        """Load a pickle file."""
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def has_approved_group(self, d):
        """Check if drug is approved or experimental, but not illicit or withdrawn."""
        gr = d['groups']
        approved_or_experimental = False
        is_withdrawn_or_illicit = False
        for g in gr:
            if g in (1, 2): 
                approved_or_experimental = True
            elif g in (0, 3):
                is_withdrawn_or_illicit = True
                break
        return approved_or_experimental and not is_withdrawn_or_illicit
    
    def get_human_targets(self, drug):
        """Extract target genes from the drug."""
        human_genes = []
        if 'targets' in drug:
            for target in drug['targets']:
                for polypeptide in target.get('polypeptides', []):
                    gene_name = polypeptide.get('gene_name')
                    if gene_name:
                        human_genes.append(gene_name)
        return human_genes
    
    def get_organisms(self, drugs):
        """Extract target organisms."""
        drug_org_dict = {} 
        for drug in drugs:
            drug_id = drug['drugbank_id']
            drug_org_dict[drug_id] = ''
            if 'targets' in drug:
                for target in drug['targets']:
                    drug_org_dict[drug_id] = target['organism']
        return drug_org_dict
    
    def normalize_value(self, value):
        """
        Normalize a value by replacing empty or invalid values with 'ABSENT'.
        
        Args:
            value: The value to normalize
            
        Returns:
            str: The normalized value or 'ABSENT' if empty/invalid
        """
        if value is None:
            return "ABSENT"
        if isinstance(value, float) and (np.isnan(value) or value == 0.0):
            return "ABSENT"
        if isinstance(value, str):
            if not value or value.strip() == '':
                return "ABSENT"
            return value
        if isinstance(value, list):
            if not value:
                return "ABSENT"
            # For gene lists, return as is (will be joined later)
            return value
        # For any other type, convert to string
        str_value = str(value).strip()
        return str_value if str_value else "ABSENT"
    
    def preproc_dataset(self):
        """Preprocess the dataset following the same logic as the original script."""
        drugs = self.load_pickle(self.drugbank_pickle)
        
        # Normalize SMILES
        for drug in drugs:
            smile = drug.get('calc_prop_smiles', '')
            if not smile or isinstance(smile, float):
                smile = ''
            drug['calc_prop_smiles'] = smile
        
        # Filter drugs
        drugs = [
            {key: drug[key] for key in ['drugbank_id', 'name', 'targets', 'calc_prop_smiles', 'drug_interactions'] if key in drug}
            for drug in drugs if self.has_approved_group(drug)
        ]
        
        print(f"Filtered drugs: {len(drugs)}")
        
        # Remove drugs without targets
        drugs = [drug for drug in drugs if self.get_human_targets(drug)]
        
        print(f"Drugs with targets: {len(drugs)}")
        
        drug_org_dict = self.get_organisms(drugs)
        drug_dict = {drug['drugbank_id']: drug for drug in drugs}
        
        res = []
        
        # Handle both dict and list formats
        dataset_list = []
        
        if isinstance(self.raw_dataset, dict):
            # If it's a dictionary, try common keys
            for key in ['train', 'test', 'val', 'validation', 'data']:
                if key in self.raw_dataset:
                    if isinstance(self.raw_dataset[key], list):
                        dataset_list.extend(self.raw_dataset[key])
                        print(f"Found {len(self.raw_dataset[key])} items in '{key}'")
            
            # If still empty, maybe all values are lists
            if not dataset_list:
                for key, value in self.raw_dataset.items():
                    if isinstance(value, list):
                        dataset_list.extend(value)
                        print(f"Found {len(value)} items in '{key}'")
        elif isinstance(self.raw_dataset, list):
            dataset_list = self.raw_dataset
            print(f"Dataset is a list with {len(dataset_list)} items")
        
        print(f"Total items to process: {len(dataset_list)}")
        
        skipped = 0
        for idx, item in enumerate(dataset_list):
            # Handle both tuple and dict formats
            if isinstance(item, (list, tuple)):
                if len(item) < 2:
                    skipped += 1
                    continue
                drug1 = item[0]
                drug2 = item[1]
                target = item[-1] if len(item) > 2 else None
            elif isinstance(item, dict):
                drug1 = item.get('drug1')
                drug2 = item.get('drug2')
                target = item.get('target')
                if drug1 is None or drug2 is None:
                    skipped += 1
                    continue
            else:
                print(f"Warning: Unknown item format at index {idx}: {type(item)}")
                skipped += 1
                continue
            
            if drug1 not in drug_dict:
                skipped += 1
                if idx < 5:  # Only print first few
                    print(f"Drug1 '{drug1}' not found in drug_dict")
                continue
                
            if drug2 not in drug_dict:
                skipped += 1
                if idx < 5:  # Only print first few
                    print(f"Drug2 '{drug2}' not found in drug_dict")
                continue
                
            cdrug1 = drug_dict[drug1]
            cdrug2 = drug_dict[drug2]
            
            # Get raw values
            smiles1 = cdrug1['calc_prop_smiles']
            smiles2 = cdrug2['calc_prop_smiles']
            genes1 = self.get_human_targets(cdrug1)
            genes2 = self.get_human_targets(cdrug2)
            org1 = drug_org_dict[drug1]
            org2 = drug_org_dict[drug2]
            
            # Normalize all values
            smiles1 = self.normalize_value(smiles1)
            smiles2 = self.normalize_value(smiles2)
            org1 = self.normalize_value(org1)
            org2 = self.normalize_value(org2)
            
            # For genes, normalize the list itself first
            genes1 = self.normalize_value(genes1)
            genes2 = self.normalize_value(genes2)
            
            # If genes are not ABSENT, they're lists, keep them as is
            # If they are ABSENT, convert to empty list for consistency
            if genes1 == "ABSENT":
                genes1 = []
            if genes2 == "ABSENT":
                genes2 = []
            
            res.append({
                "drug1": drug1,
                "drug2": drug2,
                "drug_name1": cdrug1['name'],
                "drug_name2": cdrug2['name'],
                "smiles1": smiles1,
                "smiles2": smiles2,
                "genes1": genes1,
                "genes2": genes2,
                "org1": org1,
                "org2": org2,
                "target": target
            })
        
        print(f"Skipped items: {skipped}")
        self.dataset = res
        print(f"Dataset preprocessed: {len(self.dataset)} entries")
        
        if len(self.dataset) == 0:
            print("\nERROR: No valid entries found in dataset!")
            print("Please check:")
            print("1. The input pickle file format")
            print("2. Drug IDs match between dataset and drugbank")
            print("3. Drugs pass the filtering criteria")
            exit(1)
    
    def count_tokens(self, text):
        """Count tokens in a text using tiktoken."""
        if not text or len(str(text).strip()) == 0:
            return 0
        return len(self.encoding.encode(str(text)))
 
    def analyze_token_distribution(self):
        """Analyze token distribution for each component."""
        
        # Initialize lists to collect counts
        token_counts = {
            'smiles1': [],
            'smiles2': [],
            'org1': [],
            'org2': [],
            'genes1': [],
            'genes2': [],
        }
        
        print("Analyzing token distribution...")
        
        for item in self.dataset:
            # SMILES - skip if ABSENT
            smiles1 = item['smiles1']
            smiles2 = item['smiles2']
            if smiles1 != "ABSENT":
                token_counts['smiles1'].append(self.count_tokens(smiles1))
            if smiles2 != "ABSENT":
                token_counts['smiles2'].append(self.count_tokens(smiles2))
            
            # Organisms - skip if ABSENT
            org1 = item['org1']
            org2 = item['org2']
            if org1 != "ABSENT":
                token_counts['org1'].append(self.count_tokens(org1))
            if org2 != "ABSENT":
                token_counts['org2'].append(self.count_tokens(org2))
            
            # Genes - skip if empty list
            if item['genes1']:  # non-empty list
                genes1_str = "|".join(item['genes1'])
                token_counts['genes1'].append(self.count_tokens(genes1_str))
            if item['genes2']:  # non-empty list
                genes2_str = "|".join(item['genes2'])
                token_counts['genes2'].append(self.count_tokens(genes2_str))
        
        return token_counts

    def print_statistics(self, token_counts):
        """Print descriptive statistics for each component."""
        
        print("\n" + "="*80)
        print("TOKEN DISTRIBUTION BY COMPONENT")
        print("="*80)
        
        for component, counts in token_counts.items():
            if len(counts) == 0:
                print(f"\n{component.upper()}: No data")
                continue
                
            counts_array = np.array(counts)
            
            print(f"\n{component.upper()}:")
            print(f"  Count:       {len(counts)}")
            print(f"  Min:         {np.min(counts_array)}")
            print(f"  Max:         {np.max(counts_array)}")
            print(f"  Mean:        {np.mean(counts_array):.2f}")
            print(f"  Median:      {np.median(counts_array):.2f}")
            print(f"  Std Dev:     {np.std(counts_array):.2f}")
            print(f"  25th %ile:   {np.percentile(counts_array, 25):.2f}")
            print(f"  75th %ile:   {np.percentile(counts_array, 75):.2f}")
            print(f"  95th %ile:   {np.percentile(counts_array, 95):.2f}")
            print(f"  99th %ile:   {np.percentile(counts_array, 99):.2f}")
        
        # Combined statistics
        if len(self.dataset) > 0:
            print(f"\n{'COMBINED STATISTICS'.upper()}:")
            all_smiles = token_counts['smiles1'] + token_counts['smiles2']
            all_orgs = token_counts['org1'] + token_counts['org2']
            all_genes = token_counts['genes1'] + token_counts['genes2']
            
            print(f"\n  All SMILES:")
            print(f"    Count: {len(all_smiles)}, Mean: {np.mean(all_smiles):.2f}, Median: {np.median(all_smiles):.2f}, Std: {np.std(all_smiles):.2f}")
            
            print(f"\n  All Organisms:")
            print(f"    Count: {len(all_orgs)}, Mean: {np.mean(all_orgs):.2f}, Median: {np.median(all_orgs):.2f}, Std: {np.std(all_orgs):.2f}")
            
            print(f"\n  All Genes:")
            print(f"    Count: {len(all_genes)}, Mean: {np.mean(all_genes):.2f}, Median: {np.median(all_genes):.2f}, Std: {np.std(all_genes):.2f}")
            
            # Total tokens per example (only counting present values)
            total_per_example = []
            for i in range(len(self.dataset)):
                total = 0
                # Only add if index exists in the token_counts (meaning it wasn't ABSENT)
                # We need to track which examples contributed to which component
                # This is tricky - we need to recalculate
                item = self.dataset[i]
                
                if item['smiles1'] != "ABSENT":
                    total += self.count_tokens(item['smiles1'])
                if item['smiles2'] != "ABSENT":
                    total += self.count_tokens(item['smiles2'])
                if item['org1'] != "ABSENT":
                    total += self.count_tokens(item['org1'])
                if item['org2'] != "ABSENT":
                    total += self.count_tokens(item['org2'])
                if item['genes1']:
                    total += self.count_tokens("|".join(item['genes1']))
                if item['genes2']:
                    total += self.count_tokens("|".join(item['genes2']))
                
                total_per_example.append(total)
            
            print(f"\n  Total tokens per example:")
            print(f"    Mean: {np.mean(total_per_example):.2f}")
            print(f"    Median: {np.median(total_per_example):.2f}")
            print(f"    Min: {np.min(total_per_example)}")
            print(f"    Max: {np.max(total_per_example)}")

    def save_to_csv(self, token_counts, output_path):
        """Save statistics to CSV files."""
        
        if len(self.dataset) == 0:
            print("No data to save!")
            return
        
        # Create DataFrame with detailed data per example
        data_rows = []
        for i, item in enumerate(self.dataset):
            row = {}
            
            # Only include token counts if not ABSENT
            row['smiles1_tokens'] = self.count_tokens(item['smiles1']) if item['smiles1'] != "ABSENT" else None
            row['smiles2_tokens'] = self.count_tokens(item['smiles2']) if item['smiles2'] != "ABSENT" else None
            row['org1_tokens'] = self.count_tokens(item['org1']) if item['org1'] != "ABSENT" else None
            row['org2_tokens'] = self.count_tokens(item['org2']) if item['org2'] != "ABSENT" else None
            row['genes1_tokens'] = self.count_tokens("|".join(item['genes1'])) if item['genes1'] else None
            row['genes2_tokens'] = self.count_tokens("|".join(item['genes2'])) if item['genes2'] else None
            
            # Calculate total (sum of non-None values)
            row['total_tokens'] = sum(v for v in row.values() if v is not None)
            
            data_rows.append(row)
        
        df = pd.DataFrame(data_rows)
        
        # Save detailed data
        df.to_csv(output_path, index=False)
        print(f"\nDetailed data saved to: {output_path}")
        
        # Save summary statistics (excluding None/NaN values)
        summary_path = output_path.replace('.csv', '_summary.csv')
        summary_data = []
        
        for col in df.columns:
            # Remove NaN values for statistics
            valid_data = df[col].dropna()
            
            if len(valid_data) > 0:
                summary_data.append({
                    'component': col,
                    'count': len(valid_data),
                    'min': valid_data.min(),
                    'max': valid_data.max(),
                    'mean': valid_data.mean(),
                    'median': valid_data.median(),
                    'std': valid_data.std(),
                    'p25': valid_data.quantile(0.25),
                    'p75': valid_data.quantile(0.75),
                    'p95': valid_data.quantile(0.95),
                    'p99': valid_data.quantile(0.99)
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary statistics saved to: {summary_path}")


    def plot_distributions(self, token_counts, output_path=None):
        """Create distribution plots for tokens."""
        
        if len(self.dataset) == 0:
            print("No data to plot!")
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Token Distribution by Component', fontsize=16)
        
        components = ['smiles1', 'smiles2', 'org1', 'org2', 'genes1', 'genes2']
        
        for idx, component in enumerate(components):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            data = token_counts[component]
            ax.hist(data, bins=50, edgecolor='black', alpha=0.7)
            ax.set_title(f'{component.upper()}')
            ax.set_xlabel('Number of Tokens')
            ax.set_ylabel('Frequency')
            ax.axvline(np.mean(data), color='red', linestyle='--', label=f'Mean: {np.mean(data):.1f}')
            ax.axvline(np.median(data), color='green', linestyle='--', label=f'Median: {np.median(data):.1f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze token distribution in dataset components"
    )
    parser.add_argument('drugbank_pickle', type=str, 
                       help='Path to the pickle file containing the drugbank dataset')
    parser.add_argument('input_pickle', type=str, 
                       help='Path to the pickle file containing the dataset to analyze')
    parser.add_argument('--output_csv', type=str, default='token_distribution.csv',
                       help='Path to output CSV file (default: token_distribution.csv)')
    parser.add_argument('--output_plot', type=str, default='token_distribution.png',
                       help='Path to output PNG file with plots (default: token_distribution.png)')
    parser.add_argument('--encoding', type=str, default='cl100k_base',
                       help='Tiktoken encoding to use (default: cl100k_base for OpenAI models)')
    
    args = parser.parse_args()
    
    # Create the analyzer
    analyzer = TokenDistributionAnalyzer(
        args.drugbank_pickle,
        args.input_pickle,
        args.encoding
    )
    
    # Analyze the distribution
    token_counts = analyzer.analyze_token_distribution()
    
    # Print statistics
    analyzer.print_statistics(token_counts)
    
    # Save results
    analyzer.save_to_csv(token_counts, args.output_csv)
    analyzer.plot_distributions(token_counts, args.output_plot)
    
    print("\n" + "="*80)
    print("Analysis completed!")
    print("="*80)
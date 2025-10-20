import argparse
import json
import pandas as pd
from collections import defaultdict, Counter
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
import sys
from io import StringIO
import pickle

def loadPickle(input_pickle):
    with open(input_pickle, 'rb') as f:
        r = pickle.load(f)
    return r

def load_results(file_path: str) -> List[Dict]:
    return loadPickle(file_path)

def calculate_basic_metrics(results: List[Dict]) -> Dict:
    """Calculate basic performance metrics"""
    tp = sum(1 for r in results if r['target'] == 1 and r['new_target'] == 1)
    tn = sum(1 for r in results if r['target'] == 0 and r['new_target'] == 0)
    fp = sum(1 for r in results if r['target'] == 0 and r['new_target'] == 1)
    fn = sum(1 for r in results if r['target'] == 1 and r['new_target'] == 0)
    
    total = len(results)
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'sensitivity': sensitivity,
        'f1': f1,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'total': total
    }

def analyze_smiles_complexity(results: List[Dict]) -> Dict:
    """Analyze errors by SMILES complexity (length as proxy)"""
    analysis = {
        'false_positives': {'smiles1_len': [], 'smiles2_len': [], 'combined_len': []},
        'false_negatives': {'smiles1_len': [], 'smiles2_len': [], 'combined_len': []},
        'true_positives': {'smiles1_len': [], 'smiles2_len': [], 'combined_len': []},
        'true_negatives': {'smiles1_len': [], 'smiles2_len': [], 'combined_len': []}
    }
    
    for r in results:
        smiles1_len = len(r['smiles1']) if r['smiles1'] else 0
        smiles2_len = len(r['smiles2']) if r['smiles2'] else 0
        combined_len = smiles1_len + smiles2_len
        
        if r['target'] == 0 and r['new_target'] == 1:  # False Positive
            category = 'false_positives'
        elif r['target'] == 1 and r['new_target'] == 0:  # False Negative
            category = 'false_negatives'
        elif r['target'] == 1 and r['new_target'] == 1:  # True Positive
            category = 'true_positives'
        else:  # True Negative
            category = 'true_negatives'
            
        analysis[category]['smiles1_len'].append(smiles1_len)
        analysis[category]['smiles2_len'].append(smiles2_len)
        analysis[category]['combined_len'].append(combined_len)
    
    # Calculate statistics
    stats = {}
    for category, data in analysis.items():
        stats[category] = {
            'count': len(data['combined_len']),
            'avg_smiles1_len': np.mean(data['smiles1_len']) if data['smiles1_len'] else 0,
            'avg_smiles2_len': np.mean(data['smiles2_len']) if data['smiles2_len'] else 0,
            'avg_combined_len': np.mean(data['combined_len']) if data['combined_len'] else 0,
            'std_combined_len': np.std(data['combined_len']) if data['combined_len'] else 0
        }
    
    return stats

def analyze_gene_complexity(results: List[Dict]) -> Dict:
    """Analyze errors by gene target complexity"""
    analysis = {
        'false_positives': {'genes1_count': [], 'genes2_count': [], 'combined_count': []},
        'false_negatives': {'genes1_count': [], 'genes2_count': [], 'combined_count': []},
        'true_positives': {'genes1_count': [], 'genes2_count': [], 'combined_count': []},
        'true_negatives': {'genes1_count': [], 'genes2_count': [], 'combined_count': []}
    }
    
    for r in results:
        genes1_count = len(r['genes1']) if r['genes1'] else 0
        genes2_count = len(r['genes2']) if r['genes2'] else 0
        combined_count = genes1_count + genes2_count
        
        if r['target'] == 0 and r['new_target'] == 1:  # False Positive
            category = 'false_positives'
        elif r['target'] == 1 and r['new_target'] == 0:  # False Negative
            category = 'false_negatives'
        elif r['target'] == 1 and r['new_target'] == 1:  # True Positive
            category = 'true_positives'
        else:  # True Negative
            category = 'true_negatives'
            
        analysis[category]['genes1_count'].append(genes1_count)
        analysis[category]['genes2_count'].append(genes2_count)
        analysis[category]['combined_count'].append(combined_count)
    
    # Calculate statistics
    stats = {}
    for category, data in analysis.items():
        stats[category] = {
            'count': len(data['combined_count']),
            'avg_genes1_count': np.mean(data['genes1_count']) if data['genes1_count'] else 0,
            'avg_genes2_count': np.mean(data['genes2_count']) if data['genes2_count'] else 0,
            'avg_combined_count': np.mean(data['combined_count']) if data['combined_count'] else 0,
            'std_combined_count': np.std(data['combined_count']) if data['combined_count'] else 0
        }
    
    return stats

def analyze_organism_patterns(results: List[Dict]) -> Dict:
    """Analyze errors by organism patterns"""
    organism_errors = {
        'false_positives': defaultdict(int),
        'false_negatives': defaultdict(int),
        'true_positives': defaultdict(int),
        'true_negatives': defaultdict(int)
    }
    
    for r in results:
        org_pair = f"{r['org1']} - {r['org2']}"
        
        if r['target'] == 0 and r['new_target'] == 1:  # False Positive
            organism_errors['false_positives'][org_pair] += 1
        elif r['target'] == 1 and r['new_target'] == 0:  # False Negative
            organism_errors['false_negatives'][org_pair] += 1
        elif r['target'] == 1 and r['new_target'] == 1:  # True Positive
            organism_errors['true_positives'][org_pair] += 1
        else:  # True Negative
            organism_errors['true_negatives'][org_pair] += 1
    
    return organism_errors

def find_problematic_drugs(results: List[Dict], top_n: int = 10) -> Dict:
    """Find drugs that are most frequently misclassified"""
    drug_errors = defaultdict(lambda: {'fp': 0, 'fn': 0, 'total_appearances': 0})
    
    for r in results:
        drug1_name = r['drug_name1']
        drug2_name = r['drug_name2']
        
        # Count total appearances
        drug_errors[drug1_name]['total_appearances'] += 1
        drug_errors[drug2_name]['total_appearances'] += 1
        
        # Count errors
        if r['target'] == 0 and r['new_target'] == 1:  # False Positive
            drug_errors[drug1_name]['fp'] += 1
            drug_errors[drug2_name]['fp'] += 1
        elif r['target'] == 1 and r['new_target'] == 0:  # False Negative
            drug_errors[drug1_name]['fn'] += 1
            drug_errors[drug2_name]['fn'] += 1
    
    # Calculate error rates
    drug_error_rates = {}
    for drug, counts in drug_errors.items():
        if counts['total_appearances'] > 0:
            total_errors = counts['fp'] + counts['fn']
            error_rate = total_errors / counts['total_appearances']
            drug_error_rates[drug] = {
                'error_rate': error_rate,
                'total_errors': total_errors,
                'false_positives': counts['fp'],
                'false_negatives': counts['fn'],
                'total_appearances': counts['total_appearances']
            }
    
    # Get top problematic drugs
    top_problematic = sorted(drug_error_rates.items(), 
                           key=lambda x: x[1]['error_rate'], 
                           reverse=True)[:top_n]
    
    return dict(top_problematic)

def get_specific_examples(results: List[Dict], error_type: str = 'false_positives', n_examples: int = 5) -> List[Dict]:
    """Get specific examples of errors"""
    examples = []
    
    for r in results:
        if error_type == 'false_positives' and r['target'] == 0 and r['new_target'] == 1:
            examples.append(r)
        elif error_type == 'false_negatives' and r['target'] == 1 and r['new_target'] == 0:
            examples.append(r)
        
        if len(examples) >= n_examples:
            break
    
    return examples

def generate_analysis_report(model_name: str, dataset_name: str, results: List[Dict], args: argparse.Namespace) -> str:
    """Generate comprehensive analysis report as string"""
    report = StringIO()
    
    # Header
    report.write("="*60 + "\n")
    report.write("QUALITATIVE ERROR ANALYSIS REPORT\n")
    report.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.write(f"Model: {model_name}\n")
    report.write(f"Dataset: {dataset_name}\n")
    report.write("="*60 + "\n")
    
    # Basic metrics
    metrics = calculate_basic_metrics(results)
    report.write("\n1. BASIC PERFORMANCE METRICS:\n")
    report.write(f"   Accuracy: {metrics['accuracy']:.3f}\n")
    report.write(f"   Precision: {metrics['precision']:.3f}\n")
    report.write(f"   Sensitivity: {metrics['sensitivity']:.3f}\n")
    report.write(f"   F1-Score: {metrics['f1']:.3f}\n")
    report.write(f"   Total samples: {metrics['total']}\n")
    report.write(f"   TP: {metrics['tp']}, TN: {metrics['tn']}, FP: {metrics['fp']}, FN: {metrics['fn']}\n")
    
    # SMILES complexity analysis
    if args.analyze_smiles:
        report.write("\n2. SMILES COMPLEXITY ANALYSIS:\n")
        smiles_stats = analyze_smiles_complexity(results)
        for category, stats in smiles_stats.items():
            if stats['count'] > 0:
                report.write(f"   {category.replace('_', ' ').title()}: {stats['count']} cases\n")
                report.write(f"     Avg combined SMILES length: {stats['avg_combined_len']:.1f} ± {stats['std_combined_len']:.1f}\n")
    
    # Gene complexity analysis
    if args.analyze_genes:
        report.write("\n3. GENE TARGET COMPLEXITY ANALYSIS:\n")
        gene_stats = analyze_gene_complexity(results)
        for category, stats in gene_stats.items():
            if stats['count'] > 0:
                report.write(f"   {category.replace('_', ' ').title()}: {stats['count']} cases\n")
                report.write(f"     Avg combined gene targets: {stats['avg_combined_count']:.1f} ± {stats['std_combined_count']:.1f}\n")
    
    # Organism analysis
    if args.analyze_organisms:
        report.write("\n4. ORGANISM PATTERN ANALYSIS:\n")
        org_patterns = analyze_organism_patterns(results)
        for category in ['false_positives', 'false_negatives']:
            if org_patterns[category]:
                report.write(f"   Top {category.replace('_', ' ')} organism pairs:\n")
                top_orgs = sorted(org_patterns[category].items(), key=lambda x: x[1], reverse=True)[:5]
                for org_pair, count in top_orgs:
                    report.write(f"     {org_pair}: {count} cases\n")
    
    # Problematic drugs
    if args.analyze_drugs:
        report.write("\n5. MOST PROBLEMATIC DRUGS:\n")
        problematic = find_problematic_drugs(results, args.top_drugs)
        for drug, stats in problematic.items():
            if stats['total_errors'] > 0:
                report.write(f"   {drug}: {stats['error_rate']:.3f} error rate\n")
                report.write(f"     ({stats['total_errors']}/{stats['total_appearances']} errors: "
                           f"{stats['false_positives']} FP, {stats['false_negatives']} FN)\n")
    
    # Specific examples
    if args.show_examples:
        report.write("\n6. SPECIFIC ERROR EXAMPLES:\n")
        
        # False positives
        fp_examples = get_specific_examples(results, 'false_positives', args.n_examples)
        if fp_examples:
            report.write("   False Positives (predicted interaction, but no real interaction):\n")
            for i, example in enumerate(fp_examples, 1):
                report.write(f"     {i}. {example['drug_name1']} + {example['drug_name2']}\n")
                report.write(f"        Organisms: {example['org1']} - {example['org2']}\n")
                genes1_count = len(example['genes1']) if example['genes1'] else 0
                genes2_count = len(example['genes2']) if example['genes2'] else 0
                report.write(f"        Gene targets: {genes1_count} - {genes2_count}\n")
        
        # False negatives
        fn_examples = get_specific_examples(results, 'false_negatives', args.n_examples)
        if fn_examples:
            report.write("   False Negatives (missed real interactions):\n")
            for i, example in enumerate(fn_examples, 1):
                report.write(f"     {i}. {example['drug_name1']} + {example['drug_name2']}\n")
                report.write(f"        Organisms: {example['org1']} - {example['org2']}\n")
                genes1_count = len(example['genes1']) if example['genes1'] else 0
                genes2_count = len(example['genes2']) if example['genes2'] else 0
                report.write(f"        Gene targets: {genes1_count} - {genes2_count}\n")
    
    # Summary insights
    report.write("\n7. SUMMARY INSIGHTS:\n")
    if metrics['fp'] > metrics['fn']:
        report.write("   - Model tends to over-predict interactions (more false positives than false negatives)\n")
    elif metrics['fn'] > metrics['fp']:
        report.write("   - Model tends to under-predict interactions (more false negatives than false positives)\n")
    else:
        report.write("   - Model shows balanced error distribution\n")
    
    if metrics['sensitivity'] > metrics['precision']:
        report.write("   - High sensitivity but lower precision - good at catching interactions but with some false alarms\n")
    elif metrics['precision'] > metrics['sensitivity']:
        report.write("   - High precision but lower sensitivity - conservative approach, might miss some interactions\n")
    
    report.write(f"\nReport generation completed successfully.\n")
    report.write("="*60 + "\n")
    
    return report.getvalue()

def print_analysis_report(model_name: str, dataset_name: str, results: List[Dict], args: argparse.Namespace):
    """Print comprehensive analysis report"""
    report_content = generate_analysis_report(model_name, dataset_name, results, args)
    print(report_content)

def save_report_to_file(report_content: str, file_path: str, model_name: str, dataset_name: str):
    """Save report to file with proper formatting"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"\n✅ Report saved successfully to: {file_path}")
    except Exception as e:
        print(f"\n❌ Error saving report to {file_path}: {e}")

def save_structured_data(model_name: str, dataset_name: str, results: List[Dict], 
                        args: argparse.Namespace, base_path: str):
    """Save structured data as JSON for further analysis"""
    structured_data = {
        'metadata': {
            'model_name': model_name,
            'dataset_name': dataset_name,
            'analysis_timestamp': datetime.now().isoformat(),
            'total_samples': len(results),
            'analyses_performed': {
                'smiles': args.analyze_smiles,
                'genes': args.analyze_genes,
                'organisms': args.analyze_organisms,
                'drugs': args.analyze_drugs,
                'examples': args.show_examples
            }
        },
        'basic_metrics': calculate_basic_metrics(results)
    }
    
    # Add detailed analyses if requested
    if args.analyze_smiles:
        structured_data['smiles_analysis'] = analyze_smiles_complexity(results)
    
    if args.analyze_genes:
        structured_data['gene_analysis'] = analyze_gene_complexity(results)
    
    if args.analyze_organisms:
        structured_data['organism_analysis'] = analyze_organism_patterns(results)
    
    if args.analyze_drugs:
        structured_data['drug_analysis'] = find_problematic_drugs(results, args.top_drugs)
    
    if args.show_examples:
        structured_data['error_examples'] = {
            'false_positives': get_specific_examples(results, 'false_positives', args.n_examples),
            'false_negatives': get_specific_examples(results, 'false_negatives', args.n_examples)
        }
    
    # Save structured data
    json_path = base_path.replace('.txt', '.json') if base_path.endswith('.txt') else f"{base_path}.json"
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Structured data saved to: {json_path}")
    except Exception as e:
        print(f"❌ Error saving structured data to {json_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Qualitative Error Analysis for DDI Prediction Models')
    
    # Required arguments
    parser.add_argument('--results_file', type=str, required=True,
                       help='Path to JSON file containing results')
    parser.add_argument('--model_name', type=str, required=True,
                       help='Name of the model being analyzed')
    parser.add_argument('--dataset_name', type=str, required=True,
                       help='Name of the dataset being analyzed')
    
    # Analysis options
    parser.add_argument('--analyze_smiles', action='store_true',
                       help='Analyze SMILES complexity patterns')
    parser.add_argument('--analyze_genes', action='store_true',
                       help='Analyze gene target complexity patterns')
    parser.add_argument('--analyze_organisms', action='store_true',
                       help='Analyze organism patterns')
    parser.add_argument('--analyze_drugs', action='store_true',
                       help='Find most problematic drugs')
    parser.add_argument('--show_examples', action='store_true',
                       help='Show specific error examples')
    
    # Parameters
    parser.add_argument('--top_drugs', type=int, default=10,
                       help='Number of top problematic drugs to show')
    parser.add_argument('--n_examples', type=int, default=5,
                       help='Number of specific examples to show')
    
    # Output options
    parser.add_argument('--save_report', type=str,
                       help='Path to save detailed report (text file)')
    parser.add_argument('--save_structured', action='store_true',
                       help='Also save structured data as JSON')
    parser.add_argument('--all_analyses', action='store_true',
                       help='Run all available analyses')
    
    args = parser.parse_args()
    
    # If all_analyses is selected, enable all analysis types
    if args.all_analyses:
        args.analyze_smiles = True
        args.analyze_genes = True
        args.analyze_organisms = True
        args.analyze_drugs = True
        args.show_examples = True
    
    # Load results
    try:
        results = load_results(args.results_file)
        print(f"Loaded {len(results)} results from {args.results_file}")
    except Exception as e:
        print(f"Error loading results: {e}")
        return
    
    # Generate report content
    report_content = generate_analysis_report(args.model_name, args.dataset_name, results, args)
    
    # Print to console
    print(report_content)
    
    # Save report to file if requested
    if args.save_report:
        save_report_to_file(report_content, args.save_report, args.model_name, args.dataset_name)
        
        # Save structured data if requested
        if args.save_structured:
            save_structured_data(args.model_name, args.dataset_name, results, args, args.save_report)

if __name__ == "__main__":
    main()
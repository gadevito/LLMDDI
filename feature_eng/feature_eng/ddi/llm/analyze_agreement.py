import argparse
import pickle
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
from collections import defaultdict


def load_predictions(filepath: str) -> List[Dict]:
    """
    Carica file pickle con predizioni.
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def normalize_drug_name(name: str) -> str:
    """
    Normalizza il nome del farmaco per il matching.
    """
    if name is None:
        return ""
    # Converti in lowercase e rimuovi spazi extra
    return str(name).strip().lower()


def create_prediction_dict(predictions: List[Dict], filename: str) -> Dict[Tuple[str, str], Dict]:
    """
    Converte lista di predizioni in dictionary con chiave (drug1, drug2).
    """
    pred_dict = {}
    duplicates = []
    
    # Debug: mostra prime 3 entries
    print(f"\n  Debug - First 3 entries in {filename}:")
    for i, pred in enumerate(predictions[:3]):
        print(f"    [{i}] drug1='{pred['drug1']}', drug2='{pred['drug2']}', target={pred['target']}, new_target={pred['new_target']}")
    
    for pred in predictions:
        # Normalizza i nomi per il matching
        drug1_norm = normalize_drug_name(pred['drug1'])
        drug2_norm = normalize_drug_name(pred['drug2'])
        
        key = (drug1_norm, drug2_norm)
        
        if key in pred_dict:
            duplicates.append(key)
        
        pred_dict[key] = {
            'drug1_original': pred['drug1'],  # Salva anche il nome originale
            'drug2_original': pred['drug2'],
            'target': pred['target'],
            'new_target': pred['new_target']
        }
    
    if duplicates:
        print(f"  Warning: Found {len(duplicates)} duplicate drug pairs. Using last occurrence.")
    
    return pred_dict


def calculate_metrics(predictions: List[Dict]) -> Dict[str, float]:
    """
    Calcola accuracy, precision, sensitivity, F1 dalle predizioni.
    """
    y_true = [p['target'] for p in predictions]
    y_pred = [p['new_target'] for p in predictions]
    
    # Confusion matrix
    tp = sum((t == 1 and p == 1) for t, p in zip(y_true, y_pred))
    tn = sum((t == 0 and p == 0) for t, p in zip(y_true, y_pred))
    fp = sum((t == 0 and p == 1) for t, p in zip(y_true, y_pred))
    fn = sum((t == 1 and p == 0) for t, p in zip(y_true, y_pred))
    
    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'sensitivity': sensitivity,
        'f1': f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'total': total
    }


def calculate_pairwise_agreement(dict1: Dict, dict2: Dict, name1: str, name2: str) -> Tuple[float, int, List[Dict], int]:
    """
    Calcola agreement tra due dictionary di predizioni.
    """
    # Trova chiavi comuni
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    
    common_keys = keys1 & keys2
    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1
    
    # Debug dettagliato
    print(f"\n  Debug info:")
    print(f"    Keys in {name1}: {len(keys1)}")
    print(f"    Keys in {name2}: {len(keys2)}")
    print(f"    Common keys: {len(common_keys)}")
    
    # Mostra esempi di chiavi uniche
    if only_in_1:
        print(f"\n    Drug pairs ONLY in {name1}: {len(only_in_1)}")
        for i, key in enumerate(list(only_in_1)[:5]):
            print(f"      [{i}] '{key[0]}' + '{key[1]}'")
    
    if only_in_2:
        print(f"\n    Drug pairs ONLY in {name2}: {len(only_in_2)}")
        for i, key in enumerate(list(only_in_2)[:5]):
            print(f"      [{i}] '{key[0]}' + '{key[1]}'")
    
    # Mostra esempi di chiavi comuni
    if common_keys:
        print(f"\n    Common drug pairs (showing first 5):")
        for i, key in enumerate(list(common_keys)[:5]):
            print(f"      [{i}] '{key[0]}' + '{key[1]}'")
    
    # Verifica che ci siano chiavi comuni
    if not common_keys:
        raise ValueError(f"No common drug pairs found between {name1} and {name2}!")
    
    # Verifica che i target siano uguali per le chiavi comuni
    target_mismatches = []
    for key in list(common_keys)[:10]:  # Check primi 10
        if dict1[key]['target'] != dict2[key]['target']:
            target_mismatches.append(key)
    
    if target_mismatches:
        print(f"\n    ERROR: Ground truth (target) differs for {len(target_mismatches)} pairs!")
        for key in target_mismatches[:3]:
            print(f"      {key}: {dict1[key]['target']} vs {dict2[key]['target']}")
        raise ValueError("Ground truth mismatch! Files contain different datasets.")
    
    # Calcola agreement CONFRONTANDO I new_target
    n_agreements = 0
    disagreement_cases = []
    
    for key in sorted(common_keys):
        drug1_norm, drug2_norm = key
        ground_truth = dict1[key]['target']
        pred1 = dict1[key]['new_target']
        pred2 = dict2[key]['new_target']
        
        if pred1 == pred2:
            n_agreements += 1
        else:
            disagreement_cases.append({
                'drug1': dict1[key]['drug1_original'],  # Usa nomi originali per output
                'drug2': dict1[key]['drug2_original'],
                'ground_truth': ground_truth,
                'prediction_1': pred1,
                'prediction_2': pred2,
                'file_1': name1,
                'file_2': name2
            })
    
    n_total = len(common_keys)
    agreement_pct = (n_agreements / n_total) * 100
    n_disagreements = len(disagreement_cases)
    
    return agreement_pct, n_disagreements, disagreement_cases, n_total


def calculate_overall_agreement(original_files: List[str], perturbed_files: List[str]) -> Dict:
    """
    Calcola overall agreement tra multipli file original e perturbed.
    """
    print("\n" + "=" * 70)
    print("LOADING PREDICTIONS")
    print("=" * 70)
    
    # Carica tutti i file original
    original_dicts = {}
    original_metrics = {}
    for filepath in original_files:
        filename = Path(filepath).name
        print(f"\nLoading original: {filename}")
        preds = load_predictions(filepath)
        print(f"  → {len(preds)} raw predictions loaded")
        original_dicts[filename] = create_prediction_dict(preds, filename)
        original_metrics[filename] = calculate_metrics(preds)
        print(f"  → {len(original_dicts[filename])} unique drug pairs")
        metrics = original_metrics[filename]
        print(f"  → Metrics: Acc={metrics['accuracy']:.4f}, Prec={metrics['precision']:.4f}, "
              f"Sens={metrics['sensitivity']:.4f}, F1={metrics['f1']:.4f}")
    
    # Carica tutti i file perturbed
    perturbed_dicts = {}
    perturbed_metrics = {}
    for filepath in perturbed_files:
        filename = Path(filepath).name
        print(f"\nLoading perturbed: {filename}")
        preds = load_predictions(filepath)
        print(f"  → {len(preds)} raw predictions loaded")
        perturbed_dicts[filename] = create_prediction_dict(preds, filename)
        perturbed_metrics[filename] = calculate_metrics(preds)
        print(f"  → {len(perturbed_dicts[filename])} unique drug pairs")
        metrics = perturbed_metrics[filename]
        print(f"  → Metrics: Acc={metrics['accuracy']:.4f}, Prec={metrics['precision']:.4f}, "
              f"Sens={metrics['sensitivity']:.4f}, F1={metrics['f1']:.4f}")
    
    print("\n" + "=" * 70)
    print("CALCULATING AGREEMENTS")
    print("=" * 70)
    
    # Calcola agreement per ogni coppia (original, perturbed)
    all_agreements = []
    all_disagreements = []
    pairwise_results = []
    
    total_comparisons = 0
    total_disagreements = 0
    
    for orig_name, orig_dict in original_dicts.items():
        for pert_name, pert_dict in perturbed_dicts.items():
            print(f"\nComparing: {orig_name} vs {pert_name}")
            
            agreement_pct, n_disagreements, disagreement_cases, n_total = calculate_pairwise_agreement(
                orig_dict, pert_dict, orig_name, pert_name
            )
            
            print(f"\n  → Agreement: {agreement_pct:.3f}% ({n_total-n_disagreements}/{n_total})")
            
            all_agreements.append(agreement_pct)
            all_disagreements.extend(disagreement_cases)
            total_comparisons += n_total
            total_disagreements += n_disagreements
            
            pairwise_results.append({
                'original': orig_name,
                'perturbed': pert_name,
                'agreement_pct': agreement_pct,
                'n_disagreements': n_disagreements,
                'n_total': n_total,
                'disagreement_cases': disagreement_cases
            })
    
    # Calcola statistiche aggregate
    mean_agreement = np.mean(all_agreements)
    std_agreement = np.std(all_agreements)
    min_agreement = np.min(all_agreements)
    max_agreement = np.max(all_agreements)
    
    # Overall agreement (media pesata)
    overall_agreement = ((total_comparisons - total_disagreements) / total_comparisons) * 100 if total_comparisons > 0 else 0
    
    return {
        'pairwise_results': pairwise_results,
        'all_disagreements': all_disagreements,
        'original_metrics': original_metrics,
        'perturbed_metrics': perturbed_metrics,
        'statistics': {
            'mean_agreement': mean_agreement,
            'std_agreement': std_agreement,
            'min_agreement': min_agreement,
            'max_agreement': max_agreement,
            'overall_agreement': overall_agreement,
            'total_comparisons': total_comparisons,
            'total_disagreements': total_disagreements,
            'n_original_files': len(original_files),
            'n_perturbed_files': len(perturbed_files)
        }
    }


def calculate_within_group_agreement(files: List[str], group_name: str) -> Dict:
    """
    Calcola agreement tra file dello stesso gruppo.
    """
    if len(files) < 2:
        return None
    
    print(f"\n" + "=" * 70)
    print(f"WITHIN-GROUP AGREEMENT: {group_name}")
    print("=" * 70)
    
    # Carica tutti i file
    pred_dicts = {}
    for filepath in files:
        filename = Path(filepath).name
        preds = load_predictions(filepath)
        pred_dicts[filename] = create_prediction_dict(preds, filename)
    
    # Calcola agreement per ogni coppia
    from itertools import combinations
    agreements = []
    results = []
    
    for (name1, dict1), (name2, dict2) in combinations(pred_dicts.items(), 2):
        print(f"\n  Comparing: {name1} vs {name2}")
        
        agreement_pct, n_disagreements, disagreement_cases, n_total = calculate_pairwise_agreement(
            dict1, dict2, name1, name2
        )
        
        print(f"    → Agreement: {agreement_pct:.3f}% ({n_total-n_disagreements}/{n_total})")
        
        agreements.append(agreement_pct)
        results.append({
            'file1': name1,
            'file2': name2,
            'agreement_pct': agreement_pct,
            'n_disagreements': n_disagreements,
            'n_total': n_total
        })
    
    return {
        'results': results,
        'mean_agreement': np.mean(agreements),
        'std_agreement': np.std(agreements),
        'min_agreement': np.min(agreements),
        'max_agreement': np.max(agreements)
    }


def print_results(results: Dict, show_details: bool = False):
    """
    Stampa risultati in formato leggibile.
    """
    stats = results['statistics']
    
    print("\n" + "=" * 70)
    print("OVERALL AGREEMENT ANALYSIS")
    print("=" * 70)
    
    print(f"\nNumber of original files:   {stats['n_original_files']}")
    print(f"Number of perturbed files:  {stats['n_perturbed_files']}")
    print(f"Total pairwise comparisons: {len(results['pairwise_results'])}")
    print(f"Total predictions compared: {stats['total_comparisons']}")
    
    # Mostra metriche per file
    print(f"\n{'-' * 70}")
    print("PERFORMANCE METRICS BY FILE")
    print(f"{'-' * 70}")
    
    print("\nOriginal files:")
    for filename, metrics in results['original_metrics'].items():
        print(f"  {filename}:")
        print(f"    Acc={metrics['accuracy']:.4f}, Prec={metrics['precision']:.4f}, "
              f"Sens={metrics['sensitivity']:.4f}, F1={metrics['f1']:.4f}")
    
    print("\nPerturbed files:")
    for filename, metrics in results['perturbed_metrics'].items():
        print(f"  {filename}:")
        print(f"    Acc={metrics['accuracy']:.4f}, Prec={metrics['precision']:.4f}, "
              f"Sens={metrics['sensitivity']:.4f}, F1={metrics['f1']:.4f}")
    
    print(f"\n{'-' * 70}")
    print("AGREEMENT STATISTICS")
    print(f"{'-' * 70}")
    print(f"Overall agreement:          {stats['overall_agreement']:.3f}%")
    print(f"Mean pairwise agreement:    {stats['mean_agreement']:.3f}% ± {stats['std_agreement']:.3f}%")
    print(f"Range:                      [{stats['min_agreement']:.3f}%, {stats['max_agreement']:.3f}%]")
    
    print(f"\n{'-' * 70}")
    print("DISAGREEMENT STATISTICS")
    print(f"{'-' * 70}")
    print(f"Total disagreements:        {stats['total_disagreements']} ({(stats['total_disagreements']/stats['total_comparisons']*100):.3f}%)")
    
    # Conta coppie uniche che hanno almeno un disaccordo
    unique_disagreement_pairs = set((d['drug1'], d['drug2']) for d in results['all_disagreements'])
    print(f"Unique drug pairs with disagreements: {len(unique_disagreement_pairs)}")
    
    # Pairwise results table
    print(f"\n{'-' * 70}")
    print("PAIRWISE COMPARISON DETAILS")
    print(f"{'-' * 70}")
    print(f"{'Original':<35} {'Perturbed':<35} {'Agreement':<12} {'Disagr.'}")
    print(f"{'-' * 70}")
    
    for res in results['pairwise_results']:
        orig_short = res['original'][:33]
        pert_short = res['perturbed'][:33]
        print(f"{orig_short:<35} {pert_short:<35} {res['agreement_pct']:>6.3f}%      {res['n_disagreements']:>5}")
    
    # Disagreement details
    if stats['total_disagreements'] > 0:
        print(f"\n{'-' * 70}")
        print(f"DISAGREEMENT DETAILS")
        print(f"{'-' * 70}")
        
        # Raggruppa per coppia di farmaci
        disagreements_by_pair = defaultdict(list)
        for case in results['all_disagreements']:
            key = (case['drug1'], case['drug2'], case['ground_truth'])
            disagreements_by_pair[key].append(case)
        
        print(f"\nTotal disagreement cases: {len(results['all_disagreements'])}")
        print(f"Unique drug pairs: {len(disagreements_by_pair)}")
        
        if show_details:
            print(f"\nShowing first 20 unique drug pairs with disagreements:")
            for i, (key, cases) in enumerate(list(disagreements_by_pair.items())[:20]):
                drug1, drug2, gt = key
                print(f"\n[{i+1}] {drug1} + {drug2} (ground_truth={gt})")
                print(f"    Appears in {len(cases)} comparison(s):")
                for case in cases:
                    print(f"      {case['file_1']}: pred={case['prediction_1']}")
                    print(f"      {case['file_2']}: pred={case['prediction_2']}")
            
            if len(disagreements_by_pair) > 20:
                print(f"\n... and {len(disagreements_by_pair) - 20} more unique pairs")
        else:
            # Mostra solo primi 10 senza dettagli
            print(f"\nFirst 10 disagreement cases:")
            for i, case in enumerate(results['all_disagreements'][:10]):
                print(f"  [{i+1}] {case['drug1']} + {case['drug2']} (GT={case['ground_truth']})")
                print(f"      {case['file_1']}: {case['prediction_1']} → {case['file_2']}: {case['prediction_2']}")
            
            if stats['total_disagreements'] > 10:
                print(f"\n  Use --show-details to see all disagreements")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Calculate agreement percentage between multiple original and perturbed predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--original', type=str, nargs='+', required=True)
    parser.add_argument('--perturbed', type=str, nargs='+', required=True)
    parser.add_argument('--check-consistency', action='store_true')
    parser.add_argument('--show-details', action='store_true')
    parser.add_argument('--output', type=str, default=None)
    
    args = parser.parse_args()
    
    results = calculate_overall_agreement(args.original, args.perturbed)
    print_results(results, show_details=args.show_details)
    
    if args.check_consistency:
        orig_consistency = calculate_within_group_agreement(args.original, "ORIGINAL")
        if orig_consistency:
            print(f"\n{'='*70}")
            print(f"WITHIN-GROUP CONSISTENCY: ORIGINAL FILES")
            print(f"{'='*70}")
            print(f"Mean agreement: {orig_consistency['mean_agreement']:.3f}% ± {orig_consistency['std_agreement']:.3f}%")
        
        pert_consistency = calculate_within_group_agreement(args.perturbed, "PERTURBED")
        if pert_consistency:
            print(f"\n{'='*70}")
            print(f"WITHIN-GROUP CONSISTENCY: PERTURBED FILES")
            print(f"{'='*70}")
            print(f"Mean agreement: {pert_consistency['mean_agreement']:.3f}% ± {pert_consistency['std_agreement']:.3f}%")
    
    if args.output:
        with open(args.output, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nResults saved to: {args.output}")
    
    print("\n" + "=" * 70)
    print("SUMMARY FOR PAPER")
    print("=" * 70)
    stats = results['statistics']
    print(f"\nOverall agreement: {stats['overall_agreement']:.3f}%")
    print(f"Mean pairwise agreement: {stats['mean_agreement']:.3f}% ± {stats['std_agreement']:.3f}%")
    print(f"Total disagreements: {stats['total_disagreements']}/{stats['total_comparisons']} ({(stats['total_disagreements']/stats['total_comparisons']*100):.3f}%)")


if __name__ == '__main__':
    main()
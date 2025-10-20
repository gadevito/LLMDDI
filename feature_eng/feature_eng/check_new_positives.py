import argparse
import pickle
import csv
from typing import Dict, List, Set

def load_pickle_file(pickle_path: str) -> List[Dict]:
    """
    Carica il file pickle e restituisce i dati.
    
    Args:
        pickle_path: Path al file pickle
        
    Returns:
        Lista di dizionari contenenti drugbank_id e drug_interactions
    """
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data

def build_interaction_map(pickle_data: List[Dict]) -> Dict[str, Set[str]]:
    """
    Costruisce una mappa delle interazioni per lookup veloce.
    
    Args:
        pickle_data: Dati caricati dal pickle
        
    Returns:
        Dizionario con drugbank_id come chiave e set di drugbank_id con cui interagisce
    """
    interaction_map = {}
    
    for drug_entry in pickle_data:
        drugbank_id = drug_entry.get('drugbank_id')
        drug_interactions = drug_entry.get('drug_interactions', [])
        
        # Estrai tutti i drugbank_id dalle interazioni
        interacting_drugs = set()
        for interaction in drug_interactions:
            if 'drugbank_id' in interaction:
                interacting_drugs.add(interaction['drugbank_id'])
        
        interaction_map[drugbank_id] = interacting_drugs
    
    return interaction_map

def verify_interactions(csv_path: str, interaction_map: Dict[str, Set[str]], new_target_col, do_print) -> None:
    """
    Verifica se le coppie nel CSV esistono nel pickle e riporta i risultati.
    
    Args:
        csv_path: Path al file CSV
        interaction_map: Mappa delle interazioni dal pickle
    """
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        
        total_rows = 0
        found_in_pickle = 0
        not_found_in_pickle = 0
        drug1_not_exists = 0
        
        print("\n" + "="*80)
        print("VERIFICA INTERAZIONI")
        print("="*80 + "\n")
        
        for row in reader:
            total_rows += 1
            drug1 = row['drug1']
            drug2 = row['drug2']
            target = row['target']
            new_target = row[new_target_col]
            


            # Verifica se drug1 esiste nel pickle
            if drug1 not in interaction_map:
                drug1_not_exists += 1
                #print(f"⚠️  Row {total_rows}: drug1 '{drug1}' NON trovato nel pickle")
                continue
            
            # Verifica se drug2 è nelle interazioni di drug1
            if drug2 in interaction_map[drug1]:
                found_in_pickle += 1
                print(f"✅ Row {total_rows}: Interazione TROVATA - {drug1} -> {drug2}")
                #print(f"   Ground Truth: {target}, Prediction: {new_target}")
                if do_print:
                    print(f"PREDICTED BY DS {row['DS']}, GEMMA2 {row['GEMMA2']}, GPT-4o {row['GPT-4o']}, PHI3.5 {row['PHI3.5']}, QWEN2.5 {row['QWEN2.5']}")
            else:
                not_found_in_pickle += 1
                #print(f"❌ Row {total_rows}: Interazione NON trovata - {drug1} -> {drug2}")
                #print(f"   Ground Truth: {target}, Prediction: {new_target}")
        
        # Stampa statistiche finali
        print("\n" + "="*80)
        print("STATISTICHE FINALI")
        print("="*80)
        print(f"Totale righe processate: {total_rows}")
        print(f"Interazioni trovate nel pickle: {found_in_pickle} ({found_in_pickle/total_rows*100:.2f}%)")
        print(f"Interazioni NON trovate nel pickle: {not_found_in_pickle} ({not_found_in_pickle/total_rows*100:.2f}%)")
        print(f"drug1 non esistente nel pickle: {drug1_not_exists} ({drug1_not_exists/total_rows*100:.2f}%)")
        print("="*80 + "\n")

def main():
    # Setup argparse
    parser = argparse.ArgumentParser(
        description='Verifica interazioni tra file pickle e CSV'
    )
    parser.add_argument(
        '--pickle',
        type=str,
        required=True,
        help='Path al file pickle contenente le interazioni'
    )
    parser.add_argument(
        '--new_target',
        type=str,
        required=True,
        help='new_target col'
    )
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path al file CSV contenente drug1, drug2, target, new_target'
    )

    parser.add_argument('--print', action='store_true',
                       help='Print details')

    args = parser.parse_args()
    
    # Carica dati dal pickle
    print(f"Caricamento pickle file: {args.pickle}")
    pickle_data = load_pickle_file(args.pickle)
    print(f"✓ Caricati {len(pickle_data)} record dal pickle\n")
    
    # Costruisci mappa delle interazioni
    print("Costruzione mappa delle interazioni...")
    interaction_map = build_interaction_map(pickle_data)
    print(f"✓ Mappa creata con {len(interaction_map)} drug entries\n")
    
    # Verifica interazioni dal CSV
    print(f"Verifica interazioni dal CSV: {args.csv}")
    verify_interactions(args.csv, interaction_map, args.new_target, args.print)

if __name__ == "__main__":
    main()
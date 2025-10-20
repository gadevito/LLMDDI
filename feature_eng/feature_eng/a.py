"""
Script per contare i farmaci unici in file pickle.
Supporta liste di tuple (drug1, drug2, ...) e liste di dizionari con chiavi 'drug1' e 'drug2'.
"""

import pickle
import argparse
from pathlib import Path
from typing import Set, List, Union, Dict, Tuple


def extract_drugs_from_data(data: Union[List[Tuple], List[Dict]]) -> Set[str]:
    """
    Estrae i farmaci unici dai dati caricati dal file pickle.
    
    Args:
        data: Lista di tuple o lista di dizionari
        
    Returns:
        Set di farmaci unici
    """
    drugs = set()
    
    if not isinstance(data, list):
        print(f"Attenzione: il dato caricato non è una lista, tipo: {type(data)}")
        return drugs
    
    if not data:
        print("Attenzione: la lista è vuota")
        return drugs
    
    # Determina il tipo di struttura
    first_item = data[0]
    
    if isinstance(first_item, (tuple, list)):
        # Caso: lista di tuple/liste
        for item in data:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                if item[-1] == 0:
                    continue
                drugs.add(str(item[0]))
                drugs.add(str(item[1]))
            else:
                print(f"Attenzione: elemento non valido ignorato: {item}")
                
    elif isinstance(first_item, dict):
        # Caso: lista di dizionari
        for item in data:
            if isinstance(item, dict):
                if item['target'] == 0:
                    continue
                if 'drug1' in item:
                    drugs.add(str(item['drug1']))
                if 'drug2' in item:
                    drugs.add(str(item['drug2']))
            else:
                print(f"Attenzione: elemento non è un dizionario: {item}")
    else:
        print(f"Attenzione: tipo di dato non supportato: {type(first_item)}")
    
    return drugs


def process_pickle_file(filepath: Path) -> Set[str]:
    """
    Carica un file pickle ed estrae i farmaci unici.
    
    Args:
        filepath: Path del file pickle
        
    Returns:
        Set di farmaci unici
    """
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"\n{'='*60}")
        print(f"File: {filepath.name}")
        print(f"{'='*60}")
        
        drugs = extract_drugs_from_data(data)
        
        return drugs
        
    except FileNotFoundError:
        print(f"ERRORE: File non trovato: {filepath}")
        return set()
    except pickle.UnpicklingError:
        print(f"ERRORE: Impossibile leggere il file pickle: {filepath}")
        return set()
    except Exception as e:
        print(f"ERRORE: {type(e).__name__}: {e}")
        return set()


def main():
    parser = argparse.ArgumentParser(
        description='Conta i farmaci unici presenti in file pickle.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi di utilizzo:
  %(prog)s file1.pkl
  %(prog)s file1.pkl file2.pkl file3.pkl
  %(prog)s *.pkl
  
Formati supportati:
  - Lista di tuple: [(drug1, drug2), ...]
  - Lista di dizionari: [{'drug1': 'A', 'drug2': 'B'}, ...]
        """
    )
    
    parser.add_argument(
        'files',
        nargs='+',
        type=str,
        help='Uno o più file pickle da analizzare'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Mostra i farmaci trovati'
    )
    
    parser.add_argument(
        '--combined',
        '-c',
        action='store_true',
        help='Mostra anche il conteggio combinato di tutti i file'
    )
    
    args = parser.parse_args()
    
    all_drugs = set()
    
    for filepath_str in args.files:
        filepath = Path(filepath_str)
        
        drugs = process_pickle_file(filepath)
        
        print(f"Farmaci unici trovati: {len(drugs)}")
        
        if args.verbose and drugs:
            print(f"\nElenco farmaci:")
            for drug in sorted(drugs):
                print(f"  - {drug}")
        
        all_drugs.update(drugs)
    
    # Risultati combinati
    if args.combined and len(args.files) > 1:
        print(f"\n{'='*60}")
        print(f"TOTALE COMBINATO (tutti i file)")
        print(f"{'='*60}")
        print(f"Farmaci unici totali: {len(all_drugs)}")
        
        if args.verbose and all_drugs:
            print(f"\nElenco completo:")
            for drug in sorted(all_drugs):
                print(f"  - {drug}")


if __name__ == '__main__':
    main()

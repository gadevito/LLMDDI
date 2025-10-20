import argparse
import pickle
from typing import List, Dict, Any


def load_pickle(filepath: str) -> Any:
    """Carica un file pickle."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_pickle(data: Any, filepath: str) -> None:
    """Salva dati in un file pickle."""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def find_prediction(predictions: List[Dict], drug1: str, drug2: str) -> Any:
    """
    Trova la predizione per una coppia di farmaci.
    Cerca sia (drug1, drug2) che (drug2, drug1).
    """
    for pred in predictions:
        if ((pred['drug1'] == drug1 and pred['drug2'] == drug2) or
            (pred['drug1'] == drug2 and pred['drug2'] == drug1)):
            return pred.get('new_target')
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Aggiorna file pickle source con predizioni da altro file pickle'
    )
    parser.add_argument(
        '--source',
        required=True,
        help='Path al file pickle source (lista di dictionary con drug1, drug2, target, source_file)'
    )
    parser.add_argument(
        '--predictions',
        required=True,
        help='Path al file pickle con le predizioni (lista di dictionary con drug1, drug2, new_target)'
    )
    parser.add_argument(
        '--name',
        required=True,
        help='Nome breve del file per filtrare le coppie in source (valore di source_file)'
    )
    parser.add_argument(
        '--model',
        required=True,
        help='Nome del modello da usare come chiave per aggiungere le predizioni'
    )
    
    args = parser.parse_args()
    
    # Carica i file pickle
    print(f"Caricamento file source: {args.source}")
    source_data = load_pickle(args.source)
    
    print(f"Caricamento file predictions: {args.predictions}")
    predictions_data = load_pickle(args.predictions)
    
    # Validazione dei dati
    if not isinstance(source_data, list):
        raise ValueError("Il file source deve contenere una lista")
    if not isinstance(predictions_data, list):
        raise ValueError("Il file predictions deve contenere una lista")
    
    # Statistiche
    total_processed = 0
    total_updated = 0
    total_not_found = 0
    
    # Loop attraverso tutte le coppie nel file source
    for item in source_data:
        # Verifica che l'item sia un dictionary
        if not isinstance(item, dict):
            continue
        
        # Filtra per source_file
        if item.get('source_file') != args.name:
            continue
        
        total_processed += 1
        
        # Estrai drug1 e drug2
        drug1 = item.get('drug1')
        drug2 = item.get('drug2')
        
        if not drug1 or not drug2:
            print(f"Warning: Coppia mancante di drug1/drug2 in item: {item}")
            continue
        
        # Cerca la predizione
        new_target = find_prediction(predictions_data, drug1, drug2)
        
        if new_target is not None:
            # Aggiungi la chiave del modello con il valore new_target
            item[args.model] = new_target
            total_updated += 1
        else:
            total_not_found += 1
            print(f"Warning: Predizione non trovata per coppia ({drug1}, {drug2})")
    
    # Salva il file aggiornato
    print(f"\nSalvataggio file aggiornato: {args.source}")
    save_pickle(source_data, args.source)
    
    # Stampa statistiche
    print("\n" + "="*50)
    print("STATISTICHE")
    print("="*50)
    print(f"Coppie processate (con source_file={args.name}): {total_processed}")
    print(f"Coppie aggiornate: {total_updated}")
    print(f"Predizioni non trovate: {total_not_found}")
    print("="*50)
    print("\nOperazione completata con successo!")


if __name__ == "__main__":
    main()
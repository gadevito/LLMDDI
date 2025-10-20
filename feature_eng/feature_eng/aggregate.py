import argparse
import pickle
import pandas as pd
from pathlib import Path
import sys


def load_pickle(filepath):
    """Carica un file pickle e restituisce il contenuto."""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Errore nel caricamento di {filepath}: {e}")
        sys.exit(1)


def save_pickle(data, filepath):
    """Salva i dati in un file pickle."""
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Dataset salvato in: {filepath}")
    except Exception as e:
        print(f"Errore nel salvataggio di {filepath}: {e}")
        sys.exit(1)


def combine_datasets(group_files):
    """
    Combina tutti i datasets del gruppo in un unico DataFrame.
    
    Args:
        group_files: Lista di percorsi ai file pickle del gruppo
        
    Returns:
        DataFrame combinato
    """
    dataframes = []
    
    for filepath in group_files:
        print(f"Caricamento di {filepath}...")
        data = load_pickle(filepath)
        
        # Converti in DataFrame se è una lista di dizionari
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            print(f"Formato non supportato per {filepath}")
            sys.exit(1)
        
        # Verifica che contenga le colonne necessarie
        required_cols = ['drug1', 'drug2', 'target', 'new_target']
        if not all(col in df.columns for col in required_cols):
            print(f"Errore: {filepath} non contiene tutte le colonne richieste")
            print(f"Colonne presenti: {df.columns.tolist()}")
            sys.exit(1)
        
        dataframes.append(df)
    
    # Combina tutti i DataFrame
    combined = pd.concat(dataframes, ignore_index=True)
    
    # Rimuovi duplicati basati su drug1 e drug2 (mantieni il primo)
    combined = combined.drop_duplicates(subset=['drug1', 'drug2'], keep='first')
    
    print(f"Dataset combinato creato con {len(combined)} righe")
    return combined


def add_predictions_to_compare(compare_data, combined_df):
    """
    Aggiunge l'intera riga trovata in combined_df per ogni coppia in compare_data.
    
    Args:
        compare_data: Dataset da confrontare (lista di tuple/liste)
        combined_df: DataFrame combinato con le predizioni
        
    Returns:
        Lista di righe trovate (come dizionari)
    """
    print(f"Dataset compare con {len(compare_data)} righe")
    
    # Crea un dizionario per lookup veloce dal dataset combinato
    # Salva l'intera riga come dizionario
    combined_lookup = {}
    for _, row in combined_df.iterrows():
        key = (row['drug1'], row['drug2'])
        combined_lookup[key] = row.to_dict()
    
    # Trova le righe corrispondenti
    results = []
    matches = 0
    
    for t in compare_data:
        key = (t[0], t[1])
        if key in combined_lookup:
            # Aggiungi l'intera riga trovata
            results.append(combined_lookup[key])
            matches += 1
        else:
            print(f"NON TROVATO: {key}")
            # Opzionalmente puoi aggiungere None o un dizionario vuoto
            # results.append(None)
    
    print(f"Trovate {matches} corrispondenze su {len(compare_data)} righe")
    print(f"Righe senza corrispondenza: {len(compare_data) - matches}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Aggrega datasets pickle e confronta predizioni di interazioni farmacologiche',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempio di utilizzo:
  python script.py -g dataset1.pkl dataset2.pkl dataset3.pkl -c compare.pkl -o output.pkl
        """
    )
    
    parser.add_argument(
        '-g', '--group',
        nargs='+',
        required=True,
        help='Lista di file pickle da aggregare (gruppo)'
    )
    
    parser.add_argument(
        '-c', '--compare',
        required=True,
        help='File pickle da confrontare con il gruppo aggregato'
    )
    
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='File pickle di output con il dataset compare modificato'
    )
    
    args = parser.parse_args()
    
    # Verifica che i file esistano
    for filepath in args.group:
        if not Path(filepath).exists():
            print(f"Errore: il file {filepath} non esiste")
            sys.exit(1)
    
    if not Path(args.compare).exists():
        print(f"Errore: il file {args.compare} non esiste")
        sys.exit(1)
    
    # Step 1: Combina i datasets del gruppo
    print("\n=== STEP 1: Combinazione dei datasets del gruppo ===")
    combined_df = combine_datasets(args.group)
    
    # Step 2: Carica il dataset compare
    print("\n=== STEP 2: Caricamento del dataset compare ===")
    compare_data = load_pickle(args.compare)
    
    # Step 3: Aggiungi le predizioni al dataset compare
    print("\n=== STEP 3: Aggiunta delle predizioni al dataset compare ===")
    modified_compare = add_predictions_to_compare(compare_data, combined_df)
    
    # Step 4: Salva il risultato
    print("\n=== STEP 4: Salvataggio del risultato ===")
    
    save_pickle(modified_compare, args.output)
    
    print("\n=== COMPLETATO ===")
    print(f"Statistiche finali:")
    print(f"  - Datasets combinati: {len(args.group)}")
    print(f"  - Righe nel dataset combinato: {len(combined_df)}")
    print(f"  - Righe nel dataset compare (input): {len(compare_data)}")
    print(f"  - Righe nel dataset output: {len(modified_compare)}")


if __name__ == "__main__":
    main()
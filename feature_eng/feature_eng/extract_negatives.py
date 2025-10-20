import argparse
import pickle
import os
from pathlib import Path
from typing import List, Dict, Any


def load_pickle_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Carica un file pickle contenente una lista di dizionari.
    
    Args:
        filepath: Percorso del file pickle da caricare
        
    Returns:
        Lista di dizionari dal file pickle
    """
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"Errore nel caricamento di {filepath}: {e}")
        return []


def filter_and_consolidate(pickle_files: List[str]) -> List[Dict[str, Any]]:
    """
    Filtra i dati dai file pickle mantenendo solo quelli con target=0.
    
    Args:
        pickle_files: Lista di percorsi dei file pickle
        
    Returns:
        Lista consolidata di dizionari filtrati
    """
    consolidated_data = []
    
    for filepath in pickle_files:
        print(f"Elaborazione file: {filepath}")
        data = load_pickle_file(filepath)
        source_filename = os.path.basename(filepath)
        
        if source_filename.lower().find("cred") !=-1:
            SOURCE = "CREDIBLEMES"
        elif source_filename.lower().find("2011") !=-1:
            SOURCE = "DDI2011"
        elif source_filename.lower().find("2013") !=-1:
            SOURCE = "DDI2013"
        elif source_filename.lower().find("frenc") !=-1:
            SOURCE = "FRENCHDB"
        elif source_filename.lower().find("hep") !=-1:
            SOURCE = "HEP"
        elif source_filename.lower().find("hiv") !=-1:
            SOURCE = "HIV"
        elif source_filename.lower().find("keg") !=-1: 
            SOURCE = "KEGG"
        elif source_filename.lower().find("ndf") !=-1:
            SOURCE = "NDFR"
        elif source_filename.lower().find("onc") !=-1:
            SOURCE = "ONC"
        elif source_filename.lower().find("oscar") !=-1:
            SOURCE = "OSCAR"
        elif source_filename.lower().find("nlm") !=-1:
            SOURCE = "NLM"
        elif source_filename.lower().find("pkc") !=-1:
            SOURCE = "PKCORPUS"
        elif source_filename.lower().find("world") !=-1:
            SOURCE = "WORLDVISTA"
        else:
            print("NO SOURCE FOUND")
        # Filtra i record con target=0 e aggiungi il campo source_file
        for record in data:
            if isinstance(record, dict) and record.get('target') == 0:
                filtered_record = {
                    'drug1': record.get('drug1'),
                    'drug2': record.get('drug2'),
                    'target': record.get('target'),
                    'source_file': SOURCE
                }
                consolidated_data.append(filtered_record)
        
        print(f"  - Record con target=0 trovati: {sum(1 for r in data if isinstance(r, dict) and r.get('target') == 0)}")
    
    return consolidated_data


def save_pickle_file(data: List[Dict[str, Any]], output_path: str) -> None:
    """
    Salva i dati in un file pickle.
    
    Args:
        data: Lista di dizionari da salvare
        output_path: Percorso del file di output
    """
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"\nFile salvato con successo: {output_path}")
        print(f"Totale record salvati: {len(data)}")
    except Exception as e:
        print(f"Errore nel salvataggio del file: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Consolida file pickle filtrando record con target=0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempio di utilizzo:
  python script.py file1.pkl file2.pkl file3.pkl -o output.pkl
  python script.py *.pkl --output risultato.pkl
        """
    )
    
    parser.add_argument(
        'pickle_files',
        nargs='+',
        help='Lista di file pickle da elaborare'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='consolidated_target0.pkl',
        help='Nome del file pickle di output (default: consolidated_target0.pkl)'
    )
    
    args = parser.parse_args()
    
    # Verifica che i file esistano
    valid_files = []
    for filepath in args.pickle_files:
        if os.path.exists(filepath):
            valid_files.append(filepath)
        else:
            print(f"Attenzione: file non trovato - {filepath}")
    
    if not valid_files:
        print("Errore: nessun file valido trovato")
        return
    
    print(f"File da elaborare: {len(valid_files)}")
    print("-" * 60)
    
    # Elabora i file
    consolidated_data = filter_and_consolidate(valid_files)
    
    # Salva il risultato
    if consolidated_data:
        save_pickle_file(consolidated_data, args.output)
        
        # Stampa statistiche
        print("\n" + "=" * 60)
        print("STATISTICHE")
        print("=" * 60)
        print(f"File elaborati: {len(valid_files)}")
        print(f"Record totali con target=0: {len(consolidated_data)}")
        
        # Conta per file sorgente
        source_counts = {}
        for record in consolidated_data:
            source = record['source_file']
            source_counts[source] = source_counts.get(source, 0) + 1
        
        print("\nDistribuzione per file sorgente:")
        for source, count in sorted(source_counts.items()):
            print(f"  {source}: {count} record")
    else:
        print("\nNessun record con target=0 trovato nei file forniti")


if __name__ == '__main__':
    main()
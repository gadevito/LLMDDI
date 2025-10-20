import pickle
import csv
import argparse
import sys
from pathlib import Path


def load_pickle(pickle_file):
    """
    Carica i dati dal file pickle.
    
    Args:
        pickle_file (str): Percorso del file pickle
        
    Returns:
        list: Lista di dizionari
    """
    try:
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        
        if not isinstance(data, list):
            raise ValueError("Il file pickle non contiene una lista")
        
        if data and not isinstance(data[0], dict):
            raise ValueError("Il file pickle non contiene una lista di dizionari")
        
        return data
    
    except FileNotFoundError:
        print(f"Errore: File '{pickle_file}' non trovato", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Errore durante il caricamento del pickle: {e}", file=sys.stderr)
        sys.exit(1)


def pickle_to_csv(pickle_file, csv_file, delimiter=',', encoding='utf-8'):
    """
    Converte un file pickle in un file CSV.
    
    Args:
        pickle_file (str): Percorso del file pickle di input
        csv_file (str): Percorso del file CSV di output
        delimiter (str): Delimitatore per il CSV (default: ',')
        encoding (str): Encoding del file CSV (default: 'utf-8')
    """
    # Carica i dati dal pickle
    data = load_pickle(pickle_file)
    
    if not data:
        print("Attenzione: Il file pickle è vuoto", file=sys.stderr)
        # Crea comunque un file CSV vuoto
        with open(csv_file, 'w', newline='', encoding=encoding) as f:
            pass
        return
    
    # Estrai tutte le chiavi uniche da tutti i dizionari
    fieldnames = set()
    for item in data:
        fieldnames.update(item.keys())
    
    fieldnames = sorted(fieldnames)  # Ordina le colonne alfabeticamente
    
    # Scrivi il CSV
    try:
        with open(csv_file, 'w', newline='', encoding=encoding) as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            writer.writerows(data)
        
        print(f"Conversione completata con successo!")
        print(f"File CSV creato: {csv_file}")
        print(f"Numero di righe: {len(data)}")
        print(f"Numero di colonne: {len(fieldnames)}")
        
    except Exception as e:
        print(f"Errore durante la scrittura del CSV: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """
    Funzione principale con argparse.
    """
    parser = argparse.ArgumentParser(
        description='Converte un file pickle contenente liste di dizionari in un file CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi di utilizzo:
  %(prog)s input.pkl output.csv
  %(prog)s data.pickle result.csv --delimiter ";"
  %(prog)s file.pkl file.csv --encoding iso-8859-1
        """
    )
    
    parser.add_argument(
        'pickle_file',
        type=str,
        help='Percorso del file pickle di input'
    )
    
    parser.add_argument(
        'csv_file',
        type=str,
        help='Percorso del file CSV di output'
    )
    
    parser.add_argument(
        '-d', '--delimiter',
        type=str,
        default=',',
        help='Delimitatore per il CSV (default: ",")'
    )
    
    parser.add_argument(
        '-e', '--encoding',
        type=str,
        default='utf-8',
        help='Encoding del file CSV (default: "utf-8")'
    )
    
    parser.add_argument(
        '-v', '--version',
        action='version',
        version='%(prog)s 1.0'
    )
    
    args = parser.parse_args()
    
    # Verifica che il file pickle esista
    if not Path(args.pickle_file).exists():
        parser.error(f"Il file pickle '{args.pickle_file}' non esiste")
    
    # Esegui la conversione
    pickle_to_csv(
        args.pickle_file,
        args.csv_file,
        delimiter=args.delimiter,
        encoding=args.encoding
    )


if __name__ == '__main__':
    main()
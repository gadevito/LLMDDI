import argparse
import pickle
import sys
from pathlib import Path


def count_pickle_records(filepath):
    """
    Conta i record in un file pickle contenente una lista di tuple.
    
    Args:
        filepath: Percorso del file pickle
        
    Returns:
        Numero di record (tuple) nella lista
    """
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        tot = 0
        for t in data:
            print("==========")
            print(t[-1])
            # [0] drug1 [1] drug2 [3] smiles1 [4] smiles 2, [5] org1, [6] org2, [7]enc_drug1_organism [8] enc_drug2_organism
            # [9] target 1 [10] target 2
            print("==========")
            tot +=1
            if tot >10:
                break
        # Verifica che sia una lista
        if not isinstance(data, list):
            print(f"Errore: Il file non contiene una lista, ma un {type(data).__name__}", 
                  file=sys.stderr)
            return None
        
        # Conta i record
        num_records = len(data)
        
        # Verifica opzionale che contenga tuple
        if num_records > 0:
            non_tuple_count = sum(1 for item in data if not isinstance(item, tuple))
            if non_tuple_count > 0:
                print(f"Attenzione: {non_tuple_count} elementi non sono tuple", 
                      file=sys.stderr)
        
        return num_records
        
    except FileNotFoundError:
        print(f"Errore: File '{filepath}' non trovato", file=sys.stderr)
        return None
    except pickle.UnpicklingError:
        print(f"Errore: '{filepath}' non è un file pickle valido", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Errore durante la lettura del file: {e}", file=sys.stderr)
        return None


def main():
    """Funzione principale."""
    parser = argparse.ArgumentParser(
        description='Conta i record in un file pickle contenente una lista di tuple',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  %(prog)s data.pkl
  %(prog)s --verbose output.pickle
  %(prog)s -v -s /path/to/file.pkl
        """
    )
    
    parser.add_argument(
        'pickle_file',
        type=str,
        help='Percorso del file pickle da analizzare'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Mostra informazioni dettagliate'
    )
    
    parser.add_argument(
        '-s', '--show-sample',
        action='store_true',
        help='Mostra un campione del primo record'
    )
    
    args = parser.parse_args()
    
    # Verifica che il file esista
    filepath = Path(args.pickle_file)
    
    if args.verbose:
        print(f"Analizzando il file: {filepath}")
        print(f"Dimensione file: {filepath.stat().st_size:,} bytes" if filepath.exists() else "File non trovato")
        print("-" * 50)
    
    # Conta i record
    num_records = count_pickle_records(filepath)
    
    if num_records is not None:
        print(f"Numero di record: {num_records:,}")
        
        # Mostra un campione se richiesto
        if args.show_sample and num_records > 0:
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                print(f"\nPrimo record (campione):")
                print(f"  Tipo: {type(data[0]).__name__}")
                print(f"  Lunghezza: {len(data[0]) if isinstance(data[0], tuple) else 'N/A'}")
                print(f"  Contenuto: {data[0]}")
            except Exception as e:
                print(f"Impossibile mostrare il campione: {e}", file=sys.stderr)
        
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())
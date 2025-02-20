import argparse
import pickle
import csv

def extract_pharmacodynamics(pickle_file, output_csv):
    # Leggi i dati dal file pickle
    with open(pickle_file, 'rb') as f:
        drugs = pickle.load(f)
    
    # Crea un set per memorizzare i pharmacodynamics unici
    unique_pharmacodynamics = set()

    # Estrai i pharmacodynamics per ogni farmaco
    for drug in drugs:
        pharmacodynamics = drug.get('pharmacodynamics')
        if pharmacodynamics:
            unique_pharmacodynamics.add(pharmacodynamics)
    
    # Scrivi i risultati nel file CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Pharmacodynamics'])  # Intestazione
        for pd in unique_pharmacodynamics:
            csvwriter.writerow([pd])
    
    # Stampa il numero totale di pharmacodynamics unici
    print(f"Numero totale di pharmacodynamics unici: {len(unique_pharmacodynamics)}")

def main():
    parser = argparse.ArgumentParser(description='Estrai pharmacodynamics unici da un file pickle e scrivili in un file CSV.')
    parser.add_argument('pickle_file', type=str, help='Il percorso del file pickle di input.')
    parser.add_argument('output_csv', type=str, help='Il percorso del file CSV di output.')
    
    args = parser.parse_args()
    
    extract_pharmacodynamics(args.pickle_file, args.output_csv)

if __name__ == "__main__":
    main()
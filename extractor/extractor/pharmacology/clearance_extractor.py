import pickle
import csv
import argparse

def extract_clearance(pickle_file, output_csv):
    # Carica i dati dal file pickle
    with open(pickle_file, 'rb') as f:
        drugs = pickle.load(f)

    # Estrarre i valori di clearance unici
    unique_clearance = set()
    clearance_data = []

    for drug in drugs:
        clearance = drug.get('clearance')
        if clearance and clearance not in unique_clearance:
            unique_clearance.add(clearance)
            clearance_data.append({
                'drugbank_id': drug.get('drugbank_id'),
                'name': drug.get('name'),
                'clearance': clearance
            })

    # Scrivi i risultati in un file CSV
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['drugbank_id', 'name', 'clearance']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for data in clearance_data:
            writer.writerow(data)

    # Stampa il numero totale di clearance unici
    print(f"Numero totale di clearance unici: {len(unique_clearance)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Estrai i valori di clearance unici da un file pickle e scrivili in un file CSV.')
    parser.add_argument('pickle_file', type=str, help='Il percorso del file pickle di input.')
    parser.add_argument('output_csv', type=str, help='Il percorso del file CSV di output.')

    args = parser.parse_args()

    extract_clearance(args.pickle_file, args.output_csv)
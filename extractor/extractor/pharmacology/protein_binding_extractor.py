import pickle
import argparse
import csv

def main(pickle_file, output_csv):
    # Carica i dati dal file pickle
    with open(pickle_file, 'rb') as f:
        drugs = pickle.load(f)

    # Estrarre i valori unici di "protein-binding" per ciascun farmaco
    protein_binding_data = {}
    for drug in drugs:
        drug_id = drug.get('drugbank_id')
        protein_binding = drug.get('protein_binding')
        if drug_id and protein_binding:
            protein_binding_data[drug_id] = protein_binding

    # Scrivi i risultati in un file CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['drugbank_id', 'protein_binding']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for drug_id, protein_binding in protein_binding_data.items():
            writer.writerow({'drugbank_id': drug_id, 'protein_binding': protein_binding})

    # Stampa il numero totale di "protein-binding" unici
    total_protein_binding = len(protein_binding_data)
    print(f"Numero totale di 'protein-binding' unici: {total_protein_binding}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estrai valori unici di protein-binding da un file pickle e scrivi in un CSV.')
    parser.add_argument('pickle_file', help='Il file pickle da cui estrarre i dati.')
    parser.add_argument('output_csv', help='Il file CSV di output.')

    args = parser.parse_args()
    main(args.pickle_file, args.output_csv)
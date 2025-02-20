import pickle
import csv
import argparse

def extract_unique_metabolism(drugs):
    unique_metabolisms = set()
    for drug in drugs:
        metabolism = drug.get('metabolism')
        if metabolism:
            unique_metabolisms.add(metabolism)
    return unique_metabolisms

def main(pickle_file, output_csv):
    # Carica i dati dal file pickle
    with open(pickle_file, 'rb') as f:
        drugs = pickle.load(f)

    # Estrai i metabolismi unici
    unique_metabolisms = extract_unique_metabolism(drugs)

    # Scrivi i metabolismi unici in un file CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Metabolism'])  # intestazione del file CSV
        for metabolism in unique_metabolisms:
            writer.writerow([metabolism])

    # Stampa il numero totale di metabolismi unici
    print(f"Numero totale di metabolismi unici: {len(unique_metabolisms)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estrai metabolismi unici da un file pickle e scrivi in un file CSV.')
    parser.add_argument('pickle_file', type=str, help='Il file pickle di input contenente i dati dei farmaci.')
    parser.add_argument('output_csv', type=str, help='Il file CSV di output per scrivere i metabolismi unici.')

    args = parser.parse_args()

    main(args.pickle_file, args.output_csv)
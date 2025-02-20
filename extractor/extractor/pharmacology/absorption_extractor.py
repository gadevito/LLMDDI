import pickle
import csv
import argparse

def extract_absorption(pickle_file, output_csv):
    # Carica i dati dal file pickle
    with open(pickle_file, 'rb') as f:
        drugs = pickle.load(f)

    # Estrai gli absorption unici
    absorption_set = set()
    for drug in drugs:
        absorption = drug.get('absorption')
        if absorption:
            absorption_set.add(absorption)

    # Scrivi i risultati in un file CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Absorption'])
        for absorption in absorption_set:
            writer.writerow([absorption])

    # Stampa il numero totale di absorption
    print(f"Numero totale di absorption unici: {len(absorption_set)}")

def main():
    parser = argparse.ArgumentParser(description='Estrai absorption unici per ogni farmaco e scrivi in un file CSV.')
    parser.add_argument('pickle_file', type=str, help='Il file pickle di input contenente i dati dei farmaci.')
    parser.add_argument('output_csv', type=str, help='Il file CSV di output per i dati di absorption.')

    args = parser.parse_args()
    extract_absorption(args.pickle_file, args.output_csv)

if __name__ == '__main__':
    main()
import pickle
import csv
import argparse

def main(pickle_file, output_csv):
    # Carica i dati dal file pickle
    with open(pickle_file, 'rb') as f:
        drugs = pickle.load(f)

    # Estrai half-life unici per ogni farmaco
    unique_half_lives = {}
    for drug in drugs:
        drug_id = drug.get('drugbank_id')
        half_life = drug.get('half_life')

        # Aggiungi il valore di half-life se non è None e non duplicato
        if half_life and drug_id not in unique_half_lives:
            unique_half_lives[drug_id] = half_life

    # Scrivi i risultati in un file CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['drugbank_id', 'half_life']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for drug_id, half_life in unique_half_lives.items():
            writer.writerow({'drugbank_id': drug_id, 'half_life': half_life})

    # Stampa il numero totale di half-life
    total_half_lives = len(unique_half_lives)
    print(f"Numero totale di half-life unici: {total_half_lives}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estrai half-life unici da un file pickle e scrivi i risultati in un file CSV.')
    parser.add_argument('pickle_file', type=str, help='Il percorso del file pickle di input.')
    parser.add_argument('output_csv', type=str, help='Il percorso del file CSV di output.')
    args = parser.parse_args()

    main(args.pickle_file, args.output_csv)
import pickle
import csv
import argparse

def main(pickle_file, output_file):
    # Carica i dati dal file pickle
    with open(pickle_file, 'rb') as f:
        drugs = pickle.load(f)

    toxicity_set = set()

    # Estrai le informazioni di toxicity uniche per ogni farmaco
    for drug in drugs:
        toxicity = drug.get('toxicity')
        if toxicity:
            toxicity_set.add(toxicity)

    # Scrivi i risultati in un file CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['toxicity'])  # Scrivi l'intestazione
        for toxicity in toxicity_set:
            csvwriter.writerow([toxicity])

    # Stampa il numero totale di toxicity unici
    print(f"Numero totale di toxicity unici: {len(toxicity_set)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estrai le informazioni di toxicity unici per ogni farmaco.")
    parser.add_argument("pickle_file", help="Il file pickle da cui leggere i dati.")
    parser.add_argument("output_file", help="Il file CSV in cui scrivere i risultati.")

    args = parser.parse_args()

    main(args.pickle_file, args.output_file)
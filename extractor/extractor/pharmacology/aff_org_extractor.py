
import pickle
import csv
import argparse

def main(pickle_file, csv_file):
    # Carica i dati dal file pickle
    with open(pickle_file, 'rb') as f:
        drugs = pickle.load(f)

    # Inizializza un set per memorizzare gli organismi unici
    unique_affected_organisms = set()

    # Elabora ogni farmaco e raccogli gli organismi unici
    for drug in drugs:
        affected_organisms = drug.get('affected_organisms', [])
        
        # Aggiungi gli organismi al set
        unique_affected_organisms.update(affected_organisms)

    # Scrivi gli organismi unici nel file CSV
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Affected Organism'])  # Intestazione del CSV
        for organism in unique_affected_organisms:
            writer.writerow([organism])

    # Stampa il numero totale di organismi unici
    print(f"Numero totale di affected_organisms unici: {len(unique_affected_organisms)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Estrai organismi unici "affected_organisms" per ogni farmaco e scrivi i risultati in un file CSV.')
    parser.add_argument('pickle_file', help='Path al file pickle contenente i dati dei farmaci.')
    parser.add_argument('csv_file', help='Path del file CSV di output.')

    args = parser.parse_args()
    main(args.pickle_file, args.csv_file)
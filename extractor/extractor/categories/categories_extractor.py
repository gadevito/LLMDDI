import pickle
import csv
import argparse

def main(pickle_file, csv_file):
    # Carica i dati dal file pickle
    with open(pickle_file, 'rb') as f:
        drugs = pickle.load(f)
    
    # Inizializza le variabili per le statistiche
    total_categories = 0
    drugs_without_categories = 0
    unique_categories = set()

    # Itera su ogni farmaco per estrarre le categorie
    for drug in drugs:
        categories = drug.get('categories', [])
        if not categories:
            drugs_without_categories += 1
        else:
            total_categories += len(categories)
            for category in categories:
                unique_categories.add((category['mesh_id'], category['category']))

    # Calcola la media delle categorie per farmaco
    average_categories_per_drug = total_categories / len(drugs) if drugs else 0

    # Stampa le statistiche
    print(f"Numero totale di categories: {total_categories}")
    print(f"Numero di farmaci senza categories: {drugs_without_categories}")
    print(f"Media di categories per farmaco: {average_categories_per_drug:.2f}")

    # Scrivi le categorie uniche nel file CSV
    with open(csv_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['mesh_id', 'category'])
        for mesh_id, category in unique_categories:
            csvwriter.writerow([mesh_id, category])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a pickle file and extract unique categories.")
    parser.add_argument("pickle_file", help="Path to the pickle file containing drug data.")
    parser.add_argument("csv_file", help="Path to the output CSV file for storing unique categories.")
    args = parser.parse_args()
    
    main(args.pickle_file, args.csv_file)
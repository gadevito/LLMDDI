import pickle
import csv
import argparse

def main(pickle_file, output_csv_file):
    # Carica i dati dal file pickle
    with open(pickle_file, 'rb') as f:
        drugs = pickle.load(f)

    # Estrai "volume-of-distribution" unici per ogni farmaco
    volume_of_distribution_set = set()
    volume_of_distribution_data = []

    for drug in drugs:
        volume_of_distribution = drug.get('volume_of_distribution')
        if volume_of_distribution:
            # Aggiungi al set per ottenere valori unici
            volume_of_distribution_set.add(volume_of_distribution)
            # Aggiungi al dataset da scrivere nel CSV
            volume_of_distribution_data.append({
                'drugbank_id': drug.get('drugbank_id'),
                'volume_of_distribution': volume_of_distribution
            })

    # Scrivi i risultati in un file CSV
    with open(output_csv_file, 'w', newline='') as csvfile:
        fieldnames = ['drugbank_id', 'volume_of_distribution']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for data in volume_of_distribution_data:
            writer.writerow(data)

    # Stampa il numero totale di "volume-of-distribution"
    print(f"Total unique volume-of-distribution values: {len(volume_of_distribution_set)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract unique volume-of-distribution values from a pickle file and save to CSV.')
    parser.add_argument('pickle_file', type=str, help='Path to the input pickle file')
    parser.add_argument('output_csv_file', type=str, help='Path to the output CSV file')

    args = parser.parse_args()

    main(args.pickle_file, args.output_csv_file)
import pickle
import csv
import argparse

def extract_unique_routes_of_elimination(pickle_file, output_csv):
    # Carica i dati dal file pickle
    with open(pickle_file, 'rb') as f:
        drugs = pickle.load(f)

    # Crea un set per memorizzare route-of-elimination uniche
    unique_routes = set()

    # Itera su ogni farmaco e aggiungi la route-of-elimination al set
    for drug in drugs:
        route_of_elimination = drug.get('route_of_elimination')
        if route_of_elimination:
            unique_routes.add(route_of_elimination)

    # Scrivi i risultati nel file CSV
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Route of Elimination'])
        for route in unique_routes:
            csvwriter.writerow([route])

    # Stampa il numero totale di route-of-elimination
    print(f"Numero totale di route-of-elimination uniche: {len(unique_routes)}")

def main():
    parser = argparse.ArgumentParser(description="Estrai route-of-elimination unici da un file pickle e scrivi i risultati in un file CSV.")
    parser.add_argument('pickle_file', type=str, help='Percorso al file pickle contenente i dati dei farmaci.')
    parser.add_argument('output_csv', type=str, help='Percorso al file CSV di output per i risultati.')

    args = parser.parse_args()

    extract_unique_routes_of_elimination(args.pickle_file, args.output_csv)

if __name__ == '__main__':
    main()
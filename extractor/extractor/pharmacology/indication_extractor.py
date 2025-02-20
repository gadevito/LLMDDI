import pickle
import csv
import argparse
from collections import defaultdict

def extract_unique_indications(pickle_file, output_csv):
    # Load the pickled data
    with open(pickle_file, 'rb') as f:
        drugs = pickle.load(f)

    # Dictionary to store unique indications for each drug
    drug_indications = defaultdict(set)

    # Extract unique indications
    for drug in drugs:
        drug_id = drug.get('drugbank_id')
        indication = drug.get('indication')
        if drug_id and indication:
            drug_indications[drug_id].add(indication)

    # Write the results to a CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['DrugBank ID', 'Unique Indications'])

        for drug_id, indications in drug_indications.items():
            csvwriter.writerow([drug_id, '|'.join(indications)])

    # Print the total number of unique indications
    total_indications = sum(len(indications) for indications in drug_indications.values())
    print(f"Total unique indications: {total_indications}")

def main():
    parser = argparse.ArgumentParser(description='Extract unique indications for each drug and write to a CSV file.')
    parser.add_argument('pickle_file', type=str, help='Input pickle file containing drug data')
    parser.add_argument('output_csv', type=str, help='Output CSV file for unique indications')

    args = parser.parse_args()
    extract_unique_indications(args.pickle_file, args.output_csv)

if __name__ == '__main__':
    main()